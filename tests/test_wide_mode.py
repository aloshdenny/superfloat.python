"""Wide accumulation (the v4.8.1 MX engine): accumulate="wide", shift=.

The bit-exact source is Alosh's reference model, superfloat.gpu branch
v4.8.1-openframe test/v4/mxref.py.  CI cannot fetch that repo, so
tests/data/mx_golden.json holds outputs computed by mxref.Engine itself (not
by sf); the file records the sha256 of the mxref.py it came from.
"""
import asyncio
import json
import threading
import warnings
from pathlib import Path

import numpy as np
import pytest

import sf
from sf import _device, _linalg
from sf._format import FORMATS

GOLDEN = json.loads((Path(__file__).parent / "data" / "mx_golden.json").read_text())


@pytest.fixture(autouse=True)
def _quiet_underflow():
    """Random matrices underflow now and then; tests that check underflow set it."""
    with sf.errstate(under="ignore"):
        yield


# --- the model, against mxref ------------------------------------------------------------

@pytest.mark.parametrize("i", range(len(GOLDEN["cases"])))
def test_model_matches_mxref(i):
    c = GOLDEN["cases"][i]
    codes, _ = _linalg.matmul_wide_codes(FORMATS[c["bits"]], np.array(c["a"]), np.array(c["b"]),
                                         c["shift"])
    assert codes.tolist() == c["out"]


def test_golden_covers_every_block_and_extreme():
    cs = GOLDEN["cases"]
    assert {c["bits"] for c in cs} >= {16, 12, 9, 8, 6, 4}
    assert {-32, 31} <= {c["shift"] for c in cs}
    clipped = 0
    for c in cs:
        _, ev = _linalg.matmul_wide_codes(FORMATS[c["bits"]], np.array(c["a"]), np.array(c["b"]),
                                          c["shift"])
        clipped += int(ev["overflow"].sum())
    assert clipped > 0                    # saturation paths are exercised


# --- Alosh's examples (sf wide mode (MX).md) ----------------------------------------------

def test_doc_example_chain_vs_wide():
    a = sf.array([0.75, 0.75, -0.75], dtype=sf.sf16)
    b = sf.array([0.9, 0.9, 0.9], dtype=sf.sf16)
    with sf.errstate(over="ignore"):
        assert float(sf.dot(a, b)) == pytest.approx(0.32498169, abs=1e-8)
    assert float(sf.dot(a, b, accumulate="wide")) == pytest.approx(0.67498779, abs=1e-8)


def test_doc_example_shift():
    a = sf.array([0.5] * 4, dtype=sf.sf16)
    b = sf.array([0.75] * 4, dtype=sf.sf16)
    with sf.errstate(over="ignore"):
        assert float(sf.dot(a, b, accumulate="wide")) == pytest.approx(0.99996948, abs=1e-8)
    assert float(sf.dot(a, b, accumulate="wide", shift=-2)) == 0.375
    with pytest.raises(sf.SFOverflowError):
        sf.dot(a, b, accumulate="wide")           # 1.5 does not fit: overflow at POP


# --- API -------------------------------------------------------------------------------------

def _mats(seed=0, bits=16, m=3, k=19, n=5):
    rng = np.random.default_rng(seed)
    a = sf.asarray(rng.uniform(-0.3, 0.3, (m, k)), dtype=sf.dtypes[bits - 4])
    b = sf.asarray(rng.uniform(-0.3, 0.3, (k, n)), dtype=sf.dtypes[bits - 4])
    return a, b


def test_matmul_wide_equals_model_and_chain_unchanged():
    a, b = _mats()
    fmt = FORMATS[16]
    want, _ = _linalg.matmul_wide_codes(fmt, a._codes, b._codes, 0)
    assert sf.matmul(a, b, accumulate="wide")._codes.tolist() == want.tolist()
    chain, _ = _linalg.matmul_codes(fmt, a._codes, b._codes)
    assert (a @ b)._codes.tolist() == chain.tolist()                   # default stays chain
    assert sf.matmul(a, b, accumulate="chain")._codes.tolist() == chain.tolist()


def test_operator_follows_setting_and_scopes():
    a, b = _mats(1)
    wide = sf.matmul(a, b, accumulate="wide")._codes.tolist()
    assert sf.getaccumulate() == "chain"
    with sf.accumulate("wide"):
        assert sf.getaccumulate() == "wide"
        assert (a @ b)._codes.tolist() == wide
        with sf.accumulate("chain"):
            assert (a @ b)._codes.tolist() != wide
        assert sf.matmul(a, b, accumulate="chain")._codes.tolist() != wide   # call wins
    assert sf.getaccumulate() == "chain"


def test_decorator_plain_and_async():
    a, b = _mats(2)
    wide = sf.matmul(a, b, accumulate="wide")._codes.tolist()

    @sf.accumulate("wide")
    def f():
        return (a @ b)._codes.tolist(), sf.getaccumulate()

    @sf.accumulate("wide")
    async def g():
        return sf.getaccumulate()

    assert f() == (wide, "wide")
    assert asyncio.run(g()) == "wide"
    assert sf.getaccumulate() == "chain"
    with pytest.raises(TypeError):
        @sf.accumulate("wide")
        def gen():
            yield 1


def test_setaccumulate_returns_previous_and_is_thread_local():
    old = sf.setaccumulate("wide")
    try:
        assert old == "chain" and sf.getaccumulate() == "wide"
        seen = []
        t = threading.Thread(target=lambda: seen.append(sf.getaccumulate()))
        t.start(); t.join()
        assert seen == ["chain"]                  # a new thread starts at the default
    finally:
        sf.setaccumulate(old)
    assert sf.getaccumulate() == "chain"


def test_validation():
    a, b = _mats(3)
    with pytest.raises(ValueError):
        sf.matmul(a, b, accumulate="fast")
    with pytest.raises(ValueError):
        sf.accumulate("fast")
    with pytest.raises(ValueError):
        sf.matmul(a, b, shift=1)                  # shift needs wide mode
    for bad in (-33, 32):
        with pytest.raises(ValueError):
            sf.matmul(a, b, accumulate="wide", shift=bad)
    with pytest.raises(TypeError):
        sf.matmul(a, b, accumulate="wide", shift=0.5)
    with sf.errstate(over="ignore"):
        sf.matmul(a, b, accumulate="wide", shift=-32)
        sf.matmul(a, b, accumulate="wide", shift=31)


def test_addmm_wide_then_one_saturating_add():
    a, b = _mats(4)
    d = sf.asarray(np.full((3, 5), 0.25), dtype=sf.sf16)
    want = sf.matmul(a, b, accumulate="wide") + d
    assert sf.addmm(a, b, d, accumulate="wide")._codes.tolist() == want._codes.tolist()


def test_batched_and_1d():
    rng = np.random.default_rng(5)
    a = sf.asarray(rng.uniform(-0.2, 0.2, (2, 3, 10)), dtype=sf.sf16)
    b = sf.asarray(rng.uniform(-0.2, 0.2, (2, 10, 4)), dtype=sf.sf16)
    r = sf.matmul(a, b, accumulate="wide")
    for i in range(2):
        assert r._codes[i].tolist() == sf.matmul(a[i], b[i], accumulate="wide")._codes.tolist()
    v = sf.asarray(rng.uniform(-0.2, 0.2, 10), dtype=sf.sf16)
    assert sf.matmul(a[0], v, accumulate="wide").shape == (3,)


@pytest.mark.parametrize("bits", [16, 12, 9, 8, 6, 4])
def test_every_width_runs_on_its_block(bits):
    a, b = _mats(6, bits=bits)
    r = sf.matmul(a, b, accumulate="wide")
    assert r.dtype is sf.dtypes[bits - 4]
    assert _linalg.mx_block(FORMATS[bits]) == (16 if bits >= 9 else 8)


def test_wide_mode_leaves_other_ops_alone():
    x = sf.array([0.5, 0.75, -0.25], dtype=sf.sf16)
    with sf.errstate(over="ignore"):
        base = (sf.sum(x), x * x, x + x, sf.fma(x, x, x))
        with sf.accumulate("wide"):
            again = (sf.sum(x), x * x, x + x, sf.fma(x, x, x))
    for p, q in zip(base, again):
        assert np.asarray(getattr(p, "_codes", p)).tolist() == np.asarray(getattr(q, "_codes", q)).tolist()


def test_underflow_means_total_rounded_to_zero():
    # 2^-10 * 2^-10 = 2^-20: the engine keeps it (one unit of 2^-19 after its
    # rounding), but the total rounds to 0 at POP.  (2^-15 * 2^-15 would vanish
    # inside the engine instead, so the total is exactly 0: not an underflow.)
    t = sf.array([2 ** -10], dtype=sf.sf16)
    with sf.errstate(under="warn"), pytest.warns(sf.SFUnderflowWarning):
        assert float(sf.dot(t, t, accumulate="wide")) == 0.0
    with sf.errstate(under="raise"), pytest.raises(sf.SFUnderflowError):
        sf.dot(t, t, accumulate="wide")


def test_wide_never_returns_negative_zero():
    a = sf.array([-(2 ** -15)], dtype=sf.sf16)
    b = sf.array([2 ** -15], dtype=sf.sf16)
    with sf.errstate(under="ignore"):
        r = sf.dot(a, b, accumulate="wide")
    assert r._code == 0


# --- device ----------------------------------------------------------------------------------

class _NoMX(_device.MockAtreides):
    """A target without the MX engine (all registered Atreides targets have it)."""
    name = "test-no-mx"
    mx = False


@pytest.fixture
def mxdev():
    """The default target: Atreides v4.8.1, which has MX."""
    d = _device.get_device("atreides")
    assert d.mx
    return d


@pytest.fixture
def nomx():
    d = _NoMX()
    _device.register_device(d)
    yield d
    _device._registry.pop(d.name, None)


@pytest.mark.parametrize("name", ["atreides", "atreides-flags", "atreides-sim", "atreides-hw"])
def test_every_atreides_target_has_mx(name):
    assert _device.get_device(name).mx


def test_target_without_mx_refuses_wide(nomx):
    a, b = _mats(7)
    ha, hb = a.to(nomx.name), b.to(nomx.name)
    with pytest.raises(sf.SFUnsupportedError, match="MX"):
        sf.matmul(ha, hb, accumulate="wide")
    assert nomx.launches == 0                             # never fell back, never launched
    sf.matmul(ha, hb)                                     # chain still works there
    assert nomx.launches == 1


def test_device_wide_is_bit_identical_to_host(mxdev):
    a, b = _mats(8)
    host = sf.matmul(a, b, accumulate="wide", shift=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dev = sf.matmul(a.to(mxdev.name), b.to(mxdev.name), accumulate="wide", shift=1)
    assert dev.device == "atreides"
    assert dev._codes.tolist() == host._codes.tolist()


def test_device_status_rules(mxdev):
    a, b = _mats(9)
    r = sf.matmul(a.to(mxdev.name), b.to(mxdev.name), accumulate="wide")
    # small values: the host proves the +-32 accumulator cannot clip, so the
    # output scan is a valid upper bound; underflow is invisible on the chip
    assert r.status.saturated == r.status.outputs_at_max and not r.status.exact
    assert r.status.underflow is None
    with sf.config(status="replay"):
        r2 = sf.matmul(a.to(mxdev.name), b.to(mxdev.name), accumulate="wide")
    assert r2.status.exact and r2.status.underflow is not None


def test_device_unknown_when_accumulator_can_clip(mxdev):
    big = sf.asarray(np.full((1, 80), 0.99), dtype=sf.sf16)
    col = sf.asarray(np.full((80, 1), 0.99), dtype=sf.sf16)       # total ~78 > 32
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sf.matmul(big.to(mxdev.name), col.to(mxdev.name), accumulate="wide", shift=-6)
    assert r.status.saturated is None                       # clipped at 32 then scaled: not at max
    assert float(r.numpy().reshape(())) == pytest.approx(32 * 2 ** -6, abs=2 ** -14)


def test_products_below_the_engine_resolution_vanish_without_an_event():
    # On the SF16 block a product's partial-product bits below column 9 are
    # dropped, so 2^-15 * 2^-15 contributes 0 and the total is exactly 0.
    # That is not "a nonzero total rounded to 0", so no underflow is raised.
    t = sf.array([2 ** -15] * 4, dtype=sf.sf16)
    with sf.errstate(under="raise"):
        assert float(sf.dot(t, t, accumulate="wide")) == 0.0
