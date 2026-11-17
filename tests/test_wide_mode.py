"""Wide accumulation (the MX engine): accumulate="wide", shift=.

The bit-exact source is Alosh's reference model, superfloat.gpu
test/v4/mxref.py.  CI cannot fetch that repo, so the outputs are checked in,
computed by mxref.Engine itself (not by sf):

* tests/data/mx_golden.json: v4.9.2, the default chip (32-bit accumulators),
  the ``mx`` section of tests/data/sf_golden_v4.9.2.json;
* tests/data/mx_golden_v4.8.1.json: the signed-off v4.8.1 chip (Q6.19, +-32),
  the same 240 inputs.
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

DATA = Path(__file__).parent / "data"
GOLDEN = json.loads((DATA / "mx_golden.json").read_text())
GOLDEN_V481 = json.loads((DATA / "mx_golden_v4.8.1.json").read_text())
FULL_V492 = json.loads((DATA / "sf_golden_v4.9.2.json").read_text())


@pytest.fixture(autouse=True)
def _quiet_underflow():
    """Random matrices underflow now and then; tests that check underflow set it."""
    with sf.errstate(under="ignore"):
        yield


# --- the model, against mxref ------------------------------------------------------------

@pytest.mark.parametrize("i", range(len(GOLDEN["cases"])))
def test_model_matches_mxref(i):
    """v4.9.2, the default: no mx argument."""
    c = GOLDEN["cases"][i]
    codes, _ = _linalg.matmul_wide_codes(FORMATS[c["bits"]], np.array(c["a"]), np.array(c["b"]),
                                         c["shift"])
    assert codes.tolist() == c["out"]


@pytest.mark.parametrize("i", range(len(GOLDEN_V481["cases"])))
def test_model_matches_mxref_v481(i):
    c = GOLDEN_V481["cases"][i]
    codes, _ = _linalg.matmul_wide_codes(FORMATS[c["bits"]], np.array(c["a"]), np.array(c["b"]),
                                         c["shift"], mx="v4.8.1")
    assert codes.tolist() == c["out"]


def test_the_golden_files():
    assert GOLDEN["target"] == "v4.9.2" and GOLDEN_V481["target"] == "v4.8.1"
    assert GOLDEN["cases"] == FULL_V492["mx"]["cases"]         # the mx section, unchanged
    same_inputs = all(x["a"] == y["a"] and x["b"] == y["b"] and x["shift"] == y["shift"]
                      for x, y in zip(GOLDEN["cases"], GOLDEN_V481["cases"]))
    assert same_inputs and len(GOLDEN["cases"]) == len(GOLDEN_V481["cases"]) == 240
    # the chips differ only near +-32 and in the SF16 product cut: 8 of the 240 cases
    differ = sum(x["out"] != y["out"] for x, y in zip(GOLDEN["cases"], GOLDEN_V481["cases"]))
    assert differ == 8


def test_units_per_chip():
    u, v = _linalg.MX_TARGETS["v4.9.2"], _linalg.MX_TARGETS["v4.8.1"]
    assert (u[16]["W"], u[16]["F"], u[8]["W"], u[8]["F"]) == (32, 22, 32, 14)        # Q10.22, Q18.14
    assert (v[16]["W"], v[16]["F"], v[8]["W"], v[8]["F"]) == (25, 19, 20, 14)        # Q6.19, Q6.14
    assert _linalg.MX_DEFAULT == "v4.9.2" and _linalg.MX_UNITS is u
    assert _linalg.mx_limit(FORMATS[16]) == 512 and _linalg.mx_limit(FORMATS[8]) == 131072
    assert _linalg.mx_limit(FORMATS[16], "v4.8.1") == 32 == _linalg.mx_limit(FORMATS[8], "v4.8.1")
    assert _linalg.mx_version(True) == "v4.9.2" and _linalg.mx_version(False) is None
    with pytest.raises(ValueError, match="unknown MX version"):
        _linalg.mx_version("v5")


@pytest.mark.parametrize("mx,golden", [("v4.9.2", GOLDEN), ("v4.8.1", GOLDEN_V481)])
def test_golden_covers_every_block_and_extreme(mx, golden):
    cs = golden["cases"]
    assert {c["bits"] for c in cs} >= {16, 12, 9, 8, 6, 4}
    assert {-32, 31} <= {c["shift"] for c in cs}
    clipped = 0
    for c in cs:
        _, ev = _linalg.matmul_wide_codes(FORMATS[c["bits"]], np.array(c["a"]), np.array(c["b"]),
                                          c["shift"], mx)
        clipped += int(ev["overflow"].sum())
    assert clipped > 0                    # saturation paths (POP at +-max, or the accumulator) are exercised


def test_accumulator_limit_per_chip():
    """80 terms of 0.99: a total of about 78 clips v4.8.1's +-32 accumulator,
    not v4.9.2's +-512; 600 terms (about 594) clip both."""
    f = FORMATS[16]
    a = np.full((1, 80), 32440)                   # 0.99 in sf16 codes
    b = np.full((80, 1), 32440)
    r92, e92 = _linalg.matmul_wide_codes(f, a, b, -7)
    r81, e81 = _linalg.matmul_wide_codes(f, a, b, -7, mx="v4.8.1")
    assert not e92["overflow"].any() and r92[0, 0] * 2 ** -15 * 2 ** 7 == pytest.approx(78.4, abs=0.1)
    assert e81["overflow"].all() and r81[0, 0] * 2 ** -15 * 2 ** 7 == pytest.approx(32, abs=2 ** -7)
    a, b = np.full((1, 600), 32440), np.full((600, 1), 32440)
    r, e = _linalg.matmul_wide_codes(f, a, b, -10)
    assert e["overflow"].all() and r[0, 0] * 2 ** -15 * 2 ** 10 == pytest.approx(512, abs=2 ** -4)
    assert _linalg.matmul_wide_acc_safe(f, np.full((1, 80), 32440), np.full((80, 1), 32440))
    assert not _linalg.matmul_wide_acc_safe(f, np.full((1, 80), 32440), np.full((80, 1), 32440), "v4.8.1")


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
    """The default target: Atreides v4.9.2, which has MX."""
    d = _device.get_device("atreides")
    assert d.mx == "v4.9.2"
    return d


@pytest.fixture
def v481dev():
    """The signed-off v4.8.1 chip, registered under its own name."""
    d = _device.MockAtreides(mx="v4.8.1", name="test-atreides-v4.8.1")
    _device.register_device(d)
    yield d
    _device._registry.pop(d.name, None)


@pytest.fixture
def nomx():
    d = _NoMX()
    _device.register_device(d)
    yield d
    _device._registry.pop(d.name, None)


@pytest.mark.parametrize("name", ["atreides", "atreides-flags", "atreides-sim", "atreides-hw"])
def test_every_atreides_target_has_mx(name):
    assert _device.get_device(name).mx == "v4.9.2"


def test_a_target_can_model_v481(v481dev):
    assert v481dev.mx == "v4.8.1" and _device.get_device("atreides").mx == "v4.9.2"
    with pytest.raises(ValueError, match="unknown MX version"):
        _device.MockAtreides(mx="v9")
    c = next(c for c, d in zip(GOLDEN["cases"], GOLDEN_V481["cases"]) if c["out"] != d["out"] and c["bits"] == 16)
    f = FORMATS[16]
    a, b = sf.from_codes(np.array(c["a"]), sf.sf16), sf.from_codes(np.array(c["b"]), sf.sf16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with sf.errstate(all="ignore"):
            on92 = sf.matmul(a.to("atreides"), b.to("atreides"), accumulate="wide", shift=c["shift"])
            on81 = sf.matmul(a.to(v481dev.name), b.to(v481dev.name), accumulate="wide", shift=c["shift"])
    assert on92._codes.tolist() == c["out"]
    v481 = next(d for d in GOLDEN_V481["cases"] if d["a"] == c["a"] and d["b"] == c["b"] and d["shift"] == c["shift"])
    assert on81._codes.tolist() == v481["out"] != c["out"]
    assert f.bits == 16


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
    # small values: the host proves the accumulator cannot clip, so the
    # output scan is a valid upper bound; underflow is invisible on the chip
    assert r.status.saturated == r.status.outputs_at_max and not r.status.exact
    assert r.status.underflow is None
    with sf.config(status="replay"):
        r2 = sf.matmul(a.to(mxdev.name), b.to(mxdev.name), accumulate="wide")
    assert r2.status.exact and r2.status.underflow is not None


def test_device_unknown_when_accumulator_can_clip(mxdev, v481dev):
    # v4.9.2: about 594 > 512 clips; v4.8.1: about 78 > 32 clips.  Either way the
    # clipped total, scaled down, is not at +-max, so the device cannot see it.
    for dev, k, limit, sh in ((mxdev, 600, 512, -10), (v481dev, 80, 32, -6)):
        big = sf.asarray(np.full((1, k), 0.99), dtype=sf.sf16)
        col = sf.asarray(np.full((k, 1), 0.99), dtype=sf.sf16)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sf.matmul(big.to(dev.name), col.to(dev.name), accumulate="wide", shift=sh)
        assert r.status.saturated is None
        assert float(r.numpy().reshape(())) == pytest.approx(limit * 2.0 ** sh, abs=2 ** -14)


def test_device_proves_safe_below_the_v492_limit(mxdev):
    # about 78: proved safe on v4.9.2, so the output scan is a valid upper bound
    big = sf.asarray(np.full((1, 80), 0.99), dtype=sf.sf16)
    col = sf.asarray(np.full((80, 1), 0.99), dtype=sf.sf16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sf.matmul(big.to(mxdev.name), col.to(mxdev.name), accumulate="wide", shift=-7)
    assert r.status.saturated == 0
    assert float(r.numpy().reshape(())) == pytest.approx(0.99 * 0.99 * 80 / 128, abs=2 ** -12)


def test_products_below_the_engine_resolution_vanish_without_an_event():
    # On the SF16 block a product's low partial-product columns are dropped
    # (below 6 on v4.9.2, below 9 on v4.8.1), so 2^-15 * 2^-15 contributes 0
    # and the total is exactly 0.
    # That is not "a nonzero total rounded to 0", so no underflow is raised.
    t = sf.array([2 ** -15] * 4, dtype=sf.sf16)
    with sf.errstate(under="raise"):
        assert float(sf.dot(t, t, accumulate="wide")) == 0.0


# --- kernels: sf.dot(..., accumulate="wide") traces to dot_wide ------------------------------

from sf.kernels import _cgen  # noqa: E402

BACKENDS = ["interp"] + (["c"] if _cgen.compiler() else [])


@pytest.fixture(params=BACKENDS)
def backend(request):
    with sf.config(kernel_backend=request.param):
        yield request.param


@sf.jit
def mm_wide(a_ptr, b_ptr, c_ptr, M, N, K: sf.constexpr, BM: sf.constexpr, BN: sf.constexpr,
            SHIFT: sf.constexpr):
    rm = sf.program_id(0) * BM + sf.arange(0, BM)
    rn = sf.program_id(1) * BN + sf.arange(0, BN)
    rk = sf.arange(0, K)
    a = sf.load(a_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M)
    b = sf.load(b_ptr + rk[:, None] * N + rn[None, :], mask=rn[None, :] < N)
    c = sf.dot(a, b, accumulate="wide", shift=SHIFT)
    sf.store(c_ptr + rm[:, None] * N + rn[None, :], c, mask=(rm[:, None] < M) & (rn[None, :] < N))


@sf.jit
def mm_setting(a_ptr, b_ptr, c_ptr, M: sf.constexpr, N: sf.constexpr, K: sf.constexpr):
    rm, rn, rk = sf.arange(0, M), sf.arange(0, N), sf.arange(0, K)
    a = sf.load(a_ptr + rm[:, None] * K + rk[None, :])
    b = sf.load(b_ptr + rk[:, None] * N + rn[None, :])
    sf.store(c_ptr + rm[:, None] * N + rn[None, :], a @ b)


def _host_and_dev(shape, seed, lo=-0.4, hi=0.4):
    M, N, K = shape
    rng = np.random.default_rng(seed)
    A = sf.asarray(rng.uniform(lo, hi, (M, K)), dtype=sf.sf16)
    B = sf.asarray(rng.uniform(lo, hi, (K, N)), dtype=sf.sf16)
    dev = sf.device("atreides-sim")
    return A, B, dev.upload(A), dev.upload(B), dev


@pytest.mark.parametrize("shape,shift", [((1, 1, 1), 0), ((4, 4, 8), 0), ((5, 7, 13), -1),
                                         ((9, 6, 40), -3), ((3, 8, 64), 2)])
def test_kernel_dot_wide_matches_host(backend, shape, shift):
    M, N, K = shape
    A, B, dA, dB, dev = _host_and_dev(shape, M * 100 + K)
    with sf.errstate(all="ignore"):
        ref = sf.matmul(A, B, accumulate="wide", shift=shift)
        C = dev.alloc((M, N))
        r = mm_wide[(sf.cdiv(M, 4), sf.cdiv(N, 4))](dA, dB, C, M, N, K=K, BM=4, BN=4, SHIFT=shift)
    assert np.array_equal(C.codes, ref.codes)
    assert r.backend == backend and r.counters.macs > 0
    assert any("dot_wide" in ir.dump() for ir in mm_wide.cache.values())


def test_kernel_dot_wide_events_agree_across_backends():
    # totals near 70: every output saturates (at v4.8.1's +-32 accumulator, or at
    # POP on v4.9.2, where 70 x 2^-6 > 1)
    shape = (4, 4, 80)
    A, B, dA, dB, dev = _host_and_dev(shape, 7, 0.85, 0.95)
    results = {}
    for be in BACKENDS:
        with sf.config(kernel_backend=be), sf.errstate(all="ignore"):
            C = dev.alloc((4, 4))
            r = mm_wide[(1, 1)](dA, dB, C, 4, 4, K=80, BM=4, BN=4, SHIFT=-6)
        results[be] = (C.codes.copy(), r.status.saturated, r.status.underflow)
    with sf.errstate(all="ignore"):
        ref = sf.matmul(A, B, accumulate="wide", shift=-6)
    for codes, sat, under in results.values():
        assert np.array_equal(codes, ref.codes)
        assert sat == 16 and under == 0                     # exact counts on the simulator
    assert len({(s, u) for _, s, u in results.values()}) == 1


def test_kernel_trace_follows_setting_and_cache_keeps_them_apart(backend):
    shape = (4, 4, 16)
    A, B, dA, dB, dev = _host_and_dev(shape, 11)
    with sf.errstate(all="ignore"):
        C1, C2 = dev.alloc((4, 4)), dev.alloc((4, 4))
        mm_setting[(1,)](dA, dB, C1, M=4, N=4, K=16)              # traced as chain
        with sf.accumulate("wide"):
            mm_setting[(1,)](dA, dB, C2, M=4, N=4, K=16)          # must re-trace as wide
        chain = A @ B
        wide = sf.matmul(A, B, accumulate="wide")
    assert np.array_equal(C1.codes, chain.codes)
    assert np.array_equal(C2.codes, wide.codes)
    assert not np.array_equal(chain.codes, wide.codes)


def test_kernel_shift_validation():
    A, B, dA, dB, dev = _host_and_dev((4, 4, 8), 3)
    with pytest.raises(ValueError):
        mm_wide[(1, 1)](dA, dB, dev.alloc((4, 4)), 4, 4, K=8, BM=4, BN=4, SHIFT=40)


@pytest.mark.parametrize("be", BACKENDS)
def test_kernel_dot_wide_on_a_v481_simulator(be):
    """A simulator set to the signed-off chip runs v4.8.1's units in both
    backends; the default simulator runs v4.9.2's.  Totals near 70 show it:
    v4.8.1 clips them at 32, v4.9.2 keeps them."""
    from sf.kernels import _runtime

    class SimV481(_runtime.SimulatorAtreides):
        name = "test-sim-v4.8.1"
        mx = "v4.8.1"

    dev81 = SimV481()
    _device.register_device(dev81)
    try:
        rng = np.random.default_rng(3)
        A = sf.asarray(rng.uniform(0.85, 0.95, (4, 80)), dtype=sf.sf16)
        B = sf.asarray(rng.uniform(0.85, 0.95, (80, 4)), dtype=sf.sf16)
        out = {}
        for dev in (dev81, sf.device("atreides-sim")):
            with sf.config(kernel_backend=be), sf.errstate(all="ignore"):
                C = dev.alloc((4, 4))
                mm_wide[(1, 1)](dev.upload(A), dev.upload(B), C, 4, 4, K=80, BM=4, BN=4, SHIFT=-7)
            out[dev.mx] = C.codes.copy()
        for mx, codes in out.items():
            want, _ = _linalg.matmul_wide_codes(FORMATS[16], A.codes, B.codes, -7, mx)
            assert np.array_equal(codes, want), mx
        assert not np.array_equal(out["v4.8.1"], out["v4.9.2"])
        assert (out["v4.8.1"] == 1 << 13).all()            # 32 x 2^-7 = 0.25: clipped at 32
    finally:
        _device._registry.pop(dev81.name, None)


def test_c_backend_takes_the_targets_units():
    shape = (2, 2, 8)
    A, B, dA, dB, dev = _host_and_dev(shape, 1)
    with sf.errstate(all="ignore"):
        C = dev.alloc((2, 2))
        mm_wide[(1, 1)](dA, dB, C, 2, 2, K=8, BM=2, BN=2, SHIFT=0)
    # this test's own trace: other tests trace mm_wide at other sizes and widths, in any order under xdist
    ir = next(ir for ir in mm_wide.cache.values()
              if "dot_wide" in ir.dump() and ir.constexpr.get("K") == 8 and ir.constexpr.get("BM") == 2)
    assert "0, 32, 22, 6, 80, &le" in _cgen.emit_c(ir)
    assert "0, 25, 19, 9, 1024, &le" in _cgen.emit_c(ir, "v4.8.1")
