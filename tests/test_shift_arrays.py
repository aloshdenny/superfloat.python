"""POP's shift per output (Alosh's build order, item 1).

The MX engine's POP takes one shift per column, so a wide matmul takes an int
or an int array broadcastable to the result: one shift per token and column,
as the wide-mode heuristic picks them.  Checked against the 13 matmul cases of
the heuristic section in ``sf_golden_v4.9.2.json`` (``pop_q``).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sf
import sf.kernels  # noqa: F401  (registers atreides-sim)
from sf import _linalg
from sf._format import FORMATS

GOLDEN = json.loads((Path(__file__).parent / "data" / "sf_golden_v4.9.2.json").read_text())
CASES = GOLDEN["heuristic"]["matmul"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_heuristic_golden_pop(case):
    fmt = FORMATS[case["bits"]]
    a, w = np.array(case["a_codes"]), np.array(case["w_codes_KxN"])
    shift = np.array(case["shift"])
    codes, ev = _linalg.matmul_wide_codes(fmt, a, w, shift, case["target"])
    assert codes.tolist() == case["pop_q"]
    assert np.asarray(ev["overflow"]).astype(bool).tolist() == [[bool(s) for s in r] for r in case["pop_saturated"]]


def test_the_golden_set_has_varied_and_saturating_shifts():
    assert len(CASES) == 13
    assert any(len({s for r in c["shift"] for s in r}) > 1 for c in CASES)
    assert any(any(any(r) for r in c["pop_saturated"]) for c in CASES)


@pytest.mark.parametrize("case", [c for c in CASES if c["bits"] in (8, 16)][:4], ids=lambda c: c["id"])
def test_heuristic_golden_through_sf_matmul(case):
    dt = {16: sf.sf16, 8: sf.sf8}[case["bits"]]
    fmt = FORMATS[case["bits"]]
    a = sf.SFArray._wrap(np.array(case["a_codes"]), dt)
    w = sf.SFArray._wrap(np.array(case["w_codes_KxN"]), dt)
    with sf.errstate(all="ignore"):
        y = sf.matmul(a, w, accumulate="wide", shift=np.array(case["shift"]))
    assert y.codes.tolist() == case["pop_q"]
    assert fmt.bits == case["bits"]


def _ab(m=3, k=64, n=8, device="host"):
    rng = np.random.default_rng(0)
    a = sf.array(rng.uniform(-.5, .5, (m, k)), dtype=sf.sf16, device=device)
    b = sf.array(rng.uniform(-.5, .5, (k, n)), dtype=sf.sf16, device=device)
    return a, b


def _rowwise(a, b, shift_col):
    rows = [sf.matmul(a[i:i + 1], b, accumulate="wide", shift=int(s)).codes[0]
            for i, s in enumerate(shift_col)]
    return np.stack(rows)


def test_a_shift_per_token_equals_one_call_per_row():
    a, b = _ab()
    s = np.array([[-4], [-3], [-2]])
    with sf.errstate(all="ignore"):
        assert np.array_equal(sf.matmul(a, b, accumulate="wide", shift=s).codes, _rowwise(a, b, s[:, 0]))


def test_a_shift_per_column_and_a_full_grid():
    a, b = _ab()
    cols = np.arange(-5, 3)
    grid = np.arange(24).reshape(3, 8) % 7 - 5
    with sf.errstate(all="ignore"):
        yc = sf.matmul(a, b, accumulate="wide", shift=cols)
        yg = sf.matmul(a, b, accumulate="wide", shift=grid)
        for j, s in enumerate(cols):
            assert np.array_equal(yc.codes[:, j], sf.matmul(a, b[:, j:j + 1], accumulate="wide", shift=int(s)).codes[:, 0])
        for i in range(3):
            for j in range(8):
                one = sf.matmul(a[i:i + 1], b[:, j:j + 1], accumulate="wide", shift=int(grid[i, j]))
                assert yg.codes[i, j] == one.codes[0, 0]


def test_an_all_equal_array_is_the_same_as_the_int():
    a, b = _ab()
    with sf.errstate(all="ignore"):
        assert np.array_equal(sf.matmul(a, b, accumulate="wide", shift=np.full((3, 8), -3)).codes,
                              sf.matmul(a, b, accumulate="wide", shift=-3).codes)


def test_one_d_operands_and_batches_broadcast_against_the_result_shape():
    a, b = _ab()
    with sf.errstate(all="ignore"):
        v = sf.matmul(a[0], b, accumulate="wide", shift=np.arange(-4, 4))          # result (8,)
        assert v.shape == (8,)
        assert np.array_equal(v.codes, sf.matmul(a[:1], b, accumulate="wide", shift=np.arange(-4, 4)).codes[0])
        d = sf.dot(a[0], b[:, 0], accumulate="wide", shift=np.array(-3))
        assert d == sf.dot(a[0], b[:, 0], accumulate="wide", shift=-3)
        ab = sf.SFArray._wrap(np.stack([a.codes] * 2), sf.sf16)
        y = sf.matmul(ab, b, accumulate="wide", shift=np.array([[[-4]], [[-2]]]))   # one per batch
        assert np.array_equal(y.codes[1], sf.matmul(a, b, accumulate="wide", shift=-2).codes)


def test_addmm_takes_an_array_shift():
    a, b = _ab()
    d = sf.zeros((3, 8), dtype=sf.sf16)
    s = np.array([[-4], [-3], [-2]])
    with sf.errstate(all="ignore"):
        assert np.array_equal(sf.addmm(a, b, d, accumulate="wide", shift=s).codes,
                              sf.matmul(a, b, accumulate="wide", shift=s).codes)


@pytest.mark.parametrize("device", ["atreides", "atreides-sim"])
def test_targets_match_the_host(device):
    a, b = _ab()
    s = np.arange(24).reshape(3, 8) % 5 - 4
    with sf.errstate(all="ignore"), sf.config(preflight="ignore"):
        want = sf.matmul(a, b, accumulate="wide", shift=s).codes
        got = sf.matmul(a.to(device), b.to(device), accumulate="wide", shift=s)
    assert got.device == device and np.array_equal(got.codes, want)


def test_bad_shifts_are_refused_before_anything_runs():
    a, b = _ab()
    with pytest.raises(sf.SFShapeError, match="does not broadcast to the result shape"):
        sf.matmul(a, b, accumulate="wide", shift=np.zeros((2, 8), int))
    with pytest.raises(ValueError, match="between -32 and 31, got 40"):
        sf.matmul(a, b, accumulate="wide", shift=np.array([40]))
    with pytest.raises(TypeError, match="array of float64"):
        sf.matmul(a, b, accumulate="wide", shift=np.array([1.0]))
    with pytest.raises(TypeError, match="bool"):
        sf.matmul(a, b, accumulate="wide", shift=np.array([True]))
    with pytest.raises(ValueError, match="only applies"):
        sf.matmul(a, b, accumulate="chain", shift=np.array([1]))


def test_check_shift():
    assert _linalg.check_shift(None) == 0 and _linalg.check_shift(np.array(5)) == 5
    out = _linalg.check_shift([[-32, 31]])
    assert out.dtype == np.int64 and out.tolist() == [[-32, 31]]
    with pytest.raises(TypeError, match="int here"):
        _linalg.check_shift([1], allow_array=False)


def test_kernels_take_a_shift_block_only_as_a_traced_value():
    # since 0.14.0 kernels take a shift per output (tests/test_kernel_dot_wide_shift.py), but
    # only as a traced int block: a NumPy array is not traced, so it must come in as an argument
    from sf.kernels import _lang
    with pytest.raises(sf.SFCompileError, match="int buffer argument"):
        _lang.dot(None, None, accumulate="wide", shift=np.array([1, 2]))
