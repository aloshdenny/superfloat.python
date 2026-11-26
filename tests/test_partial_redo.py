"""A redo on a target relaunches only the rows and columns that hold a redone output.

An output of a wide matmul depends only on its own row of A, its own column of B and its
shift, so relaunching a gathered window (the rows with a redo x the columns with a redo) gives
the bits of relaunching the whole matmul.  These pin that, with and without a bias, for a batch,
and on the chip model when superfloat.sim is there.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

import sf
import sf.kernels  # noqa: F401  (registers atreides)

pytest.importorskip("torch")
from sf.torch import _core  # noqa: E402


def _case(seed, tokens=16, channels=24, k=64, pairs=((2, 5), (9, 17), (9, 3))):
    """Random x, w with a few tokens aligned to a few channels: those dot products sit near the
    bound, so a few scattered outputs come back at +-max and are redone."""
    rng = np.random.default_rng(seed)
    w = rng.normal(size=(channels, k))
    x = rng.normal(size=(tokens, k))
    for t, c in pairs:
        x[t] = np.sign(w[c]) * np.abs(rng.normal(size=k))
    return x, w, rng.normal(size=channels) * 0.5


class _Spy(_core.DeviceRunner):
    """Records the (rows, columns) of every wide matmul launch."""

    def __init__(self, partial):
        super().__init__("atreides")
        self.partial_redo = partial
        self.launches = []

    def matmul_wide(self, *args, **kw):
        real = sf.matmul

        def counted(A, B, **opts):
            self.launches.append((A.shape[-2], B.shape[-1]))
            return real(A, B, **opts)

        sf.matmul = counted
        try:
            return super().matmul_wide(*args, **kw)
        finally:
            sf.matmul = real


def _both(x, w, b):
    out = {}
    for partial in (True, False):
        spy = _Spy(partial)
        with sf.config(preflight="ignore"):
            out[partial] = (_core.wide_linear(x, w, b, 16, runner=spy), spy.launches)
    return out


def _same(r1, r2):
    assert np.array_equal(r1.y, r2.y)
    assert (r1.redone, r1.saturated, r1.exact, r1.shift_min, r1.shift_max) == \
           (r2.redone, r2.saturated, r2.exact, r2.shift_min, r2.shift_max)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("bias", [False, True], ids=["no-bias", "bias"])
def test_the_window_gives_the_bits_of_a_whole_relaunch(seed, bias):
    x, w, b = _case(seed)
    out = _both(x, w, b if bias else None)
    (part, lp), (whole, lw) = out[True], out[False]
    _same(part, whole)
    host = _core.wide_linear(x, w, b if bias else None, 16)
    assert np.array_equal(part.y, host.y) and part.redone == host.redone
    if part.redone:
        assert len(lp) == len(lw) == 2                        # first launch, then one relaunch
        rows, cols = lp[1]
        assert rows * cols < lw[1][0] * lw[1][1]              # the relaunch is smaller
        assert rows * cols >= part.redone


def test_the_window_is_the_redone_rows_and_columns():
    """Large codes in rows {1, 3} of A and columns {0, 2} of B: at shift 0 only those four
    outputs reach +-max, so the relaunch is a 2 x 2 window, gathered into one launch."""
    from sf._format import FORMATS
    fmt = FORMATS[16]
    ka, kb = np.zeros((4, 8), np.int64), np.ones((8, 5), np.int64)
    ka[[1, 3]] = fmt.max_code
    kb[:, [0, 2]] = fmt.max_code
    got = {}
    for partial in (True, False):
        spy = _Spy(partial)
        with sf.config(preflight="ignore"):
            y, ev = spy.matmul_wide(fmt, ka, kb, 0, 20, None)
        got[partial] = (y, ev, spy.launches)
    (yp, ep, lp), (yw, ew, lw) = got[True], got[False]
    assert np.array_equal(yp, yw) and np.array_equal(ep["used"], ew["used"]) and ep["redone"] == ew["redone"] == 4
    assert lp == [(4, 5), (2, 2)] and lw == [(4, 5), (4, 5)]


def test_a_batch_redoes_the_union_of_its_windows():
    """(batch, M, K) @ (K, N): one window covers every batch element's redone rows and columns."""
    _, ws, _ = _case(1)
    rng = np.random.default_rng(7)
    xs = []
    for t, c in ((2, 5), (9, 17), (4, 3)):                   # each element aligns a different token
        x = rng.normal(size=(16, 64))
        x[t] = np.sign(ws[c]) * np.abs(rng.normal(size=64)) * 4
        xs.append(x)
    a = np.stack(xs)
    got = {}
    for partial in (True, False):
        spy = _Spy(partial)
        with sf.config(preflight="ignore"):
            got[partial] = _core.wide_matmul(a, ws.T, 16, runner=spy)
    _same(got[True], got[False])
    assert got[True].redone > 0


def test_no_redo_means_no_relaunch():
    rng = np.random.default_rng(0)
    x, w = rng.normal(size=(4, 32)) * 0.01, rng.normal(size=(5, 32)) * 0.01
    spy = _Spy(True)
    with sf.config(preflight="ignore"):
        r = _core.wide_linear(x, w, None, 16, runner=spy)
    assert r.redone == 0 and len(spy.launches) == 1


# --- on the chip model ------------------------------------------------------------------------

@pytest.fixture(scope="module")
def chip():
    pytest.importorskip("superfloat_sim")
    if not os.environ.get("SUPERFLOAT_GPU"):
        pytest.skip("needs a superfloat.gpu checkout (SUPERFLOAT_GPU)")
    from sf.kernels import _chipmm as C
    from sf.kernels._runtime import HardwareAtreides
    mm = C.ChipMatmul()
    inner = C.library(mm, check=True)["matmul_wide"]
    cycles = []

    def counted(fmt, a, b, shift=0, mx=None):
        out = inner(fmt, a, b, shift, mx)
        cycles.append(mm.report.cycles)
        return out

    HardwareAtreides.attach_library({"matmul_wide": counted})
    yield cycles
    HardwareAtreides.attach_library(None)


def test_on_the_chip_model_the_window_is_bit_identical_and_cheaper(chip):
    x, w, b = _case(1, tokens=16, channels=64)
    res = {}
    for partial in (True, False):
        runner = _core.DeviceRunner("atreides-hw")
        runner.partial_redo = partial
        chip.clear()
        with sf.config(preflight="ignore"):
            r = _core.wide_linear(x, w, b, 16, runner=runner)
        res[partial] = (r, list(chip))
    (part, cp), (whole, cw) = res[True], res[False]
    _same(part, whole)
    assert part.redone > 0 and len(cp) == len(cw) == 2
    assert cp[1] < cw[1]                                       # the relaunch costs fewer chip cycles
