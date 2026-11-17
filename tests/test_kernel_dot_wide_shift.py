"""dot_wide with a shift per output inside @sf.jit kernels (spec 1.8.0, DSL 0.14.0).

Alosh (2026-11-18): the chip already takes a shift per output (POP's Rs per
token and column); the DSL's kernels did not.  Now ``sf.dot(x, y,
accumulate="wide", shift=s)`` takes a traced int block, and kernels take
read-only int buffers (a NumPy int array argument) to load the shifts from.
Every case must give the host's ``matmul_wide_codes`` bits, on both backends.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sf
import sf.kernels  # noqa: F401
from sf import _linalg
from sf._format import FORMATS

ROOT = Path(__file__).resolve().parent.parent
GOLD = json.loads((ROOT / "spec" / "golden" / "wide" / "dot_wide_shift.json").read_text())
DEV = "atreides-sim"
F16 = FORMATS[16]


@sf.jit
def mm_shift(a, b, s, out, M: sf.constexpr, K: sf.constexpr, N: sf.constexpr):
    i = sf.arange(0, M)[:, None]
    k = sf.arange(0, K)
    j = sf.arange(0, N)[None, :]
    x = sf.load(a + i * K + k[None, :])
    y = sf.load(b + k[:, None] * N + j)
    sh = sf.load(s + i * N + j)                                    # one shift per output, from an int buffer
    sf.store(out + i * N + j, sf.dot(x, y, accumulate="wide", shift=sh))


def _dev(codes):
    return sf.from_codes(np.asarray(codes, dtype=np.int64).reshape(-1), sf.sf16, device=DEV)


def _run(a, b, s, backend):
    M, K = a.shape
    N = b.shape[1]
    out = sf.zeros(M * N, dtype=sf.sf16, device=DEV)
    with sf.config(kernel_backend=backend), sf.errstate(all="ignore"):
        r = mm_shift[(1,)](_dev(a), _dev(b), np.broadcast_to(s, (M, N)).copy(), out, M=M, K=K, N=N)
    return out.codes.reshape(M, N), r


CASES = [c for c in GOLD["cases"] if c["bits"] == 16 and c["target"] == "v4.9.2"]


@pytest.mark.parametrize("backend", ["interp", "c"])
@pytest.mark.parametrize("i", range(len(CASES)), ids=[f"{c['kind']}-{np.shape(c['shift'])}" for c in CASES])
def test_golden_shift_cases_through_a_kernel(i, backend):
    c = CASES[i]
    a, b = np.array(c["a"]), np.array(c["b"])
    s = np.broadcast_to(np.array(c["shift"]), np.array(c["out"]).shape)
    got, r = _run(a, b, s, backend)
    assert got.tolist() == c["out"]
    assert r.status.saturated == int(np.sum(c["overflow"])) and r.status.exact


@pytest.mark.parametrize("backend", ["interp", "c"])
def test_random_blocks_match_the_host(backend):
    rng = np.random.default_rng(7)
    for M, K, N in ((1, 8, 8), (8, 64, 8), (5, 37, 3)):
        with sf.errstate(all="ignore"):
            a = sf.array(rng.uniform(-0.99, 0.99, (M, K)), dtype=sf.sf16).codes
            b = sf.array(rng.uniform(-0.99, 0.99, (K, N)), dtype=sf.sf16).codes
        s = rng.integers(-32, 32, (M, N))
        want, _ = _linalg.matmul_wide_codes(F16, a, b, s)
        got, _ = _run(a, b, s, backend)
        assert np.array_equal(got, want), (M, K, N)


@sf.jit
def mm_rowcol(a, b, out, M: sf.constexpr, K: sf.constexpr, N: sf.constexpr, PER: sf.constexpr):
    i = sf.arange(0, M)[:, None]
    k = sf.arange(0, K)
    j = sf.arange(0, N)[None, :]
    x = sf.load(a + i * K + k[None, :])
    y = sf.load(b + k[:, None] * N + j)
    sh = (i - 4) if PER == "row" else (j - 6)                          # computed in the kernel, broadcast
    sf.store(out + i * N + j, sf.dot(x, y, accumulate="wide", shift=sh))


@pytest.mark.parametrize("backend", ["interp", "c"])
@pytest.mark.parametrize("per", ["row", "col"])
def test_a_shift_block_computed_in_the_kernel_broadcasts(per, backend):
    rng = np.random.default_rng(8)
    M, K, N = 6, 24, 5
    with sf.errstate(all="ignore"):
        a = sf.array(rng.uniform(-0.9, 0.9, (M, K)), dtype=sf.sf16).codes
        b = sf.array(rng.uniform(-0.9, 0.9, (K, N)), dtype=sf.sf16).codes
    s = (np.arange(M)[:, None] - 4) if per == "row" else (np.arange(N)[None, :] - 6)
    want, _ = _linalg.matmul_wide_codes(F16, a, b, np.broadcast_to(s, (M, N)))
    out = sf.zeros(M * N, dtype=sf.sf16, device=DEV)
    with sf.config(kernel_backend=backend), sf.errstate(all="ignore"):
        mm_rowcol[(1,)](_dev(a), _dev(b), out, M=M, K=K, N=N, PER=per)
    assert np.array_equal(out.codes.reshape(M, N), want)


@pytest.mark.parametrize("backend", ["interp", "c"])
@pytest.mark.parametrize("bad", [32, -33])
def test_a_shift_out_of_range_faults_the_launch(bad, backend):
    a = np.zeros((2, 8), dtype=np.int64)
    b = np.zeros((8, 2), dtype=np.int64)
    s = np.zeros((2, 2), dtype=np.int64)
    s[1, 0] = bad
    with pytest.raises(sf.SFDeviceError, match="dot_wide shift"):
        _run(a, b, s, backend)


def test_int_buffers_are_read_only_and_checked():
    @sf.jit
    def writes(s, out):
        sf.store(s + sf.arange(0, 2), sf.load(out + sf.arange(0, 2)))

    out = sf.zeros(2, dtype=sf.sf16, device=DEV)
    with pytest.raises(sf.SFCompileError, match="read-only"):
        writes[(1,)](np.zeros(2, dtype=np.int64), out)

    @sf.jit
    def gather(src, idx, out):
        o = sf.arange(0, 4)
        j = sf.load(idx + o, mask=o < 3, other=0)                       # an int table: masked lanes read 0
        sf.store(out + o, sf.load(src + j))

    src = sf.asarray([0.5, -0.25, 0.125, 0.75], dtype=sf.sf16, device=DEV)
    for backend in ("interp", "c"):
        out = sf.zeros(4, dtype=sf.sf16, device=DEV)
        with sf.config(kernel_backend=backend):
            gather[(1,)](src, np.array([3, 1, 2, 9], dtype=np.int32), out)
        assert out.to("host").tolist() == [0.75, -0.25, 0.125, 0.5]
    with pytest.raises(sf.SFTypeError, match="must hold integers"):
        gather[(1,)](src, np.zeros(4, dtype=np.float32), sf.zeros(4, dtype=sf.sf16, device=DEV))
    with pytest.raises(sf.SFLaunchError, match="-32768..32767"):
        gather[(1,)](src, np.array([1, 2, 40000, 0]), sf.zeros(4, dtype=sf.sf16, device=DEV))

    s = np.array([3, -2, 9, 9], dtype=np.int16)
    before = s.copy()
    _, r = _run(np.zeros((1, 8), dtype=np.int64), np.zeros((8, 4), dtype=np.int64), s.reshape(1, 4), "interp")
    assert np.array_equal(s, before)                                     # the user's array is untouched


def test_traced_shift_rules():
    @sf.jit
    def chain_with_shift(a, s, out):
        i = sf.arange(0, 8)
        x = sf.load(a + i[:, None] * 8 + i[None, :])
        sf.store(out + i[:, None] * 8 + i[None, :], sf.dot(x, x, accumulate="chain", shift=sf.load(s + i)))

    a = sf.zeros(64, dtype=sf.sf16, device=DEV)
    with pytest.raises(ValueError, match="only applies"):
        chain_with_shift[(1,)](a, np.zeros(8, dtype=np.int64), a)

    @sf.jit
    def wrong_shape(a, s, out):
        i = sf.arange(0, 8)
        x = sf.load(a + i[:, None] * 8 + i[None, :])
        sh = sf.load(s + sf.arange(0, 3))
        sf.store(out + i[:, None] * 8 + i[None, :], sf.dot(x, x, accumulate="wide", shift=sh))

    with pytest.raises(sf.SFCompileError, match="does not broadcast"):
        wrong_shape[(1,)](a, np.zeros(3, dtype=np.int64), a)

    @sf.jit
    def numpy_shift(a, out):
        i = sf.arange(0, 8)
        x = sf.load(a + i[:, None] * 8 + i[None, :])
        sf.store(out + i[:, None] * 8 + i[None, :], sf.dot(x, x, accumulate="wide", shift=np.zeros((8, 8), int)))

    with pytest.raises(sf.SFCompileError, match="int buffer argument"):
        numpy_shift[(1,)](a, a)


def test_the_ir_shows_the_int_buffer_and_the_shift_operand():
    _run(np.zeros((2, 8), dtype=np.int64), np.zeros((8, 2), dtype=np.int64), np.zeros((2, 2), dtype=np.int64), "interp")
    ir = next(v for k, v in mm_shift.cache.items() if dict(k[0]).get("M") == 2)
    text = ir.dump()
    assert "ptr<int>" in text
    dw = [op for op in ir.ops if op.name == "dot_wide"]
    assert len(dw) == 1 and len(dw[0].args) == 3 and dw[0].attrs["shift"] is None
