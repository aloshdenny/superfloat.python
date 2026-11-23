"""Wide matmul as a library call on the chip model: Alosh's gemm / gemv programs on superfloat.sim.

Every output must equal sf's MX model (sf._linalg.matmul_wide_codes, v4.9.2) bit for bit.
Skipped unless both checkouts are there:
    SUPERFLOAT_SIM=/path/superfloat.sim SUPERFLOAT_GPU=/path/superfloat.gpu pytest tests/test_chipmm.py
(superfloat.sim built; superfloat.gpu at v4.9.2).  Nothing of either is in this repository.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from sf import _linalg
from sf._format import FORMATS
from sf.kernels import _chipmm as C


def test_operand_words_round_trip():
    for bits in (4, 8, 12, 16):
        lim = 2 ** (bits - 1) - 1
        for code in (-lim, -1, 0, 1, lim):
            w = C.to_word(code, bits)
            assert C.from_word(w, bits) == code and w & ((1 << (16 - bits)) - 1) == 0
    assert C.to_word(-3, 16) == 0x8003 and C.to_word(-3, 8) == 0x8300
    assert C.unit_for(8) == 8 and C.unit_for(9) == 16


@pytest.fixture(scope="module")
def mm():
    pytest.importorskip("superfloat_sim")
    if not os.environ.get("SUPERFLOAT_GPU"):
        pytest.skip("needs a superfloat.gpu checkout (SUPERFLOAT_GPU)")
    return C.ChipMatmul()


def _check(mm, a, w, s, bits):
    got = mm(a, w, s, bits)
    ref, ev = _linalg.matmul_wide_codes(FORMATS[bits], a, w, s, mx="v4.9.2")
    assert np.array_equal(got, ref), f"{int((got != ref).sum())} of {got.size} outputs differ"
    return ev


@pytest.mark.parametrize("bits", [16, 12, 9, 8, 4])
@pytest.mark.parametrize("M,K,N", [(1, 64, 8), (8, 32, 8), (3, 40, 20), (13, 72, 17)])
def test_random_matmuls_match_the_mx_model(mm, bits, M, K, N):
    rng = np.random.default_rng(bits * 1000 + M * 100 + K + N)
    lim = 2 ** (bits - 1) - 1
    _check(mm, rng.integers(-lim, lim + 1, (M, K)), rng.integers(-lim, lim + 1, (K, N)),
           rng.integers(-12, 6, (M, N)), bits)
    assert mm.report.kernel == ("gemv" if M == 1 else "gemm")


@pytest.mark.parametrize("bits", [16, 8])
def test_saturation_and_underflow_match(mm, bits):
    rng = np.random.default_rng(bits)
    lim = 2 ** (bits - 1) - 1
    big = np.full((8, 768), lim), np.full((768, 8), lim)
    assert _check(mm, *big, -9, bits)["overflow"].all()                        # the accumulator saturates
    a, w = rng.integers(-lim, lim + 1, (8, 128)), rng.integers(-lim, lim + 1, (128, 8))
    assert _check(mm, a, w, 12, bits)["overflow"].all()                        # every POP saturates
    assert _check(mm, a // 64, w // 64, -32, bits)["underflow"].any()          # nonzero totals round to 0


def test_a_gpt2_layer_splits_into_launches(mm):
    rng = np.random.default_rng(7)
    a, w = rng.integers(-32767, 32768, (1, 3072)), rng.integers(-32767, 32768, (3072, 64))
    _check(mm, a, w, rng.integers(-8, 0, (1, 64)), 16)
    r = mm.report
    assert len(r.launches) == 4                                  # 24,576 weight words a column tile: 2 a window
    assert r.cycles > 0 and r.data_reads > 3072 * 64


def test_shifts_outside_the_pop_range_are_refused(mm):
    with pytest.raises(ValueError, match="-32..31"):
        mm(np.zeros((1, 8), int), np.zeros((8, 8), int), 32, 16)


# --- the layout matches Alosh's own functions ---------------------------------------------------

def test_numpy_layout_equals_kernels_v4(mm):
    rng = np.random.default_rng(11)
    W, A = rng.integers(0, 1 << 16, (32, 8)), rng.integers(0, 1 << 16, (8, 32))
    assert C.weight_words(W).tolist() == mm.K.weight_words(W.tolist(), 16, False)
    assert C.act_words(A).tolist() == mm.K.act_words(A.tolist())
    codes = rng.integers(-2047, 2048, 50)
    assert C.encode(codes, 12).tolist() == [C.to_word(c, 12) for c in codes]
    assert C.decode(C.encode(codes, 12), 12).tolist() == codes.tolist()


# --- atreides-hw sends wide matmuls to the chip -------------------------------------------------

@pytest.fixture
def hw_chip(mm):
    from sf.kernels._runtime import HardwareAtreides
    calls = []
    inner = C.library(mm, check=True)["matmul_wide"]

    def counted(fmt, a, b, shift=0, mx=None):
        calls.append((tuple(a.shape), tuple(b.shape)))
        return inner(fmt, a, b, shift, mx)
    HardwareAtreides.attach_library({"matmul_wide": counted})
    yield calls
    HardwareAtreides.attach_library(None)


def test_device_matmul_runs_on_the_chip(hw_chip):
    import sf
    rng = np.random.default_rng(12)
    out = {}
    for dev in ("atreides-hw", "atreides-sim"):
        with sf.errstate(all="ignore"):
            a = sf.from_codes(rng.integers(-9000, 9000, (2, 8, 64)), sf.sf16, device=dev)
            b = sf.from_codes(rng.integers(-9000, 9000, (64, 24)), sf.sf16, device=dev)
            out[dev] = sf.matmul(a, b, accumulate="wide", shift=-2)
        rng = np.random.default_rng(12)
    assert np.array_equal(out["atreides-hw"].codes, out["atreides-sim"].codes)
    assert hw_chip == [((2, 8, 64), (64, 24))]
    assert not out["atreides-hw"].status.exact                # no flags on the chip: an upper bound only
    with sf.config(status="replay"), sf.errstate(all="ignore"):
        a = sf.from_codes(rng.integers(-9000, 9000, (8, 64)), sf.sf16, device="atreides-hw")
        b = sf.from_codes(rng.integers(-9000, 9000, (64, 8)), sf.sf16, device="atreides-hw")
        assert sf.matmul(a, b, accumulate="wide", shift=-2).status.exact   # replayed on the host


def test_detached_hardware_keeps_the_host_emulation(mm):
    import sf
    from sf.kernels._runtime import HardwareAtreides
    HardwareAtreides.attach_library(None)
    with sf.errstate(all="ignore"):
        a = sf.from_codes(np.ones((8, 8), int), sf.sf16, device="atreides-hw")
        assert sf.matmul(a, a, accumulate="wide").status.exact


@pytest.mark.parametrize("dim,heads,tokens", [(64, 2, 8), (64, 2, 1),
                                              pytest.param(768, 12, 8, marks=pytest.mark.slow)])
def test_a_transformer_block_on_the_chip_is_bit_identical(mm, dim, heads, tokens):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    import gpt2_block_on_chip as G
    r = G.run(tokens=tokens, dim=dim, heads=heads)
    assert r["identical"] and r["host_emulated"] == 0
    assert len(r["calls"]) == 6                               # qkv, scores, P.V, proj, fc, out: all on the chip


def test_hugging_face_gpt2_code_on_the_chip_is_bit_identical(mm):
    """transformers' own GPT-2 (Conv1D layers, its attention, the LM head), tiny and random: no download."""
    transformers = pytest.importorskip("transformers")
    import torch
    import sf
    import sf.torch as sft
    from sf.kernels._runtime import HardwareAtreides
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=96, n_positions=32)
    model = sft.convert(transformers.GPT2LMHeadModel(cfg).eval(), dtype="sf16", accumulate="wide")
    ids = torch.randint(0, 96, (1, 12))
    calls = []
    inner = C.library(mm, check=True)["matmul_wide"]
    HardwareAtreides.attach_library({"matmul_wide": lambda fmt, a, b, shift=0, mx=None:
                                     calls.append(b.shape) or inner(fmt, a, b, shift, mx)})
    try:
        with torch.no_grad():
            want = model(ids).logits
            torch._dynamo.reset()
            with sf.config(default_device="atreides-hw", preflight="ignore"):
                got = torch.compile(model, backend="superfloat")(ids).logits
    finally:
        HardwareAtreides.attach_library(None)
    assert torch.equal(got, want)
    assert sum(1 for s in calls if s[-1] == 96) >= 1            # the LM head ran on the chip too
    assert len(calls) >= 2 * 6 + 1                              # 6 matmuls a block, then the head
