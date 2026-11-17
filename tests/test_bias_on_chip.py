"""Wide mode adds a linear layer's bias on the chip after POP (DSL 0.12.0, spec 1.7.0).

y_code = add(q, round_half_even(b * 2^(P - fe))) with the saturating sfN add, per output at
its exponent fe; an output whose POP or add reaches +-max is redone at the exponent of
|x||w| + |b|.  These pin the definition, the redo cases, and host/target agreement.
"""
from __future__ import annotations

import numpy as np
import pytest

import sf
from sf import _heuristic, _vectorized as vec
from sf._format import FORMATS

pytest.importorskip("torch")
from sf.torch import _core  # noqa: E402

WIDTHS = [16, 12, 8, 6]


def _fmt(bits):
    return FORMATS[bits]


@pytest.mark.parametrize("bits", WIDTHS)
def test_the_bias_is_the_sf_add_of_aligned_codes(bits):
    """Rebuild y from the definition: POP without bias, align b to each output's exponent,
    add with the reference sfN add, and compare with wide_linear bit for bit."""
    rng = np.random.default_rng(bits)
    x, w = rng.normal(size=(5, 48)), rng.normal(size=(7, 48)) * 0.1
    b = rng.normal(size=7) * 0.3
    with sf.config(preflight="ignore"):
        r = _core.wide_linear(x, w, b, bits)
        r0 = _core.wide_linear(x, w, None, bits)
    fmt, P = _fmt(bits), bits - 1
    if r.redone == 0 and r0.redone == 0:            # same exponents for both: rebuild exactly
        q0 = np.rint(np.ldexp(r0.y, P - _exps(r0, x, w, fmt))).astype(np.int64)
        fe = _exps(r0, x, w, fmt)
        bq, _ = _core._bias_codes(fmt, b, fe)
        y, _ = vec.add(fmt, q0, np.broadcast_to(bq, q0.shape))
        assert np.array_equal(np.ldexp(np.asarray(y, dtype=np.float64), fe - P), r.y)
    np.testing.assert_allclose(r.y, x @ w.T + b, atol=2.0 ** -(bits - 4))


def _exps(r, x, w, fmt):
    qa, ea, _ = _heuristic.quantize_pow2(fmt, x, -1)
    qw, ew, _ = _heuristic.quantize_pow2(fmt, w, -1)
    qa, ea, _ = _heuristic.acc_guard(fmt, x, qa, ea, np.ascontiguousarray(qw.T))
    sh, _, _ = _heuristic.plan(fmt, qa, ea, np.ascontiguousarray(qw.T), ew)
    return ea[:, None] + ew[None, :] - sh


@pytest.mark.parametrize("bits", WIDTHS)
def test_a_bias_larger_than_the_output_scale_is_redone_not_clipped(bits):
    """Tiny activations, large bias: the first exponent cannot hold b, so the add saturates
    and the output is redone; nothing is left at +-max and the value is right."""
    rng = np.random.default_rng(100 + bits)
    x, w = rng.normal(size=(4, 64)) * 1e-3, rng.normal(size=(6, 64)) * 0.05
    b = np.array([0.9, -0.7, 0.5, -0.3, 0.2, 0.0])
    with sf.config(preflight="ignore"):
        r = _core.wide_linear(x, w, b, bits)
    assert r.redone > 0 and r.saturated == 0
    np.testing.assert_allclose(r.y, x @ w.T + b, atol=2.0 ** -(bits - 2))


def test_a_clipped_pop_pulled_back_by_the_bias_is_still_redone():
    """POP clips (an aligned dot product near its L2 bound), and a negative bias would pull
    the sum back below max, hiding the clip.  The redo must look at the POP code too."""
    K = 1024
    x = np.full((1, K), 0.5)
    w = np.full((2, K), 0.5)
    b = np.array([-200.0, 0.0])                    # dot = 256, so the right answer is 56 and 256
    with sf.config(preflight="ignore"):
        r = _core.wide_linear(x, w, b, 16)
    assert r.redone == 2
    np.testing.assert_allclose(r.y, x @ w.T + b, rtol=1e-3)


@pytest.mark.parametrize("bits", WIDTHS)
def test_the_target_gives_the_host_bits_with_a_bias(bits):
    """The bias add runs on the target (the masked add below sf16) and matches the host."""
    rng = np.random.default_rng(200 + bits)
    x, w = rng.normal(size=(6, 40)), rng.normal(size=(5, 40)) * 0.1
    b = rng.normal(size=5) * 0.5
    x[2] *= 1e-3                                   # one token whose bias dominates: redone
    with sf.config(preflight="ignore"):
        h = _core.wide_linear(x, w, b, bits)
        d = _core.wide_linear(x, w, b, bits, runner=_core.DeviceRunner("atreides"))
    assert np.array_equal(h.y, d.y) and h.redone == d.redone


def test_bias_saturated_counts_clamped_bias_codes():
    """An explicit shift that puts outputs at a small exponent clamps the bias codes there;
    the status reports it."""
    x, w = np.full((1, 8), 0.01), np.full((3, 8), 0.01)
    b = np.array([0.5, 0.0, -0.5])
    with sf.config(preflight="ignore"):
        r = _core.wide_linear(x, w, b, 16, shift=20)
    assert r.bias_saturated == 2


def test_big_layers_in_blocks_give_the_one_shot_bits_with_a_bias(monkeypatch):
    """PR 19's blocks (an LM head quantized a block of output channels at a time) with 0.12.0's
    on-chip bias: each block adds its own slice of the bias, and the result and every status field
    equal the one-shot run."""
    rng = np.random.default_rng(19)
    x, w = rng.normal(size=(5, 40)), rng.normal(size=(23, 40)) * 0.1
    x[1] *= 1e-3                                   # a token whose bias dominates: redone
    b = rng.normal(size=23) * 0.5
    with sf.config(preflight="ignore"):
        one = _core.wide_linear(x, w, b, 16)
        monkeypatch.setattr(_core, "WIDE_BLOCK", 40 * 4)          # 4 output channels per block
        blk = _core.wide_linear(x, w, b, 16)
    assert np.array_equal(one.y, blk.y)
    for f in ("saturated", "redone", "bias_saturated", "outputs", "exact", "guard_raised", "shift_min", "shift_max"):
        assert getattr(one, f) == getattr(blk, f), f
    assert one.redone > 0
