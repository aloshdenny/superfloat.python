"""LayerNorm and RMSNorm through sf's integer definition (``sf.intops.norm_rows``, ``convert(norm="integer")``).

``norm_row`` is Alosh's definition (exact MX sums, the rest in sf16 arithmetic, ``rsqrt_norm`` for 1/sqrt);
``norm_rows`` picks one exponent per row for the input and the output and runs it on float rows.
"""
from __future__ import annotations

import numpy as np
import pytest

from sf import _intops as I

torch = pytest.importorskip("torch")
import sf.torch as sft  # noqa: E402
from sf.torch import _intops as TI  # noqa: E402


def _ref(x, g, b, eps, rms):
    if rms:
        return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * g
    mu = x.mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(((x - mu) ** 2).mean(-1, keepdims=True) + eps) * g + b


@pytest.mark.parametrize("rms", [False, True], ids=["layernorm", "rmsnorm"])
@pytest.mark.parametrize("n", [64, 768, 896])
def test_rows_are_close_to_float(rms, n):
    rng = np.random.default_rng(n)
    x = rng.normal(size=(12, n)) * 3 + rng.normal(size=(12, 1))
    g, b = rng.normal(size=n) * 0.2 + 1, rng.normal(size=n) * 0.1
    y, r = I.norm_rows(x, g, b, 1e-5, rms), _ref(x, g, b, 1e-5, rms)
    assert np.abs(y - r).max() <= 2e-3 * np.abs(r).max()


def test_a_row_is_norm_row_at_the_chosen_exponents():
    rng = np.random.default_rng(1)
    x, g, b = rng.normal(size=96), rng.normal(size=96) * 0.3 + 1, rng.normal(size=96) * 0.2
    ex, eg, eb = 2, 1, 0                              # |x| < 4 here, x - mean too; |g| < 2; |b| < 1
    assert np.abs(x).max() < 4 and np.abs(x - x.mean()).max() < 4 and np.abs(g).max() < 2 and np.abs(b).max() < 1
    r = _ref(x[None], g, b, 1e-5, False)[0]
    eo = int(np.floor(np.log2(np.abs(r).max() * (1 + 2 ** -10)))) + 1
    want = I.norm_row(I.to_codes(x, ex), I.to_codes(g, eg), I.to_codes(b, eb), eo, ex, eg, eb, 1e-5, False)
    assert np.array_equal(I.norm_rows(x[None], g, b, 1e-5, False)[0], np.asarray(want) / 32768 * 2.0 ** eo)


@pytest.mark.xfail(strict=True, reason="known defect: the truncating multiplies in z * y * gamma shrink the rows")
def test_an_outlier_channel_keeps_the_scale():
    """One channel 40x the others (GPT-2's residual stream has such channels).  The input exponent follows the
    outlier, so most z codes are small, and the two Q15 multiplies in z * y * gamma truncate toward zero: the rows
    come out small (median about 0.998 here; GPT-2's ln_f about 0.988).  Rounding those two multiplies brings it
    back (ln_f 1.00006), but that changes Alosh's definition and is not done.  Strict: when the definition is
    fixed, this test passes and the xfail has to come off."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 768)) * 3
    x[:, 5] *= 40
    y, r = I.norm_rows(x, np.ones(768), np.zeros(768)), _ref(x, 1.0, 0.0, 1e-5, False)
    assert abs(np.median(y / r) - 1) < 5e-4


def test_convert_replaces_gpt2_layernorms():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=96, n_positions=32)
    m = transformers.GPT2LMHeadModel(cfg).eval()
    c = sft.convert(m, dtype="sf16", accumulate="wide", norm="integer")
    names = [n for n, w in sft.conversion_report(c).nonlinear if w == "LayerNorm (integer)"]
    assert len(names) == 5 and "transformer.ln_f" in names                    # ln_1, ln_2 per block, ln_f
    assert isinstance(c.transformer.ln_f, TI.IntNorm)
    ids = torch.randint(0, 96, (1, 12))
    with torch.no_grad():
        assert (m(ids).logits - c(ids).logits).abs().max() < 1e-2


def test_convert_replaces_hugging_face_rmsnorms():
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.Qwen2Config(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
                                   num_key_value_heads=2, vocab_size=96)
    m = transformers.Qwen2ForCausalLM(cfg).eval()
    c = sft.convert(m, dtype="sf16", accumulate="wide", norm="integer")
    assert sum(w == "RMSNorm (integer)" for _, w in sft.conversion_report(c).nonlinear) == 5
    ids = torch.randint(0, 96, (1, 12))
    with torch.no_grad():
        assert (m(ids).logits - c(ids).logits).abs().max() < 1e-2


def test_torch_rmsnorm_and_float_by_default():
    if not hasattr(torch.nn, "RMSNorm"):
        pytest.skip("torch without nn.RMSNorm")
    net = torch.nn.Sequential(torch.nn.Linear(16, 16), torch.nn.RMSNorm(16), torch.nn.LayerNorm(16))
    assert not sft.conversion_report(sft.convert(net, dtype="sf16")).nonlinear
    c = sft.convert(net, dtype="sf16", norm="integer")
    assert [w for _, w in sft.conversion_report(c).nonlinear] == ["RMSNorm (integer)", "LayerNorm (integer)"]
    x = torch.randn(3, 16)
    with torch.no_grad():
        assert (net(x) - c(x)).abs().max() < 1e-2


def test_gemma_style_rmsnorm_stays_float():
    class GemmaRMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight, self.eps = torch.nn.Parameter(torch.zeros(8)), 1e-6

    rep, why = TI.integer_norm(GemmaRMSNorm())
    assert rep is None and "1 + weight" in why


def test_norm_must_be_float_or_integer():
    with pytest.raises(ValueError, match="norm must be"):
        sft.convert(torch.nn.Linear(4, 4), dtype="sf16", norm="int")


def test_torch_compile_keeps_the_integer_ops_whole():
    """The integer definitions run in NumPy; as custom ops, torch.compile(backend="superfloat") runs them as nodes
    instead of tracing into NumPy (which used to fail with nonlinear="integer"), with the eager bits."""
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=96, n_positions=32)
    m = sft.convert(transformers.GPT2LMHeadModel(cfg).eval(), dtype="sf16", accumulate="wide",
                    nonlinear="integer", norm="integer")
    ids = torch.randint(0, 96, (1, 12))
    torch._dynamo.reset()
    with torch.no_grad():
        want = m(ids).logits
        got = torch.compile(m, backend="superfloat")(ids).logits
    assert torch.equal(got, want)
