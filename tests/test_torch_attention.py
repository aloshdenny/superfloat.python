"""Attention on the MX engine (Alosh's build order, item 4).

``sft.attention`` takes ``F.scaled_dot_product_attention``'s arguments and
runs Q K^T (the wide-mode rule) and P V (output exponent from max|V| per
channel) on the MX engine at sf16; softmax stays float.  ``sft.autocast`` and
``sft.convert`` with ``accumulate="wide"`` route a model's SDPA calls there.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
F = torch.nn.functional

import sf  # noqa: E402
import sf.kernels  # noqa: E402,F401
import sf.torch as sft  # noqa: E402
from sf.torch import _core  # noqa: E402


def _qkv(heads_kv=4, L=10, S=10, E=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(2, 4, L, E, generator=g)
    k = torch.randn(2, heads_kv, S, E, generator=g)
    v = torch.randn(2, heads_kv, S, E, generator=g)
    return q, k, v


@pytest.mark.parametrize("kw", [{}, {"is_causal": True}, {"scale": 0.3}],
                         ids=["plain", "causal", "scale"])
def test_matches_pytorch(kw):
    q, k, v = _qkv()
    y = sft.attention(q, k, v, name="t", **kw)
    assert (y - F.scaled_dot_product_attention(q, k, v, **kw)).abs().max() < 1e-3
    st = sft.last_status()
    assert st["t.scores"].accumulate == "wide" and st["t.pv"].accumulate == "wide"
    assert st["t.scores"].h is not None and st["t.pv"].h is None        # pv uses the max|V| bound
    assert st["t.scores"].dtype == st["t.pv"].dtype == "sf16"


def test_masks_and_gqa():
    q, k, v = _qkv()
    m = torch.rand(10, 10, generator=torch.Generator().manual_seed(3)) > 0.3
    m.fill_diagonal_(True)
    assert (sft.attention(q, k, v, attn_mask=m) - F.scaled_dot_product_attention(q, k, v, attn_mask=m)).abs().max() < 1e-3
    add = torch.randn(10, 10)
    assert (sft.attention(q, k, v, attn_mask=add) - F.scaled_dot_product_attention(q, k, v, attn_mask=add)).abs().max() < 1e-3
    q, k, v = _qkv(heads_kv=2)
    assert (sft.attention(q, k, v, enable_gqa=True)
            - F.scaled_dot_product_attention(q, k, v, enable_gqa=True)).abs().max() < 1e-3
    with pytest.raises(ValueError, match="not both"):
        sft.attention(q, k, v, attn_mask=m, is_causal=True)


def test_pv_output_exponent_is_v_per_channel():
    """Rows of P sum to 1, so the output's exponent per channel is V's own:
    shift = the exponent of P's row, and nothing saturates."""
    rng = np.random.default_rng(0)
    s = rng.normal(size=(3, 50))
    p = np.exp(s) / np.exp(s).sum(-1, keepdims=True)
    v = rng.normal(size=(50, 6)) * np.array([1e-3, 0.1, 1, 10, 100, 1000])   # very different channels
    r = _core.wide_matmul(p, v, 16, out_rule="max_b")
    assert r.saturated == 0 and r.redone == 0 and r.h is None
    # the guarantee: error within a step of each channel's largest |V| (2^-15 of it, plus V's own rounding)
    assert (np.abs(r.y - p @ v) / np.abs(v).max(axis=0)).max() < 2.0 ** -14
    with pytest.raises(ValueError, match="out_rule"):
        _core.wide_matmul(p, v, 16, out_rule="max")


def test_autocast_catches_sdpa_only_in_wide_mode():
    q, k, v = _qkv()
    with sft.autocast(accumulate="wide"):
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.equal(y, sft.attention(q, k, v, is_causal=True))
    assert "autocast.attention.0.pv" in sft.last_status() or any(
        n.endswith("attention.0.pv") for n in sft.last_status())
    sft.reset_status()
    with sft.autocast():                                      # chain: attention stays float
        y = F.scaled_dot_product_attention(q, k, v)
    assert torch.equal(y, F.scaled_dot_product_attention(q, k, v))
    assert not any("attention" in n for n in sft.last_status())


class Block(torch.nn.Module):
    def __init__(self, d=16, h=4):
        super().__init__()
        self.h = h
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.out = torch.nn.Linear(d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(x).split(D, -1)
        q, k, v = (t.reshape(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


def test_convert_runs_attention_on_mx_and_names_calls_per_forward():
    torch.manual_seed(0)
    m = torch.nn.Sequential(Block(), Block()).eval()
    x = torch.randn(2, 6, 16)
    c = sft.convert(m, accumulate="wide")
    assert "MX engine" in sft.conversion_report(c).attention
    for _ in range(2):                                        # names restart at 0 every forward
        sft.reset_status()
        y = c(x)
        names = sorted(n for n in sft.last_status() if n.startswith("attention"))
        assert names == ["attention.0.pv", "attention.0.scores", "attention.1.pv", "attention.1.scores"]
    assert (y - m(x)).abs().max() < 2e-3
    off = sft.convert(m, accumulate="wide", attention=False)
    assert sft.conversion_report(off).attention.startswith("float")
    sft.reset_status()
    off(x)
    assert not any(n.startswith("attention") for n in sft.last_status())
    chain = sft.convert(m)                                    # chain: not converted
    assert sft.conversion_report(chain).attention.startswith("float")
    with pytest.raises(ValueError, match="attention must be"):
        sft.convert(m, accumulate="wide", attention="always")


def test_the_mode_is_left_even_when_the_forward_raises():
    class Boom(torch.nn.Module):
        def forward(self, x):
            raise RuntimeError("boom")

    c = sft.convert(torch.nn.Sequential(Boom()), accumulate="wide")
    with pytest.raises(RuntimeError, match="boom"):
        c(torch.randn(2))
    q, k, v = _qkv()                                          # outside a forward: plain PyTorch again
    sft.reset_status()
    F.scaled_dot_product_attention(q, k, v)
    assert not sft.last_status()


def test_gradients_flow():
    q, k, v = (t.requires_grad_() for t in _qkv())
    sft.attention(q, k, v, is_causal=True).sum().backward()
    assert q.grad is not None and k.grad is not None and v.grad is not None
    assert q.grad.shape == q.shape


@pytest.mark.parametrize("device", ["atreides", "atreides-sim"])
def test_compile_backend_runs_attention_on_the_target(device):
    torch.manual_seed(0)
    m = torch.nn.Sequential(Block()).eval()
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        want = sft.attention(*_split(m, x), is_causal=True)
        f = torch.compile(lambda q, k, v: sft.attention(q, k, v, is_causal=True), backend="superfloat")
        torch._dynamo.reset()
        with sf.config(default_device=device, preflight="ignore"):
            got = f(*_split(m, x))
    assert torch.equal(got, want)
    ops = {op for _, op, _, _ in sft.compile_report().on_device}
    assert ops == {"matmul_wide", "matmul_wide_pv"}


def _split(m, x):
    blk = m[0]
    B, T, D = x.shape
    q, k, v = blk.qkv(x).split(D, -1)
    return tuple(t.reshape(B, T, blk.h, D // blk.h).transpose(1, 2) for t in (q, k, v))
