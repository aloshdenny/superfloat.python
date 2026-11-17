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


# --- nn.MultiheadAttention ------------------------------------------------------------------------

@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("acc", ["wide", "chain"])
def test_multihead_attention_matches_pytorch(batch_first, acc):
    torch.manual_seed(0)
    m = torch.nn.MultiheadAttention(16, 4, batch_first=batch_first).eval()
    x = torch.randn(2, 7, 16) if batch_first else torch.randn(7, 2, 16)
    kp = torch.zeros(2, 7, dtype=torch.bool)
    kp[0, 5:] = True
    causal = torch.ones(7, 7, dtype=torch.bool).triu(1)
    c = sft.convert(torch.nn.Sequential(m), accumulate=acc)
    assert sft.conversion_report(c).converted == [("0", "MultiheadAttention")]
    tol = 2e-4 if acc == "wide" else 2e-3
    with torch.no_grad():
        for kw in ({}, {"key_padding_mask": kp}, {"attn_mask": causal},
                   {"attn_mask": causal, "key_padding_mask": kp, "average_attn_weights": False}):
            (ro, rw), (so, sw) = m(x, x, x, **kw), c[0](x, x, x, **kw)
            assert so.shape == ro.shape and sw.shape == rw.shape
            assert (so - ro).abs().max() < tol and (sw - rw).abs().max() < tol
        assert c[0](x, x, x, need_weights=False)[1] is None


def test_multihead_attention_names_and_wide_status():
    torch.manual_seed(1)
    m = torch.nn.MultiheadAttention(16, 4).eval()
    x = torch.randn(5, 2, 16)
    c = sft.convert(torch.nn.Sequential(m), accumulate="wide")
    sft.reset_status()
    with torch.no_grad():
        c[0](x, x, x)
    assert sorted(sft.last_status()) == ["0.k", "0.out", "0.pv", "0.q", "0.scores", "0.v"]
    assert all(s.accumulate == "wide" for s in sft.last_status().values())


def test_multihead_attention_variants():
    torch.manual_seed(2)
    m = torch.nn.MultiheadAttention(16, 4, kdim=8, vdim=12).eval()          # separate projections
    q, k, v = torch.randn(5, 2, 16), torch.randn(6, 2, 8), torch.randn(6, 2, 12)
    c = sft.convert(torch.nn.Sequential(m), accumulate="wide")
    with torch.no_grad():
        assert (c[0](q, k, v)[0] - m(q, k, v)[0]).abs().max() < 2e-4
        x1 = torch.randn(5, 16)                                              # unbatched
        m1 = torch.nn.MultiheadAttention(16, 4).eval()
        c1 = sft.convert(torch.nn.Sequential(m1), accumulate="wide")[0]
        assert (c1(x1, x1, x1)[0] - m1(x1, x1, x1)[0]).abs().max() < 2e-4
    skipped = sft.conversion_report(sft.convert(torch.nn.Sequential(
        torch.nn.MultiheadAttention(16, 4, add_bias_kv=True)))).skipped
    assert skipped == [("0", "MultiheadAttention", "bias_k / bias_v are not supported yet; runs in float")]


def test_multihead_attention_trains():
    torch.manual_seed(3)
    m = torch.nn.MultiheadAttention(16, 4, dropout=0.1)
    c = sft.convert(torch.nn.Sequential(m), accumulate="wide").train()
    x = torch.randn(5, 2, 16, requires_grad=True)
    c[0](x, x, x)[0].sum().backward()
    assert x.grad is not None and c[0].in_proj_weight.grad is not None and c[0].out_proj.weight.grad is not None


# --- P @ V's output exponent from the keys a row can see (spec 1.9.0) ------------------------------------------------

def test_pv_output_exponent_counts_only_visible_keys():
    import numpy as np
    from sf.torch import _core
    p = np.array([[1.0, 0, 0], [0.5, 0.5, 0], [0.2, 0.3, 0.5]])
    v = np.array([[0.1, 2.0], [0.4, -3.0], [40.0, 0.25]])
    e = _core.visible_max_exp(p, v)
    assert e.tolist() == [[-3, 2], [-1, 2], [6, 2]]                       # row 0 sees key 0 only, ...
    big = _core.visible_max_exp(np.broadcast_to(p, (2, 3, 3)), np.broadcast_to(v, (2, 3, 2)), chunk=4)
    assert (big == e).all()


def test_pv_rows_do_not_see_the_exponent_of_masked_keys():
    torch.manual_seed(0)
    q, k, v = (torch.randn(1, 2, 16, 8) for _ in range(3))
    v[..., 12:, :] *= 30                                                  # large values only in the last keys
    import numpy as np
    from sf.torch import _core
    calls = []
    real = _core.visible_max_exp
    _core.visible_max_exp = lambda p, b, chunk=1 << 24: calls.append(real(p, b)) or calls[-1]
    try:
        sft.attention(q, k, v, is_causal=True, name="vis")
    finally:
        _core.visible_max_exp = real
    eo = calls[-1]
    allkeys = np.ceil(np.log2(np.abs(v.numpy()).max(axis=-2)))           # what the old rule took for every row
    assert (eo[..., :12, :] < allkeys[..., None, :]).any()               # early rows keep finer exponents
