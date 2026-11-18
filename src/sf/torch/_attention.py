"""Attention on the MX engine (Alosh's build order, item 4).

``sft.attention`` takes the arguments of ``F.scaled_dot_product_attention``
and runs both of its matmuls on the MX engine at sf16:

1. Scores ``(Q * scale) @ K^T``: one power-of-two exponent per query row
   and per key, output exponents from the wide-mode rule (``eo = l2 - h``).
2. Mask and softmax in float on the host, like LayerNorm.
3. ``P @ V`` (``sf::matmul_wide_pv``): V quantized per key; each row folds
   its visible keys' exponents into P exactly and takes the output exponent
   from the exact bound over those keys, so nothing saturates and keys
   appended after a row never change it (``_core._wide_pv``, DSL 0.16).

``sft.autocast(accumulate="wide")`` and ``sft.convert(..., accumulate="wide")``
route a model's ``F.scaled_dot_product_attention`` calls here, so a Hugging
Face LLM's attention runs on the MX engine with no change to the model.
Attention stays sf16 whatever the layers' dtype (the sf8 / sf16 split).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

ATTENTION_BITS = 16


def attention(query, key, value, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False,
              scale: float | None = None, enable_gqa: bool = False, *, name: str = "attention",
              need_weights: bool = False, softmax: str = "float"):
    """``F.scaled_dot_product_attention`` with both matmuls on the MX engine.

    Statuses are recorded as ``{name}.scores`` and ``{name}.pv``.  Shapes
    follow PyTorch: query (..., L, E), key (..., S, E), value (..., S, Ev);
    with ``enable_gqa`` the key and value heads (dim -3) are repeated to the
    query's.  ``dropout_p`` applies to P in float (training only).
    ``need_weights=True`` also returns P (after softmax, before dropout).
    ``softmax="integer"``: P from sf's integer softmax (``sf.intops``, the
    chip's definition) instead of float on the host.
    """
    if softmax not in ("float", "integer"):
        raise ValueError(f"sft.attention: softmax must be 'float' or 'integer', got {softmax!r}")
    if query.dim() < 2:
        raise ValueError("sft.attention: query must have at least 2 dimensions")
    if attn_mask is not None and is_causal:
        raise ValueError("sft.attention: pass attn_mask or is_causal, not both (as PyTorch)")
    if (not enable_gqa and query.dim() >= 3 and key.dim() >= 3 and key.shape[-3] != query.shape[-3]
            and 1 not in (key.shape[-3], query.shape[-3])):
        # PyTorch refuses this too; caught here so the error names the heads, not a numpy broadcast deep inside
        raise ValueError(f"sft.attention: {query.shape[-3]} query heads and {key.shape[-3]} key heads without "
                         "enable_gqa=True")
    if enable_gqa and query.dim() >= 3 and key.shape[-3] != query.shape[-3]:
        hq, hk = query.shape[-3], key.shape[-3]
        if hq % hk:
            raise ValueError(f"sft.attention: {hq} query heads is not a multiple of {hk} key heads")
        key = key.repeat_interleave(hq // hk, dim=-3)
        value = value.repeat_interleave(hq // hk, dim=-3)
    L, S = query.shape[-2], key.shape[-2]
    s = (1.0 / math.sqrt(query.shape[-1])) if scale is None else scale
    scores = torch.ops.sf.matmul_wide(query * s, key.transpose(-2, -1), ATTENTION_BITS, None,
                                      f"{name}.scores")
    if is_causal:
        keep = torch.ones(L, S, dtype=torch.bool, device=scores.device).tril()
        scores = scores.masked_fill(~keep, float("-inf"))
    elif attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask
    if softmax == "integer":
        from ._intops import int_softmax
        p = int_softmax(scores, dim=-1)
    else:
        p = torch.softmax(scores, dim=-1)
    pd = F.dropout(p, dropout_p) if dropout_p > 0.0 else p
    out = torch.ops.sf.matmul_wide_pv(pd, value, ATTENTION_BITS, f"{name}.pv")
    return (out, p) if need_weights else out


_SDPA_KW = ("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")


def from_sdpa_call(args, kwargs, name, softmax: str = "float"):
    """Run an intercepted ``F.scaled_dot_product_attention`` call here."""
    q, k, v, *rest = args
    kw = dict(zip(_SDPA_KW, rest))
    kw.update({k_: v_ for k_, v_ in kwargs.items() if k_ in _SDPA_KW})
    return attention(q, k, v, name=name, softmax=softmax, **kw)


class AttentionMode(TorchFunctionMode):
    """Catches ``F.scaled_dot_product_attention`` only; names the calls
    ``{prefix}attention.{i}`` in call order."""

    def __init__(self, prefix: str = "", softmax: str = "float"):
        super().__init__()
        self.prefix = prefix
        self.count = 0
        self.softmax = softmax

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.scaled_dot_product_attention:
            name = f"{self.prefix}attention.{self.count}"
            self.count += 1
            return from_sdpa_call(args, kwargs, name, self.softmax)
        return func(*args, **kwargs)


def install(model: torch.nn.Module, softmax: str = "float") -> None:
    """Make ``model``'s forward catch its attention calls (``sft.convert``):
    the mode is entered before the root module's forward and left after it,
    also when it raises.  Calls are named ``attention.0``, ``attention.1``,
    ... in the order the forward makes them."""
    state = {}

    def pre(module, args):
        mode = AttentionMode(softmax=softmax)
        mode.__enter__()
        state.setdefault("modes", []).append(mode)

    def post(module, args, output):
        modes = state.get("modes")
        if modes:
            modes.pop().__exit__(None, None, None)

    model._sf_attention_hooks = (model.register_forward_pre_hook(pre),
                                 model.register_forward_hook(post, always_call=True))


def mha_unsupported(m: torch.nn.MultiheadAttention) -> str | None:
    """Why ``m`` cannot be converted, or None."""
    if m.bias_k is not None or m.bias_v is not None:
        return "bias_k / bias_v are not supported yet; runs in float"
    if m.add_zero_attn:
        return "add_zero_attn is not supported yet; runs in float"
    return None


class SFMultiheadAttention(torch.nn.Module):
    """``nn.MultiheadAttention`` with its projections as SF linears and, in
    wide mode, its attention on the MX engine (``sft.attention``).  Keeps the
    original parameter objects.  Same call and returns as PyTorch:
    ``forward(query, key, value, key_padding_mask=None, need_weights=True,
    attn_mask=None, average_attn_weights=True, is_causal=False)``.
    Statuses: ``{name}.q``, ``.k``, ``.v``, ``.out``, and in wide mode
    ``{name}.scores`` / ``.pv``."""

    def __init__(self, m: torch.nn.MultiheadAttention, name: str, cfg):
        super().__init__()
        why = mha_unsupported(m)
        if why:
            raise ValueError(f"sft: cannot convert {name}: {why}")
        self.embed_dim, self.num_heads, self.head_dim = m.embed_dim, m.num_heads, m.head_dim
        self.batch_first, self.dropout = m.batch_first, m.dropout
        self.same = m._qkv_same_embed_dim
        if self.same:
            self.in_proj_weight = m.in_proj_weight
        else:
            self.q_proj_weight, self.k_proj_weight, self.v_proj_weight = m.q_proj_weight, m.k_proj_weight, m.v_proj_weight
        self.in_proj_bias = m.in_proj_bias
        self.out_proj = m.out_proj
        self.sf = cfg
        self.sf_name = name

    def _proj(self, x, i, part):
        from ._ops import call_linear
        E = self.embed_dim
        if self.same:
            w = self.in_proj_weight[i * E:(i + 1) * E]
        else:
            w = (self.q_proj_weight, self.k_proj_weight, self.v_proj_weight)[i]
        b = None if self.in_proj_bias is None else self.in_proj_bias[i * E:(i + 1) * E]
        return call_linear(x, w, b, self.sf, f"{self.sf_name}.{part}")

    def forward(self, query, key, value, key_padding_mask=None, need_weights: bool = True, attn_mask=None,
                average_attn_weights: bool = True, is_causal: bool = False):
        from ._ops import call_linear
        unbatched = query.dim() == 2
        if unbatched:
            query, key, value = query.unsqueeze(1), key.unsqueeze(1), value.unsqueeze(1)
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask.unsqueeze(0)
        if self.batch_first and not unbatched:
            query, key, value = (t.transpose(0, 1) for t in (query, key, value))
        L, N, _ = query.shape                                      # (L, N, E) from here
        S = key.shape[0]
        H, hd = self.num_heads, self.head_dim
        q, k, v = (self._proj(t, i, p) for i, (t, p) in enumerate(((query, "q"), (key, "k"), (value, "v"))))
        q, k, v = (t.reshape(t.shape[0], N, H, hd).permute(1, 2, 0, 3) for t in (q, k, v))   # (N, H, len, hd)

        mask = None                                                 # additive, broadcast to (N, H, L, S)
        if attn_mask is None and is_causal:
            attn_mask = torch.ones(L, S, dtype=torch.bool, device=q.device).triu(1)
        if attn_mask is not None:
            am = attn_mask
            if am.dim() == 3:
                am = am.reshape(N, H, L, S)
            mask = (torch.zeros(am.shape, dtype=q.dtype).masked_fill(am, float("-inf"))
                    if am.dtype == torch.bool else am.to(q.dtype))
        if key_padding_mask is not None:
            kp = key_padding_mask.reshape(N, 1, 1, S)
            kp = (torch.zeros(kp.shape, dtype=q.dtype).masked_fill(kp, float("-inf"))
                  if kp.dtype == torch.bool else kp.to(q.dtype))
            mask = kp if mask is None else mask + kp

        p_drop = self.dropout if self.training else 0.0
        if self.sf.accumulate == "wide":
            r = attention(q, k, v, attn_mask=mask, dropout_p=p_drop, name=self.sf_name, need_weights=need_weights)
        else:
            r = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=p_drop)
            if need_weights:
                s = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
                r = (r, torch.softmax(s if mask is None else s + mask, dim=-1))
        out, w = r if need_weights else (r, None)
        out = out.permute(2, 0, 1, 3).reshape(L, N, self.embed_dim)
        out = call_linear(out, self.out_proj.weight, self.out_proj.bias, self.sf, f"{self.sf_name}.out")
        if self.batch_first and not unbatched:
            out = out.transpose(0, 1)
        if unbatched:
            out = out.squeeze(1)
        if w is not None:
            w = w.mean(dim=1) if average_attn_weights else w
            if unbatched:
                w = w.squeeze(0)
        return out, w

    def extra_repr(self) -> str:
        return (f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, batch_first={self.batch_first}, "
                f"{self.sf.describe()}, name={self.sf_name!r}")
