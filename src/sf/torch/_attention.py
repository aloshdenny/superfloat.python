"""Attention on the MX engine (Alosh's build order, item 4).

``sft.attention`` takes the arguments of ``F.scaled_dot_product_attention``
and runs both of its matmuls on the MX engine at sf16:

1. Scores ``(Q * scale) @ K^T``: one power-of-two exponent per query row
   and per key, output exponents from the wide-mode rule (``eo = l2 - h``).
2. Mask and softmax in float on the host, like LayerNorm.
3. ``P @ V``: one exponent per query row of P and per channel of V over the
   keys.  Rows of P sum to 1, so |out[:, c]| <= max |V[:, c]|: the output
   exponent per channel is V's own (``sf::matmul_wide_pv``), a tight bound
   that needs no learning.  The redo at l2 stays as a guard.

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
              scale: float | None = None, enable_gqa: bool = False, *, name: str = "attention"):
    """``F.scaled_dot_product_attention`` with both matmuls on the MX engine.

    Statuses are recorded as ``{name}.scores`` and ``{name}.pv``.  Shapes
    follow PyTorch: query (..., L, E), key (..., S, E), value (..., S, Ev);
    with ``enable_gqa`` the key and value heads (dim -3) are repeated to the
    query's.  ``dropout_p`` applies to P in float (training only).
    """
    if query.dim() < 2:
        raise ValueError("sft.attention: query must have at least 2 dimensions")
    if attn_mask is not None and is_causal:
        raise ValueError("sft.attention: pass attn_mask or is_causal, not both (as PyTorch)")
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
    p = torch.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        p = F.dropout(p, dropout_p)
    return torch.ops.sf.matmul_wide_pv(p, value, ATTENTION_BITS, f"{name}.pv")


_SDPA_KW = ("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")


def from_sdpa_call(args, kwargs, name):
    """Run an intercepted ``F.scaled_dot_product_attention`` call here."""
    q, k, v, *rest = args
    kw = dict(zip(_SDPA_KW, rest))
    kw.update({k_: v_ for k_, v_ in kwargs.items() if k_ in _SDPA_KW})
    return attention(q, k, v, name=name, **kw)


class AttentionMode(TorchFunctionMode):
    """Catches ``F.scaled_dot_product_attention`` only; names the calls
    ``{prefix}attention.{i}`` in call order."""

    def __init__(self, prefix: str = ""):
        super().__init__()
        self.prefix = prefix
        self.count = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is F.scaled_dot_product_attention:
            name = f"{self.prefix}attention.{self.count}"
            self.count += 1
            return from_sdpa_call(args, kwargs, name)
        return func(*args, **kwargs)


def install(model: torch.nn.Module) -> None:
    """Make ``model``'s forward catch its attention calls (``sft.convert``):
    the mode is entered before the root module's forward and left after it,
    also when it raises.  Calls are named ``attention.0``, ``attention.1``,
    ... in the order the forward makes them."""
    state = {}

    def pre(module, args):
        mode = AttentionMode()
        mode.__enter__()
        state.setdefault("modes", []).append(mode)

    def post(module, args, output):
        modes = state.get("modes")
        if modes:
            modes.pop().__exit__(None, None, None)

    model._sf_attention_hooks = (model.register_forward_pre_hook(pre),
                                 model.register_forward_hook(post, always_call=True))
