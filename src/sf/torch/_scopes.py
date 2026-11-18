"""The parts of a model that are not ``nn.Linear`` but are still matmuls.

``sft.convert`` swaps the layers it knows for SF modules.  Three kinds of
matmul live inside modules it cannot swap, because the model calls the
torch function directly:

* **MoE experts.**  Hugging Face MoE models (Mixtral, Qwen3-MoE,
  DeepSeek-V3, ...) keep every expert in one 3-D parameter (``gate_up_proj``
  (E, out, in), ``down_proj``) and call ``F.linear`` on one expert's slice,
  or a grouped matmul over all of them.
* **MoE routers.**  A module with a 2-D ``weight`` that calls ``F.linear``.
* **Eager attention.**  ``torch.matmul(q, k^T)`` and ``torch.matmul(P, v)``
  inside an attention module that does not use
  ``F.scaled_dot_product_attention`` (``attn_implementation="eager"``,
  Gemma 2 with logit soft-capping, older models).

For each, ``convert`` registers a forward pre-hook and a forward hook on the
module.  While the module runs, a ``TorchFunctionMode`` sends exactly those
calls to the SF ops (``sf::linear_wide`` / ``sf::matmul_wide`` /
``sf::matmul_wide_pv``, or the chain ops).  Only calls whose weight is the
module's own parameter (experts, routers) or that happen inside an attention
module (and not in a rotary-embedding module under it) are taken; every
other call passes through unchanged.  The settings live on the module
(``_sf_scope``), not in the hooks' closures, so a deep copy keeps working.

The module also reads weights that are stored in a quantized format
(``read_weight``): an FP8 weight with block scales (DeepSeek-V3 style,
``weight_scale_inv``) is dequantized to float and then quantized to sf like
any other weight; every other non-float weight (int8, packed 4-bit, GPTQ /
AWQ ``qweight``) is refused with the reason, so the layer runs as it was
instead of giving a wrong answer.
"""
from __future__ import annotations

import copy
import math
import re
import threading
import types

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from ._ops import call_linear, call_matmul

ATTENTION_BITS = 16

_LINEAR = {F.linear}
_MATMUL = {torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__, torch.bmm, torch.Tensor.bmm}

_tl = threading.local()                     # .suspend: rotary modules under an attention scope


def _suspended() -> int:
    return getattr(_tl, "suspend", 0)


# --- weights in other formats ----------------------------------------------------------------

_FP8 = tuple(getattr(torch, n) for n in ("float8_e4m3fn", "float8_e5m2", "float8_e4m3fnuz", "float8_e5m2fnuz")
             if hasattr(torch, n))
_SCALE_NAMES = ("weight_scale_inv", "weight_scale")


def _scale_of(m):
    for n in _SCALE_NAMES:
        s = getattr(m, n, None)
        if isinstance(s, torch.Tensor):
            return n, s
    return None, None


def dequantize_block_scaled(w: torch.Tensor, s: torch.Tensor, block=None) -> torch.Tensor:
    """``w * s`` with one scale per block of ``w`` (2-D), as float32.  ``s`` is
    a scalar (per tensor) or (ceil(out / b0), ceil(in / b1)); ``block`` is
    (b0, b1), found from the shapes when not given."""
    out, inn = w.shape
    wf = w.to(torch.float32)
    sf_ = s.to(torch.float32)
    if sf_.numel() == 1:
        return wf * sf_.reshape(())
    if sf_.dim() != 2:
        raise ValueError(f"block scales of shape {tuple(s.shape)} for a weight of shape {tuple(w.shape)}")
    b0, b1 = block if block is not None else (math.ceil(out / sf_.shape[0]), math.ceil(inn / sf_.shape[1]))
    if math.ceil(out / b0) != sf_.shape[0] or math.ceil(inn / b1) != sf_.shape[1]:
        raise ValueError(f"block scales of shape {tuple(s.shape)} do not tile a {tuple(w.shape)} weight "
                         f"in {b0}x{b1} blocks")
    full = sf_.repeat_interleave(b0, 0)[:out].repeat_interleave(b1, 1)[:, :inn]
    return wf * full


def read_weight(m: torch.nn.Module):
    """How ``convert`` should read the weight of a linear-like module.

    Returns ``(weight, note, problem)``: ``weight`` is None to use the
    module's own parameter, or a float32 tensor read from a quantized format
    (with ``note`` saying which); ``problem`` is a reason the layer cannot be
    converted (it then runs as it was)."""
    w = getattr(m, "weight", None)
    if not isinstance(w, torch.Tensor):
        return None, None, "no weight tensor"
    from . import _mxfp4
    if _mxfp4.is_mxfp4_linear(m):
        try:
            return _mxfp4.read_mxfp4_linear(m), "MXFP4 weights, read with their E8M0 scale per block of 32", None
        except (TypeError, ValueError) as e:
            return None, None, f"MXFP4 weights that could not be read: {e}"
    if w.dtype in _FP8:
        name, s = _scale_of(m)
        if s is None:
            return None, None, f"{w.dtype} weights with no scale (weight_scale_inv) next to them"
        if w.dim() != 2:
            return None, None, f"{w.dtype} weights of shape {tuple(w.shape)}"
        blk = getattr(m, "block_size", None)
        try:
            dq = dequantize_block_scaled(w.detach(), s.detach(), tuple(blk) if blk else None)
        except ValueError as e:
            return None, None, str(e)
        b = "per tensor" if s.numel() == 1 else "blocks of {}x{}".format(
            *(blk or (math.ceil(w.shape[0] / s.shape[0]), math.ceil(w.shape[1] / s.shape[1]))))
        return dq, f"FP8 weights with scales in {b}, read through {name}", None
    if not w.is_floating_point():
        return None, None, (f"weights stored as {w.dtype} (a quantized format sf cannot read); "
                            f"runs as it was")
    return None, None, None


def custom_forward(m: torch.nn.Module, base: type) -> bool:
    """True when ``m`` is a subclass of ``base`` with its own ``forward``: sf
    cannot know what it computes from the weight alone."""
    return type(m) is not base and type(m).forward is not base.forward


_PACKED = tuple(d for d in (torch.int8, torch.uint8, torch.int16, torch.int32,
                            getattr(torch, "int4", None), getattr(torch, "uint4", None)) if d is not None)
_QUANT_BUFFERS = re.compile(r"^(qweight|qzeros|g_idx|packed_weight|weight_packed|wscales|scales)$")


def quantized_storage(m: torch.nn.Module):
    """Own integer tensors (2-D or more) of a module sf does not know, such as
    GPTQ / AWQ ``qweight``: the reason it stays as it was, or None."""
    if "mxfp4" in type(m).__name__.lower():
        return ("holds MXFP4 weights packed for GPU kernels; load the checkpoint into a float model first "
                "(sft.load_mxfp4, or transformers' Mxfp4Config(dequantize=True)); runs as it was")
    for n, t in list(m.named_parameters(recurse=False)) + list(m.named_buffers(recurse=False)):
        if t is None:
            continue
        if t.dim() >= 2 and (t.dtype in _PACKED or (_QUANT_BUFFERS.match(n) and not t.is_floating_point())):
            return f"holds quantized weights ({n}: {t.dtype}) in a format sf cannot read; runs as it was"
        if t is not None and t.dtype in _FP8 and t.dim() >= 2:
            return f"holds {t.dtype} weights ({n}) sf cannot read here; runs as it was"
    return None


_NOT_MATMUL = re.compile(r"(embed|token|pos|bias|norm|scale|A_log|^D$)", re.I)


def unconverted_weights(m: torch.nn.Module):
    """Own float tensors of 2 dimensions or more that no rule converted (for
    the report): names, or [] when there are none.  Embedding tables are
    lookups, not matmuls, and are left out."""
    if isinstance(m, torch.nn.Embedding):
        return []
    return [n for n, p in m.named_parameters(recurse=False)
            if p is not None and p.dim() >= 2 and p.is_floating_point() and not _NOT_MATMUL.search(n)]


# --- which modules get a scope -----------------------------------------------------------------

def experts_params(m: torch.nn.Module):
    """The 3-D float parameters of an experts module (Hugging Face's fused
    experts), or [] if ``m`` is not one."""
    if "expert" not in type(m).__name__.lower():
        return []
    return [n for n, p in m.named_parameters(recurse=False) if p is not None and p.dim() == 3]


def is_router(m: torch.nn.Module) -> bool:
    name = type(m).__name__.lower()
    w = getattr(m, "weight", None)
    return ("router" in name) and isinstance(w, torch.nn.Parameter) and w.dim() == 2 \
        and not isinstance(m, torch.nn.Linear)


def is_linear_router(parent: torch.nn.Module, attr: str, child: torch.nn.Module) -> bool:
    """An ``nn.Linear`` that routes tokens in an MoE block (older layouts:
    ``block.gate = nn.Linear(hidden, n_experts)``)."""
    pn = type(parent).__name__.lower()
    return isinstance(child, torch.nn.Linear) and attr in ("gate", "router") and "moe" in pn


_ATTN = re.compile(r"Attention\d*$")


def is_attention_scope(m: torch.nn.Module) -> bool:
    return bool(_ATTN.search(type(m).__name__)) and not isinstance(m, torch.nn.MultiheadAttention) \
        and not type(m).__module__.startswith("sf.")


def is_rotary(m: torch.nn.Module) -> bool:
    return "rotary" in type(m).__name__.lower()


def hf_experts(m: torch.nn.Module) -> bool:
    """A Hugging Face experts module (``use_experts_implementation``): its
    layout is described by flags, so sf can run it with ``sf_experts_forward``."""
    return all(hasattr(m, a) for a in ("has_gate", "has_bias", "is_transposed")) and hasattr(m, "num_experts")


def _expert_proj(self, x, pn, e):
    w = getattr(self, pn)[e]
    b = getattr(self, pn + "_bias")[e] if self.has_bias else None
    cfg, name = self._sf_experts
    if self.is_transposed:                                  # stored (in, out)
        w = w.t()
    return call_linear(x, w, b, cfg, f"{name}.{pn}[{e}]")


def sf_experts_forward(self, hidden_states, top_k_index, top_k_weights):
    """The experts of a Hugging Face MoE block with every projection on sf:
    the same maths as the model's own per-expert loop (layout from the
    module's flags: gate, bias, transposed, post-expert norm, its own
    ``_apply_gate``), one SF linear per expert and projection.  The loop is
    over expert numbers (Python ints), so every op has a fixed name.

    Under ``torch.compile`` each expert runs on every token and its output
    is multiplied by the token's routing weight for it (0 when the token
    was not sent there): no shape depends on the routing, so the graph does
    not break and every projection goes to the target.  Wide mode computes
    each token's row on its own (its exponent, its norms, its redo), so the
    rows are the bits the sparse loop gives, and adding exact zeros in
    expert order leaves the float sums as they were; it costs
    num_experts / top_k times the host time."""
    out = torch.zeros_like(hidden_states)
    if torch.compiler.is_compiling():
        for e in range(self.num_experts):
            hit = top_k_index == e
            w = torch.where(hit, top_k_weights, torch.zeros_like(top_k_weights)).sum(-1)
            if self.has_gate:
                h = self._apply_gate(_expert_proj(self, hidden_states, "gate_up_proj", e))
            else:
                h = self.act_fn(_expert_proj(self, hidden_states, "up_proj", e))
            y = _expert_proj(self, h, "down_proj", e)
            if getattr(self, "has_post_expert_norm", False):
                y = self.post_expert_norm(y)
            out = out + torch.where(hit.any(-1, keepdim=True), (y * w[:, None]).to(out.dtype), torch.zeros_like(out))
        return out
    for e in range(self.num_experts):
        top_k_pos, token_idx = torch.where((top_k_index == e).transpose(0, 1))
        if token_idx.numel() == 0:
            continue
        x = hidden_states[token_idx]
        if self.has_gate:
            h = self._apply_gate(_expert_proj(self, x, "gate_up_proj", e))
        else:
            h = self.act_fn(_expert_proj(self, x, "up_proj", e))
        y = _expert_proj(self, h, "down_proj", e)
        if getattr(self, "has_post_expert_norm", False):
            y = self.post_expert_norm(y)
        y = y * top_k_weights[token_idx, top_k_pos, None]
        out.index_add_(0, token_idx, y.to(out.dtype))
    return out


def install_experts(m: torch.nn.Module, name: str, cfg) -> None:
    """Run ``m`` (a Hugging Face experts module) with ``sf_experts_forward``.
    Only this instance changes; its parameters and state dict keys stay."""
    m._sf_experts = (cfg, name)
    m.forward = types.MethodType(sf_experts_forward, m)


def eager_experts(m: torch.nn.Module) -> bool:
    """Make a Hugging Face experts module use its own loop over experts
    (``F.linear`` on one expert's slice) instead of a grouped matmul, which
    has no SF version.  Changes this module's view of the config only."""
    cfg = getattr(m, "config", None)
    impl = getattr(cfg, "_experts_implementation", None)
    if cfg is None or impl in (None, "eager"):
        return False
    c = copy.copy(cfg)
    try:
        c._experts_implementation = "eager"
    except Exception:                                       # noqa: BLE001  (a read-only config)
        return False
    m.config = c
    return True


# --- the mode and the hooks -----------------------------------------------------------------

def _owner(t: torch.Tensor):
    while getattr(t, "_base", None) is not None:
        t = t._base
    return t


_SOFTMAX = {F.softmax, torch.softmax, torch.Tensor.softmax, torch.nn.functional.softmax}


def _param_for(module, w):
    """The name of ``module``'s parameter that ``w`` is (2-D) or one expert's
    slice of (3-D), found by shape, so it traces under ``torch.compile``;
    None if none fits."""
    for n, p in module.named_parameters(recurse=False):
        if p is None or not p.is_floating_point():
            continue
        if tuple(w.shape) == tuple(p.shape) or (p.dim() == 3 and tuple(w.shape) == tuple(p.shape[1:])):
            return n, p
    return None


def _expert_index(p, w):
    """Which expert's slice ``w`` is (eager runs only: it reads storage offsets)."""
    if torch.compiler.is_compiling() or p.dim() != 3 or not p.stride(0):
        return None
    try:
        if _owner(w) is not p:
            return None
        return (w.storage_offset() - p.storage_offset()) // p.stride(0)
    except Exception:                                       # noqa: BLE001
        return None


class ScopeMode(TorchFunctionMode):
    """While one scoped module runs: experts / router: ``F.linear`` (and
    ``x @ w``) with one of the module's own weights; attention: every
    ``torch.matmul`` between activations, the one after a softmax as P @ V.
    Decisions use shapes and call order only, so ``torch.compile`` traces
    the same choices the eager run makes."""

    def __init__(self, module):
        super().__init__()
        self.module = module
        self.kind, self.name, self.cfg = module._sf_scope
        self.after_softmax = False

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if _suspended():
            return func(*args, **kwargs)
        if self.kind in ("experts", "router"):
            if func in _LINEAR:
                x = args[0]
                w = args[1] if len(args) > 1 else kwargs.get("weight")
                b = args[2] if len(args) > 2 else kwargs.get("bias")
                if isinstance(w, torch.Tensor) and w.dim() == 2 and w.is_floating_point() and x.is_floating_point():
                    own = _param_for(self.module, w)
                    if own is not None:
                        return call_linear(x, w, b, self.cfg, self._label(own, w))
            elif func in _MATMUL and len(args) == 2 and not kwargs:
                a, w = args
                if (isinstance(w, torch.Tensor) and w.dim() == 2 and w.is_floating_point() and a.dim() >= 2
                        and a.is_floating_point()):
                    own = _param_for(self.module, w)
                    if own is not None:
                        return call_matmul(a, w, self.cfg, self._label(own, w))
        elif self.kind == "attention":
            if func in _SOFTMAX:
                self.after_softmax = True
                if getattr(self.module, "_sf_softmax", "float") == "integer":
                    from ._intops import int_softmax
                    x = args[0]
                    dim = args[1] if len(args) > 1 else kwargs.get("dim", -1)
                    dt = kwargs.get("dtype") or next((a for a in args[2:] if isinstance(a, torch.dtype)), None)
                    return int_softmax(x if dt is None else x.to(dt), dim=dim)
            elif func in _MATMUL and len(args) == 2 and not kwargs:
                a, b = args
                if (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor) and a.dim() >= 2 and b.dim() >= 2
                        and a.is_floating_point() and b.is_floating_point()):
                    if self.after_softmax:
                        # P @ V: rows of P are probabilities, so each output channel's exponent is V's
                        # (the redo at l2 still guards a model that rescales P in between)
                        self.after_softmax = False
                        return torch.ops.sf.matmul_wide_pv(a, b, ATTENTION_BITS, f"{self.name}.pv")
                    return torch.ops.sf.matmul_wide(a, b, ATTENTION_BITS, None, f"{self.name}.scores")
        return func(*args, **kwargs)

    def _label(self, own, w):
        if self.kind == "router":
            return self.name
        n, p = own
        idx = _expert_index(p, w)
        return f"{self.name}.{n}" + (f"[{idx}]" if idx is not None else "")


def _pre(module, args, kwargs=None):
    mode = ScopeMode(module)
    mode.__enter__()
    module.__dict__.setdefault("_sf_modes", []).append(mode)


def _post(module, args, *rest):
    modes = module.__dict__.get("_sf_modes")
    if modes:
        modes.pop().__exit__(None, None, None)


def _suspend_pre(module, args):
    _tl.suspend = _suspended() + 1


def _suspend_post(module, args, output):
    _tl.suspend = max(0, _suspended() - 1)


def add_scope(module, kind: str, name: str, cfg) -> None:
    module._sf_scope = (kind, name, cfg)
    if getattr(module, "_sf_scope_hooks", None):
        return
    module._sf_scope_hooks = (module.register_forward_pre_hook(_pre),
                              module.register_forward_hook(_post, always_call=True))


def add_suspend(module) -> None:
    if getattr(module, "_sf_suspend_hooks", None):
        return
    module._sf_suspend_hooks = (module.register_forward_pre_hook(_suspend_pre),
                                module.register_forward_hook(_suspend_post, always_call=True))
