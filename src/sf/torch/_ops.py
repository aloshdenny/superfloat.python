"""Torch custom ops in the ``sf`` namespace.

* ``sf::quantize(x, bits, mode, scale)``: fake quantization (spec 2.0).
  Values snapped to SF codes, float storage, clipped straight-through
  gradient (1 where the input was in range, 0 where it saturated).
* ``sf::linear(x, weight, bias, bits, wscale, headroom, name)``: chip-exact
  SF linear with the scale policy of ``_core``.  Straight-through gradient
  through the quantization, i.e. the gradient of ``x @ W_q^T + b`` with the
  dequantized weights ``W_q``.
* ``sf::matmul(a, b, bits, headroom, name)``: chip-exact SF matmul of two
  activations, straight-through gradient.
* ``sf::linear_wide(x, weight, bias, bits, shift, name)`` and
  ``sf::matmul_wide(a, b, bits, shift, name)``: the same on the MX engine
  (``accumulate="wide"``), with the wide-mode shift rule; ``shift`` (an int
  or None) overrides the rule.  Separate ops, so chain graphs are unchanged.

Each op is one opaque node in a ``torch.compile`` graph, which is what lets
the ``superfloat`` backend send exactly these to the chip.  Eager execution
runs the host reference; the backend swaps in a device runner.

Output dtype: ``float32`` for float16 / bfloat16 / float32 inputs and
``float64`` for float64.  SF16 values need 16 significant bits; bfloat16 has
8 and float16 11, so neither can hold them exactly.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .. import _linalg
from . import _core
from ._status import SFLayerStatus, record

_FMT_NAMES = {b: f"sf{b}" for b in range(4, 17)}


def out_dtype(dt: torch.dtype) -> torch.dtype:
    return torch.float64 if dt == torch.float64 else torch.float32


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", torch.float64).numpy()


def _record(name, op, runner, bits, r: _core.CoreResult):
    record(SFLayerStatus(
        name=name, op=op, placement=runner.placement, dtype=_FMT_NAMES[bits], outputs=r.outputs,
        saturated=r.saturated, exact=r.exact, underflow=r.underflow,
        input_underflow=r.input_underflow, weight_underflow=r.weight_underflow,
        bias_saturated=r.bias_saturated, headroom_bits=r.headroom_bits, act_scale=r.act_scale,
        accumulate=r.accumulate, shift_min=r.shift_min, shift_max=r.shift_max, h=r.h,
        redone=r.redone, flushed=r.flushed, acc_clipped=r.acc_clipped, guard_raised=r.guard_raised))


def _mx_of(runner):
    """The MX version the runner's target has (the host follows the default device)."""
    from .. import _array, _device
    if runner.placement == "host":
        return _array._host_mx()
    mx = _device.get_device(runner.placement).mx
    if not mx:
        import sf
        raise sf.SFUnsupportedError(f"accumulate=\"wide\" needs an MX engine; {runner.placement} has none")
    return _linalg.mx_version(mx)


# --- implementations shared by eager and the compile backend ---------------------------

def linear_impl(x, weight, bias, bits, wscale, headroom, name, runner):
    K = weight.shape[1]
    N = weight.shape[0]
    if x.shape[-1] != K:
        raise ValueError(f"sf.linear: input has {x.shape[-1]} features, weight expects {K}")
    lead = x.shape[:-1]
    r = _core.linear(_np(x).reshape(-1, K), _np(weight), None if bias is None else _np(bias),
                     bits, wscale, headroom, runner)
    _record(name, "linear", runner, bits, r)
    return torch.from_numpy(r.y.reshape(*lead, N)).to(out_dtype(x.dtype))


class WeightCache:
    """Weights prepared once for the host's wide path (``_core.PreparedWeight``).

    A model's weights do not change between forwards, but preparing them
    (quantize per channel, norms, compact codes) cost most of the time of a
    forward: about 20 s of each generated token on Qwen2.5-0.5B.  Entries
    are keyed by the tensor's storage, shape, strides and version counter,
    so a weight changed in place (an optimizer step) is prepared again, and
    each entry holds a weak reference to the tensor that owns the memory: a
    new weight allocated where a freed one was is never mistaken for it.
    Least recently used entries go first once the budget is reached:
    ``SF_WEIGHT_CACHE_MB`` (default 4096; 0 turns the cache off).  3 bytes
    per weight, so 4 GB holds a 1.3B-parameter model."""

    def __init__(self):
        import os
        from collections import OrderedDict
        self.budget = int(float(os.environ.get("SF_WEIGHT_CACHE_MB", "4096")) * 2 ** 20)
        self.items = OrderedDict()
        self.bytes = 0
        self.hits = self.misses = 0

    @staticmethod
    def key(w, bits, mx):
        return (w.untyped_storage().data_ptr(), w.storage_offset(), tuple(w.shape), tuple(w.stride()),
                w._version, w.dtype, bits, mx)

    @staticmethod
    def owner(w):
        """The tensor that owns ``w``'s memory (a parameter, for the slices
        and transposes layers pass in)."""
        while w._base is not None:
            w = w._base
        return w

    def get(self, w, bits, mx):
        import weakref
        if self.budget <= 0 or w.device.type != "cpu":
            return None
        k = self.key(w, bits, mx)
        own = self.owner(w)
        hit = self.items.get(k)
        if hit is not None:
            pw, ref = hit
            if ref() is own:
                self.items.move_to_end(k)
                self.hits += 1
                return pw
            self.items.pop(k)                                   # same address, different tensor
            self.bytes -= pw.nbytes
        self.misses += 1
        pw = _core.prepare_weight(_np(w), bits)
        if pw is None or pw.nbytes > self.budget:
            return None
        while self.items and self.bytes + pw.nbytes > self.budget:
            _, (old, _) = self.items.popitem(last=False)
            self.bytes -= old.nbytes
        self.items[k] = (pw, weakref.ref(own))
        self.bytes += pw.nbytes
        return pw

    def clear(self):
        self.items.clear()
        self.bytes = 0
        self.hits = self.misses = 0


weight_cache = WeightCache()


def linear_wide_impl(x, weight, bias, bits, shift, name, runner):
    K, N = weight.shape[1], weight.shape[0]
    if x.shape[-1] != K:
        raise ValueError(f"sf.linear: input has {x.shape[-1]} features, weight expects {K}")
    lead = x.shape[:-1]
    mx = _mx_of(runner)
    pw = weight_cache.get(weight, bits, mx) if isinstance(runner, _core.HostRunner) else None
    r = _core.wide_linear(_np(x).reshape(-1, K), None if pw is not None else _np(weight),
                          None if bias is None else _np(bias), bits, shift, runner, mx, prepared=pw)
    _record(name, "linear", runner, bits, r)
    return torch.from_numpy(r.y.reshape(*lead, N)).to(out_dtype(x.dtype))


def matmul_wide_impl(a, b, bits, shift, name, runner, out_rule="heuristic"):
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("sf.matmul expects operands with at least 2 dimensions")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"sf.matmul: shapes {tuple(a.shape)} and {tuple(b.shape)} are not aligned")
    r = _core.wide_matmul(_np(a), _np(b), bits, shift, runner, _mx_of(runner), out_rule)
    _record(name, "matmul", runner, bits, r)
    return torch.from_numpy(np.ascontiguousarray(r.y)).to(out_dtype(torch.promote_types(a.dtype, b.dtype)))


def matmul_impl(a, b, bits, headroom, name, runner):
    if a.dim() < 2 or b.dim() < 2:
        raise ValueError("sf.matmul expects operands with at least 2 dimensions")
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(f"sf.matmul: shapes {tuple(a.shape)} and {tuple(b.shape)} are not aligned")
    r = _core.matmul(_np(a), _np(b), bits, headroom, runner)
    _record(name, "matmul", runner, bits, r)
    return torch.from_numpy(np.ascontiguousarray(r.y)).to(out_dtype(torch.promote_types(a.dtype, b.dtype)))


# --- sf::quantize ------------------------------------------------------------------------------

@torch.library.custom_op("sf::quantize", mutates_args=())
def quantize(x: torch.Tensor, bits: int, mode: str, scale: float) -> torch.Tensor:
    import sf
    vals, codes, s, ev = _core.fake_quantize(_np(x), bits, mode, scale)
    fmt = sf.dtypes[bits - 4]._fmt
    from .._array import _handle_events
    _handle_events(ev, fmt, "quantize", int(np.size(codes)))      # Stage 1 policy, same events
    return torch.from_numpy(vals).to(out_dtype(x.dtype))


@quantize.register_fake
def _(x, bits, mode, scale):
    return x.new_empty(x.shape, dtype=out_dtype(x.dtype))


def _quantize_setup(ctx, inputs, output):
    x, bits, mode, scale = inputs
    ctx.save_for_backward(x)
    ctx.bits, ctx.mode, ctx.scale = bits, mode, scale


def _quantize_backward(ctx, g):
    (x,) = ctx.saved_tensors
    import sf
    fmt = sf.dtypes[ctx.bits - 4]._fmt
    xv = x.detach().to(torch.float64)
    if ctx.mode == "dynamic":
        finite = xv[torch.isfinite(xv)]
        amax = float(finite.abs().max()) if finite.numel() else 0.0
        s = amax / (fmt.max_code / fmt.scale) if amax > 0 else 1.0
    elif ctx.mode == "fixed":
        s = ctx.scale
    else:
        s = 1.0
    threshold = 1.0 - 2.0 ** -fmt.bits
    inside = (xv / s).abs() < threshold                  # clipped straight-through estimator
    return (g * inside.to(g.dtype)).to(x.dtype), None, None, None


torch.library.register_autograd("sf::quantize", _quantize_backward, setup_context=_quantize_setup)


# --- sf::linear ---------------------------------------------------------------------------------

@torch.library.custom_op("sf::linear", mutates_args=())
def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], bits: int,
           wscale: str, headroom: str, name: str) -> torch.Tensor:
    return linear_impl(x, weight, bias, bits, wscale, headroom, name, _core.HostRunner())


@linear.register_fake
def _(x, weight, bias, bits, wscale, headroom, name):
    return x.new_empty(tuple(x.shape[:-1]) + (weight.shape[0],), dtype=out_dtype(x.dtype))


def _linear_setup(ctx, inputs, output):
    x, weight, bias, bits, wscale, headroom, name = inputs
    ctx.save_for_backward(x, weight)
    ctx.has_bias = bias is not None
    ctx.bits, ctx.wscale = bits, wscale


def _linear_backward(ctx, g):
    x, weight = ctx.saved_tensors
    import sf
    fmt = sf.dtypes[ctx.bits - 4]._fmt
    w_q = torch.from_numpy(_core.dequantize_weight(fmt, _np(weight), ctx.wscale)).to(g.dtype)
    g2 = g.reshape(-1, g.shape[-1])
    x2 = x.detach().reshape(-1, x.shape[-1]).to(g.dtype)
    grad_x = (g2 @ w_q).reshape(x.shape).to(x.dtype) if ctx.needs_input_grad[0] else None
    grad_w = (g2.T @ x2).to(weight.dtype) if ctx.needs_input_grad[1] else None
    grad_b = g2.sum(0).to(weight.dtype) if (ctx.has_bias and ctx.needs_input_grad[2]) else None
    return grad_x, grad_w, grad_b, None, None, None, None


torch.library.register_autograd("sf::linear", _linear_backward, setup_context=_linear_setup)


# --- sf::matmul ---------------------------------------------------------------------------------

@torch.library.custom_op("sf::matmul", mutates_args=())
def matmul(a: torch.Tensor, b: torch.Tensor, bits: int, headroom: str, name: str) -> torch.Tensor:
    return matmul_impl(a, b, bits, headroom, name, _core.HostRunner())


@matmul.register_fake
def _(a, b, bits, headroom, name):
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return a.new_empty(tuple(batch) + (a.shape[-2], b.shape[-1]),
                       dtype=out_dtype(torch.promote_types(a.dtype, b.dtype)))


def _reduce_to(t: torch.Tensor, shape) -> torch.Tensor:
    """Sum a broadcast gradient back to ``shape``."""
    while t.dim() > len(shape):
        t = t.sum(0)
    for i, n in enumerate(shape):
        if n == 1 and t.shape[i] != 1:
            t = t.sum(i, keepdim=True)
    return t


def _matmul_setup(ctx, inputs, output):
    a, b, bits, headroom, name = inputs
    ctx.save_for_backward(a, b)


def _matmul_backward(ctx, g):
    a, b = ctx.saved_tensors
    ad, bd = a.detach().to(g.dtype), b.detach().to(g.dtype)
    grad_a = _reduce_to(g @ bd.transpose(-1, -2), a.shape).to(a.dtype) if ctx.needs_input_grad[0] else None
    grad_b = _reduce_to(ad.transpose(-1, -2) @ g, b.shape).to(b.dtype) if ctx.needs_input_grad[1] else None
    return grad_a, grad_b, None, None, None


torch.library.register_autograd("sf::matmul", _matmul_backward, setup_context=_matmul_setup)

# --- wide mode: sf::linear_wide, sf::matmul_wide ----------------------------------------------

@torch.library.custom_op("sf::linear_wide", mutates_args=())
def linear_wide(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor], bits: int,
                shift: Optional[int], name: str) -> torch.Tensor:
    return linear_wide_impl(x, weight, bias, bits, shift, name, _core.HostRunner())


@linear_wide.register_fake
def _(x, weight, bias, bits, shift, name):
    return x.new_empty(tuple(x.shape[:-1]) + (weight.shape[0],), dtype=out_dtype(x.dtype))


def _linear_wide_setup(ctx, inputs, output):
    x, weight, bias, bits, shift, name = inputs
    ctx.save_for_backward(x, weight)
    ctx.has_bias = bias is not None
    ctx.bits = bits


def _linear_wide_backward(ctx, g):
    """Straight-through: the gradient of x @ W_q^T + b, W_q the weights as
    wide mode quantizes them (one power-of-two exponent per output channel)."""
    x, weight = ctx.saved_tensors
    import sf
    from .. import _heuristic
    fmt = sf.dtypes[ctx.bits - 4]._fmt
    qw, ew, _ = _heuristic.quantize_pow2(fmt, _np(weight), -1)
    w_q = torch.from_numpy(np.ldexp(qw.astype(np.float64), (ew - (fmt.bits - 1))[:, None])).to(g.dtype)
    g2 = g.reshape(-1, g.shape[-1])
    x2 = x.detach().reshape(-1, x.shape[-1]).to(g.dtype)
    grad_x = (g2 @ w_q).reshape(x.shape).to(x.dtype) if ctx.needs_input_grad[0] else None
    grad_w = (g2.T @ x2).to(weight.dtype) if ctx.needs_input_grad[1] else None
    grad_b = g2.sum(0).to(weight.dtype) if (ctx.has_bias and ctx.needs_input_grad[2]) else None
    return grad_x, grad_w, grad_b, None, None, None


torch.library.register_autograd("sf::linear_wide", _linear_wide_backward, setup_context=_linear_wide_setup)


@torch.library.custom_op("sf::matmul_wide", mutates_args=())
def matmul_wide(a: torch.Tensor, b: torch.Tensor, bits: int, shift: Optional[int], name: str) -> torch.Tensor:
    return matmul_wide_impl(a, b, bits, shift, name, _core.HostRunner())


@matmul_wide.register_fake
def _(a, b, bits, shift, name):
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    return a.new_empty(tuple(batch) + (a.shape[-2], b.shape[-1]),
                       dtype=out_dtype(torch.promote_types(a.dtype, b.dtype)))


def _matmul_wide_backward(ctx, g):
    a, b = ctx.saved_tensors
    ad, bd = a.detach().to(g.dtype), b.detach().to(g.dtype)
    grad_a = _reduce_to(g @ bd.transpose(-1, -2), a.shape).to(a.dtype) if ctx.needs_input_grad[0] else None
    grad_b = _reduce_to(ad.transpose(-1, -2) @ g, b.shape).to(b.dtype) if ctx.needs_input_grad[1] else None
    return grad_a, grad_b, None, None, None


torch.library.register_autograd("sf::matmul_wide", _matmul_wide_backward, setup_context=_matmul_setup)


def matmul_wide_pv_impl(p, v, bits, name, runner):
    return matmul_wide_impl(p, v, bits, None, name, runner, out_rule="max_b")


@torch.library.custom_op("sf::matmul_wide_pv", mutates_args=())
def matmul_wide_pv(p: torch.Tensor, v: torch.Tensor, bits: int, name: str) -> torch.Tensor:
    """Attention's P @ V on the MX engine: rows of P are probabilities, so
    each output channel's exponent is V's, pow2_exp(max |V[:, c]|)."""
    return matmul_wide_pv_impl(p, v, bits, name, _core.HostRunner())


@matmul_wide_pv.register_fake
def _(p, v, bits, name):
    batch = torch.broadcast_shapes(p.shape[:-2], v.shape[:-2])
    return p.new_empty(tuple(batch) + (p.shape[-2], v.shape[-1]),
                       dtype=out_dtype(torch.promote_types(p.dtype, v.dtype)))


def _pv_setup(ctx, inputs, output):
    p, v, bits, name = inputs
    ctx.save_for_backward(p, v)


torch.library.register_autograd("sf::matmul_wide_pv", lambda ctx, g: _matmul_wide_backward(ctx, g)[:2] + (None, None),
                                setup_context=_pv_setup)


# --- one entry point for modules, autocast and the functional API ------------------------------

def call_linear(x, weight, bias, cfg, name):
    if cfg.accumulate == "wide":
        return torch.ops.sf.linear_wide(x, weight, bias, cfg.bits, cfg.shift, name)
    return torch.ops.sf.linear(x, weight, bias, cfg.bits, cfg.scale, cfg.headroom, name)


def call_matmul(a, b, cfg, name):
    if cfg.accumulate == "wide":
        return torch.ops.sf.matmul_wide(a, b, cfg.bits, cfg.shift, name)
    return torch.ops.sf.matmul(a, b, cfg.bits, cfg.headroom, name)


SF_OPS = {"sf.linear": "linear", "sf.linear.default": "linear",
          "sf.matmul": "matmul", "sf.matmul.default": "matmul",
          "sf.linear_wide": "linear_wide", "sf.linear_wide.default": "linear_wide",
          "sf.matmul_wide": "matmul_wide", "sf.matmul_wide.default": "matmul_wide",
          "sf.matmul_wide_pv": "matmul_wide_pv", "sf.matmul_wide_pv.default": "matmul_wide_pv"}

# (bits, name) argument positions of each op, for the compile backend
SF_OP_ARGS = {"linear": (3, 6), "matmul": (2, 4), "linear_wide": (3, 5), "matmul_wide": (2, 4),
              "matmul_wide_pv": (2, 3)}
