"""sf.torch: Superfloat for PyTorch (spec section 8, Stage 2).

Integrates with PyTorch rather than cloning it: PyTorch owns autograd,
modules, optimisers, data loading and serialisation; SF owns numerics and
offload.

>>> import torch, sf
>>> import sf.torch as sft
>>> model = sft.convert(model, dtype=sf.sf16, scale="per_tensor")   # host emulation, chip-exact
>>> fast = torch.compile(model, backend="superfloat")                # offload to Atreides
>>> with torch.no_grad():
...     out = fast(x)
>>> sft.last_status()                                                # post-op events per layer

Milestones: 2.0 ``quantize`` (fake quantization, straight-through gradient);
2.1 ``convert`` / ``autocast`` (SF layers with a scale policy); 2.2 the
``superfloat`` compile backend with a coverage report.  2.3 (Atreides as a
PyTorch ``PrivateUse1`` device) is deferred, as the spec recommends, until
2.2 coverage is high.
"""
from __future__ import annotations

import numpy as np
import torch

from . import _ops  # noqa: F401  (registers the sf:: custom ops)
from ._autocast import autocast
from ._backend import GraphReport, compile_report, compile_reports, superfloat  # noqa: F401
from ._modules import ConversionReport, SFConfig, SFConv2d, SFLinear, conversion_report, convert
from ._status import SFLayerStatus, capture_status, last_status, reset_status


def _bits(dtype) -> int:
    import sf
    return sf.info(sf.dtype(dtype)).bits


def quantize(t: torch.Tensor, dtype="sf16", scale=None) -> torch.Tensor:
    """Fake quantization: values snapped to the SF grid, float storage,
    clipped straight-through gradient (spec 2.0).

    ``scale=None``: plain cast, the values must already be in (-1, 1); out of
    range, NaN and underflow follow ``sf.errstate`` exactly like Stage 1
    (``sf.sf16(1.5)`` raises, so does ``quantize(torch.tensor([1.5]))``).
    ``scale="dynamic"``: divide by ``max|t| / max`` first (never overflows).
    ``scale=<float>``: divide by that scale first.  The result is
    ``scale * code / 2^(N-1)``, float32 (float64 for float64 input).
    """
    if not isinstance(t, torch.Tensor):
        raise TypeError("sft.quantize expects a torch.Tensor")
    if not t.is_floating_point():
        t = t.to(torch.float64)
    if scale is None:
        return torch.ops.sf.quantize(t, _bits(dtype), "none", 1.0)
    if scale == "dynamic":
        return torch.ops.sf.quantize(t, _bits(dtype), "dynamic", 0.0)
    return torch.ops.sf.quantize(t, _bits(dtype), "fixed", float(scale))


def linear(x: torch.Tensor, weight: torch.Tensor, bias=None, dtype="sf16", scale: str = "per_tensor",
           headroom: str = "bound", name: str = "linear") -> torch.Tensor:
    """Functional chip-exact SF linear (what ``SFLinear`` calls)."""
    return torch.ops.sf.linear(x, weight, bias, _bits(dtype), scale, headroom, name)


def matmul(a: torch.Tensor, b: torch.Tensor, dtype="sf16", headroom: str = "bound",
           name: str = "matmul") -> torch.Tensor:
    """Functional chip-exact SF matmul of two activations (both dynamic
    per-tensor scale), for custom modules such as attention.  1-D operands
    follow ``torch.matmul``."""
    sq_a, sq_b = a.dim() == 1, b.dim() == 1
    a2 = a.unsqueeze(0) if sq_a else a
    b2 = b.unsqueeze(-1) if sq_b else b
    y = torch.ops.sf.matmul(a2, b2, _bits(dtype), headroom, name)
    if sq_a:
        y = y.squeeze(-2)
    if sq_b:
        y = y.squeeze(-1)
    return y


def to_sf(t: torch.Tensor, dtype="sf16", device: str = "host"):
    """Tensor -> ``SFArray`` (codes), quantized exactly like ``sf.asarray``."""
    import sf
    return sf.asarray(t.detach().to("cpu", torch.float64).numpy(), dtype=sf.dtype(dtype), device=device)


def from_sf(a, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """``SFArray`` or SF scalar -> tensor.  float32 holds every sf4..sf16
    value exactly; bfloat16 and float16 do not and are refused."""
    import sf
    if dtype in (torch.bfloat16, torch.float16):
        raise sf.SFTypeError(f"{dtype} cannot hold SF values exactly; use float32 or float64")
    if isinstance(a, sf.SFScalar):
        return torch.tensor(float(a), dtype=dtype)
    return torch.from_numpy(np.asarray(a.numpy())).to(dtype)


__all__ = ["quantize", "linear", "matmul", "to_sf", "from_sf", "SFLinear", "SFConv2d", "SFConfig", "convert",
           "conversion_report", "ConversionReport", "autocast", "last_status", "reset_status", "capture_status",
           "SFLayerStatus", "compile_report", "compile_reports", "GraphReport"]
