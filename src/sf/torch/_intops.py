"""Torch wrappers for sf's integer softmax, GELU and SiLU (``sf.intops``): float in, float out, the bits in
between from the integer definitions, exactly as the chip will compute them.  For accuracy runs on the host.

* ``int_softmax(scores, dim=-1)``: scores quantized at scale 2^k (k = 4, redone at 5, 6 for a row that would
  saturate); -inf keys are masked (P = 0, not summed).  P keeps one power-of-two exponent per row.
* ``int_gelu(x)``: GPT-2's tanh GELU (``gelu_new``); ``int_silu(x)``: x sigmoid(x).  One exponent per row (last
  axis), from the row's max |x|, clamped to 0..6.

* ``int_norm(x, weight, bias, eps, rms)``: LayerNorm or RMSNorm over the last axis (``sf.intops.norm_rows``).

No gradient: these are inference references (the autograd graph is cut, as on the chip).  Each is a custom op
(``sf::int_softmax``, ``sf::int_xsig``, ``sf::int_norm``), so ``torch.compile`` keeps it whole.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .. import _intops


def _np(t: torch.Tensor) -> np.ndarray:
    return t.detach().to("cpu", torch.float64).numpy()


# The integer definitions run in NumPy, so each is a custom op: torch.compile keeps it as one opaque node (run on
# the host, as in eager) instead of tracing into the NumPy code, which Dynamo cannot do.

@torch.library.custom_op("sf::int_softmax", mutates_args=())
def _int_softmax_op(scores: torch.Tensor, dim: int) -> torch.Tensor:
    x = scores.movedim(dim, -1)
    p = _intops.softmax_rows(_np(x))
    return torch.from_numpy(p).to(scores.dtype).to(scores.device).movedim(-1, dim).contiguous()


@_int_softmax_op.register_fake
def _(scores, dim):
    return torch.empty_like(scores)


@torch.library.custom_op("sf::int_xsig", mutates_args=())
def _int_xsig_op(x: torch.Tensor, cubic: bool) -> torch.Tensor:
    return torch.from_numpy(_intops.xsig_rows(_np(x), cubic)).to(x.dtype).to(x.device)


@_int_xsig_op.register_fake
def _(x, cubic):
    return torch.empty_like(x)


@torch.library.custom_op("sf::int_norm", mutates_args=())
def _int_norm_op(x: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], eps: float,
                 rms: bool) -> torch.Tensor:
    n = x.shape[-1]
    g = np.ones(n) if weight is None else _np(weight)
    b = None if bias is None else _np(bias)
    return torch.from_numpy(_intops.norm_rows(_np(x), g, b, eps, rms)).to(x.dtype).to(x.device)


@_int_norm_op.register_fake
def _(x, weight, bias, eps, rms):
    return torch.empty_like(x)


def int_softmax(scores: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return _int_softmax_op(scores, dim % scores.dim())


def int_gelu(x: torch.Tensor) -> torch.Tensor:
    return _int_xsig_op(x, True)


def int_silu(x: torch.Tensor) -> torch.Tensor:
    return _int_xsig_op(x, False)


class IntGELU(torch.nn.Module):
    """GPT-2's tanh GELU through sf's integer definition (replaces ``gelu_new`` / ``GELU(approximate="tanh")``)."""

    def forward(self, x):
        return int_gelu(x)


class IntSiLU(torch.nn.Module):
    """SiLU through sf's integer definition."""

    def forward(self, x):
        return int_silu(x)


def int_norm(x: torch.Tensor, weight=None, bias=None, eps: float = 1e-5, rms: bool = False) -> torch.Tensor:
    """LayerNorm (or RMSNorm, ``rms``) over the last axis through sf's integer definition (``sf.intops.norm_rows``)."""
    return _int_norm_op(x, weight, bias, float(eps), bool(rms))


class IntNorm(torch.nn.Module):
    """A LayerNorm or RMSNorm through sf's integer definition, with the weights of the module it replaces."""

    def __init__(self, weight, bias, eps: float, rms: bool):
        super().__init__()
        self.weight = None if weight is None else torch.nn.Parameter(weight.detach().clone(), requires_grad=False)
        self.bias = None if bias is None else torch.nn.Parameter(bias.detach().clone(), requires_grad=False)
        self.eps, self.rms = float(eps), rms

    def forward(self, x):
        return int_norm(x, self.weight, self.bias, self.eps, self.rms)

    def extra_repr(self):
        return f"{'RMSNorm' if self.rms else 'LayerNorm'}, eps={self.eps}"


def integer_norm(m: torch.nn.Module):
    """(replacement, kind) for a norm module with an integer definition, (None, why) for a norm without one,
    (None, None) for anything else.  LayerNorm over one axis; RMSNorm as torch and Hugging Face write it
    (weight times the normalised row; Gemma's 1 + weight form is not that)."""
    name = type(m).__name__
    if isinstance(m, torch.nn.LayerNorm):
        if len(m.normalized_shape) != 1:
            return None, "LayerNorm over several axes: stays float"
        return IntNorm(m.weight, m.bias, m.eps, False), "LayerNorm (integer)"
    rms_cls = getattr(torch.nn, "RMSNorm", None)
    if rms_cls is not None and isinstance(m, rms_cls):
        if len(m.normalized_shape) != 1:
            return None, "RMSNorm over several axes: stays float"
        eps = m.eps if m.eps is not None else torch.finfo(torch.float32).eps
        return IntNorm(m.weight, None, eps, True), "RMSNorm (integer)"
    if name.endswith("RMSNorm"):
        if "Gemma" in name:
            return None, "Gemma's RMSNorm scales by 1 + weight: stays float"
        w, eps = getattr(m, "weight", None), getattr(m, "variance_epsilon", getattr(m, "eps", None))
        if isinstance(w, torch.Tensor) and w.dim() == 1 and eps is not None:
            return IntNorm(w, None, eps, True), "RMSNorm (integer)"
        return None, f"{name}: not the weight x normalised-row form sf knows; stays float"
    return None, None


_TANH_GELU = {"NewGELUActivation", "GELUTanh", "PytorchGELUTanh", "FastGELUActivation", "GELUPytorchTanh"}
_SILU = {"SiLUActivation", "SiLU"}
_ERF_GELU = {"GELUActivation", "GELU"}


def integer_activation(m: torch.nn.Module):
    """(replacement, kind) for an activation module with an integer definition, (None, why) for one without,
    (None, None) for anything else."""
    name = type(m).__name__
    if isinstance(m, torch.nn.GELU):
        if m.approximate == "tanh":
            return IntGELU(), "GELU (tanh)"
        return None, "GELU with erf: sf has the tanh form only; stays float"
    if isinstance(m, torch.nn.SiLU) or name in _SILU:
        return IntSiLU(), "SiLU"
    if name in _TANH_GELU:
        return IntGELU(), "GELU (tanh)"
    if name in _ERF_GELU:
        return None, "GELU with erf: sf has the tanh form only; stays float"
    return None, None
