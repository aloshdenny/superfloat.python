"""Torch wrappers for sf's integer softmax, GELU and SiLU (``sf.intops``): float in, float out, the bits in
between from the integer definitions, exactly as the chip will compute them.  For accuracy runs on the host.

* ``int_softmax(scores, dim=-1)``: scores quantized at scale 2^k (k = 4, redone at 5, 6 for a row that would
  saturate); -inf keys are masked (P = 0, not summed).  P keeps one power-of-two exponent per row.
* ``int_gelu(x)``: GPT-2's tanh GELU (``gelu_new``); ``int_silu(x)``: x sigmoid(x).  One exponent per row (last
  axis), from the row's max |x|, clamped to 0..6.

No gradient: these are inference references (the autograd graph is cut, as on the chip).  Each is a custom op
(``sf::int_softmax``, ``sf::int_xsig``), so ``torch.compile`` keeps it whole.
"""
from __future__ import annotations

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
