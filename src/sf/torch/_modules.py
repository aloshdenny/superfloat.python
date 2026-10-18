"""SF layers and model conversion (spec 2.1)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _core


def _bits_of(dtype) -> int:
    import sf
    return sf.info(sf.dtype(dtype)).bits


@dataclass
class SFConfig:
    bits: int = 16
    scale: str = "per_tensor"         # weight scale: "per_tensor" | "per_channel"
    headroom: str = "bound"            # "bound" (no saturation, provable) | "none"

    def __post_init__(self):
        _core._check(self.bits, self.scale, self.headroom)

    @property
    def dtype_name(self) -> str:
        return f"sf{self.bits}"


class SFLinear(nn.Module):
    """``nn.Linear`` with chip-exact SF arithmetic.

    Parameters stay float (so training works through the straight-through
    gradient); every forward quantizes the weight with the configured scale
    policy and runs the SF accumulation chain.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *,
                 dtype="sf16", scale: str = "per_tensor", headroom: str = "bound",
                 name: str = "", device=None, param_dtype=None) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.sf = SFConfig(_bits_of(dtype), scale, headroom)
        self.sf_name = name or f"linear_{id(self):x}"
        kw = {"device": device, "dtype": param_dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **kw))
        self.bias = nn.Parameter(torch.empty(out_features, **kw)) if bias else None
        ref = nn.Linear(in_features, out_features, bias, **kw)
        with torch.no_grad():
            self.weight.copy_(ref.weight)
            if bias:
                self.bias.copy_(ref.bias)

    @classmethod
    def from_float(cls, m: nn.Linear, name: str, cfg: SFConfig) -> "SFLinear":
        """Wrap ``m``'s own parameter objects (identity preserved, so an
        optimizer built before conversion keeps working and ``requires_grad``
        is whatever it was), and copy its training / eval state."""
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.in_features, new.out_features = m.in_features, m.out_features
        new.sf = cfg
        new.sf_name = name or f"linear_{id(new):x}"
        new.weight = m.weight
        new.bias = m.bias
        new.train(m.training)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.sf.linear(x, self.weight, self.bias, self.sf.bits, self.sf.scale,
                                   self.sf.headroom, self.sf_name)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, dtype={self.sf.dtype_name}, scale={self.sf.scale}, "
                f"headroom={self.sf.headroom}, name={self.sf_name!r}")


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


class SFConv2d(nn.Module):
    """``nn.Conv2d`` (groups=1, zero padding) as unfold + SF linear: the patch
    extraction is data movement on the host, the reduction is the SF chain."""

    def __init__(self, conv: nn.Conv2d, name: str, cfg: SFConfig) -> None:
        super().__init__()
        if not convertible_conv(conv)[0]:
            raise ValueError(convertible_conv(conv)[1])
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        self.kernel_size = _pair(conv.kernel_size)
        self.stride, self.padding, self.dilation = _pair(conv.stride), _pair(conv.padding), _pair(conv.dilation)
        self.sf = cfg
        self.sf_name = name
        self.weight = conv.weight                     # same parameter objects as the source conv
        self.bias = conv.bias
        self.train(conv.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        kh, kw = self.kernel_size
        cols = F.unfold(x, (kh, kw), dilation=self.dilation, padding=self.padding, stride=self.stride)
        oh = (h + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        ow = (w + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
        y = torch.ops.sf.linear(cols.transpose(1, 2), self.weight.reshape(self.out_channels, -1),
                                self.bias, self.sf.bits, self.sf.scale, self.sf.headroom, self.sf_name)
        return y.transpose(1, 2).reshape(n, self.out_channels, oh, ow)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, dtype={self.sf.dtype_name}, "
                f"scale={self.sf.scale}, headroom={self.sf.headroom}, name={self.sf_name!r}")


def convertible_conv(c: nn.Conv2d) -> tuple[bool, str]:
    if c.groups != 1:
        return False, f"groups={c.groups} (only groups=1 is converted)"
    if c.padding_mode != "zeros":
        return False, f"padding_mode={c.padding_mode!r} (only 'zeros' is converted)"
    if isinstance(c.padding, str):
        return False, f"padding={c.padding!r} (only explicit integer padding is converted)"
    return True, ""


@dataclass
class ConversionReport:
    converted: list = field(default_factory=list)     # (name, kind)
    skipped: list = field(default_factory=list)       # (name, kind, reason)
    shared: list = field(default_factory=list)        # (alias name, converted name): same module

    def __repr__(self) -> str:
        lines = [f"ConversionReport: {len(self.converted)} converted, {len(self.skipped)} skipped"
                 + (f", {len(self.shared)} shared alias(es)" if self.shared else "")]
        lines += [f"  + {n} ({k})" for n, k in self.converted]
        lines += [f"  = {a} is {n} (shared module, converted once)" for a, n in self.shared]
        lines += [f"  - {n} ({k}): {r}" for n, k, r in self.skipped]
        return "\n".join(lines)


def convert(model: nn.Module, dtype="sf16", scale: str = "per_tensor", headroom: str = "bound",
            inplace: bool = False) -> nn.Module:
    """Swap ``nn.Linear`` and ``nn.Conv2d`` for SF versions with a scale policy.

    Other modules (activations, normalisation, pooling, softmax, ...) stay as
    they are and run in float on the host; ``sft.conversion_report(model)``
    lists what was converted and what was skipped, and why.
    """
    cfg = SFConfig(_bits_of(dtype), scale, headroom)
    if not inplace:
        model = copy.deepcopy(model)
    report = ConversionReport()
    done: dict[int, tuple[str, nn.Module | None]] = {}     # id(source) -> (name, replacement)
    seen: set[int] = set()                                  # containers already visited

    def replacement(child: nn.Module, full: str):
        """The SF module for ``child``, or None if it stays; converts each
        distinct module once and returns the same object for every alias."""
        key = id(child)
        if key in done:
            name, rep = done[key]
            if rep is not None:
                report.shared.append((full, name))
            return rep, True
        rep = None
        if isinstance(child, (SFLinear, SFConv2d)):
            report.converted.append((full, type(child).__name__ + " (already SF)"))
        elif isinstance(child, nn.Linear):
            rep = SFLinear.from_float(child, full, cfg)
            report.converted.append((full, "Linear"))
        elif isinstance(child, nn.Conv2d):
            ok, why = convertible_conv(child)
            if ok:
                rep = SFConv2d(child, full, cfg)
                report.converted.append((full, "Conv2d"))
            else:
                report.skipped.append((full, "Conv2d", why))
        elif isinstance(child, (nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                                nn.ConvTranspose3d, nn.MultiheadAttention, nn.Bilinear,
                                nn.LSTM, nn.GRU, nn.RNN)):
            report.skipped.append((full, type(child).__name__, "no SF version yet; runs in float"))
        else:
            done[key] = (full, None)
            return None, False                              # a container: recurse into it
        done[key] = (full, rep)
        return rep, True

    def visit(parent: nn.Module, prefix: str):
        if id(parent) in seen:
            return
        seen.add(id(parent))
        # parent._modules, not named_children(): the latter yields a module
        # registered under two names only once, and the second alias would
        # keep its float version
        for child_name, child in list(parent._modules.items()):
            if child is None:
                continue
            full = f"{prefix}.{child_name}" if prefix else child_name
            rep, leaf = replacement(child, full)
            if rep is not None:
                setattr(parent, child_name, rep)
            elif not leaf:
                visit(child, full)

    rep, leaf = replacement(model, "")
    if rep is not None:
        model = rep
    elif not leaf:
        visit(model, "")
    model._sf_conversion_report = report
    return model


def conversion_report(model: nn.Module) -> ConversionReport:
    r = getattr(model, "_sf_conversion_report", None)
    if r is None:
        raise ValueError("model was not produced by sft.convert")
    return r
