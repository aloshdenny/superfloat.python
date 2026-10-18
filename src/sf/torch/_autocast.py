"""``sft.autocast``: run linear algebra inside a block with SF semantics.

Intercepts ``F.linear``, ``F.conv2d`` (groups=1), ``torch.matmul`` / ``@`` /
``mm`` / ``bmm`` through a ``TorchFunctionMode`` and routes them to the SF
custom ops, so an unmodified model's linear algebra runs chip-exact SF while
everything else stays in float:

>>> with sft.autocast(sf.sf16):
...     y = model(x)
"""
from __future__ import annotations

import functools
import inspect
import itertools

import torch
import torch.nn.functional as F
from torch.overrides import TorchFunctionMode

from ._modules import SFConfig, _bits_of

_MATMULS = {torch.matmul, torch.Tensor.matmul, torch.Tensor.__matmul__, torch.mm, torch.bmm,
            torch.Tensor.mm, torch.Tensor.bmm}


class _SFMode(TorchFunctionMode):
    def __init__(self, cfg: SFConfig, prefix: str):
        super().__init__()
        self.cfg = cfg
        self.prefix = prefix
        self.counter = itertools.count()

    def _name(self, kind):
        return f"{self.prefix}{kind}.{next(self.counter)}"

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        cfg = self.cfg
        if func is F.linear:
            x, w = args[0], args[1]
            b = args[2] if len(args) > 2 else kwargs.get("bias")
            return torch.ops.sf.linear(x, w, b, cfg.bits, cfg.scale, cfg.headroom, self._name("linear"))
        if func in _MATMULS:
            a, b = args[0], args[1]
            if a.dim() == 0 or b.dim() == 0:
                return func(*args, **kwargs)
            sq_a, sq_b = a.dim() == 1, b.dim() == 1
            a2 = a.unsqueeze(0) if sq_a else a
            b2 = b.unsqueeze(-1) if sq_b else b
            y = torch.ops.sf.matmul(a2, b2, cfg.bits, cfg.headroom, self._name("matmul"))
            if sq_a:
                y = y.squeeze(-2)
            if sq_b:
                y = y.squeeze(-1)
            return y
        if func is F.conv2d:
            return self._conv2d(func, args, kwargs)
        return func(*args, **kwargs)

    def _conv2d(self, func, args, kwargs):
        names = ["input", "weight", "bias", "stride", "padding", "dilation", "groups"]
        p = dict(zip(names, args))
        p.update(kwargs)
        groups = p.get("groups", 1)
        padding = p.get("padding", 0)
        if groups != 1 or isinstance(padding, str):
            return func(*args, **kwargs)                    # not converted: float
        x, w, b = p["input"], p["weight"], p.get("bias")
        stride, dilation = _pair(p.get("stride", 1)), _pair(p.get("dilation", 1))
        padding = _pair(padding)
        n, _, h, wd = x.shape
        oc, _, kh, kw = w.shape
        cols = F.unfold(x, (kh, kw), dilation=dilation, padding=padding, stride=stride)
        oh = (h + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
        ow = (wd + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
        y = torch.ops.sf.linear(cols.transpose(1, 2), w.reshape(oc, -1), b, self.cfg.bits,
                                self.cfg.scale, self.cfg.headroom, self._name("conv2d"))
        return y.transpose(1, 2).reshape(n, oc, oh, ow)


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


class autocast:
    """Context manager and decorator (plain and ``async`` functions)."""

    def __init__(self, dtype="sf16", scale: str = "per_tensor", headroom: str = "bound",
                 prefix: str = "autocast.") -> None:
        self.cfg = SFConfig(_bits_of(dtype), scale, headroom)
        self.prefix = prefix
        self._modes = []

    def __enter__(self):
        mode = _SFMode(self.cfg, self.prefix)
        mode.__enter__()
        self._modes.append(mode)
        return self

    def __exit__(self, *exc):
        self._modes.pop().__exit__(*exc)

    def __call__(self, fn):
        kw = dict(dtype=self.cfg.dtype_name, scale=self.cfg.scale, headroom=self.cfg.headroom,
                  prefix=self.prefix)
        if inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn) \
                or inspect.isgeneratorfunction(fn):
            # torch function modes live on a thread-local stack, not a context
            # variable: held across an await or a yield, the mode would apply
            # to whatever else runs on the thread in between.
            raise TypeError("sft.autocast cannot decorate coroutine or generator functions; "
                            "use 'with sft.autocast(...)' around synchronous code")

        @functools.wraps(fn)
        def wrapper(*a, **k):
            with autocast(**kw):
                return fn(*a, **k)
        return wrapper
