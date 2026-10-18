"""sf.config: runtime knobs that are not numeric error policy.

* ``preflight``: what to do when the host can *prove*, before launch, that a
  heavy op saturates on every element (``"raise"`` | ``"warn"`` |
  ``"ignore"``).  Default ``"raise"``: the chip never receives work whose
  answer is already known to be wrong.
* ``suspect``: what to do when the golden-vector probe (a sample, not a
  proof) finds that every probed output saturates: ``"warn"`` (default:
  ``SFPreflightWarning``, then launch), ``"raise"`` (refuse), ``"verify"``
  (replay the whole op on the host reference and refuse only if every output
  really saturates), ``"ignore"``.
* ``probe`` / ``probe_size``: run the probe for ops the range bound cannot
  decide, and how many elements it samples (default 256).
* ``status``: how saturation is counted on a target without sticky flags:
  ``"scan"`` (default; count outputs at ±max, an upper bound for elementwise
  ops and *unknown* for reductions) or ``"replay"`` (recompute the events on
  the host reference, exact, at the cost of running the op twice).
* ``offload_threshold``: element count above which a host op runs on
  ``default_device`` instead (``None`` = never).
* ``kernel_backend``: how the simulator executes SF IR: ``"interp"`` (the
  host reference, default) or ``"c"`` (SF IR -> C with the ``__sf16_*``
  builtins, compiled with the system C compiler).  Both are bit-identical.
* ``kernel_device``: the target for a kernel launch whose arguments carry no
  device array (default ``"atreides-sim"``).

``sf.config(...)`` applies for the rest of the context; ``with
sf.config(...)`` scopes it; as a decorator it wraps plain and ``async``
functions.  Context-local like ``errstate``.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
from types import MappingProxyType
from typing import Mapping

_DEFAULTS: Mapping[str, object] = MappingProxyType({
    "preflight": "raise",
    "suspect": "warn",
    "probe": True,
    "probe_size": 256,
    "status": "scan",
    "offload_threshold": None,
    "default_device": "atreides",
    "kernel_backend": "interp",
    "kernel_device": "atreides-sim",
})
_KEYS = tuple(_DEFAULTS)
_CHOICES = {
    "preflight": ("raise", "warn", "ignore"),
    "suspect": ("raise", "warn", "verify", "ignore"),
    "status": ("scan", "replay"),
    "kernel_backend": ("interp", "c"),
}

_state: contextvars.ContextVar[Mapping[str, object]] = contextvars.ContextVar(
    "sf_config", default=_DEFAULTS)


def _check(k, v):
    if k not in _KEYS:
        raise ValueError(f"unknown config key {k!r}; expected one of {_KEYS}")
    if k in _CHOICES and v not in _CHOICES[k]:
        raise ValueError(f"{k} must be one of {_CHOICES[k]}, got {v!r}")
    if k == "probe_size" and (not isinstance(v, int) or v < 1):
        raise ValueError("probe_size must be a positive int")
    if k == "offload_threshold" and v is not None and (not isinstance(v, int) or v < 0):
        raise ValueError("offload_threshold must be None or a non-negative int")


def get(key: str):
    return _state.get()[key]


def getconfig() -> dict:
    return dict(_state.get())


class config:
    """Set runtime options; a call, a context manager or a decorator."""

    def __init__(self, **kw):
        for k, v in kw.items():
            _check(k, v)
        self._kw = kw
        self._tokens: contextvars.ContextVar[tuple] = contextvars.ContextVar(
            "sf_config_tokens", default=())
        # a plain call applies immediately for the rest of this context
        self._tokens.set(self._tokens.get() + (self._apply(),))
        self._init_live = True                 # the construction-time application is still on the stack

    def _apply(self):
        new = dict(_state.get())
        new.update(self._kw)
        return _state.set(MappingProxyType(new))

    def __enter__(self):
        if self._init_live:
            self._init_live = False               # ``with sf.config(...)``: already applied
        else:
            self._tokens.set(self._tokens.get() + (self._apply(),))
        return self

    def __exit__(self, *exc):
        stack = self._tokens.get()
        if not stack:
            raise RuntimeError("config.__exit__ called without a matching __enter__")
        self._init_live = False
        self._tokens.set(stack[:-1])
        _state.reset(stack[-1])

    def __call__(self, fn):
        # as a decorator the construction-time application is undone: the
        # options apply inside each call only
        if self._init_live:
            self.__exit__(None, None, None)
        if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
            raise TypeError("sf.config cannot decorate a generator function")
        kw = self._kw
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a, **k):
                c = config(**kw)
                try:
                    return await fn(*a, **k)
                finally:
                    c.__exit__(None, None, None)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*a, **k):
            c = config(**kw)
            try:
                return fn(*a, **k)
            finally:
                c.__exit__(None, None, None)
        return wrapper

    def __repr__(self):
        return f"sf.config({getconfig()})"
