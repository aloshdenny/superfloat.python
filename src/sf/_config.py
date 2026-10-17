"""sf.config: runtime knobs that are not numeric error policy.

* ``preflight``: what to do when the host can prove, before launch, that a
  heavy op will saturate on every element (``"raise"`` | ``"warn"`` |
  ``"ignore"``).  Default ``"raise"``: the chip never receives work whose
  answer is already known to be wrong.
* ``probe``: run the golden-vector probe for ops the range bound cannot
  decide (``True`` / ``False``).  The probe runs a sample of the real inputs
  through the host reference; if it saturates, the op is treated like a
  proven event under the ``preflight`` policy.  Default ``True``.
* ``probe_size``: how many input elements the probe samples (default 256).
* ``offload_threshold``: element count above which a *host* op is run on the
  default device instead (``None`` = never, the default).
"""
from __future__ import annotations

import contextvars
from types import MappingProxyType
from typing import Mapping

_DEFAULTS: Mapping[str, object] = MappingProxyType({
    "preflight": "raise",
    "probe": True,
    "probe_size": 256,
    "offload_threshold": None,
    "default_device": "atreides",
})
_KEYS = tuple(_DEFAULTS)

_state: contextvars.ContextVar[Mapping[str, object]] = contextvars.ContextVar(
    "sf_config", default=_DEFAULTS)


def _check(k, v):
    if k not in _KEYS:
        raise ValueError(f"unknown config key {k!r}; expected one of {_KEYS}")
    if k == "preflight" and v not in ("raise", "warn", "ignore"):
        raise ValueError("preflight must be 'raise', 'warn' or 'ignore'")
    if k == "probe_size" and (not isinstance(v, int) or v < 1):
        raise ValueError("probe_size must be a positive int")


def get(key: str):
    return _state.get()[key]


def getconfig() -> dict:
    return dict(_state.get())


class config:
    """Set runtime options; usable as a call, a context manager or a decorator.

    >>> sf.config(preflight="warn")                 # permanent for this context
    >>> with sf.config(probe=False): ...            # scoped
    """

    def __init__(self, **kw):
        for k, v in kw.items():
            _check(k, v)
        self._kw = kw
        self._prev = getconfig()
        new = dict(self._prev)
        new.update(kw)
        self._token = _state.set(MappingProxyType(new))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        _state.reset(self._token)

    def __call__(self, fn):
        import functools
        _state.reset(self._token)            # a decorator should not also apply globally

        @functools.wraps(fn)
        def wrapper(*a, **k):
            with config(**self._kw):
                return fn(*a, **k)
        return wrapper

    def __repr__(self):
        return f"sf.config({getconfig()})"
