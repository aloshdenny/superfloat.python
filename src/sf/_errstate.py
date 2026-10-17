"""Error policy: sf.seterr / sf.geterr / sf.errstate (spec section 3.3).

Keys and values mirror numpy.errstate: keys ``over``, ``under``, ``divide``,
``invalid``; values ``"raise"``, ``"warn"``, ``"ignore"``.  The value an op
returns is always the hardware value (saturated, truncated, ...); the policy
only decides whether the host also raises or warns.

The state lives in a ``contextvars.ContextVar`` so ``with sf.errstate(...)``
is safe across threads and asyncio tasks.
"""
from __future__ import annotations

import contextvars
import functools
import os
import sys
import warnings
from types import MappingProxyType
from typing import Callable, Mapping

from . import _errors

KEYS = ("over", "under", "divide", "invalid")
VALUES = ("raise", "warn", "ignore")

DEFAULTS: Mapping[str, str] = MappingProxyType(
    {"over": "raise", "under": "warn", "divide": "raise", "invalid": "raise"}
)

_state: contextvars.ContextVar[Mapping[str, str]] = contextvars.ContextVar(
    "sf_errstate", default=DEFAULTS
)

# event name -> (policy key, error class, warning class)
EVENTS: dict[str, tuple[str, type, type]] = {
    "overflow": ("over", _errors.SFOverflowError, _errors.SFOverflowWarning),
    "underflow": ("under", _errors.SFUnderflowError, _errors.SFUnderflowWarning),
    "divide": ("divide", _errors.SFZeroDivisionError, _errors.SFZeroDivisionWarning),
    "invalid": ("invalid", _errors.SFInvalidError, _errors.SFInvalidWarning),
}


def _check(key: str, value: str | None) -> None:
    if key not in KEYS:
        raise ValueError(f"unknown errstate key {key!r}; expected one of {KEYS}")
    if value is not None and value not in VALUES:
        raise ValueError(f"errstate value for {key!r} must be one of {VALUES}, got {value!r}")


def geterr() -> dict[str, str]:
    """Current error policy as a plain dict."""
    return dict(_state.get())


def seterr(all: str | None = None, **kw: str | None) -> dict[str, str]:
    """Set the error policy; returns the previous policy (like numpy.seterr)."""
    for k, v in kw.items():
        _check(k, v)
    if all is not None:
        _check("over", all)
    old = geterr()
    new = dict(old)
    if all is not None:
        for k in KEYS:
            new[k] = all
    for k, v in kw.items():
        if v is not None:
            new[k] = v
    _state.set(MappingProxyType(new))
    return old


class errstate:
    """Context manager / decorator that scopes a policy change.

    >>> with sf.errstate(over="ignore"):
    ...     sf.sf16(0.75) + 0.5          # saturates silently
    """

    def __init__(self, all: str | None = None, **kw: str | None) -> None:
        for k, v in kw.items():
            _check(k, v)
        if all is not None:
            _check("over", all)
        self._all = all
        self._kw = kw
        self._tokens: list[contextvars.Token] = []

    def __enter__(self) -> "errstate":
        new = dict(_state.get())
        if self._all is not None:
            for k in KEYS:
                new[k] = self._all
        for k, v in self._kw.items():
            if v is not None:
                new[k] = v
        self._tokens.append(_state.set(MappingProxyType(new)))
        return self

    def __exit__(self, *exc: object) -> None:
        _state.reset(self._tokens.pop())

    def __call__(self, fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*a, **k):
            with self:
                return fn(*a, **k)
        return wrapper


_PKG_DIR = os.path.dirname(os.path.abspath(__file__))


def _user_stacklevel() -> int:
    """Stack level of the first frame outside the sf package.

    Used so that warnings point at the Python line that caused the event,
    the way the spec asks, instead of at the package internals.
    """
    level = 1
    frame = sys._getframe(1)
    while frame is not None:
        fname = frame.f_code.co_filename
        if not os.path.abspath(fname).startswith(_PKG_DIR):
            return level
        frame = frame.f_back
        level += 1
    return level


def handle(event: str | None, message: str) -> None:
    """Apply the current policy to one numeric event.

    ``event`` is one of ``None`` (nothing happened), ``"overflow"``,
    ``"underflow"``, ``"divide"``, ``"invalid"``.  Raises, warns or does
    nothing; never changes the value the caller is about to return.
    """
    if event is None:
        return
    key, err_cls, warn_cls = EVENTS[event]
    policy = _state.get()[key]
    if policy == "raise":
        raise err_cls(message)
    if policy == "warn":
        warnings.warn(message, warn_cls, stacklevel=_user_stacklevel())
    # "ignore": return the hardware value silently
