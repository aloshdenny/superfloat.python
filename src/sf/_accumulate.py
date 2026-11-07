"""How matrix multiplies and dot products accumulate: sf.accumulate,
sf.setaccumulate, sf.getaccumulate.

* ``"chain"`` (default): a chain of fma on the FMA units.  Every product is
  truncated and every partial sum saturates at +-1, at data width.
* ``"wide"``: the v4.8.1 MX engine.  Products go into a wide accumulator
  (range +-32) and the total is rounded once, half to even, scaled by
  ``2^shift`` (see ``_linalg.matmul_wide_codes``).

Only ``@``, ``matmul``, ``dot`` and ``addmm`` read this setting.  Elementwise
ops, ``sum`` and everything else keep their rules.

Strongest first: ``accumulate=`` on the call, the innermost
``sf.accumulate`` block or decorator, ``sf.setaccumulate``, the default.
The state lives in a ``contextvars.ContextVar`` like ``sf.errstate``, so each
thread and asyncio task has its own setting.  It is separate from
``errstate`` because it changes the bits.
"""
from __future__ import annotations

import contextvars
import functools
import inspect
from typing import Callable

MODES = ("chain", "wide")
DEFAULT = "chain"

_state: contextvars.ContextVar[str] = contextvars.ContextVar("sf_accumulate", default=DEFAULT)


def check(mode) -> str:
    if mode not in MODES:
        raise ValueError(f"accumulate must be one of {MODES}, got {mode!r}")
    return mode


def resolve(mode) -> str:
    """The mode a call uses: its own ``accumulate=`` if given, else the setting."""
    return _state.get() if mode is None else check(mode)


def getaccumulate() -> str:
    """The accumulation mode in force here: ``"chain"`` or ``"wide"``."""
    return _state.get()


def setaccumulate(mode: str) -> str:
    """Set the mode for the rest of this context; returns the previous one."""
    old = _state.get()
    _state.set(check(mode))
    return old


class accumulate:
    """Context manager / decorator that scopes the accumulation mode.

    >>> with sf.accumulate("wide"):
    ...     y = a @ b                    # runs on the MX engine

    As a decorator it works on plain and ``async def`` functions; each call
    gets its own context entry.  Generator functions are rejected.
    """

    def __init__(self, mode: str) -> None:
        self._mode = check(mode)
        self._tokens: contextvars.ContextVar[tuple] = contextvars.ContextVar(
            "sf_accumulate_tokens", default=())

    def __enter__(self) -> "accumulate":
        token = _state.set(self._mode)
        self._tokens.set(self._tokens.get() + (token,))
        return self

    def __exit__(self, *exc: object) -> None:
        stack = self._tokens.get()
        if not stack:
            raise RuntimeError("accumulate.__exit__ called without a matching __enter__")
        self._tokens.set(stack[:-1])
        _state.reset(stack[-1])

    def __call__(self, fn: Callable) -> Callable:
        if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
            raise TypeError("sf.accumulate cannot decorate a generator function; "
                            "use 'with sf.accumulate(...)' inside the generator body")
        mode = self._mode
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a, **k):
                with accumulate(mode):
                    return await fn(*a, **k)
            return awrapper

        @functools.wraps(fn)
        def wrapper(*a, **k):
            with accumulate(mode):
                return fn(*a, **k)
        return wrapper

    def __repr__(self) -> str:
        return f"sf.accumulate({self._mode!r})"
