"""Per-layer post-op status for converted models (``sft.last_status()``)."""
from __future__ import annotations

import contextvars
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class SFLayerStatus:
    """What one SF layer call reported.

    ``saturated``: outputs whose accumulation chain or bias add saturated.
    Exact on the host emulation; on a device without sticky flags it is exact
    only when the pre-flight bound proved it zero, otherwise ``None``
    (unknown).  With ``headroom="bound"`` it is always an exact 0.

    ``underflow``: outputs in which at least one nonzero product truncated to
    0 (precision loss, not an error); ``None`` on a device.
    ``input_underflow`` / ``weight_underflow``: nonzero inputs / weights that
    quantized to 0.  ``bias_saturated``: bias values that did not fit the
    accumulator scale.  ``headroom_bits``: activation bits given up so the
    accumulation cannot saturate.  ``act_scale``: the float activation scale
    ``a`` used for this call (input value = a * code / 2^(N-1)).
    """

    name: str
    op: str
    placement: str
    dtype: str
    outputs: int
    saturated: int | None
    exact: bool
    underflow: int | None
    input_underflow: int
    weight_underflow: int
    bias_saturated: int
    headroom_bits: float
    act_scale: float = 1.0

    @property
    def ok(self) -> bool | None:
        if self.bias_saturated:
            return False
        if self.saturated is None:
            return None
        return self.saturated == 0

    def __repr__(self) -> str:
        sat = "unknown" if self.saturated is None else str(self.saturated)
        return (f"SFLayerStatus({self.name!r}, {self.op} on {self.placement}, {self.dtype}, "
                f"saturated={sat}, underflow={self.underflow}, "
                f"headroom_bits={self.headroom_bits:.2f})")


_lock = threading.Lock()
_last: dict[str, SFLayerStatus] = {}

# request-scoped capture: a context-local stack of dicts (threads and asyncio
# tasks each see their own), filled alongside the process-wide registry
_captures: contextvars.ContextVar[tuple] = contextvars.ContextVar("sf_status_captures", default=())


def record(st: SFLayerStatus) -> None:
    with _lock:
        _last[st.name] = st
    for sink in _captures.get():
        sink[st.name] = st


class capture_status:
    """Collect the statuses of the SF layers that run inside this context,
    for this thread or task only:

    >>> with sft.capture_status() as st:
    ...     out = model(x)
    >>> st["encoder.fc"].saturated

    Unlike ``last_status()``, two requests served concurrently with models
    that share layer names cannot overwrite each other's entries.  Usable as
    a decorator on plain and ``async`` functions (the captured dict is then
    available as ``fn.last``).
    """

    def __init__(self) -> None:
        self.statuses: dict[str, SFLayerStatus] = {}
        self._tokens: list = []

    def __enter__(self) -> dict[str, SFLayerStatus]:
        self._tokens.append(_captures.set(_captures.get() + (self.statuses,)))
        return self.statuses

    def __exit__(self, *exc) -> None:
        _captures.reset(self._tokens.pop())

    def __call__(self, fn):
        import functools
        import inspect
        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def awrapper(*a, **k):
                with capture_status() as st:
                    out = await fn(*a, **k)
                awrapper.last = st
                return out
            awrapper.last = {}
            return awrapper

        @functools.wraps(fn)
        def wrapper(*a, **k):
            with capture_status() as st:
                out = fn(*a, **k)
            wrapper.last = st
            return out
        wrapper.last = {}
        return wrapper


def last_status(model=None) -> dict[str, SFLayerStatus]:
    """Status of the most recent call of every SF layer, by layer name.

    The registry is process-wide and keyed by layer name, so two models with
    the same structure share names.  Pass ``model`` to get only that model's
    SF layers (by their ``sf_name``), in module order.
    """
    with _lock:
        snapshot = dict(_last)
    if model is None:
        return snapshot
    names = [getattr(m, "sf_name") for m in model.modules() if hasattr(m, "sf_name")]
    return {n: snapshot[n] for n in names if n in snapshot}


def reset_status() -> None:
    with _lock:
        _last.clear()
