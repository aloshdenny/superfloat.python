"""Post-op status of a heavy op (spec section 3.4)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SFStatus:
    """What the runtime knows about a heavy op after it ran.

    ``saturated``: number of outputs whose computation saturated.  It is
    exact when the target reports sticky flags, when the pre-flight bound
    proved saturation impossible, or when ``sf.config(status="replay")``
    recomputed the events on the host.  Otherwise:

    * for an elementwise op it is an *upper bound* taken from scanning the
      output for ±max (an exact ±max input is legal and is counted);
    * for a reduction (``sum``, ``dot``, ``@``) it is **unknown** (``None``):
      a partial sum can saturate and come back into range, so the final
      output says nothing about the chain.

    ``underflow``: number of outputs where a nonzero product truncated to 0.
    Exact under the same conditions as ``saturated`` (flags or replay); 0
    for an op that cannot underflow (add, sub, sum, neg, abs, div, sqrt);
    otherwise ``None`` (unknown): a product that truncates to 0 leaves no
    trace in the output.

    ``outputs_at_max``: the raw scan count, always available.
    ``div0`` and ``invalid`` are read off the inputs on the host and exact.
    ``exact`` is True when ``saturated`` is an exact count.
    """

    saturated: int | None = 0
    div0: int = 0
    invalid: int = 0
    exact: bool = False
    outputs_at_max: int = 0
    underflow: int | None = 0

    @property
    def known(self) -> bool:
        """False when saturation or underflow could not be determined."""
        return self.saturated is not None and self.underflow is not None

    @property
    def ok(self) -> bool | None:
        """True: no event occurred.  False: an event occurred, or the upper
        bound is nonzero.  None: saturation is unknown and nothing else went
        wrong."""
        if self.div0 or self.invalid or self.saturated or self.underflow:
            return False
        if self.saturated is None or self.underflow is None:
            return None
        return True                           # an upper bound of 0 is a proof of 0

    def __repr__(self) -> str:
        if self.saturated is None:
            sat = f"saturated=unknown (outputs_at_max={self.outputs_at_max})"
        elif self.exact:
            sat = f"saturated={self.saturated}"
        else:
            sat = f"saturated<={self.saturated}"
        under = "unknown" if self.underflow is None else self.underflow
        return (f"SFStatus({sat}, underflow={under}, div0={self.div0}, invalid={self.invalid}, "
                f"exact={self.exact})")
