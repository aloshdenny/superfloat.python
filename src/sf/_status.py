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

    ``outputs_at_max``: the raw scan count, always available.
    ``div0`` and ``invalid`` are read off the inputs on the host and exact.
    ``exact`` is True when ``saturated`` is an exact count.
    """

    saturated: int | None = 0
    div0: int = 0
    invalid: int = 0
    exact: bool = False
    outputs_at_max: int = 0

    @property
    def known(self) -> bool:
        """False when saturation could not be determined (reduction on a target without flags)."""
        return self.saturated is not None

    @property
    def ok(self) -> bool | None:
        """True: no event occurred.  False: an event occurred, or the upper
        bound is nonzero.  None: saturation is unknown and nothing else went
        wrong."""
        if self.div0 or self.invalid:
            return False
        if self.saturated is None:
            return None
        return self.saturated == 0            # an upper bound of 0 is a proof of 0

    def __repr__(self) -> str:
        if self.saturated is None:
            sat = f"saturated=unknown (outputs_at_max={self.outputs_at_max})"
        elif self.exact:
            sat = f"saturated={self.saturated}"
        else:
            sat = f"saturated<={self.saturated}"
        return f"SFStatus({sat}, div0={self.div0}, invalid={self.invalid}, exact={self.exact})"
