"""Post-op status of a heavy op (spec section 3.4)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SFStatus:
    """What the runtime knows about a heavy op after it ran.

    ``saturated`` is a count of outputs at ±max.  On Atreides v1 there is no
    saturation flag, so the runtime scans the output after readback and the
    count is an *upper bound* (an exact ±max is legal, and partial sums that
    saturated and came back into range are missed); ``exact`` is then False.
    When the pre-flight bound proves saturation impossible, or the target
    exposes sticky flags, ``exact`` is True.

    ``div0`` and ``invalid`` are decided on the host from the inputs (a zero
    divisor, a negative sqrt argument), so they are always exact.
    """

    saturated: int = 0
    div0: int = 0
    invalid: int = 0
    exact: bool = False

    @property
    def ok(self) -> bool:
        return self.saturated == 0 and self.div0 == 0 and self.invalid == 0

    def __repr__(self) -> str:
        sat = f"saturated={self.saturated}" if self.exact else f"saturated<={self.saturated}"
        return f"SFStatus({sat}, div0={self.div0}, invalid={self.invalid}, exact={self.exact})"
