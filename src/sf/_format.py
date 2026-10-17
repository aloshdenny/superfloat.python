"""Numeric format definitions for sfN, N = 4 to 16.

sfN is signed fixed point Q1.(N-1) in sign-magnitude: 1 sign bit and N-1
fraction bits.  value = (-1)^s * m / 2^(N-1), m in [0, 2^(N-1) - 1].

Inside the host reference every value is held as a *signed integer code*
k in [-M, M] with M = 2^(N-1) - 1.  The sign-magnitude bit pattern the
hardware uses is only produced at the device boundary (``to_bits`` /
``from_bits``).  Negative zero (sign set, m = 0) is canonicalised to +0 on
the way in, so a code of 0 always means +0.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

MIN_BITS = 4
MAX_BITS = 16


@dataclass(frozen=True)
class SFFormat:
    """Constants for one width.  Immutable; one shared instance per width."""

    bits: int

    def __post_init__(self) -> None:
        if not (MIN_BITS <= self.bits <= MAX_BITS):
            raise ValueError(f"sf width must be {MIN_BITS}..{MAX_BITS}, got {self.bits}")

    # --- derived constants -------------------------------------------------
    @property
    def name(self) -> str:
        return f"sf{self.bits}"

    @property
    def frac_bits(self) -> int:
        return self.bits - 1

    @property
    def scale(self) -> int:
        """2^(N-1): one unit of value in integer codes."""
        return 1 << self.frac_bits

    @property
    def max_code(self) -> int:
        """M = 2^(N-1) - 1, the largest magnitude code."""
        return self.scale - 1

    @property
    def codes(self) -> int:
        """Number of distinct values (negative zero is canonicalised away)."""
        return 2 * self.max_code + 1

    @property
    def lsb(self) -> Fraction:
        return Fraction(1, self.scale)

    @property
    def max(self) -> Fraction:
        return Fraction(self.max_code, self.scale)

    @property
    def min(self) -> Fraction:
        return -self.max

    @property
    def cast_overflow_threshold(self) -> Fraction:
        """|v| >= this overflows when cast from a host type (1 - 2^-N)."""
        return Fraction(1) - Fraction(1, 1 << self.bits)

    @property
    def cast_underflow_threshold(self) -> Fraction:
        """0 < |v| <= this rounds to 0 when cast from a host type (2^-N)."""
        return Fraction(1, 1 << self.bits)

    # --- code helpers ------------------------------------------------------
    def check_code(self, code: int) -> int:
        if not isinstance(code, int) or isinstance(code, bool):
            raise TypeError(f"code must be an int, got {type(code).__name__}")
        if not (-self.max_code <= code <= self.max_code):
            raise ValueError(f"code {code} is outside {self.name} range ±{self.max_code}")
        return code

    def to_bits(self, code: int) -> int:
        """Signed code -> sign-magnitude bit pattern (what the device stores)."""
        self.check_code(code)
        if code < 0:
            return (1 << self.frac_bits) | (-code)
        return code

    def from_bits(self, bits: int) -> int:
        """Sign-magnitude bit pattern -> signed code.  -0 becomes +0."""
        if not isinstance(bits, int) or isinstance(bits, bool):
            raise TypeError(f"bits must be an int, got {type(bits).__name__}")
        if not (0 <= bits < (1 << self.bits)):
            raise ValueError(f"bit pattern {bits:#x} does not fit in {self.bits} bits")
        sign = bits >> self.frac_bits
        mag = bits & self.max_code
        return -mag if sign else mag

    def to_fraction(self, code: int) -> Fraction:
        return Fraction(code, self.scale)

    def to_float(self, code: int) -> float:
        # Exact: |code| < 2^15 and the denominator is a power of two.
        return code / self.scale

    def __repr__(self) -> str:
        return f"SFFormat({self.name})"


FORMATS: dict[int, SFFormat] = {n: SFFormat(n) for n in range(MIN_BITS, MAX_BITS + 1)}


def widen_code(code: int, src: SFFormat, dst: SFFormat) -> int:
    """Exact widening of a code from a narrower to a wider format."""
    if dst.bits < src.bits:
        raise ValueError("widen_code cannot narrow")
    return code << (dst.bits - src.bits)
