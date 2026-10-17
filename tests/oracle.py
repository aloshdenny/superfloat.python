"""Independent oracle for the tests (spec section 10).

``fractions.Fraction`` decides whether an event should happen and what the
exact result is.  This module never touches integer codes or the sf
package internals: it works on exact rational *values* and applies the
rules of spec section 3.1 literally.  The reference model in
``sf._reference`` works on integer codes.  Agreement between the two is
what the tests assert.

Every function returns ``(value: Fraction, event: str | None)`` where the
value is what the hardware would hold (saturated / truncated).
"""
from __future__ import annotations

import math
from fractions import Fraction


class Fmt:
    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.lsb = Fraction(1, 2 ** (bits - 1))
        self.max = 1 - self.lsb
        self.max_code = 2 ** (bits - 1) - 1

    def values(self):
        """Every representable value, ascending."""
        return [k * self.lsb for k in range(-self.max_code, self.max_code + 1)]

    # --- primitives ---------------------------------------------------------
    def saturate(self, v: Fraction):
        if v > self.max:
            return self.max, "overflow"
        if v < -self.max:
            return -self.max, "overflow"
        return v, None

    def trunc(self, v: Fraction) -> Fraction:
        """Truncate toward zero to a multiple of 1 LSB."""
        q = abs(v) / self.lsb
        q = math.floor(q)
        r = q * self.lsb
        return -r if v < 0 else r

    def round_half_even(self, v: Fraction) -> Fraction:
        q = v / self.lsb
        return round(q) * self.lsb           # Fraction.__round__ is half-even

    # --- ops ----------------------------------------------------------------
    def cast(self, v):
        """Host scalar -> SF.  ``v`` may be a float (nan / inf allowed) or a Fraction."""
        if isinstance(v, float):
            if math.isnan(v):
                return Fraction(0), "invalid"
            if math.isinf(v):
                return (self.max if v > 0 else -self.max), "overflow"
            v = Fraction(v)
        r = self.round_half_even(v)
        r, ev = self.saturate(r)
        if ev:
            return r, ev
        if r == 0 and v != 0:
            return r, "underflow"
        return r, None

    def add(self, a, b):
        return self.saturate(a + b)

    def sub(self, a, b):
        return self.saturate(a - b)

    def neg(self, a):
        return -a, None

    def abs(self, a):
        return abs(a), None

    def mul(self, a, b):
        exact = a * b
        r = self.trunc(exact)
        if r == 0 and exact != 0:
            return r, "underflow"
        return r, None

    def div(self, a, b):
        if b == 0:
            if a == 0:
                return Fraction(0), "divide"
            return (self.max if a > 0 else -self.max), "divide"
        exact = a / b
        if abs(a) >= abs(b):                   # |exact| >= 1: not representable
            return (self.max if exact > 0 else -self.max), "overflow"
        return self.trunc(exact), None

    def fma(self, a, b, c):
        p, ev_p = self.mul(a, b)
        s, ev_s = self.add(p, c)
        return s, (ev_s or ev_p)

    def sqrt(self, a):
        if a < 0:
            return Fraction(0), "invalid"
        # trunc(sqrt(a)): largest multiple of lsb whose square is <= a
        q = math.isqrt(math.floor(a / (self.lsb * self.lsb)))
        return q * self.lsb, None

    def pow_int(self, a, n):
        r, event = a, None
        for _ in range(n - 1):
            r, ev = self.mul(r, a)
            event = event or ev
        return r, event


BINARY = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
