"""Bit-exact host reference model (spec sections 2 and 3.1).

Every function here works on *signed integer codes* and returns
``(code, event)`` where ``event`` is ``None`` or one of ``"overflow"``,
``"underflow"``, ``"divide"``, ``"invalid"``.  The returned code is always
the hardware value: saturated on overflow, 0 on underflow, the proposed
device result on divide-by-zero / domain errors.

Nothing here raises or warns for numeric events; that is the error policy's
job (``_errstate.handle``).  Keeping the two apart is what lets the same
model serve as oracle for the compiler backends and the device later.

Semantics, with M = 2^(N-1) - 1 and S = 2^(N-1):

    cast(v)      round_half_even(v * S), saturate to ±M
    a + b        k_a + k_b, saturate            (exact otherwise)
    a - b        k_a - k_b, saturate
    -a, |a|      -k_a, |k_a|                     (never overflow)
    a * b        trunc(k_a * k_b / S)            (never overflows; may underflow)
    a / b        trunc(k_a * S / k_b)            (overflows when |a| >= |b|)
    fma(a,b,c)   k_c + trunc(k_a * k_b / S), saturate
    sqrt(a)      trunc(sqrt(a) * S) = isqrt(k_a * S)

Rounding: casts from host types round to nearest, ties to even.  Device
arithmetic truncates toward zero (current Atreides RTL).
"""
from __future__ import annotations

import math
import numbers
from decimal import Decimal
from fractions import Fraction
from typing import Optional, Tuple

from ._format import SFFormat

Event = Optional[str]
Result = Tuple[int, Event]


# --- helpers ---------------------------------------------------------------

def _trunc_div(num: int, den: int) -> int:
    """Integer division truncating toward zero (C semantics), den != 0."""
    q = abs(num) // abs(den)
    return q if (num < 0) == (den < 0) else -q


def _sign(k: int) -> int:
    return (k > 0) - (k < 0)


def to_exact(v) -> Fraction | float:
    """Convert a host scalar to an exact Fraction.

    Returns a float only for NaN / ±inf, which have no exact value.
    Accepts int, bool, float, Fraction, Decimal, str (parsed exactly) and
    NumPy integer / floating scalars.
    """
    if isinstance(v, numbers.Integral):          # int, bool, numpy ints
        return Fraction(int(v))
    if isinstance(v, numbers.Rational):          # Fraction and friends
        return Fraction(int(v.numerator), int(v.denominator))
    if isinstance(v, Decimal):
        if v.is_nan():
            return math.nan
        if v.is_infinite():
            return math.inf if v > 0 else -math.inf
        return Fraction(v)
    if isinstance(v, str):
        try:
            return Fraction(v.strip())
        except ValueError:
            raise TypeError(f"cannot parse {v!r} as an SF value") from None
    if isinstance(v, numbers.Real):              # float, numpy floats
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return f
        return Fraction(f)                       # exact binary value
    # numpy integer scalars are numbers.Integral; anything else is a TypeError
    raise TypeError(f"cannot convert {type(v).__name__} to an SF value")


# --- casts -----------------------------------------------------------------

def cast_exact(fmt: SFFormat, v: Fraction) -> Result:
    """Quantize an exact rational to ``fmt`` (round half to even)."""
    k = round(v * fmt.scale)                     # Fraction.__round__ is half-even
    if k > fmt.max_code:
        return fmt.max_code, "overflow"
    if k < -fmt.max_code:
        return -fmt.max_code, "overflow"
    if k == 0 and v != 0:
        return 0, "underflow"
    return k, None


def cast(fmt: SFFormat, v) -> Result:
    """Quantize a host scalar (see ``to_exact``) to ``fmt``."""
    x = to_exact(v)
    if isinstance(x, float):                     # nan or ±inf
        if math.isnan(x):
            return 0, "invalid"
        return (fmt.max_code if x > 0 else -fmt.max_code), "overflow"
    return cast_exact(fmt, x)


# --- arithmetic ------------------------------------------------------------

def _saturate(fmt: SFFormat, k: int) -> Result:
    if k > fmt.max_code:
        return fmt.max_code, "overflow"
    if k < -fmt.max_code:
        return -fmt.max_code, "overflow"
    return k, None


def add(fmt: SFFormat, ka: int, kb: int) -> Result:
    return _saturate(fmt, ka + kb)


def sub(fmt: SFFormat, ka: int, kb: int) -> Result:
    return _saturate(fmt, ka - kb)


def neg(fmt: SFFormat, ka: int) -> Result:
    return -ka, None


def abs_(fmt: SFFormat, ka: int) -> Result:
    return abs(ka), None


def mul(fmt: SFFormat, ka: int, kb: int) -> Result:
    p = ka * kb
    k = _trunc_div(p, fmt.scale)
    if k == 0 and p != 0:
        return 0, "underflow"
    return k, None


def div(fmt: SFFormat, ka: int, kb: int) -> Result:
    """a / b.  Divide-by-zero returns the proposed device value
    (sign(a) * max, and 0 for 0 / 0); |a| >= |b| saturates."""
    if kb == 0:
        return _sign(ka) * fmt.max_code, "divide"
    if ka == 0:
        return 0, None
    if abs(ka) >= abs(kb):
        return (fmt.max_code if _sign(ka) == _sign(kb) else -fmt.max_code), "overflow"
    return _trunc_div(ka * fmt.scale, kb), None   # never underflows: |result| >= 1


def fma(fmt: SFFormat, ka: int, kb: int, kc: int) -> Result:
    """a * b + c with the product truncated before the add.

    Gives the same bits as ``add(mul(a, b), c)``.  If both the product
    underflows and the sum overflows, overflow is reported (it is the
    event the final value shows).
    """
    kp, ev_p = mul(fmt, ka, kb)
    k, ev_s = add(fmt, kp, kc)
    return k, (ev_s or ev_p)


def sqrt(fmt: SFFormat, ka: int) -> Result:
    """sqrt(a), truncated toward zero.  Negative input is a domain error
    and returns the proposed device value 0."""
    if ka < 0:
        return 0, "invalid"
    # trunc(sqrt(ka / S) * S) = floor(sqrt(ka * S)), exact in integers.
    return math.isqrt(ka * fmt.scale), None


def pow_int(fmt: SFFormat, ka: int, n: int) -> Result:
    """a ** n for integer n >= 1 as a left-to-right chain of ``mul``.
    Reports the first underflow seen (the chain can only go to 0).

    The chain stops once the running value is 0, so the cost is bounded by
    the number of steps it takes to reach 0 (at most 2^(N-1) - 1, for
    a = max), whatever ``n`` is."""
    if n < 1:
        raise ValueError("pow_int needs n >= 1")
    k, event = ka, None
    for _ in range(n - 1):
        if k == 0:
            break                      # stays 0 from here on; the chain is finite anyway
        k, ev = mul(fmt, k, ka)
        event = event or ev
    return k, event
