"""NumPy-vectorised form of the reference ops on int64 code arrays.

Same semantics as ``_reference`` (one op, one definition), written over
arrays.  This is what host arrays (Stage 1.1), the mock target and the
nightly exhaustive sf16 run execute.  Every op returns ``(codes, events)``
with ``events`` a dict of boolean masks keyed by event name; only the events
an op can raise are present.

All intermediate products fit in int64: |k| < 2^15, so |k_a k_b| < 2^30
and |k_a S| < 2^30.
"""
from __future__ import annotations

import numpy as np

from ._format import SFFormat


def _trunc_div(num, den):
    """Elementwise integer division truncating toward zero; den != 0."""
    q = np.abs(num) // np.abs(den)
    return np.where((num < 0) != (den < 0), -q, q)


def _saturate(fmt: SFFormat, k):
    over = np.abs(k) > fmt.max_code
    return np.clip(k, -fmt.max_code, fmt.max_code), {"overflow": over}


# --- casts -----------------------------------------------------------------

def cast_array(fmt: SFFormat, values: np.ndarray):
    """Quantize a host array (float, int or bool) to codes: round half to
    even, saturate, with overflow / underflow / invalid masks."""
    values = np.asarray(values)
    if values.dtype == object:
        from . import _reference as ref
        flat = values.ravel()
        codes = np.empty(flat.shape, dtype=np.int64)
        ev = {k: np.zeros(flat.shape, dtype=bool) for k in ("overflow", "underflow", "invalid")}
        for i, v in enumerate(flat):
            k, e = ref.cast(fmt, v)
            codes[i] = k
            if e:
                ev[e][i] = True
        return codes.reshape(values.shape), {k: m.reshape(values.shape) for k, m in ev.items()}
    if values.dtype.kind in "iub":
        # decide sign and overflow in the source type; only 0 is representable
        nonzero = values != 0
        if values.dtype.kind == "u" or values.dtype.kind == "b":
            sign = nonzero.astype(np.int64)                        # unsigned: never negative
        else:
            sign = np.sign(values).astype(np.int64)
        codes = (sign * fmt.max_code).astype(np.int64)
        return codes, {"overflow": nonzero,
                       "underflow": np.zeros(values.shape, bool),
                       "invalid": np.zeros(values.shape, bool)}
    if values.dtype.kind != "f":
        raise TypeError(f"cannot cast array of dtype {values.dtype} to {fmt.name}")
    v = values.astype(np.float64)                  # exact for float16/32/64
    nan = np.isnan(v)
    # overflow is decided on the exact value, before any integer conversion:
    # |v| >= 1 - 2^-N rounds to 2^(N-1) or beyond (ties to even), including ±inf
    threshold = 1.0 - 2.0 ** -fmt.bits             # exact in float64
    over = (np.abs(v) >= threshold) & ~nan
    fits = ~over & ~nan
    k = np.zeros(v.shape, dtype=np.int64)
    k[fits] = np.rint(v[fits] * fmt.scale).astype(np.int64)   # exact scaling, half-even
    k[over] = np.sign(v[over]).astype(np.int64) * fmt.max_code
    under = fits & (k == 0) & (v != 0)
    return k, {"overflow": over, "underflow": under, "invalid": nan}


def to_float(fmt: SFFormat, codes) -> np.ndarray:
    return np.asarray(codes, dtype=np.float64) / fmt.scale


def to_bits(fmt: SFFormat, codes) -> np.ndarray:
    """Signed codes -> sign-magnitude patterns (uint16 for every width)."""
    k = np.asarray(codes, dtype=np.int64)
    return np.where(k < 0, (1 << fmt.frac_bits) | (-k), k).astype(np.uint16)


def from_bits(fmt: SFFormat, bits) -> np.ndarray:
    b = np.asarray(bits, dtype=np.int64)
    if np.any((b < 0) | (b >= (1 << fmt.bits))):
        raise ValueError(f"bit pattern does not fit in {fmt.bits} bits")
    mag = b & fmt.max_code
    return np.where(b >> fmt.frac_bits, -mag, mag).astype(np.int64)


# --- arithmetic ------------------------------------------------------------

def add(fmt: SFFormat, ka, kb):
    return _saturate(fmt, ka + kb)


def sub(fmt: SFFormat, ka, kb):
    return _saturate(fmt, ka - kb)


def neg(fmt: SFFormat, ka):
    return -np.asarray(ka), {}


def abs_(fmt: SFFormat, ka):
    return np.abs(ka), {}


def mul(fmt: SFFormat, ka, kb):
    p = np.asarray(ka) * np.asarray(kb)
    k = _trunc_div(p, fmt.scale)
    return k, {"underflow": (k == 0) & (p != 0)}


def div(fmt: SFFormat, ka, kb):
    ka, kb = np.asarray(ka), np.asarray(kb)
    zero = kb == 0
    sa, sb = np.sign(ka), np.sign(kb)
    safe_b = np.where(zero, 1, kb)
    q = _trunc_div(ka * fmt.scale, safe_b)
    over = (~zero) & (ka != 0) & (np.abs(ka) >= np.abs(kb))
    k = np.where(over, np.where(sa == sb, fmt.max_code, -fmt.max_code), q)
    k = np.where(zero, sa * fmt.max_code, k)
    return k, {"overflow": over, "divide": zero}


def fma(fmt: SFFormat, ka, kb, kc):
    kp, ev_p = mul(fmt, ka, kb)
    k, ev_s = add(fmt, kp, kc)
    return k, {"overflow": ev_s["overflow"], "underflow": ev_p["underflow"] & ~ev_s["overflow"]}


def sqrt(fmt: SFFormat, ka):
    ka = np.asarray(ka)
    neg_ = ka < 0
    x = np.where(neg_, 0, ka) * fmt.scale
    k = np.floor(np.sqrt(x.astype(np.float64))).astype(np.int64)
    # float64 sqrt is correctly rounded; fix the rare off-by-one around exact squares
    k = np.where((k + 1) * (k + 1) <= x, k + 1, k)
    k = np.where(k * k > x, k - 1, k)
    return np.where(neg_, 0, k), {"invalid": neg_}


def pow_int(fmt: SFFormat, ka, n: int):
    """a ** n, n >= 1, left-to-right chain of ``mul``; stops once every
    element is 0."""
    ka = np.asarray(ka)
    k = ka.copy()
    under = np.zeros(k.shape, dtype=bool)
    for _ in range(n - 1):
        if not np.any(k):
            break
        k, ev = mul(fmt, k, ka)
        under |= ev["underflow"]
    return k, {"underflow": under}


OPS = {"add": add, "sub": sub, "mul": mul, "div": div}
