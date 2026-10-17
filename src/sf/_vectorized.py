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
        v = values.astype(np.int64)
        nonzero = v != 0
        codes = np.where(nonzero, np.sign(v) * fmt.max_code, 0).astype(np.int64)
        return codes, {"overflow": nonzero,
                       "underflow": np.zeros(v.shape, bool),
                       "invalid": np.zeros(v.shape, bool)}
    if values.dtype.kind != "f":
        raise TypeError(f"cannot cast array of dtype {values.dtype} to {fmt.name}")
    v = values.astype(np.float64)                  # exact for float16/32/64
    nan = np.isnan(v)
    inf = np.isinf(v)
    safe = np.where(nan | inf, 0.0, v)
    k = np.rint(safe * fmt.scale).astype(np.int64)  # exact scaling, half-even
    over = (np.abs(k) > fmt.max_code) | inf
    k = np.clip(k, -fmt.max_code, fmt.max_code)
    k = np.where(inf, np.sign(np.where(nan, 0.0, v)).astype(np.int64) * fmt.max_code, k)
    under = (k == 0) & (safe != 0) & ~inf
    k = np.where(nan, 0, k)
    return k, {"overflow": over & ~nan, "underflow": under, "invalid": nan}


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
