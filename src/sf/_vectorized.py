"""NumPy-vectorised form of the reference ops on int64 code arrays.

Same semantics as ``_reference`` (one op, one definition), written over
arrays so that the nightly exhaustive sf16 run (4.3 billion pairs per
binary op) and, later, Stage 1.1 host arrays can use it.  Every function
returns ``(codes, event_mask)`` with ``event_mask`` a dict of boolean
arrays keyed by event name.

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


def add(fmt: SFFormat, ka, kb):
    return _saturate(fmt, ka + kb)


def sub(fmt: SFFormat, ka, kb):
    return _saturate(fmt, ka - kb)


def mul(fmt: SFFormat, ka, kb):
    p = ka * kb
    k = _trunc_div(p, fmt.scale)
    return k, {"underflow": (k == 0) & (p != 0)}


def div(fmt: SFFormat, ka, kb):
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
    neg = ka < 0
    k = np.floor(np.sqrt(np.where(neg, 0, ka).astype(np.float64) * fmt.scale)).astype(np.int64)
    # float64 sqrt is correctly rounded; fix the rare off-by-one at exact squares
    k = np.where((k + 1) * (k + 1) <= ka * fmt.scale, k + 1, k)
    k = np.where(k * k > ka * fmt.scale, k - 1, k)
    return np.where(neg, 0, k), {"invalid": neg}


OPS = {"add": add, "sub": sub, "mul": mul, "div": div}
