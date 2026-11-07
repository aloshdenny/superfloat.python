"""Reference accumulation: sum, dot and matmul on code arrays (spec 3.1, 6).

``sum``, ``dot`` and ``@`` are chains of fma in the target's accumulation
order.  Atreides v1 saturates per step and accumulates at data width, so a
chain is: ``acc = saturate(acc + trunc(a_k * b_k / S))`` for k in order.
Order matters under saturation ((0.9 + 0.9) - 0.9 != 0.9 + (0.9 - 0.9)), so
it is a parameter here and an attribute of the target.

Every function returns ``(codes, events)`` like ``_vectorized``; the
overflow mask marks outputs where *any* partial sum saturated, the
underflow mask outputs where any product underflowed.
"""
from __future__ import annotations

import numpy as np

from . import _vectorized as vec
from ._format import SFFormat

ORDERS = ("sequential",)


def _check_order(order: str) -> None:
    if order not in ORDERS:
        raise ValueError(f"unknown accumulation order {order!r}; known: {ORDERS}")


def sum_codes(fmt: SFFormat, a: np.ndarray, axis: int, order: str = "sequential"):
    """Saturating sum along ``axis``, sequential from index 0."""
    _check_order(order)
    a = np.asarray(a, dtype=np.int64)
    a = np.moveaxis(a, axis, 0)
    acc = np.zeros(a.shape[1:], dtype=np.int64)
    over = np.zeros(a.shape[1:], dtype=bool)
    for k in range(a.shape[0]):
        acc, ev = vec.add(fmt, acc, a[k])
        over |= ev["overflow"]
    return acc, {"overflow": over}


def dot_codes(fmt: SFFormat, a: np.ndarray, b: np.ndarray, order: str = "sequential"):
    """1-D dot: chain of fma, sequential."""
    _check_order(order)
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    acc = np.int64(0)
    over = under = False
    first = None                                   # first event in time, as a scalar chain sees it
    for k in range(a.shape[0]):
        acc, ev = vec.fma(fmt, a[k], b[k], acc)
        if first is None and ev["overflow"]:
            first = "overflow"
        elif first is None and ev["underflow"]:
            first = "underflow"
        over |= bool(ev["overflow"])
        under |= bool(ev["underflow"])
    return np.int64(acc), {"overflow": np.bool_(over), "underflow": np.bool_(under), "first": first}


def matmul_codes(fmt: SFFormat, a: np.ndarray, b: np.ndarray, order: str = "sequential"):
    """(..., M, K) @ (..., K, N): per output a chain of fma over k, sequential.

    Vectorised across the output and the batch, looped over K, so it is
    bit-identical to ``dot_codes`` on each (row, column) pair.
    """
    _check_order(order)
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    K = a.shape[-1]
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    out_shape = batch + (a.shape[-2], b.shape[-1])
    acc = np.zeros(out_shape, dtype=np.int64)
    over = np.zeros(out_shape, dtype=bool)
    under = np.zeros(out_shape, dtype=bool)
    for k in range(K):
        col = a[..., :, k][..., :, None]          # (..., M, 1)
        row = b[..., k, :][..., None, :]          # (..., 1, N)
        acc, ev = vec.fma(fmt, col, row, acc)
        over |= ev["overflow"]
        under |= ev["underflow"]
    return acc, {"overflow": over, "underflow": under}


def matmul_bound(fmt: SFFormat, a: np.ndarray, b: np.ndarray) -> int:
    """Upper bound, in accumulator code units, on |every partial sum| of
    every output of ``a @ b`` (codes, batched like ``matmul_codes``).

    Each step adds trunc(k_a k_b / S), whose magnitude is at most
    |k_a||k_b| / S, so for output (i, j) every partial sum is bounded by
    sum_k |a_ik||b_kj| / S.  That per-output sum is O(MKN) to compute, so
    this uses the O(MK + KN) relaxation

        min( max_i ||a_i||_1 * max|b| ,  max|a| * max_j ||b_:j||_1 ) // S

    which is never looser than the spec 3.4 form K * max|A| * max|B| / S.
    Exact integer arithmetic, no rounding.
    """
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if a.size == 0 or b.size == 0 or a.shape[-1] == 0:
        return 0
    abs_a, abs_b = np.abs(a), np.abs(b)
    row_l1 = int(abs_a.sum(axis=-1).max())          # max over (batch, i) of sum_k |a_ik|
    col_l1 = int(abs_b.sum(axis=-2).max())          # max over (batch, j) of sum_k |b_kj|
    bound = min(row_l1 * int(abs_b.max()), int(abs_a.max()) * col_l1)
    return bound // fmt.scale


def matmul_bound_safe(fmt: SFFormat, a: np.ndarray, b: np.ndarray) -> bool:
    """Pre-flight proof that no partial sum of ``a @ b`` can saturate
    (spec 3.4, tightened to the L1 form of ``matmul_bound``)."""
    return matmul_bound(fmt, a, b) <= fmt.max_code


# --- wide accumulation: the v4.8.1 MX engine ------------------------------------------------
#
# Written from the units' definitions in the chip repo (superfloat.gpu, branch
# v4.8.1-openframe, test/v4/mxref.py and compiler_v4.8.1.md section 3.4) and
# checked bit for bit against mxref.Engine (tests/test_wide_mode.py).
#
# * Operands are sfN codes left-aligned in 16-bit sign-magnitude words.
#   sf9..sf16 run on the SF16 block, sf4..sf8 on the SF8 block, which reads
#   only each word's top byte.
# * SF16 block: each product is |a||w| with its partial-product bits below
#   column 9 dropped, plus 1024, cut to 19 fraction bits (not v1's exact
#   truncation).  SF8 block: exact products, 14 fraction bits.
# * K runs in chunks of 8 (one MAC).  A chunk's 8 products are summed exactly
#   and added to a W-bit two's-complement accumulator (Q6.19 = 25 bits, or
#   Q6.14 = 20 bits) that saturates near +-32 once per chunk.
# * POP: the total times 2^shift is rounded once to sfN, half to even, and
#   saturated at +-(2^(N-1) - 1); never -0.

MX_UNITS = {16: {"W": 25, "F": 19}, 8: {"W": 20, "F": 14}}
SHIFT_MIN, SHIFT_MAX = -32, 31


def mx_block(fmt: SFFormat) -> int:
    """The MX block an sfN operand runs on: 16 for sf9..sf16, 8 for sf4..sf8."""
    return 16 if fmt.bits >= 9 else 8


def _mx_mags(fmt: SFFormat, codes: np.ndarray, blk: int) -> np.ndarray:
    """|code| as the block reads it: the 15-bit magnitude field of the
    left-aligned 16-bit word, or the 7-bit field of its top byte (SF8)."""
    mag = np.abs(codes) << (16 - fmt.bits)
    return mag >> 8 if blk == 8 else mag


def _mx_products16(am: np.ndarray, wm: np.ndarray) -> np.ndarray:
    """SF16-block product magnitudes, in 19 fraction bits (see mxref.product)."""
    keep = ~np.int64((1 << 9) - 1)
    rs = np.full(np.broadcast_shapes(am.shape, wm.shape), 1024, dtype=np.int64)
    for i in range(15):
        rs += ((am >> i) & 1) * ((wm << i) & keep)
    rs &= (1 << 30) - 1
    return rs >> 11


def _mx_pop(fmt: SFFormat, acc: np.ndarray, shift: int, frac: int):
    """sfN codes of acc * 2^(shift - frac), rounded half to even and saturated.
    Returns (codes, overflow mask, underflow mask)."""
    M = fmt.max_code
    e = shift - frac + fmt.bits - 1          # exponent from accumulator units to output codes
    mag = np.abs(acc)
    if e >= 0:
        q = mag << e
    else:
        d = -e
        fl = mag >> d
        rem = mag - (fl << d)
        half = np.int64(1) << (d - 1)
        q = fl + ((rem > half) | ((rem == half) & ((fl & 1) == 1)))
    over = q > M
    q = np.minimum(q, M)
    under = (acc != 0) & (q == 0)
    return np.where(acc < 0, -q, q), over, under


def check_shift(shift) -> int:
    if shift is None:
        return 0
    if isinstance(shift, bool) or not isinstance(shift, (int, np.integer)):
        raise TypeError(f"shift must be an int, got {type(shift).__name__}")
    if not SHIFT_MIN <= int(shift) <= SHIFT_MAX:
        raise ValueError(f"shift must be between {SHIFT_MIN} and {SHIFT_MAX}, got {shift}")
    return int(shift)


def matmul_wide_codes(fmt: SFFormat, a: np.ndarray, b: np.ndarray, shift: int = 0):
    """(..., M, K) @ (..., K, N) on the MX engine, batched like ``matmul_codes``.

    Returns ``(codes, events)``: ``overflow`` marks outputs whose accumulator
    saturated at +-32 or whose POP saturated at +-max; ``underflow`` marks
    outputs whose nonzero total rounded to 0.
    """
    shift = check_shift(shift)
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    K = a.shape[-1]
    batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    out_shape = batch + (a.shape[-2], b.shape[-1])
    blk = mx_block(fmt)
    W, F = MX_UNITS[blk]["W"], MX_UNITS[blk]["F"]
    lo, hi = -(1 << (W - 1)), (1 << (W - 1)) - 1
    am, bm = _mx_mags(fmt, a, blk), _mx_mags(fmt, b, blk)
    an, bn = a < 0, b < 0
    acc = np.zeros(out_shape, dtype=np.int64)
    clipped = np.zeros(out_shape, dtype=bool)
    for k0 in range(0, K, 8):
        ca = am[..., :, k0:k0 + 8][..., :, :, None]          # (..., M, c, 1)
        cb = bm[..., k0:k0 + 8, :][..., None, :, :]          # (..., 1, c, N)
        t = _mx_products16(ca, cb) if blk == 16 else ca * cb
        neg = an[..., :, k0:k0 + 8][..., :, :, None] ^ bn[..., k0:k0 + 8, :][..., None, :, :]
        s = acc + np.where(neg, -t, t).sum(axis=-2)
        clipped |= (s < lo) | (s > hi)
        acc = np.clip(s, lo, hi)
    codes, over, under = _mx_pop(fmt, acc, shift, F)
    return codes, {"overflow": clipped | over, "underflow": under}


def matmul_wide_acc_safe(fmt: SFFormat, a: np.ndarray, b: np.ndarray) -> bool:
    """Pre-flight proof that no accumulator of ``a @ b`` can reach +-32.

    Each SF16-block product term is at most |a||w| / 2^11 + 1 (dropped bits
    only lower it; the +1024 adds under one unit), and an SF8 term is exact,
    so the L1 relaxation of ``matmul_bound`` bounds every partial total."""
    a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
    if a.size == 0 or b.size == 0 or a.shape[-1] == 0:
        return True
    blk = mx_block(fmt)
    am, bm = _mx_mags(fmt, a, blk), _mx_mags(fmt, b, blk)
    row_l1 = int(am.sum(axis=-1).max())
    col_l1 = int(bm.sum(axis=-2).max())
    l1 = min(row_l1 * int(bm.max()), int(am.max()) * col_l1)
    W = MX_UNITS[blk]["W"]
    bound = l1 // (1 << 11) + a.shape[-1] if blk == 16 else l1
    return bound <= (1 << (W - 1)) - 1
