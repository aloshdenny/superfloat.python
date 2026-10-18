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
