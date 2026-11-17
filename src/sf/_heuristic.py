"""The wide-mode shift rule (Alosh's ``sf wide mode (MX).md``, item 5).

Stateless: an output depends only on its inputs.  No calibration, nothing
learned at run time.  For a matmul of quantized activations ``a`` (one
power-of-two exponent ``ea`` per token) and weights ``w`` (one ``ew`` per
output column), with K the dot-product length:

    h        = max(0, floor(log2 K) // 2 - MARGIN)      MARGIN = 2, the shipped table
    l2[t, c] = pow2_exp(|a_t|_2 * |w_c|_2)              the exponent that can never clip
    eo[t, c] = l2[t, c] - h
    shift    = clamp(ea[t] + ew[c] - eo[t, c], -32, 31)

then POP at ``shift``, and every output that saturated at +-max is popped
again at ``l2``.  Why it works: a typical dot product of length K sits about
sqrt(K) below its L2 bound, so ``h`` buys back about half of log2 K bits.

The table is pinned by ``tests/data/sf_golden_v4.9.2.json`` (``heuristic``):
``exponent_rule`` for ``h`` and 13 matmul cases end to end.  Change it only
with the spec version, and tune it only on real model activations.
"""
from __future__ import annotations

import numpy as np

from . import _linalg
from ._format import SFFormat

MARGIN = 2
CALIBRATIONS = ("heuristic",)


def h_for(K: int) -> int:
    """The shipped number: bits taken back below the L2 bound for length K."""
    K = int(K)
    if K < 1:
        return 0
    return max(0, (K.bit_length() - 1) // 2 - MARGIN)


def pow2_exp(m) -> np.ndarray:
    """The integer e with m / 2^e < 1 (the smallest), and 0 where m is 0."""
    m = np.asarray(m, dtype=np.float64)
    out = np.zeros(m.shape, dtype=np.int64)
    nz = m > 0
    if nz.any():
        _, e = np.frexp(m[nz])           # m = f * 2^e with 0.5 <= f < 1, exact
        out[nz] = e
    return out


def quantize_pow2(fmt: SFFormat, x: np.ndarray, axis: int):
    """Codes with one power-of-two exponent per slice along ``axis``.

    ``e = pow2_exp(max |x|)`` over the slice (0 if all zero), so
    ``|x| / 2^e < 1``; ``q = round_half_even(x / 2^e * 2^(N-1))``, clamped to
    +-max.  Value = ``q * 2^(e - (N-1))``.  Returns (codes, e, underflow):
    ``underflow`` counts nonzero inputs that quantized to 0."""
    P, M = fmt.bits - 1, fmt.max_code
    x = np.asarray(x, dtype=np.float64)
    e = pow2_exp(np.abs(x).max(axis=axis) if x.shape[axis] else np.zeros(np.delete(x.shape, axis)))
    q = _round_codes(x, np.expand_dims(e, axis), P, M)
    return q, e, int(np.count_nonzero((q == 0) & (x != 0)))


def _round_codes(x, e, P, M):
    return np.clip(np.rint(np.ldexp(x, (P - e).astype(np.int64))), -M, M).astype(np.int64)


def _norms(fmt: SFFormat, codes: np.ndarray, axis: int) -> np.ndarray:
    """L2 norms of code vectors, in code units (exact sums of squares)."""
    c = codes.astype(np.float64)
    return np.sqrt(np.einsum("...i,...i->...", np.moveaxis(c, axis, -1), np.moveaxis(c, axis, -1)))


def acc_guard(fmt: SFFormat, x: np.ndarray, qa: np.ndarray, ea: np.ndarray, qb: np.ndarray, mx=None):
    """Raise per-token exponents until no accumulator can reach the unit's
    limit (+-512 on v4.9.2's SF16 block): the chip cannot report that clip.

    Bound per token t, in accumulator units: Cauchy-Schwarz on the
    magnitudes the block reads, |S| <= |a_t| max_c |b_c| / 2^(30 - F) + K on
    the SF16 block (each truncated product is at most one unit above its
    exact value), |a_t| max_c |b_c| on SF8.  It holds for every partial
    total, so a token that passes can never clip.  On v4.9.2 it costs one
    norm per token and has not triggered on any model measured; on v4.8.1
    (+-32) it does.

    ``qa``: (..., M, K) codes, ``ea``: (..., M), ``qb``: (..., K, N) codes.
    Returns (qa, ea, raised): ``raised`` counts tokens whose exponent went up."""
    blk = _linalg.mx_block(fmt)
    unit = _linalg.mx_units(mx)[blk]
    hi = (1 << (unit["W"] - 1)) - 1
    K = qa.shape[-1]
    bmag = _linalg._mx_mags(fmt, qb, blk).astype(np.float64)
    bmax = np.sqrt((bmag * bmag).sum(axis=-2)).max(axis=-1, keepdims=True) if bmag.size else 0.0
    P, M = fmt.bits - 1, fmt.max_code
    raised = np.zeros(ea.shape, dtype=bool)
    for _ in range(64):
        amag = _linalg._mx_mags(fmt, qa, blk).astype(np.float64)
        an = np.sqrt((amag * amag).sum(axis=-1))                  # (..., M)
        prod = an * bmax                                          # bmax: (..., 1), the largest column norm
        bound = (prod / float(1 << (30 - unit["F"])) + K) if blk == 16 else prod
        bad = bound * (1 + 1e-12) + 1 > hi
        if not bad.any():
            return qa, ea, int(np.count_nonzero(raised))
        ea = np.where(bad, ea + 1, ea)
        raised |= bad
        qa = np.where(bad[..., None], _round_codes(x, ea[..., None], P, M), qa)
    raise RuntimeError("sf: accumulator guard did not converge")    # pragma: no cover


def plan(fmt: SFFormat, qa, ea, qb, eb):
    """The rule for (..., M, K) @ (..., K, N) with token exponents ``ea``
    (..., M) and column exponents ``eb`` (..., N).

    Returns (shift, shift_l2, h): POP's shift per output, the shift at the
    safe l2 exponent for redos, and the table value used."""
    P = fmt.bits - 1
    K = qa.shape[-1]
    h = h_for(K)
    na = _norms(fmt, qa, -1) * np.ldexp(1.0, (ea - P).astype(np.int64))       # dequantized norms
    nb = _norms(fmt, qb, -2) * np.ldexp(1.0, (eb - P).astype(np.int64))
    l2 = pow2_exp(na[..., :, None] * nb[..., None, :])
    base = ea[..., :, None] + eb[..., None, :]
    shift = np.clip(base - (l2 - h), _linalg.SHIFT_MIN, _linalg.SHIFT_MAX)
    shift_l2 = np.clip(base - l2, _linalg.SHIFT_MIN, _linalg.SHIFT_MAX)
    return shift, shift_l2, h
