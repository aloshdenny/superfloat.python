"""Integer definitions of the non-matmul inference ops (Alosh, 2026-11-18): exp, softmax, GELU, SiLU,
reciprocal, rsqrt, and LayerNorm / RMSNorm for reference.

Every op is a fixed sequence of sf's own SF16 arithmetic, so it gives the same bits on any host and on the chip:

    mul(a, b)       = sign(a) sign(b) * floor(|a| |b| / 2^15)       sf16 multiply, truncated toward zero
    add(a, b)       = clamp(a + b, -32767, 32767)                    sf16 add, saturating
    fma(acc, a, b)  = add(acc, mul(a, b))
    muladd(P, v, c) = add(c, mul(P, v))                              the same multiply-add with the addend last
    sums of rows    = exact integer sums with one round-half-to-even scaling (sf's wide-mode sum and POP)

Codes are signed integers in [-32767, 32767] (value = code / 2^15).  Everything below the ``softmax`` section is
a verbatim port of ``sf_intops_ref.py`` (kept as ``tests/data/sf_intops_ref_v0.py``; the tests hold every op to
it bit for bit): the same sequences, constants and tables.

``softmax_row`` here is the fixed sequence (proposed to Alosh, 2026-11-18); his original is ``softmax_row_v0``:

1. **Sub-ulp mass in the sum.**  A key whose exp(d 2^k) is below one ulp gives e_i = 0, so its probability never
   reached the sum L and every P came out too large (the causal row in
   ``tests/data/softmax_causal_row_example.json``: P_max +0.4%, P sums to 0.9863).  A second exp at
   d + 15 ln2 / 2^k gives those keys 15 more fraction bits; they go into a third MX slot popped 15 steps lower.
2. **q below 2^14.**  When the coarse POP rounded up to a power of two, normalising could leave q just below 2^14,
   outside ``recip_norm``'s domain (it saturates).  One more POP one step higher puts it back in [2^14, 2^15).
3. **P at the row's own exponent.**  The last step shifted t = e_i / (2q) right by a - e - 1 to reach scale 1,
   so a long, flat row lost up to log2(n) bits and many P rounded to 0.  P is returned as (t, -(a - e - 1)): one
   power-of-two exponent per row, like every sf activation; the P @ V matmul takes it as the row's exponent.

Accuracy on P @ V (random V): 4.2e-3 -> 9.7e-4 on the causal row above, 2.7e-2 -> 3.3e-4 on a flat 4,096-key row.
"""
from __future__ import annotations

import numpy as np

MAXC = 32767
ONE = 32767          # 1 - 2^-15, the largest SF16 magnitude
HALF = 16384

# ---------------------------------------------------------------- primitives (numpy int64 or Python int)


def _np(a):
    return np.asarray(a, dtype=np.int64)


def mul(a, b):
    a, b = _np(a), _np(b)
    p = (np.abs(a) * np.abs(b)) >> 15
    return np.where((a < 0) ^ (b < 0), -p, p)


def add(a, b):
    return np.clip(_np(a) + _np(b), -MAXC, MAXC)


def fma(acc, a, b):
    return add(acc, mul(a, b))


def muladd(P, v, c):
    return add(c, mul(P, v))


def relu(a):
    return np.maximum(_np(a), 0)


def abs_sum(a, b):
    return np.abs(add(a, b))


def neg(a):
    return -_np(a)


# ---------------------------------------------------------------- coefficients (frozen; tune_coefs.py made them)

# exp(d) on d in (-1, 0], Horner from the top: P = C4; P = P*d + C3; ...; P = P*d + C0, then k squarings
# P = P*P + o_j with o_j in {0, 1} (a 1-ulp offset that cancels the truncation bias). One table per k; sf picks k.
EXP_TABLE = {   # tune_exp.py: min max|err| + 20 |mean err| over every d code
    0: ((32767, 32750, 16241, 5042, 839), ()),
    1: ((32767, 32749, 16241, 5039, 835), (0,)),
    2: ((32767, 32750, 16240, 5042, 843), (1, 0)),
    3: ((32767, 32750, 16241, 5042, 835), (1, 1, 0)),
    4: ((32767, 32750, 16257, 5074, 824), (1, 1, 0, 0)),
    5: ((32767, 32734, 16081, 5203, 643), (1, 0, 1, 0, 0)),
    6: ((32767, 32734, 16033, 5382, 691), (1, 0, 0, 1, 1, 0)),
}
# h(E) = (1 - E) / (2 (1 + E)) on E in [0, 1], degree 6
SIG_C = (16383, -32742, 32322, -29723, 22130, -10791, 2421)
# Newton seeds: 1/(2q) on [1/2, 1): y0 = R_A + q * R_B (R_B < 0); 1/(2 sqrt q) on [1/4, 1): y0 = S_A + q * S_B
R_A, R_B = 23362, -15676
S_A, S_B = 19098, -11469


def exp_core(d, k: int, coefs=None, offs=None):
    """exp(d * 2^k) for d code in (-1, 0] (positive d saturates toward 1). Output code, scale 2^0."""
    tc, to = EXP_TABLE[k]
    coefs = tc if coefs is None else coefs
    offs = to if offs is None else offs
    d = _np(d)
    P = np.full_like(d, coefs[-1])
    for c in coefs[-2::-1]:
        P = muladd(P, d, c)
    for o in offs[:k]:
        P = muladd(P, P, o)
    for _ in range(k - len(offs)):
        P = muladd(P, P, 0)
    return P


def sig_half(E, coefs=SIG_C):
    """(sigmoid(t) - 1/2) for E = exp(-t) code in [0, 2^15). Output code in [0, 2^14]."""
    E = _np(E)
    P = np.full_like(E, coefs[-1])
    for c in coefs[-2::-1]:
        P = muladd(P, E, c)
    return P


# GELU (tanh form, GPT-2's gelu_new): x * sigmoid(g), g = 1.5957691 x + 0.0713548 x^3.
GELU_A = 1.5957691216057308
GELU_B = 0.0713548162726009


def _gelu_consts(ex: int, k: int, cubic: bool = True):
    """Codes A' = A 2^(ex-k-j) and B' = B 2^(3ex-k-j-3c), with j doublings of the whole g and c doublings of x on the
    cubic path, so that A', B' < 1:  g / 2^k = 2^j (A' x + B' (2^c x)^3), x the code in (-1, 1)."""
    A = (GELU_A if cubic else 1.0) * 2.0 ** (ex - k)
    j = 0
    while A >= 1:
        A /= 2
        j += 1
    B = GELU_B * 2.0 ** (3 * ex - k - j) if cubic else 0.0
    c = 0
    while B >= 1:
        B /= 8
        c += 1
    return int(round(A * 32768)), int(round(B * 32768)), j, c


# squarings k per input scale 2^ex (gelu_search.py: exhaustive, min max error over every code)
GELU_K = {0: 2, 1: 2, 2: 4, 3: 4, 4: 4, 5: 4, 6: 4}
SILU_K = {0: 1, 1: 2, 2: 3, 3: 3, 4: 4, 5: 4, 6: 4}


def gelu_k(ex: int) -> int:
    return GELU_K[ex]


def _xsig(x, ex: int, k: int, cubic: bool):
    """x sigmoid(g(x)), g = A x + B x^3 (GELU) or x (SiLU); x = code 2^(ex-15), same scale out."""
    x = _np(x)
    A, B, j, b = _gelu_consts(ex, k, cubic)
    if cubic:
        xc = x
        for _ in range(b):
            xc = add(xc, xc)          # 2^c x, saturating (benign: then |g| / 2^k >= 1 and exp(-|g|) = 0)
        t = muladd(xc, xc, 0)           # (2^c x)^2
        t = muladd(t, B, 0)             # B' (2^c x)^2
        t = fma(0, t, xc)             # B' (2^c x)^3 (signed)
        t = fma(t, A, x)              # A' x + B' (2^c x)^3
    else:
        t = muladd(x, A, 0)             # A' x
    for _ in range(j):
        t = add(t, t)                 # g / 2^k, saturating (benign)
    u = abs_sum(t, 0)                 # |g| / 2^k
    E = exp_core(neg(u), k)           # exp(-|g|)
    h = sig_half(E)                   # sigmoid(|g|) - 1/2
    a = abs_sum(x, 0)                 # |x|
    y = muladd(x, HALF, 0)              # x / 2 (truncated)
    return fma(y, a, h)               # x/2 + |x| (sigmoid(|g|) - 1/2) = x sigmoid(g)


def gelu(x, ex: int, k: int = None):
    """GPT-2's tanh GELU of x = code * 2^(ex - 15), output at the same scale. Bit-exact definition."""
    return _xsig(x, ex, GELU_K[ex] if k is None else k, True)


def silu(x, ex: int, k: int = None):
    """SiLU = x sigmoid(x) of x = code * 2^(ex - 15), same scale out."""
    return _xsig(x, ex, SILU_K[ex] if k is None else k, False)


# ---------------------------------------------------------------- per-row scalars


def normalize(L):
    """q = L * 2^e in [2^14, 2^15) for 0 < L < 2^15, by the doubling loop (one CMP/BR per step; uniform per row)."""
    q, e = int(L), 0
    assert 0 < q <= MAXC
    while q < HALF:
        q = q + q
        e += 1
    return q, e


def recip_norm(q, ra=None, rb=None):
    """y ~ 1/(2 q) as a code (2^29 / q), for q in [2^14, 2^15). Seed 2 (R_A + R_B q), then two Newton steps."""
    ra = R_A if ra is None else ra
    rb = R_B if rb is None else rb
    q = _np(q)
    nq = neg(q)
    y = muladd(q, rb, ra)               # R_A + R_B q  (R_B < 0)
    y = add(y, y)                     # seed ~ 1/(2q), saturating at q = 1/2
    for _ in range(2):
        e = fma(HALF, nq, y)          # 1/2 - q y
        e = add(e, e)                 # 1 - 2 q y
        y = fma(y, y, e)              # y + y (1 - 2 q y)
    return y


QUARTER = 8192


def rsqrt_norm(q, sa=None, sb=None):
    """y ~ 1/(2 sqrt(q)) as a code (2^14 sqrt(2^15 / q)), for q in [2^13, 2^15). Seed 2 (S_A + S_B q), three steps."""
    sa = S_A if sa is None else sa
    sb = S_B if sb is None else sb
    q = _np(q)
    nq = neg(q)
    y = muladd(q, sb, sa)
    y = add(y, y)
    for _ in range(3):
        y2 = muladd(y, y, 0)            # y^2
        e = fma(QUARTER, nq, y2)      # 1/4 - q y^2
        e = add(e, e)                 # (1 - 4 q y^2) / 2
        y = fma(y, y, e)              # y + y (1 - 4 q y^2) / 2
    return y


# ---------------------------------------------------------------- exact row sums (wide-mode sum and POP)

def _word(c):
    c = int(c)
    return (0x8000 | -c) if c < 0 else c


def _rne_shift(S: int, m: int) -> int:
    """round(S * 2^m) half to even, S an integer."""
    if m >= 0:
        return S << m
    d = 1 << (-m)
    q, r = divmod(S, d)
    if 2 * r > d or (2 * r == d and q % 2):
        q += 1
    return q


def mx_sum_pop(v, shift: int, bias_code: int = 0, wshift: int = 0) -> int:
    """The wide-mode sum of v scaled by 2^-(1+wshift) (wshift <= 6: every product exact), summed exactly, then one
    rounding half to even at POP's shift, saturating. bias_code: one extra term.
    Returns the signed code round(sum(v) * 2^(shift - 1 - wshift)) (codes in units of 2^-15)."""
    assert 0 <= wshift <= 6
    S = int(np.sum(_np(v))) + int(bias_code)
    assert abs(S) << (6 - wshift) < (1 << 31), "accumulator would saturate: raise wshift"
    q = _rne_shift(S, shift - 1 - wshift)
    return int(max(-MAXC, min(MAXC, q)))


def mx_normalize(v, s0: int, bias_code: int = 0, ind=None, wshift: int = 0):
    """(q, e): POP the sum at s0 (no saturation by construction), find e with normalize() on that coarse code, then
    POP the same accumulator again at s0 + e (POP does not clear it), so q keeps the accumulator's full precision.
    q = round(sum(v)/2^(1+wshift) * 2^(s0 + e)), normally in [2^14, 2^15) (2^14 - 1 or 32767 at a rounding edge).
    ind: optional 0/1 codes accumulated in a second slot (same weights) and popped one step lower, so q gains
    sum(ind) / 2 ulps: the truncation-bias correction (see softmax_row)."""
    def pq(shift):
        c = mx_sum_pop(v, shift, bias_code, wshift)
        if ind is not None:
            c = c + mx_sum_pop(ind, shift - 1, 0, wshift)
        return c
    c = pq(s0)
    if c <= 0:
        return 0, 0
    _, e = normalize(min(c, MAXC))
    return pq(s0 + e), e


def shr_round(t, j: int):
    """Non-negative t: (t + 2^(j-1)) >> j for j >= 1 ; doubling for j < 0."""
    t = _np(t)
    if j <= 0:
        return shr(t, j)
    return (t + (1 << (j - 1))) >> j


# ---------------------------------------------------------------- composed rows


def row_max(s):
    """Exact max of SF16 codes ."""
    return int(np.max(_np(s)))


def shr(t, j: int):
    """Magnitude shift right by j (truncating) for j >= 0; saturating doubling for j < 0."""
    t = _np(t)
    if j >= 0:
        return np.sign(t) * (np.abs(t) >> j)
    for _ in range(-j):
        t = add(t, t)
    return t


def softmax_row_v0(s, k: int, debias: bool = True):
    """Softmax of one row of scores s (codes at scale 2^k, k = the scores' POP exponent). P codes at scale 1.

    m   = max_i s_i                                  (exact)
    e_i = exp_core(add(s_i, -m), k)                  (in (0, 1))
    ind_i = e_i - relu(e_i - 1)                      (1 where e_i is not 0: the e_i that carry a truncation bias)
    q, e = mx_normalize(e, 1 - a, ind), a = ceil(log2 n)   (MX all-reduce of sum e_i + sum ind_i / 2: slot 0 holds
                                                     e, slot 1 holds ind; two POPs per normalisation step, added;
                                                     q = L 2^(e - a) in ~[1/2, 1))
    y   = recip_norm(q)                              (~ 1/(2q))
    P_i = shr_round(mul(e_i, y), a - e - 1)         (P_i = e_i / L, rounded to nearest by an added 2^(j-1))
    """
    s = _np(s)
    n = len(s)
    m = row_max(s)
    ei = exp_core(add(s, -m), k)
    a = max(1, int(np.ceil(np.log2(n))))
    ind = (ei - relu(ei - 1)) if debias else None       # 1 where e_i >= 1 ulp: 
    ws = max(0, a - 10)                                 # weights 2^-(1+ws): no accumulator saturation up to n = 2^16
    q, e = mx_normalize(ei, 1 + ws - a, 0, ind, ws)
    if q <= 0:
        return np.zeros_like(s)
    y = recip_norm(q)
    t = mul(ei, y)                                   # e_i / (2q) <= 1
    return (shr_round if debias else shr)(t, a - e - 1)   # P_i = e_i / L = (e_i / 2q) 2^(e + 1 - a)


def _mean(v, n: int) -> int:
    """mean of codes v over n: POP(sum / 2, 1 - a) = sum / 2^a, then x 2^a / n (two-step when n is not 2^a)."""
    a = int(np.ceil(np.log2(n)))
    ws = max(0, a - 10)
    p = mx_sum_pop(v, 1 + ws - a, 0, ws)
    if (1 << a) == n:
        return p
    c = int(round(2.0 ** (a - 1) / n * 32768))        # 2^(a-1)/n in [1/2, 1)
    t = int(mul(p, c))
    return int(add(t, t))


def shr_signed(t, j: int):
    """Sign-magnitude shift: j >= 0 is mul(t, 2^(15-j)) (FMA by a constant: truncation toward zero, j <= 15);
    j < 0 is -j saturating doublings ."""
    t = _np(t)
    if j >= 0:
        return mul(t, 1 << (15 - j)) if j <= 15 else np.zeros_like(t)
    for _ in range(-j):
        t = add(t, t)
    return t


def norm_row(x, gamma, beta, eo: int, ex: int, eg: int, eb: int, eps: float = 1e-5, rms: bool = False):
    """LayerNorm (rms=False) or RMSNorm (rms=True, beta ignored) of one row; a deterministic integer definition.

    x codes at scale 2^ex, gamma at 2^eg, beta at 2^eb; output codes at scale 2^eo (sf chooses eo, as for POP).
      mu    = _mean(x)  (MX sum, one POP)                       z_i = add(x_i, -mu)   (RMSNorm: z = x, no beta)
      sq_i  = mul(z_i, z_i)                                      (FMA, truncated)
      S     = sum sq_i + n/2 + eps n 2^(15 - 2 ex)  (MX; the n/2 lane cancels the squares' truncation bias)
      q, e  = S / 2^a normalised by two POPs (mx_normalize), then x 2^a/n for n not a power of two, e made even
              by a third POP one step lower; q in [2^13, 2^15), v = q 2^-e
      y     = rsqrt_norm(q) ~ 1/(2 sqrt q)                       1/sqrt(v) = 2 y 2^(e/2)
      out_i = add(shr_signed(mul(mul(z_i, y), gamma_i), eo - (e/2 + 1 + eg)), shr_signed(beta_i, eo - eb))
    """
    x = _np(x)
    n = len(x)
    a = int(np.ceil(np.log2(n)))
    z = x if rms else add(x, -_mean(x, n))
    sq = mul(z, z)
    bias = int(round(n / 2 + eps * n * 2.0 ** (15 - 2 * ex)))
    lanes = []
    while bias > 0:                                   # one lane per 32767 of bias (a single MAC holds 8 lanes)
        lanes.append(min(bias, MAXC))
        bias -= lanes[-1]
    vals = list(sq) + lanes
    ws = max(0, a - 10)
    q, e = mx_normalize(vals, 1 + ws - a, 0, None, ws)
    if q <= 0:
        return np.zeros_like(x), 0
    if (1 << a) != n:                                 # x 2^a / n = 2 c, c in [1/2, 1)
        c = int(round(2.0 ** (a - 1) / n * 32768))
        q = int(mul(q, c))                            # q c in [1/4, 1): v = q 2^(1-e)
        e -= 1
        while q < HALF:                               # at most one step: uniform branch
            q, e = q + q, e + 1
    if e % 2:                                         # make e even: q in [2^13, 2^14)
        q, e = q >> 1, e - 1
    y = rsqrt_norm(q)
    t = mul(mul(z, y), _np(gamma))
    out = shr_signed(t, eo - (e // 2 + 1 + eg))
    if not rms:
        out = add(out, shr_signed(_np(beta), eo - eb))
    return out


# ---------------------------------------------------------------- the fixed softmax (sf's definition)

LN2_15 = 15 * np.log(2)


def lo_offset(k: int) -> int:
    """15 ln2 / 2^k as a score code: d + lo_offset(k) multiplies exp(d 2^k) by 2^15."""
    return int(round(LN2_15 / 2 ** k * 32768))


def softmax_row(s, k: int, visible=None):
    """Softmax of one row of scores s (codes at scale 2^k).  Returns (P codes, ep): P_i = code_i 2^(ep - 15).

    The v0 sequence (``softmax_row_v0``) with three changes (see the module notes): the e_i the main exp flushes
    are summed with 15 more fraction bits (a second exp at d + 15 ln2 / 2^k, a third MX slot popped 15 lower);
    q below 2^14 after normalising is popped once more one step higher; P keeps the row's own exponent instead of
    being shifted right to scale 1.  ``visible``: optional bool mask, False for masked keys (P = 0, not summed)."""
    s = _np(s)
    n = len(s)
    vis = np.ones(n, bool) if visible is None else np.asarray(visible, bool)
    if not vis.any():
        return np.zeros_like(s), 0
    m = int(np.max(s[vis]))
    d = add(s, -m)
    ei = np.where(vis, exp_core(d, k), 0)
    ind = ei - relu(ei - 1)
    lo = np.where(vis & (ei == 0), exp_core(add(d, lo_offset(k)), k), 0)
    a = max(1, int(np.ceil(np.log2(n))))
    ws = max(0, a - 10)

    def pq(shift):
        return (mx_sum_pop(ei, shift, 0, ws) + mx_sum_pop(ind, shift - 1, 0, ws)
                + mx_sum_pop(lo, shift - 15, 0, ws))

    s0 = 1 + ws - a
    c = pq(s0)
    if c <= 0:
        return np.zeros_like(s), 0
    _, e = normalize(min(c, MAXC))
    q = pq(s0 + e)
    if q < HALF:
        q, e = pq(s0 + e + 1), e + 1
    y = recip_norm(q)
    return mul(ei, y), e + 1 - a


def _rne(S, m):
    """round(S * 2^m) half to even, elementwise, S int64 and m int arrays (m may differ per row)."""
    S = np.asarray(S, np.int64)
    m = np.broadcast_to(np.asarray(m, np.int64), S.shape)
    up = np.left_shift(S, np.maximum(m, 0))
    d = np.left_shift(np.int64(1), np.minimum(np.maximum(-m, 0), 62))
    q, r = np.divmod(S, d)
    q = q + ((2 * r > d) | ((2 * r == d) & (q % 2 == 1)))
    return np.where(m >= 0, up, q)


def softmax(s, k, visible=None):
    """``softmax_row`` over the last axis of a (rows, n) code array, vectorized, same bits row by row.
    ``k``: one int or one per row.  Returns (P codes (rows, n), ep (rows,))."""
    s = _np(s)
    rows, n = s.shape
    kk = np.broadcast_to(np.asarray(k, np.int64), (rows,))
    vis = np.ones(s.shape, bool) if visible is None else np.broadcast_to(np.asarray(visible, bool), s.shape)
    any_vis = vis.any(axis=-1)
    m = np.where(vis, s, -MAXC - 1).max(axis=-1)
    d = add(s, -m[:, None])
    ei = np.zeros_like(s)
    lo = np.zeros_like(s)
    for kv in np.unique(kk):
        r = kk == kv
        ei[r] = exp_core(d[r], int(kv))
        lo[r] = exp_core(add(d[r], lo_offset(int(kv))), int(kv))
    ei = np.where(vis, ei, 0)
    lo = np.where(vis & (ei == 0), lo, 0)
    ind = ei - relu(ei - 1)
    a = max(1, int(np.ceil(np.log2(n))))
    ws = max(0, a - 10)
    S, Si, Sl = ei.sum(-1), ind.sum(-1), lo.sum(-1)

    def pq(shift):
        clip = lambda x: np.clip(x, -MAXC, MAXC)                     # noqa: E731  (each POP saturates)
        return clip(_rne(S, shift - 1 - ws)) + clip(_rne(Si, shift - 2 - ws)) + clip(_rne(Sl, shift - 16 - ws))

    s0 = 1 + ws - a
    c = pq(np.full(rows, s0))
    e = np.zeros(rows, np.int64)
    qq = np.minimum(c, MAXC)
    for _ in range(16):                                             # normalize(): the doubling loop, per row
        low = (qq < HALF) & (qq > 0)
        qq = np.where(low, qq + qq, qq)
        e = e + low
    q = pq(s0 + e)
    edge = (q < HALF) & (c > 0)
    q = np.where(edge, pq(s0 + e + 1), q)
    e = e + edge
    ok = (c > 0) & any_vis
    y = recip_norm(np.where(ok, q, HALF))
    t = np.where(ok[:, None], mul(ei, y[:, None]), 0)
    return t, np.where(ok, e + 1 - a, 0)


# ---------------------------------------------------------------- float rows <-> codes (for models on the host)

def row_exponent(x: np.ndarray, lo: int = 0, hi: int = 6) -> np.ndarray:
    """One power-of-two exponent per row (last axis) from max |x|, clamped to [lo, hi]: the GELU / SiLU tables
    exist for exponents 0..6 (Alosh's sf_intops_torch convention)."""
    mx = np.abs(x).max(axis=-1, keepdims=True)
    e = np.ceil(np.log2(np.maximum(mx, 1e-30) * (1 + 2 ** -14)))
    return np.clip(np.clip(e, -6, 6), lo, hi).astype(np.int64)


def to_codes(x, e):
    return np.clip(np.round(np.asarray(x, np.float64) / np.exp2(e) * 32768), -MAXC, MAXC).astype(np.int64)


def xsig_rows(x: np.ndarray, cubic: bool) -> np.ndarray:
    """GELU (tanh form, ``cubic``) or SiLU of float rows through the integer definition: each row quantized at its
    own exponent (0..6), the op run on the codes, dequantized at the same exponent."""
    x = np.asarray(x, np.float64)
    ex = row_exponent(x)
    out = np.empty_like(x)
    flat_e = ex[..., 0]
    for e in np.unique(flat_e):
        r = flat_e == e
        c = to_codes(x[r], int(e))
        out[r] = (gelu if cubic else silu)(c, int(e)) / 32768 * 2.0 ** int(e)
    return out


MASKED = -1e4


def softmax_rows(x: np.ndarray, k_min: int = 4, k_max: int = 5) -> np.ndarray:
    """Softmax of float score rows (last axis) through the integer definition.

    The scores enter with their row's max already subtracted (exact: softmax does not change), d = s - max, in
    [-2^k, 0] at scale 2^k: k = 4, or 5 when some visible key is more than 16 below the max.  A key more than 32
    below gets exp(d) < 2^-46, below anything the sum keeps (2^-30), so saturating it at -2^k changes nothing.

    Alosh's v0 convention quantized the raw scores (k = 4, redone at 5, 6): Qwen2.5-0.5B has attention scores up
    to 1728 (its first-token sink), so 6% of its rows saturate at 64 and keys above 64 collapse onto one value;
    perplexity doubles (22.4 to 45.3).  Taking the max first needs one max reduction over the wide scores
    before narrowing them (``raw=True`` keeps the v0 convention).

    Masked keys get P = 0 and are not summed: -inf, and any score at or below ``MASKED`` (-10^4), as models mask
    with the dtype's lowest finite value (``torch.finfo(dtype).min``) as well as -inf."""
    return _softmax_rows(x, k_min, k_max, raw=False)


def softmax_rows_raw(x: np.ndarray, k_min: int = 4, k_max: int = 6) -> np.ndarray:
    """Alosh's v0 convention: raw scores at k = 4, redone at 5, 6 if max |score| would saturate."""
    return _softmax_rows(x, k_min, k_max, raw=True)


def _softmax_rows(x, k_min, k_max, raw):
    x = np.asarray(x, np.float64)
    shp = x.shape
    x2 = x.reshape(-1, shp[-1])
    vis = np.isfinite(x2) & (x2 > MASKED)
    if raw:
        x2 = np.where(vis, x2, 0.0)
        amax = np.abs(x2).max(axis=-1)
    else:
        m = np.where(vis, x2, -np.inf).max(axis=-1)
        x2 = np.where(vis, x2 - np.where(np.isfinite(m), m, 0.0)[:, None], 0.0)
        amax = (-x2).max(axis=-1)
    k = np.full(len(x2), k_min, np.int64)
    for kv in range(k_min, k_max):
        k = np.where((k == kv) & (amax * 32768 / 2.0 ** kv >= MAXC + 0.5), kv + 1, k)
    codes = np.clip(np.round(x2 / np.exp2(k)[:, None] * 32768), -MAXC, MAXC).astype(np.int64)
    t, ep = softmax(codes, k, vis)
    p = t / 32768 * np.exp2(ep)[:, None]
    return np.where(vis, p, 0.0).reshape(shp)
