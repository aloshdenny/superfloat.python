"""sf integer reference for the non-matmul inference ops: exp, softmax, GELU, SiLU, reciprocal, rsqrt, norms.

Every op is a fixed sequence of sf's own SF16 arithmetic, so it gives the same bits on any host and on the chip:

    mul(a, b)       = sign(a) sign(b) * floor(|a| |b| / 2^15)       sf16 multiply, truncated toward zero
    add(a, b)       = clamp(a + b, -32767, 32767)                    sf16 add, saturating
    fma(acc, a, b)  = add(acc, mul(a, b))
    muladd(P, v, c) = add(c, mul(P, v))                              the same multiply-add with the addend last
    abs_sum(a, b)   = |add(a, b)|
    sums of rows    = exact integer sums with one round-half-to-even scaling (sf's wide-mode sum and POP)

Codes are signed integers in [-32767, 32767] (value = code / 2^15). -0 reads as 0 and is never produced.

The ops (deterministic integer functions; Python ints or int64 numpy arrays):

    exp_core(d, k)          ~ exp(d * 2^k) * 2^15 for d in (-1, 0]  (Horner degree 4, then k squarings)
    sig_half(E)             ~ (1 - E) / (2 (1 + E)) * 2^15 = (sigmoid(t) - 1/2) * 2^15 with E = exp(-t)
    gelu(x, ex)             GPT-2's tanh GELU (gelu_new) of x = code * 2^(ex-15), same scale out
    silu(x, ex)             x * sigmoid(x), same scale
    recip_norm(q)           ~ 2^15 / (2 q) for q in [2^14, 2^15)           (Newton, 2 steps)
    rsqrt_norm(q)           ~ 2^15 / (2 sqrt(q)) for q in [2^13, 2^15)     (Newton, 3 steps)
    normalize(L)            (q, e) with q = L * 2^e in [2^14, 2^15)
    softmax_row(s, k)       one attention row, scores at scale 2^k; returns P codes (scale 1)
    layernorm_row(x, ...)   and rmsnorm_row: exact row sums, the rest in sf16 arithmetic (norms stay on the host
                            in float for now; these are here for reference)

Known issues to fix before this becomes sf's definition (see ANSWERS.md): softmax loses probability mass on long
rows whose probabilities fall below one ulp, and normalisation can leave q below 2^14 (outside recip_norm's domain).
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


def softmax_row(s, k: int, debias: bool = True):
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
