/* Reference C implementation of the __sf16_* builtins (spec section 11).
 *
 * Signed codes in int64_t, Q1.15: value = k / 32768, k in [-32767, 32767].
 * Semantics are exactly those of sf/_reference.py: casts round half to
 * even, device arithmetic truncates toward zero, add/sub saturate, div
 * overflows when |a| >= |b|, fma truncates the product before the add.
 * Each builtin takes an event counter block; a NULL ev means "do not count"
 * (a predicated-off lane).  This header is the host-side stand-in for the
 * builtins the custom Clang provides for Atreides; a backend that emits
 * calls to it must agree bit for bit with the interpreter.
 */
#ifndef SF16_BUILTINS_H
#define SF16_BUILTINS_H
#include <stdint.h>
#include <math.h>

#define SF16_SCALE 32768LL
#define SF16_MAX   32767LL

typedef struct { int64_t sat, under, div0, invalid; } sf16_events;

static inline int64_t sf16_trunc_div(int64_t num, int64_t den) {
    int64_t q = (num < 0 ? -num : num) / (den < 0 ? -den : den);
    return ((num < 0) != (den < 0)) ? -q : q;
}

static inline int64_t sf16_saturate(int64_t k, sf16_events *ev) {
    if (k > SF16_MAX)  { if (ev) ev->sat++; return SF16_MAX; }
    if (k < -SF16_MAX) { if (ev) ev->sat++; return -SF16_MAX; }
    return k;
}

static inline int64_t __sf16_add(int64_t a, int64_t b, sf16_events *ev) { return sf16_saturate(a + b, ev); }
static inline int64_t __sf16_sub(int64_t a, int64_t b, sf16_events *ev) { return sf16_saturate(a - b, ev); }
static inline int64_t __sf16_neg(int64_t a) { return -a; }
static inline int64_t __sf16_abs(int64_t a) { return a < 0 ? -a : a; }

static inline int64_t __sf16_mul(int64_t a, int64_t b, sf16_events *ev) {
    int64_t p = a * b;
    int64_t k = sf16_trunc_div(p, SF16_SCALE);
    if (k == 0 && p != 0 && ev) ev->under++;
    return k;
}

static inline int64_t __sf16_div(int64_t a, int64_t b, sf16_events *ev) {
    if (b == 0) { if (ev) ev->div0++; return a > 0 ? SF16_MAX : (a < 0 ? -SF16_MAX : 0); }
    if (a == 0) return 0;
    int64_t aa = a < 0 ? -a : a, bb = b < 0 ? -b : b;
    if (aa >= bb) { if (ev) ev->sat++; return ((a < 0) == (b < 0)) ? SF16_MAX : -SF16_MAX; }
    return sf16_trunc_div(a * SF16_SCALE, b);
}

static inline int64_t __sf16_fma(int64_t a, int64_t b, int64_t c, sf16_events *ev) {
    /* product truncated first (may underflow), then a saturating add; the
       event reported is the one the final value shows (overflow wins) */
    int64_t p = a * b;
    int64_t k = sf16_trunc_div(p, SF16_SCALE);
    int64_t s = k + c;
    if (s > SF16_MAX)  { if (ev) ev->sat++; return SF16_MAX; }
    if (s < -SF16_MAX) { if (ev) ev->sat++; return -SF16_MAX; }
    if (k == 0 && p != 0 && ev) ev->under++;
    return s;
}

static inline int64_t __sf16_sqrt(int64_t a, sf16_events *ev) {
    if (a < 0) { if (ev) ev->invalid++; return 0; }
    /* floor(sqrt(a * 32768)) exactly: float sqrt then fix up */
    int64_t x = a * SF16_SCALE;
    int64_t k = (int64_t)sqrt((double)x);
    while ((k + 1) * (k + 1) <= x) k++;
    while (k * k > x) k--;
    return k;
}

/* A reduction (sum, dot) reports each event at most once per output, as the
   Stage 1.1 built-ins do: fold a per-output block into the program's block. */
/* Wide accumulation on the v4.8.1 MX engine, SF16 block (see
   _linalg.matmul_wide_codes, written from the chip repo's test/v4/mxref.py).
   x[k * xs] and y[k * ys] for k < K.  Products: the 15-bit magnitudes with
   partial-product bits below column 9 dropped, plus 1024, cut to 19 fraction
   bits.  Chunks of 8 summed exactly, then a Q6.19 (25-bit) accumulator that
   saturates once per chunk.  POP: total * 2^shift rounded half to even to
   sf16 and saturated; never -0. */
static inline int64_t __sf16_dot_wide(const int64_t *x, int64_t xs, const int64_t *y, int64_t ys,
                                      int64_t K, int64_t shift, sf16_events *ev) {
    const int64_t hi = (1LL << 24) - 1, lo = -(1LL << 24);
    int64_t acc = 0; int clipped = 0;
    for (int64_t k0 = 0; k0 < K; k0 += 8) {
        int64_t s = 0;
        for (int64_t k = k0; k < k0 + 8 && k < K; k++) {
            int64_t a = x[k * xs], b = y[k * ys];
            int64_t am = a < 0 ? -a : a, bm = b < 0 ? -b : b;
            int64_t rs = 1024;
            for (int t = 0; t < 15; t++)
                if ((am >> t) & 1) rs += (bm << t) & ~(int64_t)511;
            rs &= (1LL << 30) - 1;
            int64_t p = rs >> 11;
            s += ((a < 0) != (b < 0)) ? -p : p;
        }
        acc += s;
        if (acc > hi) { acc = hi; clipped = 1; }
        if (acc < lo) { acc = lo; clipped = 1; }
    }
    int64_t e = shift - 19 + 15, mag = acc < 0 ? -acc : acc, q;
    if (e >= 0) q = mag << e;
    else {
        int64_t d = -e, fl = mag >> d, rem = mag - (fl << d), half = 1LL << (d - 1);
        q = fl + ((rem > half) || (rem == half && (fl & 1)));
    }
    int over = q > SF16_MAX;
    if (over) q = SF16_MAX;
    if (ev) { if (clipped || over) ev->sat++; if (acc != 0 && q == 0) ev->under++; }
    return acc < 0 ? -q : q;
}

static inline void sf16_fold(sf16_events *ev, const sf16_events *local) {
    if (!ev) return;
    if (local->sat) ev->sat++;
    if (local->under) ev->under++;
    if (local->div0) ev->div0++;
    if (local->invalid) ev->invalid++;
}

/* narrowing to fewer bits is not needed on the device (sf16 only) */
#endif
