"""A compiled MX accumulator for the host (speed only; the bits are the model's).

``_linalg.matmul_wide_acc`` is the reference: NumPy, the SF16 block's
partial-product array walked bit by bit.  It is exact and slow (GPT-2 small
takes about 10 minutes per 128 tokens).  This module computes the same
integers in C, with the rows split across threads (ctypes releases the GIL).

The SF16 product in closed form.  The block adds the 15 partial-product rows
``bit_i(a) * (w << i)`` with the columns below ``tk`` dropped, plus ``comp``,
then cuts to F fraction bits.  Dropping those columns from row i removes
``(w << i) mod 2^tk``, which is 0 for i >= tk, so

    t = (comp + a*w - C[a mod 2^tk][w mod 2^tk]) >> (30 - F)
    C[x][y] = sum_{i < tk, bit_i(x)} (y << i) mod 2^tk

a table of 2^tk x 2^tk entries (64 x 64 on v4.9.2, 512 x 512 on v4.8.1).
The SF8 block is exact: t = a*w.  Accumulation, the clip at the unit's
limit once per group of 8 terms, and the clip mask are as in the reference.

``tests/test_fastmx.py`` checks this against the reference on every width,
both chips, random and edge codes and the golden files.  If no C compiler is
found, or the build fails, ``available()`` is False and sf uses the
reference, with the same bits.  ``SF_FAST_MX=0`` turns it off.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

_SRC = r"""
#include <stdint.h>
/* rows [r0, r1) of acc = A (M x K) @ B, with B given transposed (N x K).
   am / bm: magnitudes as the block reads them; an / bn: 0 or -1 (negative).
   Products are at most 2^30 - 1, so they fit int32; totals are int64. */
void mx_acc(int64_t r0, int64_t r1, int64_t K, int64_t N,
            const int32_t *am, const int32_t *an, const int32_t *bm, const int32_t *bn,
            int32_t blk16, int32_t comp, int32_t sh, int32_t tk, const int32_t *C,
            int64_t lo, int64_t hi, int64_t *acc, uint8_t *clip) {
    const int32_t msk = (1 << tk) - 1;
    const int32_t cut = (int32_t)((1u << 30) - 1u);
    for (int64_t m = r0; m < r1; m++) {
        const int32_t *ar = am + m * K, *as = an + m * K;
        for (int64_t n = 0; n < N; n++) {
            const int32_t *br = bm + n * K, *bs = bn + n * K;
            int64_t s = 0;
            uint8_t c = 0;
            for (int64_t k0 = 0; k0 < K; k0 += 8) {
                const int64_t k1 = k0 + 8 < K ? k0 + 8 : K;
                int64_t g = s;
                if (blk16) {
                    for (int64_t k = k0; k < k1; k++) {
                        const int32_t a = ar[k], w = br[k];
                        const int32_t t = ((comp + a * w - C[((a & msk) << tk) | (w & msk)]) & cut) >> sh;
                        const int32_t neg = as[k] ^ bs[k];          /* 0 or -1 */
                        g += (t ^ neg) - neg;
                    }
                } else {
                    for (int64_t k = k0; k < k1; k++) {
                        const int32_t t = ar[k] * br[k];
                        const int32_t neg = as[k] ^ bs[k];
                        g += (t ^ neg) - neg;
                    }
                }
                if (g < lo) { g = lo; c = 1; }
                else if (g > hi) { g = hi; c = 1; }
                s = g;
            }
            acc[m * N + n] = s;
            clip[m * N + n] = c;
        }
    }
}
"""

_FLAGS = ["-O3", "-shared", "-fPIC", "-std=c99"]
_lock = threading.Lock()
_lib = None
_failed = False
_tables: dict[int, np.ndarray] = {}


def _compiler():
    for cc in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cc and shutil.which(cc):
            return cc
    return None


def _load():
    global _lib, _failed
    if _lib is not None or _failed:
        return _lib
    with _lock:
        if _lib is not None or _failed:
            return _lib
        try:
            if os.environ.get("SF_FAST_MX", "1") == "0":
                raise OSError("disabled by SF_FAST_MX=0")
            cc = _compiler()
            if cc is None:
                raise OSError("no C compiler")
            banner = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
            key = hashlib.sha256("\0".join([_SRC, cc, *banner, *_FLAGS]).encode()).hexdigest()[:20]
            d = os.path.join(tempfile.gettempdir(), f"sf-fastmx-{os.getuid() if hasattr(os, 'getuid') else 0}")
            os.makedirs(d, exist_ok=True)
            so = os.path.join(d, f"mx_{key}.so")
            if not os.path.exists(so):
                fd, tmp = tempfile.mkstemp(suffix=".so", prefix=f".{key}-", dir=d)
                os.close(fd)
                src = tmp[:-3] + ".c"
                try:
                    with open(src, "w") as f:
                        f.write(_SRC)
                    p = subprocess.run([cc, *_FLAGS, src, "-o", tmp], capture_output=True, text=True)
                    if p.returncode != 0:
                        raise OSError(p.stderr)
                    os.replace(tmp, so)                        # atomic: no half-written library is loaded
                finally:
                    for f_ in (tmp, src):
                        try:
                            os.remove(f_)
                        except FileNotFoundError:
                            pass
            lib = ctypes.CDLL(so)
            i64, i32, p = ctypes.c_int64, ctypes.c_int32, ctypes.c_void_p
            lib.mx_acc.argtypes = [i64, i64, i64, i64, p, p, p, p, i32, i32, i32, i32, p, i64, i64, p, p]
            lib.mx_acc.restype = None
            _lib = lib
        except Exception:                                      # noqa: BLE001  (any failure: use the reference)
            _failed = True
        return _lib


def available() -> bool:
    return _load() is not None


def correction_table(tk: int) -> np.ndarray:
    """C[x][y] = sum over i < tk with bit i of x set of (y << i) mod 2^tk."""
    t = _tables.get(tk)
    if t is None:
        x = np.arange(1 << tk, dtype=np.int64)[:, None]
        y = np.arange(1 << tk, dtype=np.int64)[None, :]
        t = np.zeros((1 << tk, 1 << tk), dtype=np.int64)
        for i in range(tk):
            t += ((x >> i) & 1) * ((y << i) & ((1 << tk) - 1))
        t = np.ascontiguousarray(t.astype(np.int32))
        _tables[tk] = t
    return t


def _threads() -> int:
    return max(1, min(8, os.cpu_count() or 1))


def acc2d(am, an, bm, bn, blk16: bool, unit: dict, lo: int, hi: int):
    """(M, K) @ (K, N) magnitudes and sign masks -> (acc int64, clip bool), 2-D only."""
    lib = _load()
    M, K = am.shape
    N = bm.shape[1]
    A = np.ascontiguousarray(am, dtype=np.int32)
    As = -np.ascontiguousarray(an, dtype=np.int32)              # 0 or -1
    B = np.ascontiguousarray(bm.T, dtype=np.int32)              # (N, K): each column contiguous
    Bs = -np.ascontiguousarray(bn.T, dtype=np.int32)
    acc = np.empty((M, N), dtype=np.int64)
    clip = np.empty((M, N), dtype=np.uint8)
    tk = int(unit["tk"]) if blk16 else 1
    C = correction_table(tk)
    comp = int(unit["comp"]) if blk16 else 0
    sh = 30 - int(unit["F"]) if blk16 else 0
    args = (K, N, A.ctypes.data, As.ctypes.data, B.ctypes.data, Bs.ctypes.data, int(blk16), comp, sh, tk,
            C.ctypes.data, lo, hi, acc.ctypes.data, clip.ctypes.data)
    nt = min(_threads(), M)
    if nt <= 1 or M * N * K < 1 << 18:
        lib.mx_acc(0, M, *args)
    else:
        cuts = np.linspace(0, M, nt + 1).astype(int)
        with ThreadPoolExecutor(nt) as ex:
            list(ex.map(lambda i: lib.mx_acc(int(cuts[i]), int(cuts[i + 1]), *args), range(nt)))
    return acc, clip.astype(bool)
