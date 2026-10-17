#!/usr/bin/env python3
"""Exhaustive check of every operand pair for one width (spec section 10).

    python tools/exhaustive.py --bits 16          # nightly: 4.3e9 pairs per op
    python tools/exhaustive.py --bits 10 --ops add div

Two independent formulations are compared on every pair:

* the vectorised reference (``sf._vectorized``), and
* an exact integer oracle written here from the spec's formulas
  (Python ints via NumPy object arrays would be too slow, so the oracle
  uses plain int64 arithmetic, which is exact for these magnitudes).

In addition, a random sample of each chunk is checked against the scalar
reference (``sf._reference``) and the Fraction oracle used by the test
suite, which ties the vectorised path back to the scalar semantics.

Exit status is nonzero on the first mismatch, with the offending codes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from fractions import Fraction

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import sf                                   # noqa: E402
from sf import _reference as ref            # noqa: E402
from sf import _vectorized as vec           # noqa: E402
from sf._format import FORMATS              # noqa: E402
from oracle import Fmt                      # noqa: E402


def oracle_int(op: str, fmt, ka, kb):
    """Spec section 3.1 formulas on int64 arrays, written independently."""
    M, S = fmt.max_code, fmt.scale
    if op in ("add", "sub"):
        exact = ka + kb if op == "add" else ka - kb
        over = np.abs(exact) > M
        return np.clip(exact, -M, M), over, np.zeros_like(over), np.zeros_like(over)
    if op == "mul":
        p = ka * kb
        q = np.sign(p) * (np.abs(p) // S)
        under = (p != 0) & (q == 0)
        return q, np.zeros_like(under), under, np.zeros_like(under)
    if op == "div":
        zero = kb == 0
        b = np.where(zero, 1, kb)
        n = ka * S
        q = np.sign(n) * np.sign(b) * (np.abs(n) // np.abs(b))
        over = ~zero & (ka != 0) & (np.abs(ka) >= np.abs(kb))
        sat = np.where(np.sign(ka) == np.sign(kb), M, -M)
        r = np.where(zero, np.sign(ka) * M, np.where(over, sat, q))
        return r, over, np.zeros_like(over), zero
    raise ValueError(op)


def check_scalar_sample(op: str, fmt, ka, kb, rk, n=64, rng=None):
    """Tie a few pairs back to the scalar reference and the Fraction oracle."""
    frac = Fmt(fmt.bits)
    idx = rng.integers(0, ka.size, n)
    for i in idx:
        a, b = int(ka[i]), int(kb[i])
        k_ref, ev_ref = getattr(ref, op)(fmt, a, b)
        v_or, ev_or = getattr(frac, op)(Fraction(a, fmt.scale), Fraction(b, fmt.scale))
        assert k_ref == int(rk[i]), (op, a, b, k_ref, int(rk[i]))
        assert Fraction(k_ref, fmt.scale) == v_or and ev_ref == ev_or, (op, a, b)


def run(bits: int, ops: list[str], chunk: int, seed: int) -> int:
    fmt = FORMATS[bits]
    M = fmt.max_code
    n = 2 * M + 1
    total = n * n
    rng = np.random.default_rng(seed)
    codes = np.arange(-M, M + 1, dtype=np.int64)
    for op in ops:
        t0 = time.time()
        done = 0
        rows_per_chunk = max(1, chunk // n)
        for start in range(0, n, rows_per_chunk):
            a_rows = codes[start:start + rows_per_chunk]
            ka = np.repeat(a_rows, n)
            kb = np.tile(codes, a_rows.size)
            rk, rev = vec.OPS[op](fmt, ka, kb)
            ok, over, under, zero = oracle_int(op, fmt, ka, kb)
            bad = np.nonzero(rk != ok)[0]
            if bad.size:
                i = bad[0]
                print(f"MISMATCH {op} sf{bits}: a={ka[i]} b={kb[i]} ref={rk[i]} oracle={ok[i]}")
                return 1
            for name, mask in (("overflow", over), ("underflow", under), ("divide", zero)):
                got = rev.get(name, np.zeros_like(mask))
                bad = np.nonzero(got != mask)[0]
                if bad.size:
                    i = bad[0]
                    print(f"EVENT MISMATCH {op} sf{bits} {name}: a={ka[i]} b={kb[i]} "
                          f"ref={bool(got[i])} oracle={bool(mask[i])}")
                    return 1
            check_scalar_sample(op, fmt, ka, kb, rk, rng=rng)
            done += ka.size
            if bits >= 14:
                el = time.time() - t0
                print(f"\r{op} sf{bits}: {done / total:6.1%}  {el:6.0f}s  "
                      f"eta {el / done * (total - done):6.0f}s", end="", flush=True)
        print(f"\r{op} sf{bits}: {total:,} pairs OK in {time.time() - t0:.1f}s" + " " * 20)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bits", type=int, default=16)
    p.add_argument("--ops", nargs="+", default=["add", "sub", "mul", "div"], choices=list(vec.OPS))
    p.add_argument("--chunk", type=int, default=1 << 24, help="pairs per chunk")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    sys.exit(run(a.bits, a.ops, a.chunk, a.seed))


if __name__ == "__main__":
    main()
