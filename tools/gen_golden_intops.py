"""Write spec/golden/intops/*.json: golden vectors for sf's integer exp, GELU, SiLU, reciprocal, rsqrt and softmax
(sf._intops, ported bit for bit from Alosh's sf_intops_ref.py).  ``--check`` compares instead of writing.

Codes are SF16 codes (value = code / 2^15, scaled by 2^ex where an op takes an exponent).  Softmax cases carry
both the v0 output (Alosh's original, P at scale 1) and the fixed one (P codes and the row's exponent ep).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from sf import _intops as I  # noqa: E402
from spec.ops import SPEC_VERSION  # noqa: E402

OUT = ROOT / "spec" / "golden" / "intops"
DEF = "sf._intops: Alosh's integer definitions (sf_intops_ref.py, 2026-11-18), ported bit for bit"


def _codes(rng, lo, hi, n, edges):
    c = set(int(x) for x in edges if lo <= x <= hi)
    c.update(int(x) for x in rng.integers(lo, hi + 1, n))
    return sorted(c)


def exp_cases(rng):
    out = []
    for k in range(7):
        d = _codes(rng, -32767, 0, 300, [-32767, -32766, -16384, -1, 0, -I.lo_offset(k), -2 * I.lo_offset(k)])
        out.append({"k": k, "d": d, "out": I.exp_core(np.array(d), k).tolist()})
    return out


def xsig_cases(rng, fn):
    out = []
    for ex in range(7):
        x = _codes(rng, -32767, 32767, 400, [-32767, -16384, -1, 0, 1, 16384, 32767])
        out.append({"ex": ex, "x": x, "out": fn(np.array(x), ex).tolist()})
    return out


def softmax_cases(rng):
    cases = []
    row = json.loads((ROOT / "tests" / "data" / "softmax_causal_row_example.json").read_text())
    rows = [("causal row from Alosh (942 visible keys)", np.array(row["scores"]), 4)]
    for n, sd in ((1, 1.0), (2, 1.0), (7, 0.5), (8, 2.0), (64, 1.0), (100, 3.0), (1024, 1.5), (4096, 0.5)):
        rows.append((f"random n={n} sd={sd}", rng.normal(0, sd, n), 4))
    rows.append(("wide row at k=6", rng.normal(0, 15, 256), 6))
    for what, x, k in rows:
        s = np.clip(np.round(x / 2 ** k * 32768), -32767, 32767).astype(np.int64)
        t, ep = I.softmax_row(s, k)
        cases.append({"what": what, "k": k, "s": s.tolist(), "v0": I.softmax_row_v0(s, k).tolist(),
                      "p": t.tolist(), "ep": int(ep)})
    s = np.clip(np.round(rng.normal(0, 1, 40) / 16 * 32768), -32767, 32767).astype(np.int64)
    vis = np.arange(40) < 23                                           # a causal row: 23 visible keys
    t, ep = I.softmax_row(s, 4, vis)
    cases.append({"what": "masked keys (visible = first 23)", "k": 4, "s": s.tolist(), "visible": vis.tolist(),
                  "p": t.tolist(), "ep": int(ep)})
    return cases


def build():
    rng = np.random.default_rng(20261118)
    head = {"spec_version": SPEC_VERSION, "definition": DEF,
            "format": "SF16 codes: signed ints in [-32767, 32767], value = code / 2^15 (times 2^ex or 2^ep)"}
    q_recip = list(range(I.HALF, I.MAXC + 1, 13)) + [I.HALF, I.MAXC]
    q_rsqrt = list(range(I.QUARTER, I.MAXC + 1, 23)) + [I.QUARTER, I.MAXC]
    return {
        "exp.json": {**head, "op": "exp_core(d, k) ~ exp(d 2^k) for d in (-1, 0]", "cases": exp_cases(rng)},
        "gelu.json": {**head, "op": "gelu(x, ex): GPT-2's tanh GELU of x = code 2^(ex-15)", "cases": xsig_cases(rng, I.gelu)},
        "silu.json": {**head, "op": "silu(x, ex): x sigmoid(x)", "cases": xsig_cases(rng, I.silu)},
        "recip.json": {**head, "op": "recip_norm(q) ~ 2^29 / q for q in [2^14, 2^15)",
                       "cases": [{"q": q_recip, "out": I.recip_norm(np.array(q_recip)).tolist()}]},
        "rsqrt.json": {**head, "op": "rsqrt_norm(q) ~ 2^14 sqrt(2^15 / q) for q in [2^13, 2^15)",
                       "cases": [{"q": q_rsqrt, "out": I.rsqrt_norm(np.array(q_rsqrt)).tolist()}]},
        "softmax.json": {**head, "op": "softmax_row(s, k): P codes at the row's exponent ep (fixed); v0 = Alosh's "
                                       "original at scale 1", "cases": softmax_cases(rng)},
    }


def main():
    check = "--check" in sys.argv
    files = build()
    bad = []
    for name, data in files.items():
        text = json.dumps(data, separators=(",", ":")) + "\n"
        p = OUT / name
        if check:
            if not p.exists() or p.read_text() != text:
                bad.append(name)
        else:
            OUT.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
            print(f"wrote {p.relative_to(ROOT)} ({len(text) // 1024} KB)")
    if check and bad:
        print("out of date:", ", ".join(bad))
        sys.exit(1)


if __name__ == "__main__":
    main()
