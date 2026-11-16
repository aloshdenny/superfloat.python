"""Golden vectors for wide mode beyond dot_wide's scalar shift (Alosh's item 9).

    python tools/gen_golden_wide.py           # write spec/golden/wide/dot_wide_shift.json and sft_wide.json
    python tools/gen_golden_wide.py --check   # exit 1 if the files differ from what this writes

Source: sf's host model of the MX engine, which matches the emulator's 240
dot_wide cases (scalar shift) and the 13 heuristic cases of
``sf_golden_v4.9.2.json`` bit for bit.  These files extend that to shift
arrays, every width, the full sft wide path (status included), attention
and one small model.  Next step: confirm them on the RTL.

Floats are written with Python's repr, which round-trips float64 exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sf import _linalg  # noqa: E402
from sf._format import FORMATS  # noqa: E402
from spec.ops import SPEC_VERSION  # noqa: E402

STATUS_FIELDS = ("saturated", "exact", "underflow", "input_underflow", "weight_underflow", "outputs",
                 "accumulate", "shift_min", "shift_max", "h", "redone", "flushed", "acc_clipped",
                 "guard_raised")


def _codes(rng, fmt, shape, kind):
    M = fmt.max_code
    if kind == "full":
        return rng.integers(-M, M + 1, shape)
    if kind == "small":
        return rng.integers(-max(1, M >> 6), max(1, M >> 6) + 1, shape)
    c = rng.integers(-M, M + 1, shape)
    c[rng.random(shape) < 0.6] = 0                                   # sparse
    return c


def dot_wide_shift():
    rng = np.random.default_rng(20261116)
    cases = []
    for mx in ("v4.9.2", "v4.8.1"):
        for bits in (4, 6, 8, 10, 12, 16):
            fmt = FORMATS[bits]
            for kind, (M, K, N) in (("full", (3, 9, 4)), ("small", (2, 33, 5)), ("sparse", (4, 64, 3))):
                a = _codes(rng, fmt, (M, K), kind)
                b = _codes(rng, fmt, (K, N), kind)
                for layout in ("token", "column", "grid", "scalar"):
                    if layout == "token":
                        s = rng.integers(-8, 3, (M, 1))
                    elif layout == "column":
                        s = rng.integers(-8, 3, (N,))
                    elif layout == "grid":
                        s = rng.integers(-32, 32, (M, N)) if kind == "full" else rng.integers(-9, 4, (M, N))
                    else:
                        s = int(rng.integers(-6, 2))
                    q, ev = _linalg.matmul_wide_codes(fmt, a, b, s, mx)
                    cases.append({"target": mx, "bits": bits, "kind": kind,
                                  "a": a.tolist(), "b": b.tolist(),
                                  "shift": s if isinstance(s, int) else s.tolist(),
                                  "out": q.tolist(),
                                  "overflow": np.asarray(ev["overflow"]).astype(int).tolist(),
                                  "underflow": np.asarray(ev["underflow"]).astype(int).tolist()})
    return {"op": "dot_wide", "spec_version": SPEC_VERSION, "what": "shift as an array",
            "source": "sf host model of the MX engine (matches the emulator's dot_wide and heuristic golden "
                      "cases bit for bit); to confirm on the RTL",
            "format": "a: (M, K) codes, b: (K, N) codes, shift: an int or an array broadcastable to (M, N), "
                      "every element -32..31. out = POP(a @ b, shift), overflow / underflow as 0/1 masks. "
                      "Widths sf4-sf8 use the SF8 block, sf9-sf16 the SF16 block.",
            "cases": cases}


def _status(st):
    return {f: getattr(st, f) for f in STATUS_FIELDS}


def sft_wide():
    import torch
    import sf.torch as sft
    from sf.torch import _core

    rng = np.random.default_rng(7)
    layers = []
    specs = [("sf16 plain", 16, (3, 64), (5, 64), 1.0, None),
             ("sf16 one outlier token", 16, (4, 96), (6, 96), 1.0, "outlier"),
             ("sf8 plain", 8, (3, 48), (4, 48), 1.0, None),
             ("sf12 sparse", 12, (3, 128), (4, 128), 0.5, "sparse"),
             ("sf16 K=1024 near 1 (guard raises)", 16, (1, 1024), (2, 1024), None, "const"),
             ("sf16 with bias", 16, (2, 32), (3, 32), 1.0, "bias")]
    for desc, bits, xs, ws, sc, extra in specs:
        if extra == "const":
            x, w = np.full(xs, 0.9), np.full(ws, 0.9)
        else:
            x, w = rng.normal(size=xs) * sc, rng.normal(size=ws) * 0.1
        if extra == "outlier":
            x[1] *= 40
        if extra == "sparse":
            x[rng.random(xs) < 0.8] = 0.0
        b = rng.normal(size=ws[0]) * 0.05 if extra == "bias" else None
        r = _core.wide_linear(x, w, b, bits)
        layers.append({"what": desc, "bits": bits, "x": x.tolist(), "w": w.tolist(),
                       "b": None if b is None else b.tolist(), "y": r.y.tolist(),
                       "status": {f: getattr(r, f) for f in STATUS_FIELDS}})

    attn = []
    for desc, kw, hk in (("plain", {}, 2), ("causal", {"is_causal": True}, 2), ("gqa causal", {"is_causal": True, "enable_gqa": True}, 1)):
        g = torch.Generator().manual_seed(len(attn) + 1)
        q = torch.randn(1, 2, 6, 8, generator=g, dtype=torch.float64)
        k = torch.randn(1, hk, 6, 8, generator=g, dtype=torch.float64)
        v = torch.randn(1, hk, 6, 8, generator=g, dtype=torch.float64)
        sft.reset_status()
        y = sft.attention(q, k, v, name="golden", **kw)
        st = sft.last_status()
        attn.append({"what": desc, "kwargs": kw, "q": q.tolist(), "k": k.tolist(), "v": v.tolist(),
                     "y": y.tolist(), "status": {n: _status(st[f"golden.{n}"]) for n in ("scores", "pv")}})

    model = tiny_model(torch)
    x = torch.randn(1, 6, 16, generator=torch.Generator().manual_seed(99), dtype=torch.float64)
    outs = {}
    for dtype in ("sf16", "sf8"):
        c = sft.convert(model, dtype=dtype, accumulate="wide")
        sft.reset_status()
        with torch.no_grad():
            outs[dtype] = {"y": c(x).tolist(),
                           "status": {n: _status(s) for n, s in sorted(sft.last_status().items())},
                           "kept_sf16": [n for n, _ in sft.conversion_report(c).kept_sf16]}
    return {"spec_version": SPEC_VERSION,
            "source": "sf host model (sf.torch wide mode, the shift rule of sf._heuristic); to confirm on the RTL",
            "format": "linear: y = sft wide linear(x, w, b) in float64, status = every wide SFLayerStatus field. "
                      "attention: sft.attention(q, k, v, **kwargs), statuses of {name}.scores and {name}.pv. "
                      "model: tiny_model() in tools/gen_golden_wide.py (float64 state below), "
                      "sft.convert(dtype, accumulate='wide') on x.",
            "linear": layers, "attention": attn,
            "model": {"state": {k: v.tolist() for k, v in model.state_dict().items()},
                      "x": x.tolist(), "outputs": outs}}


def tiny_model(torch):
    """One pre-norm transformer block and an LM head tied to the embedding
    of a 24-word vocabulary: linears, attention (SDPA) and a kept head."""
    nn, F = torch.nn, torch.nn.functional

    class Tiny(nn.Module):
        def __init__(self, d=16, h=2, vocab=24):
            super().__init__()
            self.h = h
            self.emb = nn.Embedding(vocab, d)
            self.ln = nn.LayerNorm(d)
            self.qkv = nn.Linear(d, 3 * d)
            self.proj = nn.Linear(d, d)
            self.up = nn.Linear(d, 2 * d)
            self.down = nn.Linear(2 * d, d)
            self.head = nn.Linear(d, vocab, bias=False)
            self.head.weight = self.emb.weight

        def forward(self, x):
            B, T, D = x.shape
            q, k, v = self.qkv(self.ln(x)).split(D, -1)
            q, k, v = (t.reshape(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
            x = x + self.down(F.gelu(self.up(x)))
            return self.head(x)

    torch.manual_seed(1234)
    return Tiny().double().eval()


def main(argv):
    out = {ROOT / "spec" / "golden" / "wide" / "dot_wide_shift.json": dot_wide_shift(),
           ROOT / "spec" / "golden" / "wide" / "sft_wide.json": sft_wide()}
    bad = []
    for path, data in out.items():
        text = json.dumps(data, separators=(",", ":")) + "\n"
        if "--check" in argv:
            if not path.exists() or path.read_text() != text:
                bad.append(path.name)
        else:
            path.write_text(text)
            print(f"wrote {path.relative_to(ROOT)} ({len(text) // 1024} KB)")
    if bad:
        print("out of date:", ", ".join(bad))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
