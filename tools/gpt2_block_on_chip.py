"""One GPT-2-sized transformer block on the chip model, against sf's eager emulation.

    SUPERFLOAT_GPU=<superfloat.gpu checkout> PYTHONPATH=<superfloat.sim>/python \\
        python tools/gpt2_block_on_chip.py [--tokens 8] [--dim 768] [--heads 12]

Random weights in GPT-2's shapes (no download): LayerNorm, QKV, causal
attention, the output projection and the 4x MLP, converted with
sft.convert(accumulate="wide").  The compile backend sends every wide matmul
to atreides-hw, where Alosh's gemm / gemv programs run it on superfloat.sim
(sf.kernels._chipmm); each call is also checked against the host MX model.
Prints whether the block's output is bit-identical to the eager emulation,
every chip call, and the chip cycles.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sf  # noqa: E402
import sf.torch as sft  # noqa: E402
from sf.kernels import _chipmm as C  # noqa: E402
from sf.kernels._runtime import HardwareAtreides  # noqa: E402


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.d, self.h = d, heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        for lin in (self.qkv, self.proj, self.fc, self.out):           # GPT-2's initial scales
            nn.init.normal_(lin.weight, std=0.02)
            nn.init.normal_(lin.bias, std=0.01)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=-1)
        q, k, v = (t.view(B, T, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        a = sft.attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(a)
        return x + self.out(torch.nn.functional.gelu(self.fc(self.ln2(x))))


def run(tokens: int = 8, dim: int = 768, heads: int = 12, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    model = sft.convert(Block(dim, heads).eval(), accumulate="wide")
    x = torch.randn(1, tokens, dim)
    mm = C.ChipMatmul()
    inner = C.library(mm, check=True)["matmul_wide"]
    calls = []

    def counted(fmt, a, b, shift=0, mx=None):
        out = inner(fmt, a, b, shift, mx)
        calls.append(dict(bits=fmt.bits, a=tuple(a.shape), b=tuple(b.shape), cycles=mm.report.cycles,
                          launches=len(mm.report.launches)))
        return out

    old = HardwareAtreides._library
    try:
        with torch.no_grad():
            want = model(x)
            HardwareAtreides.attach_library({"matmul_wide": counted})
            torch._dynamo.reset()
            with sf.config(default_device="atreides-hw", preflight="ignore"):
                t = time.time()
                got = torch.compile(model, backend="superfloat")(x)
                seconds = time.time() - t
    finally:
        HardwareAtreides.attach_library(old)
    rep = sft.compile_report()
    return dict(identical=torch.equal(got, want), calls=calls, seconds=seconds,
                on_device=len(rep.on_device), host_emulated=len(rep.host_emulated))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--heads", type=int, default=12)
    a = ap.parse_args(argv)
    r = run(a.tokens, a.dim, a.heads)
    print(f"block output bit-identical to the eager emulation: {r['identical']}")
    print(f"{r['on_device']} SF ops on atreides-hw, {r['host_emulated']} host-emulated, "
          f"{len(r['calls'])} chip calls (each checked against the host MX model), {r['seconds']:.1f} s")
    for c in r["calls"]:
        print(f"  sf{c['bits']} {c['a']} @ {c['b']}: {c['cycles']:,} cycles in {c['launches']} launches")
    print(f"chip cycles for the block: {sum(c['cycles'] for c in r['calls']):,}")
    return 0 if r["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
