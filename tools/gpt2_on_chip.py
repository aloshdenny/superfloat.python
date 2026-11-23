"""Real GPT-2 on the chip model: every wide matmul through Alosh's gemm / gemv programs on superfloat.sim.

    HF_HOME=<cache with openai-community/gpt2 and wikitext> SUPERFLOAT_GPU=<superfloat.gpu checkout> \\
    PYTHONPATH=<superfloat.sim>/python python tools/gpt2_on_chip.py --tokens 32 [--windows 1] [--no-check]

Converts GPT-2 with sft.convert(dtype="sf16", accumulate="wide") and runs the
first windows of WikiText-2 test twice: the eager host emulation, and
torch.compile(backend="superfloat") on atreides-hw with the chip library
attached.  Prints whether the logits are bit-identical, the loss and
perplexity of both, every chip call's shape count, and the chip cycles.
With --check (default) every chip call is also checked against the host MX model.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import sf  # noqa: E402
import sf.torch as sft  # noqa: E402
from sf.kernels import _chipmm as C  # noqa: E402
from sf.kernels._runtime import HardwareAtreides  # noqa: E402

MODEL = "openai-community/gpt2"


def wikitext_ids():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return AutoTokenizer.from_pretrained(MODEL)("\n\n".join(ds["text"]), return_tensors="pt").input_ids


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=32, help="tokens per window")
    ap.add_argument("--windows", type=int, default=1)
    ap.add_argument("--dtype", default="sf16")
    ap.add_argument("--no-check", action="store_true", help="do not also check each chip call on the host")
    ap.add_argument("--out", help="write the result as JSON here")
    a = ap.parse_args(argv)

    from transformers import AutoModelForCausalLM
    model = sft.convert(AutoModelForCausalLM.from_pretrained(MODEL).eval(), dtype=a.dtype, accumulate="wide")
    ids = wikitext_ids()
    mm = C.ChipMatmul()
    inner = C.library(mm, check=not a.no_check)["matmul_wide"]
    shapes, cycles, launches = Counter(), [0], [0]

    def counted(fmt, x, w, shift=0, mx=None):
        out = inner(fmt, x, w, shift, mx)
        shapes[(fmt.bits, tuple(x.shape), tuple(w.shape))] += 1
        cycles[0] += mm.report.cycles
        launches[0] += len(mm.report.launches)
        return out

    res = dict(model=MODEL, dtype=a.dtype, tokens_per_window=a.tokens, windows=[])
    nll = {"host": 0.0, "chip": 0.0}
    n = 0
    old = HardwareAtreides._library
    first_report = len(sft.compile_reports())
    try:
        HardwareAtreides.attach_library({"matmul_wide": counted})
        with torch.no_grad():
            compiled = torch.compile(model, backend="superfloat")
            for w in range(a.windows):
                chunk = ids[:, w * a.tokens:(w + 1) * a.tokens]
                t = time.time()
                host = model(chunk, labels=chunk)
                th = time.time() - t
                c0 = cycles[0]
                t = time.time()
                with sf.config(default_device="atreides-hw", preflight="ignore"):
                    chip = compiled(chunk, labels=chunk)
                tc = time.time() - t
                same = torch.equal(host.logits, chip.logits)
                k = chunk.shape[1] - 1
                nll["host"] += host.loss.item() * k
                nll["chip"] += chip.loss.item() * k
                n += k
                res["windows"].append(dict(identical_logits=same, host_loss=host.loss.item(), chip_loss=chip.loss.item(),
                                           host_seconds=round(th, 1), chip_seconds=round(tc, 1),
                                           chip_cycles=cycles[0] - c0))
                print(f"window {w}: logits bit-identical {same}; loss host {host.loss.item():.6f} chip "
                      f"{chip.loss.item():.6f}; {cycles[0] - c0:,} chip cycles; {tc:.0f} s", flush=True)
    finally:
        HardwareAtreides.attach_library(old)
    reps = sft.compile_reports()[first_report:]              # GPT-2 compiles as several graphs
    on_device, host_emulated = sum(len(r.on_device) for r in reps), sum(len(r.host_emulated) for r in reps)
    res.update(predicted_tokens=n, ppl_host=math.exp(nll["host"] / n), ppl_chip=math.exp(nll["chip"] / n),
               chip_cycles=cycles[0], chip_launches=launches[0], chip_calls=sum(shapes.values()),
               checked_on_host=not a.no_check, graphs=len(reps), on_device=on_device, host_emulated=host_emulated,
               shapes={f"sf{b} {x} @ {w}": c for (b, x, w), c in sorted(shapes.items())})
    print(f"perplexity over {n} predicted tokens: host {res['ppl_host']:.4f}, chip {res['ppl_chip']:.4f}")
    print(f"{res['chip_calls']} chip calls, {res['chip_launches']:,} launches, {res['chip_cycles']:,} chip cycles; "
          f"{res['graphs']} graphs: {res['on_device']} SF ops on device, {res['host_emulated']} host-emulated")
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0 if all(w["identical_logits"] for w in res["windows"]) else 1


if __name__ == "__main__":
    sys.exit(main())
