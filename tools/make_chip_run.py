"""Write a run file for a chip or RTL testbench, or check the memory dump it sends back.

    python tools/make_chip_run.py                  # write examples/chip_run/axpy_n600.run.json
    python tools/make_chip_run.py --check RUN DUMP # compare a testbench dump with RUN's expected words

A run file holds everything the chip needs for one call: the program, the data
memory before the first launch, and each launch (thread count and the
parameter-block words to write first), plus the expected output words from
atreides-sim.  Whoever runs the chip loads it, runs the launches in order, and
sends back the data memory as a dump (the same "ranges" format, a
{"address": word} object, or [[address, word], ...]).  No compiler is involved:
the kernel is Kartik's hand-written axpy (tests/data/atreides_kernel_axpy_v0.3.json).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sf  # noqa: E402
from sf.kernels import _driver as D  # noqa: E402

KERNEL = ROOT / "tests" / "data" / "atreides_kernel_axpy_v0.3.json"
OUT = ROOT / "examples" / "chip_run" / "axpy_n600.run.json"
N, SEED, BLOCK = 600, 20261122, 16


@sf.jit
def axpy(a, x_ptr, y_ptr, out_ptr, n, BLOCK: sf.constexpr):
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    mask = offs < n
    x = sf.load(x_ptr + offs, mask=mask)
    y = sf.load(y_ptr + offs, mask=mask)
    sf.store(out_ptr + offs, sf.fma(a, x, y), mask=mask)


def build() -> str:
    """The axpy run file as text: n = 600, so three launches (240, 240, 120 threads)."""
    rng = np.random.default_rng(SEED)
    with sf.errstate(all="ignore"):
        x = sf.asarray(rng.uniform(-0.9, 0.9, N), dtype=sf.sf16, device="atreides-sim")
        y = sf.asarray(rng.uniform(-0.9, 0.9, N), dtype=sf.sf16, device="atreides-sim")
        a = sf.sf16(float(rng.uniform(-0.9, 0.9)))
        out = sf.zeros(N, "sf16", device="atreides-sim")
        axpy[(sf.cdiv(N, BLOCK),)](a, x, y, out, N, BLOCK=BLOCK)          # the expected bits: atreides-sim
    args = {"a": a, "x_ptr": x, "y_ptr": y, "out_ptr": sf.zeros(N, "sf16"), "n": N}
    plan = D.plan_launch(json.loads(KERNEL.read_text()), args)
    run = D.to_run_file(plan, expected={"out_ptr": out.codes})
    run["inputs"] = {"n": N, "a_code": int(a.code), "seed": SEED, "made_by": "tools/make_chip_run.py"}
    return json.dumps(run, indent=1) + "\n"


def main(argv) -> int:
    if argv[:1] == ["--check"]:
        run, dump = (json.loads(Path(f).read_text()) for f in argv[1:3])
        bad = D.check_dump(run, dump)
        print("every output word matches" if not bad else f"{len(bad)} words differ:\n  " + "\n  ".join(bad[:20]))
        return 1 if bad else 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(build())
    print("wrote", OUT.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
