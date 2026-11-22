"""A stand-in for ``superfloat-compile`` in the driver tests.

``--version`` prints a version.  Otherwise it reads SF IR JSON on stdin,
checks it is SF IR, and prints the hand-written axpy kernel file
(tests/data/atreides_kernel_axpy_v0.2.json) with the requested target.
FAKE_COMPILE_COUNT names a file to which each compile appends a line, so
tests can count runs; FAKE_COMPILE_FAIL makes it fail like a compiler bug;
FAKE_COMPILE_WRONG_TARGET makes it write a kernel for another target;
FAKE_COMPILE_VERSION replaces its version line.  FAKE_COMPILE_LAUNCH (JSON) is merged into
the kernel file's launch rule, which is then labelled 0.3-draft.  It takes ``--target=T`` as
the real one does, and writes its argument list to FAKE_COMPILE_COUNT too.
"""
import json
import os
import sys
from pathlib import Path

SAMPLE = Path(__file__).resolve().parent / "data" / "atreides_kernel_axpy_v0.2.json"


def main(argv):
    if "--version" in argv:
        print(os.environ.get("FAKE_COMPILE_VERSION", "superfloat-compile 0.0.0 (fake)"))
        return 0
    if os.environ.get("FAKE_COMPILE_FAIL"):
        print("superfloat-compile: assertion failed: lowering of fma", file=sys.stderr)
        return 3
    ir = json.loads(sys.stdin.read())
    if ir.get("format") != "sfir":
        print("superfloat-compile: input is not SF IR", file=sys.stderr)
        return 2
    count = os.environ.get("FAKE_COMPILE_COUNT")
    if count:
        with open(count, "a") as f:
            f.write(ir["kernel"] + " " + " ".join(argv) + "\n")
    k = json.loads(SAMPLE.read_text())
    target = next((a.split("=", 1)[1] for a in argv if a.startswith("--target=")), None)
    if target is None:
        print("superfloat-compile: --target=<target> is required", file=sys.stderr)
        return 2
    k["target"]["isa"] = os.environ.get("FAKE_COMPILE_WRONG_TARGET") or target
    if os.environ.get("FAKE_COMPILE_LAUNCH"):              # e.g. {"lanes_per_program": 16, ...}
        k["launch"].update(json.loads(os.environ["FAKE_COMPILE_LAUNCH"]))
        k["format_version"] = "0.3-draft"
    print(json.dumps(k))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
