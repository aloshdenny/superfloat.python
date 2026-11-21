"""A stand-in for ``superfloat-compile`` in the driver tests.

``--version`` prints a version.  Otherwise it reads SF IR JSON on stdin,
checks it is SF IR, and prints the hand-written axpy kernel file
(tests/data/atreides_kernel_axpy_draft.json) with the requested target.
FAKE_COMPILE_COUNT names a file to which each compile appends a line, so
tests can count runs; FAKE_COMPILE_FAIL makes it fail like a compiler bug.
"""
import json
import os
import sys
from pathlib import Path

SAMPLE = Path(__file__).resolve().parent / "data" / "atreides_kernel_axpy_draft.json"


def main(argv):
    if "--version" in argv:
        print("fake-superfloat-compile 0.0")
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
            f.write(ir["kernel"] + "\n")
    k = json.loads(SAMPLE.read_text())
    target = argv[argv.index("--target") + 1] if "--target" in argv else None
    k["target"]["isa"] = target
    print(json.dumps(k))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
