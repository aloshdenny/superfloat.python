#!/usr/bin/env python3
"""Regenerate tests/golden: full code tables for sf4..sf10 and SHA-256
checksums for every width.  Run only when the format definition changes;
the golden file is the baseline every later milestone is checked against.

    python tools/make_golden.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import sf  # noqa: E402
from test_1_0_1_types_casts import golden_table  # noqa: E402


def main() -> None:
    out = os.path.join(ROOT, "tests", "golden")
    os.makedirs(out, exist_ok=True)
    checksums = {}
    for dt in sf.dtypes:
        bits = sf.info(dt).bits
        table = golden_table(bits)
        checksums[f"sf{bits}"] = hashlib.sha256(table.encode()).hexdigest()
        if bits <= 10:
            with open(os.path.join(out, f"sf{bits}.tsv"), "w") as f:
                f.write(table)
    with open(os.path.join(out, "checksums.json"), "w") as f:
        json.dump(checksums, f, indent=2, sort_keys=True)
        f.write("\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
