"""The lowering spec (spec/ops.py) stays complete, current and correct."""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sf import _linalg, _vectorized as vec
from sf._format import FORMATS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spec.ops import OPS, SPEC_VERSION  # noqa: E402

GOLDEN = sorted((ROOT / "spec" / "golden").glob("*.json"))


def _interp_ops():
    src = (ROOT / "src" / "sf" / "kernels" / "_interp.py").read_text()
    names = set(re.findall(r'\bn == "([a-z_]+)"', src))
    for group in re.findall(r"\bn in \(([^)]*)\)", src):
        names |= set(re.findall(r'"([a-z_]+)"', group))
    return names


def test_every_interpreter_op_has_a_spec_entry():
    spec = {o["name"] for o in OPS}
    assert _interp_ops() - spec == set()
    assert spec - _interp_ops() == set()


def test_entries_are_complete():
    need = {"name", "version", "category", "signature", "operands", "result", "shape", "semantics", "events",
            "memory", "faults", "predication", "hw_v1", "mlir", "python", "rewrites"}
    major_minor = tuple(int(x) for x in SPEC_VERSION.split(".")[:2])
    for o in OPS:
        assert need <= o.keys(), o["name"]
        assert o["predication"] in ("pure", "events", "memory")
        assert set(o["events"]) <= {"overflow", "underflow", "divide", "invalid"}
        assert tuple(int(x) for x in o["version"].split(".")[:2]) <= major_minor, o["name"]
        assert "—" not in json.dumps(o)


def test_generated_files_are_current():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_spec.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout


def test_compiler_pages_are_in_the_nav():
    nav = (ROOT / "website" / "docs.json").read_text()
    for p in ("overview", "ops", "python-api", "versioning"):
        assert f'"compiler/{p}"' in nav


def _ev1(ev, i):
    hits = [k for k, m in ev.items() if np.asarray(m).reshape(-1)[i]]
    return hits


@pytest.mark.parametrize("path", GOLDEN, ids=lambda p: p.stem)
def test_golden_vectors_match_the_kernels(path):
    """The vectors come from the exact oracle; the interpreter and C back end use these vectorized kernels."""
    g = json.loads(path.read_text())
    op = g["op"]
    for c in g["cases"]:
        f = FORMATS[c["bits"]]
        if op in ("add", "sub", "mul", "div", "fma", "sqrt"):
            r, ev = getattr(vec, op)(f, *[np.array([x]) for x in c["in"]])
            got_ev = _ev1(ev, 0)
            assert int(r[0]) == c["out"], c
            if op == "div" and "divide" in got_ev:
                got_ev = ["divide"]              # divide wins over overflow, as in the scalar catalogue
            assert got_ev == ([c["event"]] if c["event"] else []) or (op == "fma" and c["event"] in got_ev), c
        elif op in ("neg", "abs"):
            r = getattr(vec, op if op == "neg" else "abs_")(f, np.array(c["in"]))
            r = r[0] if isinstance(r, tuple) else r
            assert int(np.asarray(r)[0]) == c["out"]
        elif op == "widen":
            assert c["in"][0] << (c["to_bits"] - c["bits"]) == c["out"]
        elif op == "narrow":
            g2 = FORMATS[c["to_bits"]]
            r, ev = vec.cast_array(g2, np.array([c["in"][0] / (1 << (c["bits"] - 1))]))
            assert int(r[0]) == c["out"], c
        elif op == "sum":
            r, ev = _linalg.sum_codes(f, np.array(c["in"][0]), 0)
            assert int(r) == c["out"] and bool(ev["overflow"]) == ("overflow" in c["events"]), c
        elif op == "dot":
            a, b = (np.array(x) for x in c["in"])
            r, ev = _linalg.matmul_codes(f, a[None, :], b[:, None])
            assert int(r[0, 0]) == c["out"], c
            assert {k for k, m in ev.items() if m.any()} == set(c["events"]), c
        else:
            pytest.fail(f"no checker for {op}")
