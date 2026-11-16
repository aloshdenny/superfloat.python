"""Replay spec/golden/wide/dot_wide_shift.json and spec/golden/wide/sft_wide.json
(Alosh's item 9) bit for bit, on the host and on the Atreides targets, and
check that tools/gen_golden_wide.py still writes exactly these files."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sf
import sf.kernels  # noqa: F401
from sf._format import FORMATS

ROOT = Path(__file__).resolve().parent.parent
DW = json.loads((ROOT / "spec" / "golden" / "wide" / "dot_wide_shift.json").read_text())
SW = json.loads((ROOT / "spec" / "golden" / "wide" / "sft_wide.json").read_text())
DTYPES = {b: sf.dtypes[b - 4] for b in range(4, 17)}


def _arr(codes, bits, device="host"):
    return sf.from_codes(np.asarray(codes, dtype=np.int64), DTYPES[bits], device=device)


@pytest.mark.parametrize("i", range(len(DW["cases"])))
def test_dot_wide_shift_arrays_on_host(i):
    c = DW["cases"][i]
    from sf import _linalg
    q, ev = _linalg.matmul_wide_codes(FORMATS[c["bits"]], np.array(c["a"]), np.array(c["b"]),
                                      c["shift"] if isinstance(c["shift"], int) else np.array(c["shift"]),
                                      c["target"])
    assert q.tolist() == c["out"]
    assert np.asarray(ev["overflow"]).astype(int).tolist() == c["overflow"]
    assert np.asarray(ev["underflow"]).astype(int).tolist() == c["underflow"]


@pytest.mark.parametrize("device", ["atreides", "atreides-sim"])
def test_dot_wide_shift_arrays_on_the_targets(device):
    """Every v4.9.2 case through sf.matmul on the target (the mock's MX version)."""
    n = 0
    for c in DW["cases"]:
        if c["target"] != "v4.9.2":
            continue
        s = c["shift"] if isinstance(c["shift"], int) else np.array(c["shift"])
        with sf.errstate(all="ignore"), sf.config(preflight="ignore"):
            y = sf.matmul(_arr(c["a"], c["bits"], device), _arr(c["b"], c["bits"], device),
                          accumulate="wide", shift=s)
        assert y.device == device and y.codes.tolist() == c["out"], (c["bits"], c["kind"])
        n += 1
    assert n == 72


def test_coverage():
    cases = DW["cases"]
    assert {c["bits"] for c in cases} == {4, 6, 8, 10, 12, 16}
    assert {c["target"] for c in cases} == {"v4.9.2", "v4.8.1"}
    assert sum(any(map(any, c["overflow"])) for c in cases) > 20
    assert any(c["status"]["guard_raised"] and c["status"]["redone"] for c in SW["linear"])


torch = pytest.importorskip("torch")
import sf.torch as sft  # noqa: E402
from sf.torch import _core  # noqa: E402

FIELDS = ("saturated", "exact", "underflow", "input_underflow", "weight_underflow", "outputs",
          "accumulate", "shift_min", "shift_max", "h", "redone", "flushed", "acc_clipped", "guard_raised")


@pytest.mark.parametrize("i", range(len(SW["linear"])), ids=[c["what"] for c in SW["linear"]])
def test_sft_wide_linear_and_every_status_field(i):
    c = SW["linear"][i]
    r = _core.wide_linear(np.array(c["x"]), np.array(c["w"]), None if c["b"] is None else np.array(c["b"]),
                          c["bits"])
    assert r.y.tolist() == c["y"]
    assert {f: getattr(r, f) for f in FIELDS} == c["status"]


def test_sft_wide_linear_on_the_chip_gives_the_same_values():
    for c in SW["linear"]:
        if c["bits"] != 16:
            continue
        with sf.config(preflight="ignore"):
            r = _core.wide_linear(np.array(c["x"]), np.array(c["w"]),
                                  None if c["b"] is None else np.array(c["b"]), 16,
                                  runner=_core.DeviceRunner("atreides"))
        assert r.y.tolist() == c["y"] and r.redone == c["status"]["redone"]


@pytest.mark.parametrize("i", range(len(SW["attention"])), ids=[c["what"] for c in SW["attention"]])
def test_attention(i):
    c = SW["attention"][i]
    t = {k: torch.tensor(c[k], dtype=torch.float64) for k in ("q", "k", "v")}
    sft.reset_status()
    y = sft.attention(t["q"], t["k"], t["v"], name="golden", **c["kwargs"])
    assert y.tolist() == c["y"]
    st = sft.last_status()
    for part in ("scores", "pv"):
        assert {f: getattr(st[f"golden.{part}"], f) for f in FIELDS} == c["status"][part]


@pytest.mark.parametrize("dtype", ["sf16", "sf8"])
def test_small_model_end_to_end(dtype):
    sys.path.insert(0, str(ROOT / "tools"))
    from gen_golden_wide import tiny_model
    m = tiny_model(torch)
    m.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in SW["model"]["state"].items()})
    want = SW["model"]["outputs"][dtype]
    c = sft.convert(m, dtype=dtype, accumulate="wide")
    assert [n for n, _ in sft.conversion_report(c).kept_sf16] == want["kept_sf16"]
    sft.reset_status()
    with torch.no_grad():
        y = c(torch.tensor(SW["model"]["x"], dtype=torch.float64))
    assert y.tolist() == want["y"]
    st = sft.last_status()
    assert {n: {f: getattr(s, f) for f in FIELDS} for n, s in st.items()} == want["status"]


def test_the_generator_writes_exactly_these_files():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_golden_wide.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
