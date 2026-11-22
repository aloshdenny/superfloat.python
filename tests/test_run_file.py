"""Run files: the whole launch as one file for a testbench we do not drive, and the dump check.

The round trip is plan -> run file -> plan -> toy model of the chip -> dump ->
check_dump, so the file alone is enough to run the kernel and check every bit.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import sf
from sf.kernels import _driver as D

from atreides_toy import run_toy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import make_chip_run  # noqa: E402

RUN = ROOT / "examples" / "chip_run" / "axpy_n600.run.json"


def _dump(run):
    plan = D.from_run_file(run)
    return D._ranges(run_toy(plan, plan.memory))


def test_example_run_file_is_current():
    assert RUN.read_text() == make_chip_run.build()


def test_run_file_alone_runs_and_checks_every_bit():
    run = json.loads(RUN.read_text())
    assert [L["threads"] for L in run["launches"]] == [240, 240, 120]
    assert [L["first_block"] for L in run["launches"]] == [0, 30, 60]
    assert D.check_dump(run, _dump(run)) == []


def test_one_wrong_bit_is_reported():
    run = json.loads(RUN.read_text())
    mem = D._unranges(_dump(run))
    out = run["outputs"][0]
    mem[out["base"] + 7] ^= 1
    bad = D.check_dump(run, {str(a): w for a, w in mem.items()})     # the {"address": word} form
    assert len(bad) == 1 and "out_ptr[7]" in bad[0]
    mem.pop(out["base"] + 9)
    assert len(D.check_dump(run, [[a, w] for a, w in mem.items()])) == 2   # a missing word too


def test_run_file_round_trips_a_plan():
    k = json.loads((ROOT / "tests" / "data" / "atreides_kernel_axpy_v0.3.json").read_text())
    with sf.errstate(all="ignore"):
        x = sf.array(np.linspace(-0.9, 0.9, 20), dtype=sf.sf16)
    plan = D.plan_launch(k, {"a": sf.sf16(0.5), "x_ptr": x, "y_ptr": x, "out_ptr": sf.zeros(20, "sf16"), "n": 20})
    back = D.from_run_file(json.loads(json.dumps(D.to_run_file(plan))))
    assert back.program == plan.program and back.memory == plan.memory
    assert [(L.threads, L.block_offset, L.writes) for L in back.launches] == \
           [(L.threads, L.block_offset, L.writes) for L in plan.launches]
    with pytest.raises(sf.SFLaunchError, match="no expected words"):
        D.check_dump(D.to_run_file(plan), {})


def test_check_command(tmp_path):
    run = json.loads(RUN.read_text())
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps(_dump(run)))
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "make_chip_run.py"), "--check", str(RUN), str(dump)],
                       capture_output=True, text=True)
    assert r.returncode == 0 and "every output word matches" in r.stdout
