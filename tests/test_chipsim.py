"""The driver on superfloat.sim, Alosh's cycle-accurate model of the chip (lockstep-checked against the v4.9.2 RTL).

Skipped unless superfloat.sim is importable: build it in its checkout and run
    SUPERFLOAT_SIM=/path/to/superfloat.sim pytest tests/test_chipsim.py
The kernel file is Kartik's hand-written axpy (format 0.3) through the stand-in
compiler; no part of superfloat.sim is in this repository.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("superfloat_sim")

import sf  # noqa: E402
from sf.kernels import _driver as D  # noqa: E402
from sf.kernels._chipsim import SuperfloatSimRunner  # noqa: E402
from sf.kernels._runtime import HardwareAtreides  # noqa: E402

from test_driver_wiring import FAKE, _launch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "examples" / "chip_run" / "axpy_n600.run.json"


@pytest.fixture
def chip(tmp_path, monkeypatch):
    """atreides-hw = the driver with the stand-in compiler and superfloat.sim as the runner."""
    monkeypatch.setattr(D, "_CACHE_DIR", str(tmp_path / "cache"))
    runner = SuperfloatSimRunner()
    driver = D.AtreidesDriver(runner=runner, compiler=FAKE)
    monkeypatch.setattr(HardwareAtreides, "_driver", driver)
    return driver


def test_the_run_file_gives_every_expected_bit():
    run = json.loads(RUN.read_text())
    runner = SuperfloatSimRunner()
    assert D.check_dump(run, runner(D.from_run_file(run))) == []
    assert [r["writes"] for r in runner.launches] == [240, 240, 120]   # one store per live thread
    assert runner.cycles > 0


@pytest.mark.parametrize("n", [1, 8, 20, 240, 241, 600, 1000])
def test_axpy_on_the_chip_model_matches_atreides_sim(chip, n):
    out_hw, _ = _launch("atreides-hw", n, seed=100 + n)
    out_sim, _ = _launch("atreides-sim", n, seed=100 + n)
    assert np.array_equal(out_hw.codes, out_sim.codes)
    assert len(chip.runner.launches) == len(chip.last_plan.launches) == -(-n // 240)


def test_a_slow_host_changes_cycles_not_bits():
    plan = D.from_run_file(json.loads(RUN.read_text()))
    fast, slow = SuperfloatSimRunner(), SuperfloatSimRunner(host_delay=4, host_seed=7)
    assert fast(plan) == slow(plan)
    assert slow.cycles > fast.cycles


def test_sim_command_runs_and_checks_a_run_file():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "make_chip_run.py"), "--sim", str(RUN)],
                       capture_output=True, text=True, env={**__import__("os").environ,
                                                            "PYTHONPATH": ":".join(sys.path)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert "every output word matches" in r.stdout and "cycles" in r.stdout
