"""The whole chain on the hardware device: kernel[grid](...) on atreides-hw traces,
writes SF IR JSON, compiles (cached), plans memory and launches, runs, and
writes the outputs back into the user's arrays.

Here the compiler is the stand-in (tests/fake_superfloat_compile.py, which
returns Kartik's hand-written axpy) and the runner is the toy model of its 7
instructions (tests/atreides_toy.py).  Swapping in the real compiler and a
real runner (ISA model, testbench, board) changes nothing else.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import sf
from sf.kernels import _driver as D
from sf.kernels._runtime import HardwareAtreides

from atreides_toy import toy_runner  # noqa: E402

HERE = Path(__file__).resolve().parent
FAKE = [sys.executable, str(HERE / "fake_superfloat_compile.py")]


@sf.jit
def axpy(a, x_ptr, y_ptr, out_ptr, n, BLOCK: sf.constexpr):
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    mask = offs < n
    x = sf.load(x_ptr + offs, mask=mask)
    y = sf.load(y_ptr + offs, mask=mask)
    sf.store(out_ptr + offs, sf.fma(a, x, y), mask=mask)


@pytest.fixture
def hw(tmp_path, monkeypatch):
    """atreides-hw with the driver attached: stand-in compiler, toy runner, private cache."""
    monkeypatch.setattr(D, "_CACHE_DIR", str(tmp_path / "cache"))
    count = tmp_path / "count"
    monkeypatch.setenv("FAKE_COMPILE_COUNT", str(count))
    driver = D.AtreidesDriver(runner=toy_runner, compiler=FAKE)
    monkeypatch.setattr(HardwareAtreides, "_driver", driver)
    driver.compiles = lambda: len(count.read_text().splitlines()) if count.exists() else 0
    return driver


def _launch(device, n, seed, grid=None):
    rng = np.random.default_rng(seed)
    with sf.errstate(all="ignore"):
        x = sf.asarray(rng.uniform(-0.9, 0.9, n), dtype=sf.sf16, device=device)
        y = sf.asarray(rng.uniform(-0.9, 0.9, n), dtype=sf.sf16, device=device)
        a = sf.sf16(float(rng.uniform(-0.9, 0.9)))
        out = sf.zeros(n, "sf16", device=device)
        res = axpy[grid or (sf.cdiv(n, 8),)](a, x, y, out, n, BLOCK=8)
    return out, res


@pytest.mark.parametrize("n", [5, 20, 248, 600])
def test_axpy_on_hardware_matches_the_simulator(hw, n):
    out_hw, res = _launch("atreides-hw", n, seed=n)
    out_sim, res_sim = _launch("atreides-sim", n, seed=n)
    assert np.array_equal(out_hw.codes, out_sim.codes)
    assert res.device == "atreides-hw"
    assert res.counters.programs == res_sim.counters.programs == -(-n // 8)   # counted the same way
    assert len(hw.last_plan.launches) == -(-n // 248)


@pytest.mark.parametrize("grid", [(1,), (4,), (3, 2)])
def test_a_grid_that_does_not_match_the_threads_is_refused(hw, grid):
    with pytest.raises(sf.SFLaunchError, match="programs"):     # n = 20 needs exactly 3 programs of 8
        _launch("atreides-hw", 20, seed=8, grid=grid)


def test_events_are_unknown_on_hardware(hw):
    _, res = _launch("atreides-hw", 20, seed=1)
    assert res.counters.saturated is None and res.status.saturated is None


def test_replay_checks_the_hardware_bits(hw):
    with sf.config(status="replay"):
        _, res = _launch("atreides-hw", 20, seed=2)
    assert res.status.exact and res.status.saturated is not None


def test_replay_catches_wrong_bits(hw, monkeypatch):
    def broken(plan):
        mem = toy_runner(plan)
        out = plan.buffers["out_ptr"]
        mem[out.base] ^= 1                                  # one wrong bit in the first output
        return mem
    monkeypatch.setattr(hw, "runner", broken)
    with sf.config(status="replay"), pytest.raises(sf.SFDeviceError, match="differ from the host"):
        _launch("atreides-hw", 20, seed=3)


def test_one_compile_for_many_launches(hw):
    for n, seed in ((20, 4), (20, 5), (600, 6)):            # the IR is the same: n is a runtime value
        _launch("atreides-hw", n, seed)
    assert hw.compiles() == 1


def test_no_driver_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(HardwareAtreides, "_driver", None)
    with pytest.raises(sf.SFDeviceError, match="no Atreides hardware is attached"):
        _launch("atreides-hw", 8, seed=7)
