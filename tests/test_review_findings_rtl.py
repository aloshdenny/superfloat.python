"""Regressions from the v1 RTL report (superfloat.gpu, 2026-10-19 and 2026-10-27).

The framework ran on the signed-off v1 chip RTL in Verilator. Framework findings:

1. a hardware driver could not say "I don't know": a launch status was always
   "exact" with the driver's counts, so a v1 driver's chip saturations came back
   as "0 saturated, exact";
2. a heavy op's status had no underflow, so a device multiply that truncated to 0
   was never reported;
3. the C backend's build cache was not safe across processes;
4. torch 2.11 broke compiled graphs and compile_report showed only the last one
   (see test_compile_report_covers_every_graph_of_a_call);
5. a device layer's saturation count added the matmul's and the bias add's counts
   and still claimed to be exact.

Plus the hardware gap: v1 has no divide or square-root instruction, so a target
must be able to declare the ops it has.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import warnings

import numpy as np
import pytest

import sf
from sf import _device
from sf.kernels._interp import Counters, Interpreter


# --- shared: an interpreter-backed driver that reports like v1 silicon ------------------------

class V1Driver:
    """Runs kernels correctly but, like v1 silicon, cannot read any event back."""

    def __init__(self, counts="none", corrupt=False):
        self.counts, self.corrupt, self.calls = counts, corrupt, 0

    def launch(self, ir, grid, block, buffers, scalars):
        self.calls += 1
        c = Interpreter(ir, buffers, scalars, grid, sf.device("atreides-hw").limits).run()
        if self.corrupt:
            for b in buffers.values():
                b.reshape(-1)[0] ^= 1
        if self.counts == "none":
            return Counters(programs=c.programs, saturated=None, underflow=None, div0=None, invalid=None)
        if self.counts == "bound":
            return Counters(programs=c.programs, saturated=c.saturated + 3, underflow=None, exact=False)
        return c


@pytest.fixture
def hw():
    yield sf.device("atreides-hw")
    sf.HardwareAtreides.attach(None)


@sf.jit
def axpy(a, x_ptr, y_ptr, out_ptr, n, BLOCK: sf.constexpr):
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    m = offs < n
    sf.store(out_ptr + offs, sf.fma(a, sf.load(x_ptr + offs, mask=m), sf.load(y_ptr + offs, mask=m)), mask=m)


def _axpy_inputs(dev_name, n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = sf.asarray(rng.uniform(-0.9, 0.9, n), dtype=sf.sf16, device=dev_name)
    y = sf.asarray(rng.uniform(-0.9, 0.9, n), dtype=sf.sf16, device=dev_name)
    return x, y, sf.zeros((n,), device=dev_name)


def _host_axpy_events(x, y, a=0.9):
    with sf.errstate(all="ignore"):
        ref = sf.fma(sf.sf16(a), x.to("host"), y.to("host"))
    hx, hy = x.to("host").codes, y.to("host").codes
    prod = (sf.sf16(a).code * hx) // 32768
    prod = np.where((sf.sf16(a).code * hx < 0) & ((sf.sf16(a).code * hx) % 32768 != 0), prod + 1, prod)
    return ref, int(np.count_nonzero(np.abs(prod + hy) > 32767))


# --- 1. a driver can say "I don't know" ------------------------------------------------------------

def test_driver_without_counts_gives_unknown_status_not_exact_zero(hw):
    sf.HardwareAtreides.attach(V1Driver("none"))
    x, y, out = _axpy_inputs("atreides-hw")
    ref, expected_sat = _host_axpy_events(x, y)
    assert expected_sat > 0                                    # the case the report hit
    with warnings.catch_warnings():
        warnings.simplefilter("error")                         # nothing claimed, nothing raised
        r = axpy[(sf.cdiv(200, 64),)](sf.sf16(0.9), x, y, out, 200, BLOCK=64)
    assert r.status.saturated is None and r.status.underflow is None
    assert r.status.exact is False and r.status.ok is None and not r.status.known
    assert np.array_equal(out.codes, ref.codes)


def test_replay_recovers_exact_counts_and_raises(hw):
    sf.HardwareAtreides.attach(V1Driver("none"))
    x, y, out = _axpy_inputs("atreides-hw")
    _, expected_sat = _host_axpy_events(x, y)
    with sf.config(status="replay"):
        with pytest.raises(sf.SFOverflowError, match=f"saturation in {expected_sat} lane"):
            axpy[(sf.cdiv(200, 64),)](sf.sf16(0.9), x, y, out, 200, BLOCK=64)
        with sf.errstate(over="ignore"):
            r = axpy[(sf.cdiv(200, 64),)](sf.sf16(0.9), x, y, out, 200, BLOCK=64)
    assert r.status.saturated == expected_sat and r.status.exact and r.status.underflow is not None


def test_replay_catches_wrong_bits_from_the_target(hw):
    sf.HardwareAtreides.attach(V1Driver("none", corrupt=True))
    x, y, out = _axpy_inputs("atreides-hw")
    with sf.config(status="replay"), sf.errstate(all="ignore"):
        with pytest.raises(sf.SFDeviceError, match="differ from the host reference"):
            axpy[(sf.cdiv(200, 64),)](sf.sf16(0.9), x, y, out, 200, BLOCK=64)


def test_driver_with_a_bound_warns_like_a_scan(hw):
    sf.HardwareAtreides.attach(V1Driver("bound"))
    x, y, out = _axpy_inputs("atreides-hw")
    with pytest.warns(sf.SFDeviceOverflowWarning, match="<="):
        r = axpy[(sf.cdiv(200, 64),)](sf.sf16(0.9), x, y, out, 200, BLOCK=64)  # over="raise" only warns
    assert r.status.exact is False and r.status.saturated > 0


def test_simulator_status_is_still_exact():
    dev = sf.device("atreides-sim")
    x, y, out = _axpy_inputs("atreides-sim")
    _, expected_sat = _host_axpy_events(x, y)
    with sf.errstate(over="ignore", under="ignore"):
        r = axpy[(sf.cdiv(200, 64),)](sf.sf16(0.9), x, y, out, 200, BLOCK=64)
    assert r.status.exact and r.status.saturated == expected_sat and r.status.underflow is not None
    assert dev.last is r


# --- 2. underflow is part of a heavy op's status ---------------------------------------------------

def test_device_underflow_is_counted_with_flags():
    a = sf.asarray(np.array([0.001, 0.5, 0.002, 0.3]), dtype=sf.sf16, device="atreides-flags")
    b = sf.asarray(np.array([0.001, 0.5, 0.003, 0.3]), dtype=sf.sf16, device="atreides-flags")
    with pytest.warns(sf.SFUnderflowWarning, match="underflow in 2 element"):   # default: warn
        r = a * b
    assert r.status.underflow == 2 and r.status.exact and r.status.ok is False
    with sf.errstate(under="raise"):
        with pytest.raises(sf.SFUnderflowError):
            a * b
    with sf.errstate(under="ignore"):
        assert (a * b).codes.tolist()[0] == 0


def test_device_underflow_is_unknown_without_flags_and_zero_where_impossible():
    a = sf.asarray(np.array([0.001, 0.25]), dtype=sf.sf16, device="atreides")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = a * a
    assert m.status.underflow is None and m.status.ok is None
    s = a + a
    assert s.status.underflow == 0
    with sf.config(status="replay"), sf.errstate(under="ignore"):
        assert (a * a).status.underflow == 1


def test_underflow_in_matmul_status():
    A = sf.asarray(np.full((2, 2), 0.004), dtype=sf.sf16, device="atreides-flags")
    with sf.errstate(under="ignore"):
        st = (A @ A).status
    assert st.underflow == 4 and st.exact
    assert "underflow=4" in repr(st)


# --- 3. the C build cache is safe across processes --------------------------------------------------

@pytest.mark.skipif(not __import__("sf.kernels._cgen", fromlist=["x"]).compiler(), reason="no C compiler")
def test_c_cache_concurrent_builds(tmp_path):
    script = textwrap.dedent(f"""
        import sys, numpy as np, sf
        from sf.kernels import _cgen
        _cgen._CACHE_DIR = {str(tmp_path)!r}
        @sf.jit
        def k(x, out, BLOCK: sf.constexpr):
            o = sf.arange(0, BLOCK)
            sf.store(out + o, sf.load(x + o) * 0.5 + 0.125)
        dev = sf.device("atreides-sim")
        x = dev.upload(np.full(64, 0.25)); out = dev.alloc(64)
        with sf.config(kernel_backend="c"):
            k[(1,)](x, out, BLOCK=64)
        assert out.codes.tolist() == [8192] * 64, out.codes[:4]
    """)
    procs = [subprocess.Popen([sys.executable, "-c", script], stderr=subprocess.PIPE) for _ in range(12)]
    errs = [p.communicate()[1].decode() for p in procs]
    assert all(p.returncode == 0 for p in procs), "\n".join(e for e in errs if e)[-2000:]
    names = os.listdir(tmp_path)
    assert len([n for n in names if n.endswith(".so")]) == 1
    assert not [n for n in names if n.startswith(".")]                 # no temp files left behind


# --- 5. a layer's saturation is the union of its steps --------------------------------------------

def _layer(dev):
    from sf.torch import _core
    rng = np.random.default_rng(0)
    x, W, b = rng.normal(0, 3, (16, 32)), rng.normal(0, 1, (8, 32)), rng.normal(0, 2, 8)
    runner = None if dev == "host" else _core.DeviceRunner(dev)
    return _core.linear(x, W, b, 16, headroom="none", runner=runner)


def test_layer_saturation_matches_host_with_flags():
    pytest.importorskip("torch")
    host, flags = _layer("host"), _layer("atreides-flags")
    assert host.saturated == 12 and host.exact
    assert flags.saturated == host.saturated and flags.exact           # was 15, "exact"
    assert np.array_equal(flags.y, host.y)


def test_layer_saturation_from_replay_matches_host():
    pytest.importorskip("torch")
    with sf.config(status="replay"):
        r = _layer("atreides")
    assert r.saturated == 12 and r.exact


def test_merged_counts_are_not_called_exact():
    from sf.torch._core import _merge
    assert _merge(5, 4) == (9, False)                                  # may overlap: a bound
    assert _merge(5, 0) == (5, True) and _merge(0, 4) == (4, True)
    assert _merge(None, 3) == (None, False)
    m, ex = _merge(np.array([1, 1, 0], bool), np.array([1, 0, 1], bool))
    assert m.tolist() == [True, True, True] and ex


# --- v1 has no divide or square root: targets declare their ops -------------------------------------

class V1Mock(_device.MockAtreides):
    name = "atreides-v1-ops"
    supported_ops = frozenset({"add", "sub", "mul", "fma", "neg", "abs", "sum", "matmul", "pow"})


@pytest.fixture
def v1():
    dev = V1Mock()
    _device.register_device(dev)
    yield dev
    _device._registry.pop(dev.name, None)


def test_unsupported_ops_are_refused_before_launch(v1):
    a = sf.asarray(np.array([0.25, 0.5]), dtype=sf.sf16, device=v1.name)
    b = sf.asarray(np.array([0.5, 0.75]), dtype=sf.sf16, device=v1.name)
    n0 = v1.launches
    with pytest.raises(sf.SFUnsupportedError, match="no instruction on atreides-v1-ops"):
        a / b
    with pytest.raises(sf.SFUnsupportedError, match="sqrt"):
        sf.sqrt(a)
    assert v1.launches == n0
    with sf.errstate(over="ignore"):
        assert (a + b).device == v1.name and v1.launches == n0 + 1


def test_known_answers_still_come_first_on_a_target_without_divide(v1):
    a = sf.asarray(np.array([0.25, 0.5]), dtype=sf.sf16, device=v1.name)
    z = sf.asarray(np.array([0.0, 0.5]), dtype=sf.sf16, device=v1.name)
    with pytest.raises(sf.SFZeroDivisionError):
        a / z
