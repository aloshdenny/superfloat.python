"""Regression tests for the independent Stage 1.1 review."""
from __future__ import annotations

import asyncio
import threading
import time
import warnings

import numpy as np
import pytest

import sf

DEV = "atreides"


def launches():
    return sf.get_device(DEV).launches


# P1: a probe hit is a suspicion, never a proof ------------------------------------

def test_probe_does_not_reject_valid_work():
    a = sf.from_codes(np.full(1000, 20000), sf.sf16, device=DEV)
    bc = np.full(1000, 20000)
    bc[500] = 0
    b = sf.from_codes(bc, sf.sf16, device=DEV)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = a + b                                           # launched, with a warning
    assert int(r.codes[500]) == 20000 and r.status.outputs_at_max == 999
    assert any(issubclass(x.category, sf.SFPreflightWarning) for x in w)
    with sf.config(suspect="verify"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            r = a + b                                       # replay: not every output, so launch
        assert int(r.codes[500]) == 20000
        assert not any(issubclass(x.category, sf.SFPreflightWarning) for x in w)
        mixed = sf.from_codes(np.where(np.arange(1000) % 2, 20000, -20000), sf.sf16, device=DEV)
        with pytest.raises(sf.SFOverflowError, match="host replay"):   # bound cannot prove, replay can
            mixed + mixed
    with sf.config(suspect="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("error", sf.SFPreflightWarning)
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always", sf.SFDeviceOverflowWarning)
                a + b
    with sf.config(suspect="raise"):
        n0 = launches()
        with pytest.raises(sf.SFOverflowError, match="suspected"):
            a + b
        assert launches() == n0


# P1: saturation status is honest in both directions -------------------------------

def test_reduction_saturation_is_unknown_without_flags():
    x = sf.array([0.75, 0.75, -0.75], dtype=sf.sf16, device=DEV)
    s = x.sum()                                             # two partial sums saturate, final is 8191
    assert s.codes.tolist() == 32767 - 24576 and s.status.saturated is None
    assert s.status.ok is None and not s.status.known and s.status.outputs_at_max == 0
    assert "unknown" in repr(s.status)
    y = sf.from_codes(np.array([32767, 0]), sf.sf16, device=DEV)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no device warning for an exact max
        s2 = y.sum()
    assert s2.status.saturated is None and s2.status.outputs_at_max == 1
    # dot and @ likewise
    with warnings.catch_warnings(), sf.config(suspect="ignore"):
        warnings.simplefilter("error", sf.SFDeviceOverflowWarning)
        d = sf.dot(x, sf.array([0.9, 0.9, 0.9], dtype=sf.sf16, device=DEV))
        m = x.reshape(1, 3) @ sf.full((3, 1), 0.9, sf.sf16, device=DEV)
    assert d.status.saturated is None and m.status.saturated is None
    # the range bound still gives exact zeros where it applies
    assert sf.full((4,), 0.1, sf.sf16, device=DEV).sum().status == \
        sf.SFStatus(saturated=0, div0=0, invalid=0, exact=True, outputs_at_max=0)


def test_replay_and_flags_give_exact_counts():
    x = sf.array([0.75, 0.75, -0.75], dtype=sf.sf16, device=DEV)
    with sf.config(status="replay"):
        with pytest.raises(sf.SFOverflowError):             # exact, so the policy is honoured
            x.sum()
        with sf.errstate(over="ignore"):
            s = x.sum()
        assert s.status == sf.SFStatus(saturated=1, div0=0, invalid=0, exact=True, outputs_at_max=0)
        assert sf.from_codes(np.array([32767, 0]), sf.sf16, device=DEV).sum().status.saturated == 0
    f = sf.array([0.75, 0.75, -0.75], dtype=sf.sf16, device="atreides-flags")
    with sf.errstate(over="ignore"):
        assert f.sum().status.saturated == 1 and f.sum().status.exact
        assert sf.from_codes(np.array([32767, 0]), sf.sf16, device="atreides-flags").sum().status.saturated == 0
    # a flagged device's flags come through Device.run, not from scanning the output
    codes, flags = sf.get_device("atreides-flags").run("add", sf.sf16._fmt,
                                                       np.array([30000, 1]), np.array([30000, 1]))
    assert flags["overflow"].tolist() == [True, False] and codes.tolist() == [32767, 2]
    assert sf.get_device(DEV).run("add", sf.sf16._fmt, np.array([1]), np.array([1]))[1] is None


def test_elementwise_scan_is_a_valid_upper_bound():
    a = sf.array([0.75, 0.25, -0.75], dtype=sf.sf16, device=DEV)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y = a + sf.from_codes(np.array([32767 - 24576, 0, -24576]), sf.sf16, device=DEV)
    # element 0 is an exact max (legal), element 2 saturated: the bound is 2, the truth is 1
    assert y.status.saturated == 2 and not y.status.exact and y.status.ok is False
    with sf.config(status="replay"), sf.errstate(over="ignore"):
        y = a + sf.from_codes(np.array([32767 - 24576, 0, -24576]), sf.sf16, device=DEV)
    assert y.status.saturated == 1 and y.status.exact


# P1: casts keep the sign of large inputs ----------------------------------------------

def test_array_casts_match_scalar_casts_on_extremes():
    values = [1e300, -1e300, 1e16, -1e16, 2.0, -2.0, 0.9999847412109375, -0.9999847412109375,
              0.9999847412109374, 0.75, 2 ** -17, -(2 ** -17), 2 ** -16, 3 * 2 ** -17, 0.0, -0.0,
              float("inf"), float("-inf")]
    with sf.errstate(all="ignore"):
        arr = sf.array(values, dtype=sf.sf16).codes.tolist()
        expect = [sf.sf16(v).code for v in values]
    assert arr == expect
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no NumPy RuntimeWarning leaks
        with sf.errstate(all="ignore"):
            sf.array([1e300, float("nan"), float("inf")], dtype=sf.sf16)
    with sf.errstate(all="ignore"):
        assert sf.array(np.array([2 ** 63, 2 ** 64 - 1, 0], dtype=np.uint64), dtype=sf.sf16).codes.tolist() \
            == [32767, 32767, 0]
        assert sf.array(np.array([-(2 ** 63), 2 ** 63 - 1], dtype=np.int64), dtype=sf.sf16).codes.tolist() \
            == [-32767, 32767]
        assert sf.array(np.array([True, False]), dtype=sf.sf8).codes.tolist() == [127, 0]
        assert sf.array(np.array([1e30, -1e30], dtype=np.float32), dtype=sf.sf16).codes.tolist() == [32767, -32767]
    for bits in (4, 8, 12, 16):
        dt = sf.dtypes[bits - 4]
        thr = 1 - 2.0 ** -bits
        with sf.errstate(all="ignore"):
            assert sf.array([thr, -thr, np.nextafter(thr, 0)], dtype=dt).codes.tolist() == \
                [sf.info(dt).max_code, -sf.info(dt).max_code, sf.info(dt).max_code]
        with pytest.raises(sf.SFOverflowError):
            sf.array([thr], dtype=dt)


def test_from_codes_validates_in_the_source_type():
    with pytest.raises(sf.SFInvalidError):
        sf.from_codes(np.array([2 ** 64 - 1], dtype=np.uint64), sf.sf16)
    with pytest.raises(sf.SFInvalidError):
        sf.from_codes(np.array([2 ** 63 - 1], dtype=np.int64), sf.sf16)
    with pytest.raises(sf.SFInvalidError):
        sf.from_codes(np.array([32768], dtype=np.uint16), sf.sf16)
    assert sf.from_codes(np.array([32767, 5], dtype=np.uint64), sf.sf16).codes.tolist() == [32767, 5]
    assert sf.from_codes(np.array([-7], dtype=np.int8), sf.sf4).codes.tolist() == [-7]
    assert sf.from_codes(np.array([], dtype=np.uint64), sf.sf16).size == 0


# P2: sf.config is reusable, nestable, async- and thread-safe ---------------------------------

def test_config_reuse_nesting_async_threads():
    c = sf.config(probe=False)
    c.__exit__(None, None, None)                            # undo the plain-call application
    assert sf.getconfig()["probe"] is True
    with c:
        assert sf.getconfig()["probe"] is False
        with c:
            with sf.config(probe_size=7):
                assert sf.getconfig() == {**sf.getconfig(), "probe": False, "probe_size": 7}
            assert sf.getconfig()["probe_size"] == 256
        assert sf.getconfig()["probe"] is False
    assert sf.getconfig()["probe"] is True

    @sf.config(preflight="warn")
    async def f():
        await asyncio.sleep(0)
        return sf.getconfig()["preflight"]

    async def main():
        r = await asyncio.gather(*(f() for _ in range(4)))
        return r, sf.getconfig()["preflight"]

    assert asyncio.run(main()) == (["warn"] * 4, "raise")

    @sf.config(preflight="warn")
    def g():
        time.sleep(0.003)
        return sf.getconfig()["preflight"]

    results, errors = [], []

    def worker():
        try:
            results.append(g())
        except BaseException as e:                          # noqa: BLE001
            errors.append(repr(e))
    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and results == ["warn"] * 8 and sf.getconfig()["preflight"] == "raise"
    with pytest.raises(TypeError):
        @sf.config(probe=False)
        def gen():
            yield 1
    es = sf.config(probe=False)
    es.__exit__(None, None, None)
    with pytest.raises(RuntimeError):
        es.__exit__(None, None, None)
    with pytest.raises(ValueError):
        sf.config(suspect="maybe")
    with pytest.raises(ValueError):
        sf.config(status="guess")
    with pytest.raises(ValueError):
        sf.config(offload_threshold=-1)


# P2: masked division is a heavy op on device arrays ---------------------------------------------

def test_masked_div_goes_through_the_device_path():
    x = sf.array([0.5, 0.25, 0.1], dtype=sf.sf16, device=DEV)
    y = sf.array([0.25, 0, 0.5], dtype=sf.sf16, device=DEV)
    n0 = launches()
    mask = np.array([False, False, True])
    r = sf.div(x, y, where=mask)
    assert launches() == n0 + 1 and r.device == DEV and r.status is not None
    assert r.codes.tolist() == [0, 0, (x[2] / y[2]).code] and r.status.ok
    # live elements still carry their events, masked ones never do
    with pytest.raises(sf.SFZeroDivisionError, match="status.div0"):
        sf.div(x, y, where=np.array([True, True, True]))
    with pytest.raises(sf.SFOverflowError, match="pre-flight"):     # 0.5 / 0.25 proven, all live
        sf.div(x, y, where=np.array([True, False, False]))
    assert launches() == n0 + 1
    # host form unchanged
    h = sf.div(x.to("host"), y.to("host"), where=mask)
    assert h.device == "host" and h.status is None and h.codes.tolist() == r.codes.tolist()
    with sf.config(offload_threshold=1):
        assert sf.div(x.to("host"), y.to("host"), where=mask).device == DEV


# P2: offloading host sum / dot / matmul uses the default device ---------------------------------

def test_offload_threshold_for_reductions():
    n0 = launches()
    with sf.config(offload_threshold=2):
        s = sf.array([0.1, 0.1, 0.1], dtype=sf.sf16).sum()
        d = sf.dot(sf.array([0.1, 0.1, 0.1], dtype=sf.sf16), sf.array([0.1, 0.1, 0.1], dtype=sf.sf16))
        m = sf.zeros((2, 2)) @ sf.zeros((2, 2))
        v = sf.zeros((3,)) @ sf.zeros((3, 2))
    assert launches() == n0 + 4
    assert all(r.device == DEV and r.status is not None for r in (s, d, m, v))
    assert s.codes.tolist() == sf.array([0.1, 0.1, 0.1], dtype=sf.sf16).sum().code
    # a device that rejects the dtype refuses before dispatch
    with sf.config(offload_threshold=1):
        with pytest.raises(sf.SFUnsupportedError):
            sf.array([0.1, 0.1], dtype=sf.sf8).sum()
        with pytest.raises(sf.SFUnsupportedError):
            sf.zeros((2, 2), sf.sf8) @ sf.zeros((2, 2), sf.sf8)
    assert launches() == n0 + 4
