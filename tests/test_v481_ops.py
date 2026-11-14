"""Atreides v4.8.1 has no SF divide and no square root (ISA section 5).

Every Atreides target refuses those two on device arrays before anything is
launched, and never falls back to the host silently.  Host arrays are not
affected, and a host op is never offloaded to a target that cannot run it.
"""
from __future__ import annotations

import numpy as np
import pytest

import sf
import sf.kernels  # noqa: F401  (registers atreides-sim and atreides-hw)
from sf import _device

ATREIDES = ("atreides", "atreides-flags", "atreides-sim", "atreides-hw")
V481_OPS = {"add", "sub", "mul", "fma", "neg", "abs", "pow", "sum", "matmul", "matmul_wide"}


def _pair(device):
    a = sf.array([0.25, 0.5], dtype=sf.sf16, device=device)
    b = sf.array([0.5, 0.75], dtype=sf.sf16, device=device)
    return a, b


@pytest.mark.parametrize("name", ATREIDES)
def test_every_atreides_target_declares_the_v481_ops(name):
    assert sf.get_device(name).supported_ops == V481_OPS


def test_the_declared_ops_are_the_built_ins_minus_divide_and_sqrt():
    assert set(_device._KERNELS) - V481_OPS == {"div", "sqrt"}


@pytest.mark.parametrize("name", ATREIDES)
def test_divide_and_sqrt_are_refused_before_launch(name):
    dev = sf.get_device(name)
    a, b = _pair(name)
    n0 = dev.launches
    with pytest.raises(sf.SFUnsupportedError, match=f"no instruction on {name}") as e:
        a / b
    assert "nothing was launched" in str(e.value) and "supported: abs, add" in str(e.value)
    with pytest.raises(sf.SFUnsupportedError, match="sqrt"):
        sf.sqrt(a)
    with pytest.raises(sf.SFUnsupportedError):
        sf.div(a, b, where=np.array([False, False]))        # even with nothing live
    assert dev.launches == n0


@pytest.mark.parametrize("name", ATREIDES)
def test_the_ops_the_chip_has_still_run(name):
    dev = sf.get_device(name)
    a, b = _pair(name)
    h_a, h_b = a.to("host"), b.to("host")
    n0 = dev.launches
    with sf.errstate(all="ignore"), sf.config(preflight="ignore"):
        for fn in (lambda x, y: x + y, lambda x, y: x - y, lambda x, y: x * y,
                   lambda x, y: sf.fma(x, y, y), lambda x, y: -x, lambda x, y: abs(x),
                   lambda x, y: x ** 3, lambda x, y: x.sum(), lambda x, y: sf.dot(x, y),
                   lambda x, y: sf.dot(x, y, accumulate="wide")):
            d, h = fn(a, b), fn(h_a, h_b)
            dc = d.codes if isinstance(d, sf.SFArray) else np.asarray(d.code)
            hc = h.codes if isinstance(h, sf.SFArray) else np.asarray(h.code)
            assert np.array_equal(dc, hc)
    assert dev.launches == n0 + 10


def test_known_answers_still_come_first():
    """A divide by zero or a square root of a negative is reported as that,
    because it is read off the inputs before the target is asked."""
    a, _ = _pair("atreides")
    z = sf.array([0.0, 0.5], dtype=sf.sf16, device="atreides")
    with pytest.raises(sf.SFZeroDivisionError, match="status.div0"):
        a / z
    with pytest.raises(sf.SFInvalidError, match="status.invalid"):
        sf.sqrt(-a)


def test_host_arrays_are_not_affected():
    a, b = _pair("host")
    q, r = a / b, sf.sqrt(a)
    assert q.device == "host" and q.codes.tolist() == [16384, 21845]
    assert r.device == "host" and r.codes.tolist() == [16384, 23170]


def test_a_host_op_is_not_offloaded_to_a_target_that_cannot_run_it():
    dev = sf.get_device("atreides")
    a, b = _pair("host")
    want_q, want_r = (a / b).codes.tolist(), sf.sqrt(a).codes.tolist()
    n0 = dev.launches
    with sf.config(offload_threshold=1):                    # default_device is atreides
        q, r = a / b, sf.sqrt(a)
        assert q.device == "host" and q.codes.tolist() == want_q
        assert r.device == "host" and r.codes.tolist() == want_r
        assert dev.launches == n0                            # stayed on the host
        with sf.errstate(all="ignore"):
            assert (a + b).device == "atreides"              # an op the chip has still offloads
        assert dev.launches == n0 + 1


def test_a_host_wide_matmul_is_not_offloaded_to_a_target_without_mx():
    class NoMX(sf.MockAtreides):
        name = "no-mx"
        mx = False

    dev = NoMX()
    sf.register_device(dev)
    try:
        a = sf.array([[0.5, 0.25]], dtype=sf.sf16)
        b = sf.array([[0.5], [0.25]], dtype=sf.sf16)
        want = sf.matmul(a, b, accumulate="wide").codes.tolist()
        with sf.config(offload_threshold=0, default_device="no-mx"):
            y = sf.matmul(a, b, accumulate="wide")
        assert y.device == "host" and y.codes.tolist() == want and dev.launches == 0
    finally:
        _device._registry.pop("no-mx", None)
