"""Milestone 1.1.3: matmul, the device path, pre-flight and result.status.

Unit: shape errors on the host before launch; status counts on inputs known
to saturate.
Mix/regression: matmul equals per-output dot; host and the target are
bit-identical on the same inputs.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import sf
from conftest import dtype_of

DEV = "atreides"


def launches():
    return sf.get_device(DEV).launches


def rand(shape, lo, hi, seed, dtype=sf.sf16, device="host"):
    v = np.random.default_rng(seed).uniform(lo, hi, shape)
    with sf.errstate(under="ignore"):
        return sf.asarray(v, dtype=dtype, device=device)


# --- matmul semantics on the host --------------------------------------------------

@pytest.mark.parametrize("bits", [4, 8, 16])
def test_matmul_equals_per_output_dot(bits):
    dt = dtype_of(bits)
    m = sf.info(dt).max_code
    rng = np.random.default_rng(bits)
    for (M, K, N) in [(1, 1, 1), (2, 3, 4), (5, 7, 3), (4, 40, 4)]:
        for scale in (m // 8, m):
            a = sf.from_codes(rng.integers(-scale, scale + 1, (M, K)), dt)
            b = sf.from_codes(rng.integers(-scale, scale + 1, (K, N)), dt)
            with sf.errstate(all="ignore"):
                c = a @ b
                assert c.shape == (M, N) and c.dtype is dt
                for i in range(M):
                    for j in range(N):
                        assert int(c.codes[i, j]) == sf.dot(a[i], b[:, j]).code
                assert sf.matmul(a, b).codes.tolist() == c.codes.tolist()
                assert sf.addmm(a, b, 0).codes.tolist() == c.codes.tolist()
                d = sf.from_codes(rng.integers(-scale, scale + 1, (N,)), dt)
                assert sf.addmm(a, b, d).codes.tolist() == (c + d).codes.tolist()


@sf.errstate(under="ignore")
def test_matmul_vector_forms_and_batching():
    rng = np.random.default_rng(1)
    A = sf.from_codes(rng.integers(-2000, 2000, (3, 4)), sf.sf16)
    B = sf.from_codes(rng.integers(-2000, 2000, (4, 5)), sf.sf16)
    v = sf.from_codes(rng.integers(-2000, 2000, (4,)), sf.sf16)
    u = sf.from_codes(rng.integers(-2000, 2000, (3,)), sf.sf16)
    assert (A @ v).shape == (3,) and (A @ v).codes.tolist() == [sf.dot(A[i], v).code for i in range(3)]
    assert (u @ A).shape == (4,) and (u @ A).codes.tolist() == [sf.dot(u, A[:, j]).code for j in range(4)]
    assert isinstance(v @ v, sf.SFScalar) and (v @ v).code == sf.dot(v, v).code
    Ab = sf.from_codes(rng.integers(-2000, 2000, (6, 3, 4)), sf.sf16)
    Bb = sf.from_codes(rng.integers(-2000, 2000, (6, 4, 5)), sf.sf16)
    C = Ab @ Bb
    assert C.shape == (6, 3, 5)
    for k in range(6):
        assert C[k].codes.tolist() == (Ab[k] @ Bb[k]).codes.tolist()
    Cb = Ab @ B                                               # broadcast the right operand
    assert Cb.shape == (6, 3, 5) and Cb[2].codes.tolist() == (Ab[2] @ B).codes.tolist()
    assert (A @ sf.array([[0.5], [0.25], [0.125], [0]], dtype=sf.sf8)).dtype is sf.sf16


def test_shape_errors_before_launch():
    n0 = launches()
    A = sf.zeros((3, 4), device=DEV)
    for bad in (sf.zeros((3, 5), device=DEV), sf.zeros((5,), device=DEV), sf.zeros((2, 3, 5), device=DEV)):
        with pytest.raises(sf.SFShapeError):
            A @ bad
    with pytest.raises(sf.SFShapeError):
        sf.zeros((2, 3, 4), device=DEV) @ sf.zeros((3, 4, 5), device=DEV)   # batch mismatch
    with pytest.raises(sf.SFShapeError):
        A + sf.zeros((4, 3), device=DEV)
    with pytest.raises(sf.SFShapeError):
        sf.sf16(0.5) @ A
    assert launches() == n0                                   # nothing reached the device


# --- placement and the device interface ---------------------------------------------

def test_placement_and_transfer():
    x = sf.array([0.5, 0.25], dtype=sf.sf16, device=DEV)
    assert x.device == DEV and x.status is None and x.to("host").device == "host"
    assert x.numpy().tolist() == [0.5, 0.25] and x[0] == 0.5 and x.codes.tolist() == [16384, 8192]
    y = x.to("host").to(DEV)
    assert y.device == DEV and y.codes.tolist() == x.codes.tolist()
    with pytest.raises(sf.SFUnsupportedError):
        sf.array([0.5], dtype=sf.sf8, device=DEV)              # only sf16 on Atreides v1
    with pytest.raises(sf.SFUnsupportedError):
        sf.array([0.5], dtype=sf.sf8).to(DEV)
    with pytest.raises(sf.SFUnsupportedError):
        sf.array([0.5], dtype=sf.sf16, device="nowhere")
    assert "atreides" in sf.devices() and "host" in sf.devices()
    assert repr(sf.get_device(DEV)) == "<sf device atreides>"
    assert "device='atreides'" in repr(x)


def test_heavy_ops_bit_identical_to_host():
    """The gate for 1.1: host and target agree on every op, including on
    inputs that saturate, underflow and divide by zero."""
    rng = np.random.default_rng(3)
    shapes = [(64,), (8, 9), (3, 4, 5)]
    for shape in shapes:
        ka = rng.integers(-32767, 32768, shape)
        kb = rng.integers(-32767, 32768, shape)
        kb.reshape(-1)[0] = 0
        a_h, b_h = sf.from_codes(ka, sf.sf16), sf.from_codes(kb, sf.sf16)
        a_d, b_d = a_h.to(DEV), b_h.to(DEV)
        with sf.errstate(all="ignore"), sf.config(preflight="ignore"):
            for fn in (lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / b,
                       lambda a, b: sf.fma(a, b, b), lambda a, b: -a, lambda a, b: abs(a),
                       lambda a, b: sf.sqrt(a), lambda a, b: a ** 3, lambda a, b: a.sum(),
                       lambda a, b: a.sum(axis=0), lambda a, b: a + 0.25, lambda a, b: 0.5 * a):
                h = fn(a_h, b_h)
                d = fn(a_d, b_d)
                hc = h.codes if isinstance(h, sf.SFArray) else np.asarray(h.code)
                dc = d.codes if isinstance(d, sf.SFArray) else np.asarray(d.code)
                assert np.array_equal(hc, dc)
                assert isinstance(d, sf.SFArray) and d.device == DEV and d.status is not None
    for (M, K, N) in [(4, 4, 4), (16, 64, 8), (1, 300, 1)]:
        a = rand((M, K), -0.9, 0.9, M)
        b = rand((K, N), -0.9, 0.9, N + 10)
        with sf.errstate(all="ignore"), sf.config(preflight="ignore"):
            h = a @ b
            d = a.to(DEV) @ b.to(DEV)
            assert np.array_equal(h.codes, d.codes) and d.status.saturated >= 0
            v = a[0]
            assert sf.dot(v.to(DEV), b[:, 0].to(DEV)).codes.tolist() == sf.dot(v, b[:, 0]).code
            assert np.array_equal((a.to(DEV) @ b).codes, h.codes)          # mixed placement is heavy


# --- status --------------------------------------------------------------------------

def test_status_fields_and_policy():
    a = sf.array([0.75, 0.25, -0.75, 0.5], dtype=sf.sf16, device=DEV)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        y = a + a
    assert y.codes.tolist() == [32767, 16384, -32767, 32767]       # hardware value, no raise
    assert y.status == sf.SFStatus(saturated=3, div0=0, invalid=0, exact=False)
    assert repr(y.status) == "SFStatus(saturated<=3, div0=0, invalid=0, exact=False)"
    assert len(w) == 1 and issubclass(w[0].category, sf.SFDeviceOverflowWarning)
    assert not y.status.ok
    # v1 cannot raise on saturation (upper bound), even with over='raise'
    with sf.errstate(over="raise"):
        with pytest.warns(sf.SFDeviceOverflowWarning):
            a + a
    with sf.errstate(over="ignore"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            (a + a)
    # the count is an upper bound: an exact ±max is legal
    mx = sf.full((2,), sf.info(sf.sf16).max, sf.sf16, device=DEV)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        z = mx * sf.full((2,), sf.info(sf.sf16).max, sf.sf16, device=DEV)
    assert z.status.saturated == 0 and z.status.exact                   # mul: proven safe
    with pytest.warns(sf.SFDeviceOverflowWarning, match="<=1"):
        s = sf.array([0.25, -0.5], dtype=sf.sf16, device=DEV) + \
            sf.from_codes(np.array([32767 - 8192, 100]), sf.sf16, device=DEV)
    assert s.codes.tolist() == [32767, -16384 + 100]
    # an exact ±max from a non-saturating add is still counted (upper bound)
    assert s.status.saturated == 1 and not s.status.exact
    # div0 and invalid are exact, read off the inputs, and honour the policy fully
    b = sf.array([0.5, 0, 0.25, 0], dtype=sf.sf16, device=DEV)
    n0 = launches()
    with pytest.raises(sf.SFZeroDivisionError, match="status.div0"):
        a / b                                                      # known on the host, before launch
    assert launches() == n0
    a2 = sf.array([0.75, 0.25, -0.1, 0.5], dtype=sf.sf16, device=DEV)
    with sf.errstate(divide="ignore"):
        with pytest.warns(sf.SFDeviceOverflowWarning):            # 0.75 / 0.5 saturates
            q = a2 / b
    assert q.status.div0 == 2 and q.codes.tolist()[1] == 32767 and q.codes.tolist()[3] == 32767
    n = sf.array([0.25, -0.25], dtype=sf.sf16, device=DEV)
    with pytest.raises(sf.SFInvalidError, match="status.invalid"):
        sf.sqrt(n)
    with sf.errstate(invalid="ignore"):
        r = sf.sqrt(n)
    assert r.status == sf.SFStatus(saturated=0, div0=0, invalid=1, exact=True)
    assert r.codes.tolist() == [16384, 0]
    # spec section 6 example shape
    h = rand((16, 32), -0.3, 0.3, 1, device=DEV)
    w_ = rand((32, 8), -0.3, 0.3, 2, device=DEV)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y = h @ w_
    assert isinstance(y.status, sf.SFStatus) and y.numpy().shape == (16, 8) and y.device == DEV


# --- pre-flight: decide on the host, launch only the undecidable ------------------------------

def test_preflight_safe_skips_scan_and_marks_exact():
    a = sf.array([0.25, 0.125], dtype=sf.sf16, device=DEV)
    y = a + a
    assert y.status == sf.SFStatus(0, 0, 0, exact=True)
    small = sf.full((4, 4), 0.01, sf.sf16, device=DEV)
    z = small @ small                                                  # K max|A| max|B| <= max
    assert z.status.exact and z.status.saturated == 0
    assert (small.sum()).status.exact


def test_preflight_refuses_certain_saturation_without_launch():
    n0 = launches()
    big = sf.full((4, 4), 0.9, sf.sf16, device=DEV)
    with pytest.raises(sf.SFOverflowError, match="pre-flight"):
        big + big                                                   # every element overflows
    with pytest.raises(sf.SFOverflowError, match="not launched"):
        big @ big
    with pytest.raises(sf.SFOverflowError, match="pre-flight"):
        big / sf.full((4, 4), 0.5, sf.sf16, device=DEV)             # |a| >= |b| everywhere
    assert launches() == n0                                          # the chip never saw them
    with sf.config(preflight="warn"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = big + big
        assert [x.category for x in w] == [sf.SFOverflowWarning, sf.SFDeviceOverflowWarning]
        assert "pre-flight" in str(w[0].message)
        assert launches() == n0 + 1 and y.codes.tolist() == [[32767] * 4] * 4
    with sf.config(preflight="ignore"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            y = big + big
        assert launches() == n0 + 2 and y.status.saturated == 16
        assert all(issubclass(x.category, sf.SFDeviceOverflowWarning) for x in w)
    assert sf.getconfig()["preflight"] == "raise"


def test_preflight_probe_can_be_disabled():
    rng = np.random.default_rng(9)
    a = sf.asarray(rng.uniform(0.6, 0.9, (8, 8)), dtype=sf.sf16, device=DEV)
    b = sf.asarray(rng.uniform(0.6, 0.9, (8, 8)), dtype=sf.sf16, device=DEV)
    n0 = launches()
    with pytest.raises(sf.SFOverflowError, match="golden-vector probe"):
        a @ b
    assert launches() == n0
    with sf.config(probe=False):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            y = a @ b
        assert launches() == n0 + 1 and y.status.saturated == 64
    with pytest.raises(ValueError):
        sf.config(preflight="maybe")
    with pytest.raises(ValueError):
        sf.config(bogus=1)


def test_mixed_sign_inputs_are_not_certain():
    """Only the host-provable cases are refused; mixed data launches."""
    a = sf.array([0.9, -0.9, 0.9, -0.9], dtype=sf.sf16, device=DEV)
    b = sf.array([0.9, 0.9, -0.9, -0.9], dtype=sf.sf16, device=DEV)
    n0 = launches()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y = a + b
    assert launches() == n0 + 1 and y.codes.tolist() == [32767, 0, 0, -32767]
    assert y.status.saturated == 2


def test_offload_threshold():
    n0 = launches()
    a = sf.array([0.25] * 10, dtype=sf.sf16)
    with sf.config(offload_threshold=4):
        y = a + a
    assert launches() == n0 + 1 and y.device == DEV and y.status is not None
    z = a + a
    assert launches() == n0 + 1 and z.device == "host"


def test_register_device():
    class Flagged(sf.MockAtreides):
        name = "atreides-flags"
        sticky_flags = True

    sf.register_device(Flagged())
    try:
        a = sf.array([0.75, 0.25], dtype=sf.sf16, device="atreides-flags")
        with pytest.raises(sf.SFOverflowError):                 # with flags, exact, so raise is honoured
            a + a
        with sf.errstate(over="ignore"):
            y = a + a
        assert y.status.exact and y.status.saturated == 1
    finally:
        sf.get_device("atreides-flags")   # still registered; harmless
