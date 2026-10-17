"""Milestone 1.0.3: mul and fma.

Unit: mul never overflows (exhaustive); underflow threshold; truncation is
sign-symmetric.
Mix/regression: a * b + c equals sf.fma(a, b, c) bit for bit; precedence
a + b * c; record, never assert, where a * (b + c) differs from
a * b + a * c.
"""
from __future__ import annotations

import operator
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import sf
from conftest import EVENT_ERROR, dtype_of, pairs, run, singles
from oracle import Fmt


def test_mul_exhaustive_or_sampled(width, dt, oracle):
    """Value and event agree with the oracle on every pair; mul never overflows."""
    lsb, scale = oracle.lsb, 2 ** (width - 1)
    for k_a, k_b in pairs(width):
        a, b = sf.from_code(k_a, dt), sf.from_code(k_b, dt)
        expect, ev = oracle.mul(k_a * lsb, k_b * lsb)
        r, raised = run(operator.mul, a, b)
        assert ev != "overflow"
        assert raised is EVENT_ERROR[ev], (a, b)
        assert type(r) is dt and r.exact == expect, (a, b)
        # underflow exactly when a, b != 0 and |k_a k_b| < 2^(N-1)
        assert (ev == "underflow") == (k_a != 0 and k_b != 0 and abs(k_a * k_b) < scale)
        # truncation toward zero: |r| <= |a b| < |r| + lsb
        assert abs(r.exact) <= abs(a.exact * b.exact) < abs(r.exact) + lsb


def test_mul_is_sign_symmetric(width, dt):
    with sf.errstate(under="ignore"):
        for k_a, k_b in pairs(width):
            a, b = sf.from_code(k_a, dt), sf.from_code(k_b, dt)
            r = (a * b).code
            assert ((-a) * b).code == -r
            assert (a * (-b)).code == -r
            assert ((-a) * (-b)).code == r
            assert (b * a).code == r


def test_underflow_threshold(width, dt):
    i = sf.info(dt)
    scale = 2 ** (width - 1)
    # the smallest nonzero product that survives is 2^(N-1) in code units
    for k_a in singles(width):
        if k_a == 0:
            continue
        # largest |k_b| that still underflows against k_a
        kb = (scale - 1) // abs(k_a)
        if 1 <= kb <= i.max_code:
            with pytest.raises(sf.SFUnderflowError):
                with sf.errstate(under="raise"):
                    sf.from_code(k_a, dt) * sf.from_code(kb, dt)
            with pytest.warns(sf.SFUnderflowWarning):            # default policy warns
                assert (sf.from_code(k_a, dt) * sf.from_code(kb, dt)).code == 0
        if kb + 1 <= i.max_code:
            assert (sf.from_code(k_a, dt) * sf.from_code(kb + 1, dt)).code != 0
    if width == 16:
        with pytest.warns(sf.SFUnderflowWarning):
            assert (sf.sf16(0.005) * 0.005).code == 0                  # the spec's example
        assert (sf.sf16(0.005) * 0.01).code == 1                      # 0.00005 * 32768 ≈ 1.6 -> 1


def test_mul_by_zero_and_literals(width, dt):
    for k in singles(width):
        a = sf.from_code(k, dt)
        assert (a * 0).code == 0 and (0 * a).code == 0 and (a * sf.from_code(0, dt)).code == 0
    x = sf.sf16(0.5)
    assert (x * 0.5).code == 0x2000 and (0.5 * x).code == 0x2000
    assert (np.float32(0.5) * x).code == 0x2000
    assert type(sf.sf8(0.5) * x) is sf.sf16
    with pytest.raises(sf.SFOverflowError, match="literal"):
        x * 2                                              # 2 is not representable
    with pytest.raises(sf.SFOverflowError, match="literal"):
        x * 1
    y = sf.sf16(0.5)
    y *= 0.25
    assert y.code == 0x1000


# --- fma ---------------------------------------------------------------------------

def test_fma_equals_mul_then_add(width, dt, oracle):
    """a * b + c gives the same bits as sf.fma(a, b, c), and the same event."""
    m = sf.info(dt).max_code
    rnd = np.random.default_rng(300 + width)
    n = 4000 if width >= 11 else 20000
    triples = [tuple(int(v) for v in rnd.integers(-m, m + 1, 3)) for _ in range(n)]
    corners = [-m, -1, 0, 1, m]
    triples += [(a, b, c) for a in corners for b in corners for c in corners]
    for k_a, k_b, k_c in triples:
        a, b, c = (sf.from_code(k, dt) for k in (k_a, k_b, k_c))
        expect, ev = oracle.fma(a.exact, b.exact, c.exact)
        r1, e1 = run(lambda: a * b + c)
        r2, e2 = run(sf.fma, a, b, c)
        assert r1.code == r2.code and r1.exact == expect
        assert e1 is e2 is EVENT_ERROR[ev]
        assert type(r2) is dt


def test_fma_signature_and_literals():
    a, b, c = sf.sf16(0.5), sf.sf16(0.25), sf.sf16(0.125)
    assert sf.fma(a, b, c).code == (a * b + c).code
    assert sf.fma(a, 0.25, 0.125).code == sf.fma(0.5, b, 0.125).code == sf.fma(0.5, 0.25, c).code
    assert type(sf.fma(sf.sf8(0.5), a, 0)) is sf.sf16
    with pytest.raises(sf.SFTypeError):
        sf.fma(0.5, 0.25, 0.125)              # no SF operand to fix the type
    with pytest.raises(sf.SFOverflowError):
        sf.fma(a, 0.5, 0.75)                  # 0.25 + 0.75 = 1.0
    with sf.errstate(over="ignore"):
        assert sf.fma(a, 0.5, 0.75).code == 32767
    with pytest.warns(sf.SFUnderflowWarning):
        sf.fma(sf.sf16(0.001), 0.001, 0.5)    # product underflows, add is fine


def test_precedence():
    a, b, c = sf.sf16(0.5), sf.sf16(0.25), sf.sf16(0.125)
    assert (a + b * c).code == (a + (b * c)).code
    assert (a * b + c).code == ((a * b) + c).code
    assert (a - b * c).code == (a - (b * c)).code
    assert (-a * b).code == ((-a) * b).code == -(a * b).code
    assert (a + b * c).code == 0x4000 + 0x400
    assert (a * b * c).code == ((a * b) * c).code            # left associative


def test_distributivity_recorded_not_asserted(width, dt, capsys):
    """Spec: record where a * (b + c) differs from a * b + a * c; never assert."""
    m = sf.info(dt).max_code
    rnd = np.random.default_rng(400 + width)
    n = 3000
    differs = 0
    with sf.errstate(all="ignore"):
        for _ in range(n):
            a, b, c = (sf.from_code(int(k), dt) for k in rnd.integers(-m, m + 1, 3))
            if (a * (b + c)).code != (a * b + a * c).code:
                differs += 1
    print(f"sf{width}: a*(b+c) != a*b + a*c in {differs}/{n} random triples")
    assert differs >= 0                                       # recorded only


@settings(max_examples=500, deadline=None)
@given(bits=st.integers(4, 16), ka=st.integers(-32767, 32767), kb=st.integers(-32767, 32767))
def test_mul_hypothesis(bits, ka, kb):
    dt, fmt = dtype_of(bits), Fmt(bits)
    m = sf.info(dt).max_code
    ka, kb = ka % (2 * m + 1) - m, kb % (2 * m + 1) - m
    a, b = sf.from_code(ka, dt), sf.from_code(kb, dt)
    expect, ev = fmt.mul(a.exact, b.exact)
    r, raised = run(operator.mul, a, b)
    assert r.exact == expect and raised is EVENT_ERROR[ev]
    assert (b * r.dtype(0)).code == 0


# --- integer powers ---------------------------------------------------------------------

def test_pow():
    x = sf.sf16(0.5)
    assert (x ** 1).code == x.code and (x ** 2).code == 0x2000 and (x ** 3).code == 0x1000
    assert (sf.sf16(-0.5) ** 2).code == 0x2000 and (sf.sf16(-0.5) ** 3).code == -0x1000
    assert type(x ** 2) is sf.sf16
    with pytest.warns(sf.SFUnderflowWarning):
        assert (x ** 20).code == 0
    with pytest.raises(sf.SFOverflowError):
        x ** 0                                               # 1 is not representable
    for bad in (-1, 0.5, 2.0, Fraction(1, 2), True):
        with pytest.raises(sf.SFUnsupportedError):
            x ** bad
    with pytest.raises(sf.SFUnsupportedError):
        2 ** x
    with pytest.raises(sf.SFUnsupportedError):
        pow(x, 2, 3)
    # x ** k is the left-to-right chain of multiplies
    y = sf.sf16(0.9)
    assert (y ** 4).code == (y * y * y * y).code
