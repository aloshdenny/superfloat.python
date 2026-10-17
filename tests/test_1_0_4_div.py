"""Milestone 1.0.4: div.

Unit: |a| >= |b| overflows, including a / a; b = 0 raises; 0 / b == 0;
sign rules; no underflow.
Mix/regression: random BODMAS trees over + - * / ( ) (test_bodmas.py);
x / (y - y) raises at the right op; |(a / b) * b - a| <= 1 LSB when |a| < |b|.
"""
from __future__ import annotations

import operator

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import sf
from conftest import EVENT_ERROR, dtype_of, pairs, run, singles
from oracle import Fmt


def test_div_exhaustive_or_sampled(width, dt, oracle):
    lsb = oracle.lsb
    m = sf.info(dt).max_code
    for k_a, k_b in pairs(width):
        a, b = sf.from_code(k_a, dt), sf.from_code(k_b, dt)
        expect, ev = oracle.div(k_a * lsb, k_b * lsb)
        r, raised = run(operator.truediv, a, b)
        assert raised is EVENT_ERROR[ev], (a, b)
        assert type(r) is dt and r.exact == expect, (a, b)
        if k_b == 0:
            assert ev == "divide"
            assert r.code == (0 if k_a == 0 else (m if k_a > 0 else -m))   # proposed device value
        elif k_a == 0:
            assert ev is None and r.code == 0                               # 0 / b == 0
        elif abs(k_a) >= abs(k_b):
            assert ev == "overflow"
            assert r.code == (m if (k_a > 0) == (k_b > 0) else -m)
        else:
            assert ev is None and r.code != 0                               # never underflows
            # sign rules and truncation toward zero
            assert (r.code > 0) == ((k_a > 0) == (k_b > 0))
            assert abs(r.exact) <= abs(a.exact / b.exact) < abs(r.exact) + lsb
            # |(a / b) * b - a| <= 1 LSB
            with sf.errstate(under="ignore"):
                back = r * b
            assert abs(back.exact - a.exact) <= lsb, (a, b, r, back)


def test_a_over_a_overflows(width, dt):
    m = sf.info(dt).max_code
    for k in singles(width):
        if k == 0:
            continue
        a = sf.from_code(k, dt)
        with pytest.raises(sf.SFOverflowError):
            a / a
        with sf.errstate(over="ignore"):
            assert (a / a).code == m                 # 1.0 saturates to max
            assert (a / -a).code == -m


def test_divide_by_zero(width, dt):
    z = dt(0)
    for k in singles(width):
        a = sf.from_code(k, dt)
        with pytest.raises(sf.SFZeroDivisionError):
            a / z
        with pytest.raises(sf.SFZeroDivisionError):
            a / 0
        with pytest.raises(sf.SFZeroDivisionError):
            a / 0.0
        with pytest.raises(sf.SFZeroDivisionError):
            sf.div(a, 0)
        with pytest.raises(ZeroDivisionError):           # base type, like Python
            a / z
    # a tiny literal divisor underflows to 0 first (warn), then divides by zero
    with sf.errstate(under="ignore"):
        with pytest.raises(sf.SFZeroDivisionError):
            dt(0.5) / 1e-12
    with sf.errstate(divide="warn"):
        with pytest.warns(sf.SFZeroDivisionWarning):
            assert (dt(0.5) / z).code == sf.info(dt).max_code
    with sf.errstate(divide="ignore"):
        assert (z / z).code == 0


def test_zero_numerator_and_sign_rules():
    a, b = sf.sf16(0.25), sf.sf16(0.5)
    assert (sf.sf16(0) / b).code == 0
    assert (0 / b).code == 0 and (0.0 / b).code == 0
    assert (a / b).code == 0x4000 and (-a / b).code == -0x4000
    assert (a / -b).code == -0x4000 and (-a / -b).code == 0x4000
    assert (0.25 / b).code == 0x4000 and (np.float32(0.25) / b).code == 0x4000
    assert type(sf.sf8(0.25) / b) is sf.sf16 and (sf.sf8(0.25) / b).code == 0x4000
    assert sf.div(a, b).code == (a / b).code and sf.div(0.25, b).code == 0x4000
    with pytest.raises(sf.SFTypeError):
        sf.div(0.25, 0.5)
    x = sf.sf16(0.25)
    x /= 0.5
    assert x.code == 0x4000
    # the spec's examples
    a, b = sf.sf16(0.5), sf.sf16(0.25)
    assert (b / a).code == 0x4000                      # sf16(0.5)
    with pytest.raises(sf.SFOverflowError):
        a / b                                          # 2.0 is outside sf16


def test_x_over_y_minus_y_raises_at_the_right_op():
    x, y = sf.sf16(0.5), sf.sf16(0.25)
    with pytest.raises(sf.SFZeroDivisionError):
        x / (y - y)
    # the subtraction itself is fine
    assert (y - y).code == 0
    # and a guarded form never raises
    assert (y / x if x != 0 else y).code == 0x4000


def test_truncation_toward_zero_examples():
    # 1/3 = 0.333... -> trunc; -1/3 -> -trunc (sign symmetric)
    a, b = sf.sf16(0.25), sf.sf16(0.75)
    q = a / b
    assert q.code == (0x2000 * 0x8000) // 0x6000 == 10922
    assert (-a / b).code == -10922 and (a / -b).code == -10922
    assert float(q) < 1 / 3 < float(q) + 2 ** -15


@settings(max_examples=500, deadline=None)
@given(bits=st.integers(4, 16), ka=st.integers(-32767, 32767), kb=st.integers(-32767, 32767))
def test_div_hypothesis(bits, ka, kb):
    dt, fmt = dtype_of(bits), Fmt(bits)
    m = sf.info(dt).max_code
    ka, kb = ka % (2 * m + 1) - m, kb % (2 * m + 1) - m
    a, b = sf.from_code(ka, dt), sf.from_code(kb, dt)
    expect, ev = fmt.div(a.exact, b.exact)
    r, raised = run(operator.truediv, a, b)
    assert r.exact == expect and raised is EVENT_ERROR[ev]


# --- sqrt (section 3.1 proposed semantics; needed for the invalid example) --------------

def test_sqrt(width, dt, oracle):
    m = sf.info(dt).max_code
    for k in singles(width):
        a = sf.from_code(k, dt)
        expect, ev = oracle.sqrt(a.exact)
        r, raised = run(sf.sqrt, a)
        assert raised is EVENT_ERROR[ev] and r.exact == expect
        if k < 0:
            assert ev == "invalid" and r.code == 0
        else:
            # monotonic, truncated, and sqrt(x) >= x on [0, 1)
            assert r.exact * r.exact <= a.exact < (r.exact + oracle.lsb) ** 2
            assert r.exact >= a.exact
    assert sf.sqrt(sf.from_code(0, dt)).code == 0
    assert sf.sqrt(sf.from_code(m, dt)).code == m           # isqrt((S-1) S) == S-1


def test_sqrt_examples():
    assert sf.sqrt(sf.sf16(0.25)).code == 0x4000
    with pytest.raises(sf.SFInvalidError):
        sf.sqrt(sf.sf16(-0.25))                         # the spec's example
    with pytest.raises(ValueError):
        sf.sqrt(sf.sf16(-0.25))
    with sf.errstate(invalid="ignore"):
        assert sf.sqrt(sf.sf16(-0.25)).code == 0
    with pytest.raises(sf.SFTypeError):
        sf.sqrt(0.25)
