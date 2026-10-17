"""Milestone 1.0.5: comparisons and conditionals.

Unit: comparisons are exact, even against out-of-range literals (x < 1 is
always True); total order on codes; -0 equals +0.
Mix/regression: ``a / b if b != 0 else a`` never raises; property tests over
the whole 1.0 grammar (test_bodmas.py).
"""
from __future__ import annotations

import math
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import sf
from conftest import dtype_of, pairs, singles


def test_total_order_on_codes(width, dt):
    codes = sorted(set(singles(width)))
    prev = None
    for k in codes:
        x = sf.from_code(k, dt)
        assert x == x and x <= x and x >= x and not (x < x) and not (x != x)
        if prev is not None:
            assert prev < x and prev <= x and x > prev and x >= prev
            assert prev != x and not (prev == x) and not (prev > x)
        prev = x
    assert sorted((sf.from_code(k, dt) for k in reversed(codes)), key=lambda v: v.code) == \
        sorted(sf.from_code(k, dt) for k in reversed(codes))


def test_comparisons_match_exact_values(width, dt):
    for k_a, k_b in pairs(width):
        a, b = sf.from_code(k_a, dt), sf.from_code(k_b, dt)
        assert (a < b) == (k_a < k_b) and (a <= b) == (k_a <= k_b)
        assert (a > b) == (k_a > k_b) and (a >= b) == (k_a >= k_b)
        assert (a == b) == (k_a == k_b) and (a != b) == (k_a != k_b)
        # against the exact value as float / Fraction (every value is exact in float64)
        assert (a < float(b)) == (k_a < k_b) and (float(a) < b) == (k_a < k_b)
        assert (a == b.exact) == (k_a == k_b)


def test_out_of_range_literals_do_not_quantize_or_raise():
    for dt in sf.dtypes:
        i = sf.info(dt)
        x = sf.from_code(i.max_code, dt)
        assert x < 1 and x < 1.0 and x <= 1 and not (x >= 1) and x != 1
        assert -x > -1 and sf.from_code(-i.max_code, dt) > -1
        assert x < 2 ** 40 and x > -(2 ** 40) and x < Fraction(10 ** 30, 3)
        assert x < float("inf") and x > float("-inf") and not (x > float("inf"))
        # the spec's example: x < 1 is always True
        for k in (-i.max_code, 0, i.max_code):
            assert sf.from_code(k, dt) < 1
        # comparisons never go through the error policy
        with sf.errstate(all="raise"):
            assert x < 1 and x != 5 and x > -3


def test_exact_not_quantized():
    x = sf.sf16(0.1)                     # code 3277 = 0.100006103515625
    assert x != 0.1 and x > 0.1          # the stored value is not the float 0.1
    assert x == Fraction(3277, 32768) and x == 0.100006103515625
    assert x != Fraction(1, 10) and x > Fraction(1, 10)
    assert sf.sf16(0.5) == 0.5 == sf.sf8(0.5) == Fraction(1, 2) == Decimal("0.5")
    assert sf.sf16(0.5) == np.float32(0.5) and np.float64(0.5) == sf.sf16(0.5)
    # a literal that would underflow or overflow if quantized is simply compared
    assert sf.sf16(0) < 1e-9 and sf.sf16(2 ** -15) > 1e-9 and sf.sf16(0) > -1e-9
    with sf.errstate(all="raise"):
        assert sf.sf16(0.5) < 1e9


def test_nan_and_foreign_types():
    x = sf.sf16(0.5)
    nan = float("nan")
    assert not (x == nan) and (x != nan)
    assert not (x < nan) and not (x > nan) and not (x <= nan) and not (x >= nan)
    assert not (x == np.float32("nan")) and not (x == Decimal("NaN"))
    assert not (x == "0.5") and (x != "0.5") and not (x == None) and x != [0.5]
    with pytest.raises(TypeError):
        x < "0.5"
    with pytest.raises(TypeError):
        x < None


def test_negative_zero_equals_zero():
    for dt in sf.dtypes:
        assert dt(-0.0) == dt(0.0) == 0 == -0.0 == dt(0)
        assert not (dt(-0.0) < dt(0.0)) and not (dt(-0.0) > dt(0.0))
        assert hash(dt(-0.0)) == hash(dt(0.0)) == hash(0)
        assert (-dt(0)) == dt(0) and (-dt(0)).sign == 0


def test_bool_min_max_where_conditional():
    a, b, z = sf.sf16(0.5), sf.sf16(-0.25), sf.sf16(0)
    assert bool(a) and bool(b) and not bool(z)
    assert min(a, b) is b and max(a, b) is a
    assert max(a, b, 0.75) == 0.75 and min(a, b, -1) == -1       # literals compare exactly
    assert sf.where(True, a, b) is a and sf.where(False, a, b) is b
    assert sf.where(a > b, a, b) is a and sf.where(z, a, b) is b
    assert (a if a > b else b) is a
    assert sf.where(b != 0, 0.25, 0) == 0.25
    assert [x for x in (a, b, z) if x] == [a, b]
    assert {a: 1}[0.5] == 1                                      # hash agrees with ==
    assert a in {0.5, 0.25} and b in [sf.sf8(-0.25)]


def test_guarded_division_never_raises(width, dt):
    for k_a, k_b in pairs(width):
        a, b = sf.from_code(k_a, dt), sf.from_code(k_b, dt)
        if abs(k_a) >= abs(k_b):
            continue                            # overflow is a different event
        r = a / b if b != 0 else a              # must never raise
        assert type(r) is dt
    for k in singles(width):
        a = sf.from_code(k, dt)
        assert (a / dt(0) if dt(0) != 0 else a) is a
        # sf.where evaluates both branches: the division raises first
        with pytest.raises(sf.SFZeroDivisionError):
            sf.where(dt(0) != 0, a / dt(0), a)


@settings(max_examples=400, deadline=None)
@given(bits=st.integers(4, 16), k=st.integers(-32767, 32767),
       v=st.floats(allow_nan=False, width=64))
def test_comparisons_against_floats_hypothesis(bits, k, v):
    dt = dtype_of(bits)
    m = sf.info(dt).max_code
    x = sf.from_code(k % (2 * m + 1) - m, dt)
    e = x.exact
    if math.isinf(v):
        assert (x < v) == (v > 0) and (x > v) == (v < 0) and x != v
    else:
        f = Fraction(v)
        assert (x < v) == (e < f) and (x <= v) == (e <= f) and (x == v) == (e == f)
        assert (x > v) == (e > f) and (x >= v) == (e >= f) and (x != v) == (e != f)
        assert (v < x) == (f < e) and (v == x) == (f == e)
