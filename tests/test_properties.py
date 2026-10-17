"""Algebraic properties (spec section 10).

Asserted: a + b == b + a, a * b == b * a, a - a == 0, a + 0 == a,
a * 0 == 0, -(-a) == a, saturation is monotonic.
Recorded, never asserted: associativity and distributivity, which fail
under saturation and truncation.
"""
from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import sf
from conftest import dtype_of, pairs, singles


def test_commutativity(width, dt):
    with sf.errstate(all="ignore"):
        for k_a, k_b in pairs(width):
            a, b = sf.from_code(k_a, dt), sf.from_code(k_b, dt)
            assert (a + b).code == (b + a).code
            assert (a * b).code == (b * a).code


def test_identities(width, dt):
    z = dt(0)
    for k in singles(width):
        a = sf.from_code(k, dt)
        assert (a - a).code == 0
        assert (a + z).code == k and (z + a).code == k
        assert (a * z).code == 0 and (z * a).code == 0
        assert (-(-a)).code == k
        assert abs(a).code == abs(k) and abs(-a).code == abs(k)
        assert (a - z).code == k and (z - a).code == -k


def test_saturation_is_monotonic(width, dt):
    """If a <= a' then a + b <= a' + b and a * b <= a' * b for b >= 0,
    with saturation and truncation (both are monotone maps)."""
    m = sf.info(dt).max_code
    rnd = np.random.default_rng(500 + width)
    with sf.errstate(all="ignore"):
        for _ in range(3000 if width > 10 else 1500):
            k1, k2, kb = (int(v) for v in rnd.integers(-m, m + 1, 3))
            if k1 > k2:
                k1, k2 = k2, k1
            a1, a2, b = sf.from_code(k1, dt), sf.from_code(k2, dt), sf.from_code(kb, dt)
            assert (a1 + b) <= (a2 + b)
            assert (a1 - b) <= (a2 - b)
            if kb >= 0:
                assert (a1 * b) <= (a2 * b)
                if kb > 0 and abs(k1) < kb and abs(k2) < kb:
                    assert (a1 / b) <= (a2 / b)
            else:
                assert (a1 * b) >= (a2 * b)
        # saturation at the top is a fixed point
        mx = sf.from_code(m, dt)
        for kb in range(0, min(m, 50) + 1):
            assert (mx + sf.from_code(kb, dt)).code == m
            assert (-mx - sf.from_code(kb, dt)).code == -m


def test_associativity_and_distributivity_recorded(width, dt, capsys):
    m = sf.info(dt).max_code
    rnd = np.random.default_rng(600 + width)
    n = 3000
    assoc_add = assoc_mul = distrib = 0
    with sf.errstate(all="ignore"):
        for _ in range(n):
            a, b, c = (sf.from_code(int(k), dt) for k in rnd.integers(-m, m + 1, 3))
            assoc_add += ((a + b) + c).code != (a + (b + c)).code
            assoc_mul += ((a * b) * c).code != (a * (b * c)).code
            distrib += (a * (b + c)).code != (a * b + a * c).code
    print(f"sf{width}: (a+b)+c != a+(b+c): {assoc_add}/{n}; (a*b)*c != a*(b*c): {assoc_mul}/{n}; "
          f"a*(b+c) != a*b+a*c: {distrib}/{n}")
    # recorded only: these are expected to fail under saturation and truncation
    assert assoc_add >= 0 and assoc_mul >= 0 and distrib >= 0


def test_the_spec_associativity_example():
    with sf.errstate(over="ignore"):
        x = sf.sf16(0.9)
        assert float((x + x) - x) == pytest.approx(0.09997, abs=1e-5)
        assert float(x + (x - x)) == pytest.approx(0.9, abs=1e-4)
        assert ((x + x) - x).code != (x + (x - x)).code


@settings(max_examples=300, deadline=None)
@given(bits=st.integers(4, 16), ka=st.integers(-32767, 32767), kb=st.integers(-32767, 32767),
       kc=st.integers(-32767, 32767))
def test_properties_hypothesis(bits, ka, kb, kc):
    dt = dtype_of(bits)
    m = sf.info(dt).max_code
    a, b, c = (sf.from_code(k % (2 * m + 1) - m, dt) for k in (ka, kb, kc))
    with sf.errstate(all="ignore"):
        assert (a + b).code == (b + a).code and (a * b).code == (b * a).code
        assert (a - a).code == 0 and (-(-a)).code == a.code
        assert (a * b + c).code == sf.fma(a, b, c).code
        assert -(a * b) == (-a) * b == a * (-b)
        assert (a + b) - b == a or abs(a + b) == sf.info(dt).max        # exact unless saturated
