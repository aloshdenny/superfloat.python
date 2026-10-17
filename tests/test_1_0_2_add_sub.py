"""Milestone 1.0.2: add, sub, neg, abs.

Unit: overflow exactly at |k_a ± k_b| = M + 1; a - a == 0; a - b == a + (-b).
Mix/regression: chains like a + b - c; saturation is not undone:
(max + lsb) - lsb == max - lsb.
"""
from __future__ import annotations

import operator

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import sf
from conftest import EVENT_ERROR, dtype_of, pairs, run, singles


def test_add_sub_exhaustive_or_sampled(width, dt, oracle):
    """Every pair (sf4..sf10) or a sample (sf11..sf16) against the oracle:
    same value, and an SFOverflowError exactly when the oracle says so."""
    m = sf.info(dt).max_code
    lsb = oracle.lsb
    for k_a, k_b in pairs(width):
        a, b = sf.from_code(k_a, dt), sf.from_code(k_b, dt)
        va, vb = k_a * lsb, k_b * lsb
        for op, oracle_op in ((operator.add, oracle.add), (operator.sub, oracle.sub)):
            expect, ev = oracle_op(va, vb)
            r, raised = run(op, a, b)
            assert raised is EVENT_ERROR[ev], (a, op.__name__, b)
            assert type(r) is dt
            assert r.exact == expect, (a, op.__name__, b)
        # overflow happens exactly when |k_a + k_b| > M, never before
        assert (abs(k_a + k_b) > m) == (oracle.add(va, vb)[1] == "overflow")


def test_overflow_edge_exactly_at_M_plus_1(width, dt):
    m = sf.info(dt).max_code
    for k_a in singles(width):
        # the largest k_b that still fits, and the first that does not
        a = sf.from_code(k_a, dt)
        fit = m - k_a                        # k_a + fit == M exactly
        if -m <= fit <= m:
            assert (a + sf.from_code(fit, dt)).code == m
        if -m <= fit + 1 <= m:
            with pytest.raises(sf.SFOverflowError):
                a + sf.from_code(fit + 1, dt)
        fit_neg = -m - k_a                   # k_a + fit_neg == -M exactly
        if -m <= fit_neg <= m:
            assert (a + sf.from_code(fit_neg, dt)).code == -m
        if -m <= fit_neg - 1 <= m:
            with pytest.raises(sf.SFOverflowError):
                a + sf.from_code(fit_neg - 1, dt)


def test_neg_abs_never_overflow(width, dt):
    for k in singles(width):
        a = sf.from_code(k, dt)
        assert (-a).code == -k and abs(a).code == abs(k) and (+a) is a
        assert (-(-a)).code == k
        assert (a - a).code == 0
        assert (a + 0).code == k and (0 + a).code == k and (a - 0).code == k
        assert (0 - a).code == -k
        assert type(-a) is dt and type(abs(a)) is dt


def test_sub_equals_add_neg(width, dt):
    with sf.errstate(over="ignore"):
        for k_a, k_b in pairs(width):
            a, b = sf.from_code(k_a, dt), sf.from_code(k_b, dt)
            assert (a - b).code == (a + (-b)).code


def test_saturation_is_not_undone(width, dt):
    i = sf.info(dt)
    mx, lsb = sf.from_code(i.max_code, dt), sf.from_code(1, dt)
    with sf.errstate(over="ignore"):
        assert ((mx + lsb) - lsb).code == i.max_code - 1   # (max + lsb) - lsb == max - lsb
        assert ((-mx - lsb) + lsb).code == -i.max_code + 1
        # the spec's example: (0.9 + 0.9) - 0.9 != 0.9 + (0.9 - 0.9)
        x = dt(0.9) if i.bits >= 5 else dt(0.875)
        left = (x + x) - x
        right = x + (x - x)
        assert right.code == x.code
        assert left.code == i.max_code - x.code
        if i.bits == 16:
            assert float(left) == pytest.approx(0.09997, abs=1e-5)
            assert float(right) == pytest.approx(0.9, abs=1e-4)


def test_chains(width, dt, oracle):
    rnd = np.random.default_rng(width)
    m = sf.info(dt).max_code
    with sf.errstate(over="ignore"):
        for _ in range(2000):
            ks = rnd.integers(-m, m + 1, 4).tolist()
            a, b, c, d = (sf.from_code(int(k), dt) for k in ks)
            # a + b - c + d evaluated left to right with saturation at each step
            v, _ = oracle.add(a.exact, b.exact)
            v, _ = oracle.sub(v, c.exact)
            v, _ = oracle.add(v, d.exact)
            assert (a + b - c + d).exact == v
            v2, _ = oracle.sub(b.exact, c.exact)
            v2, _ = oracle.add(a.exact, v2)
            assert (a + (b - c)).exact == v2


def test_literal_and_width_promotion():
    a = sf.sf16(0.5)
    # literals take the SF operand's type and are quantized first
    assert (a + 0.25).code == 0x6000 and (0.25 + a).code == 0x6000
    assert (a - 0.75).code == -0x2000 and (0.75 - a).code == 0x2000
    assert type(a + 0.25) is sf.sf16
    # x + 1 overflows at the literal, before the add
    with pytest.raises(sf.SFOverflowError, match="literal 1 is not representable"):
        a + 1
    with pytest.raises(sf.SFOverflowError, match="literal"):
        sf.sf16(0) + 1          # even though 0 + 1 would "fit" if 1 were representable
    # a tiny literal underflows (warns) and the op then sees 0
    with pytest.warns(sf.SFUnderflowWarning):
        assert (a + 1e-9).code == a.code
    # width promotion: the wider type wins, widening is exact
    b = sf.sf8(0.25)
    assert type(a + b) is sf.sf16 and type(b + a) is sf.sf16
    assert (a + b).code == 0x6000 and (b - a).code == -0x2000
    assert type(sf.sf4(0.5) + sf.sf5(0.25)) is sf.sf5
    c = sf.sf12(0.5)
    assert (c + sf.sf4(0.25)).code == 0x600
    # numpy scalars behave as literals, on either side
    assert (np.float32(0.25) + a).code == 0x6000
    assert (a - np.float64(0.25)).code == 0x2000
    assert (np.int8(0) + a).code == a.code
    with pytest.raises(sf.SFOverflowError):
        np.int8(1) + a
    # in-place forms
    x = sf.sf16(0.5)
    x += 0.25
    assert x.code == 0x6000 and type(x) is sf.sf16
    x -= sf.sf8(0.5)
    assert x.code == 0x2000
    # foreign types are rejected with TypeError (NotImplemented on both sides)
    with pytest.raises(TypeError):
        a + "0.5"
    with pytest.raises(TypeError):
        [1] + a


@settings(max_examples=500, deadline=None)
@given(bits=st.integers(4, 16), ka=st.integers(-32767, 32767), kb=st.integers(-32767, 32767))
def test_add_sub_hypothesis(bits, ka, kb):
    from oracle import Fmt
    dt, fmt = dtype_of(bits), Fmt(bits)
    m = sf.info(dt).max_code
    ka, kb = ka % (2 * m + 1) - m, kb % (2 * m + 1) - m
    a, b = sf.from_code(ka, dt), sf.from_code(kb, dt)
    with sf.errstate(all="ignore"):
        assert (a + b).exact == fmt.add(a.exact, b.exact)[0]
        assert (a - b).exact == fmt.sub(a.exact, b.exact)[0]
        assert (a + b).code == (b + a).code
