"""Milestone 1.1.2: reductions and dot.

Unit: saturating accumulation in target order; empty reductions.
Mix/regression: sf.dot(x, y) equals a chain of Stage 1.0 sf.fma calls in
target order.
"""
from __future__ import annotations

import numpy as np
import pytest

import sf
from conftest import dtype_of, run


def chain_sum(dt, codes):
    acc = dt(0)
    for k in codes:
        acc = acc + dt.from_code(int(k))
    return acc


def chain_dot(dt, ka, kb):
    acc = dt(0)
    for a, b in zip(ka, kb):
        acc = sf.fma(dt.from_code(int(a)), dt.from_code(int(b)), acc)
    return acc


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_sum_is_a_sequential_saturating_chain(bits):
    dt = dtype_of(bits)
    m = sf.info(dt).max_code
    rng = np.random.default_rng(bits)
    for n in (0, 1, 2, 5, 17, 64):
        for scale in (m // 8, m):
            codes = rng.integers(-scale, scale + 1, n)
            x = sf.from_codes(codes, dt)
            with sf.errstate(over="ignore"):
                expect = chain_sum(dt, codes)
                got = x.sum()
                assert sf.sum(x).code == expect.code
            assert isinstance(got, sf.SFScalar) and got.code == expect.code
            # the event matches the chain's event
            r, raised = run(lambda: x.sum())
            _, raised2 = run(chain_sum, dt, codes)
            assert raised is raised2
    assert sf.zeros((0,), dt).sum().code == 0                     # empty sum is 0


def test_sum_axes():
    m = sf.from_codes(np.arange(1, 13).reshape(3, 4) * 1000, sf.sf16)
    with sf.errstate(over="ignore"):
        s0, s1, s_all, s_neg = m.sum(axis=0), m.sum(axis=1), m.sum(), m.sum(axis=-1)
    assert s0.shape == (4,) and s1.shape == (3,) and isinstance(s_all, sf.SFScalar)
    with sf.errstate(over="ignore"):
        assert s0.codes.tolist() == [chain_sum(sf.sf16, m.codes[:, j]).code for j in range(4)]
        assert s1.codes.tolist() == [chain_sum(sf.sf16, m.codes[i]).code for i in range(3)]
        assert s_neg.codes.tolist() == s1.codes.tolist()
        assert s_all.code == chain_sum(sf.sf16, m.codes.reshape(-1)).code   # C order, sequential
    with pytest.raises(sf.SFShapeError):
        m.sum(axis=2)
    # saturation is order dependent and the order is sequential from index 0
    x = sf.array([0.9, 0.9, -0.9], dtype=sf.sf16)
    with sf.errstate(over="ignore"):
        assert x.sum().code == (sf.sf16(0.9) + 0.9 - 0.9).code != sf.sf16(0.9).code
    with pytest.raises(sf.SFOverflowError, match="sum"):
        x.sum()


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_dot_equals_fma_chain(bits):
    dt = dtype_of(bits)
    m = sf.info(dt).max_code
    rng = np.random.default_rng(100 + bits)
    for n in (0, 1, 3, 16, 100):
        for scale in (m // 16, m):
            ka = rng.integers(-scale, scale + 1, n)
            kb = rng.integers(-scale, scale + 1, n)
            x, y = sf.from_codes(ka, dt), sf.from_codes(kb, dt)
            r, raised = run(sf.dot, x, y)
            r2, raised2 = run(chain_dot, dt, ka, kb)
            assert isinstance(r, sf.SFScalar) and r.code == r2.code
            # same first error class in catalogue order for a single output
            assert raised is raised2
    with pytest.raises(sf.SFShapeError):
        sf.dot(sf.zeros((3,)), sf.zeros((4,)))
    # mixed widths promote
    assert sf.dot(sf.array([0.5], dtype=sf.sf8), sf.array([0.5], dtype=sf.sf16)).dtype is sf.sf16
    with pytest.raises(sf.SFTypeError):
        sf.dot([0.5, 0.5], [0.5, 0.5])                   # host lists are not SF arrays


def test_dot_commutes_and_matches_matmul_forms():
    x = sf.array([0.5, -0.25, 0.125], dtype=sf.sf16)
    y = sf.array([0.25, 0.5, -0.5], dtype=sf.sf16)
    assert sf.dot(x, y).code == sf.dot(y, x).code == (x @ y).code
    assert (x @ y).code == sf.fma(x[2], y[2], sf.fma(x[1], y[1], sf.fma(x[0], y[0], 0))).code


def test_max_min_argmax():
    x = sf.array([0.5, -0.75, 0.25, 0.5], dtype=sf.sf16)
    assert x.max() == 0.5 and x.min() == -0.75 and x.argmax() == 0 and x.argmin() == 1
    assert sf.max(x) == 0.5 and sf.min(x) == -0.75 and sf.argmax(x) == 0 and sf.argmin(x) == 1
    m = sf.array([[0.5, -0.75], [0.25, 0.5]], dtype=sf.sf16)
    assert m.max(axis=0).codes.tolist() == [16384, 16384] and m.min(axis=1).codes.tolist() == [-24576, 8192]
    assert m.argmax(axis=1).tolist() == [0, 1]
    with pytest.raises(sf.SFShapeError):
        sf.zeros((0,)).max()
    # max/min raise nothing, even at the boundaries
    with sf.errstate(all="raise"):
        assert sf.full((3,), sf.info(sf.sf16).max, sf.sf16).max().code == 32767


def test_array_fma_matches_scalar():
    rng = np.random.default_rng(5)
    ka, kb, kc = (rng.integers(-32767, 32768, 50) for _ in range(3))
    a, b, c = (sf.from_codes(k, sf.sf16) for k in (ka, kb, kc))
    with sf.errstate(all="ignore"):
        r = sf.fma(a, b, c)
        assert r.codes.tolist() == [sf.fma(sf.from_code(int(x), sf.sf16), sf.from_code(int(y), sf.sf16),
                                           sf.from_code(int(z), sf.sf16)).code for x, y, z in zip(ka, kb, kc)]
        assert sf.fma(a, 0.5, c).codes.tolist() == (a * 0.5 + c).codes.tolist()
        assert sf.fma(a, b, 0).codes.tolist() == (a * b).codes.tolist()
