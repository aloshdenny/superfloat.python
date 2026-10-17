"""Milestone 1.1.1: arrays and elementwise ops.

Unit: each element equals the Stage 1.0 scalar op.
Mix/regression: every Stage 1.0 operator re-run on shapes (), (n,), (m, n),
with broadcasting, literals, promotion and the error policy.
"""
from __future__ import annotations

import operator
import pickle
from fractions import Fraction

import numpy as np
import pytest

import sf
from conftest import EVENT_ERROR, dtype_of, run

SHAPES = [(), (7,), (4, 5)]
WIDTHS = [4, 8, 16]


def rand_codes(bits, shape, seed, small=False):
    m = sf.info(dtype_of(bits)).max_code
    rng = np.random.default_rng(seed)
    hi = min(m, 200) if small else m
    return rng.integers(-hi, hi + 1, size=shape, dtype=np.int64)


def arr(bits, codes):
    return sf.from_codes(np.asarray(codes), dtype_of(bits))


# --- construction -----------------------------------------------------------

def test_construction_paths():
    x = sf.array([0.10, -0.25, 0.50], dtype=sf.sf16)
    assert isinstance(x, sf.SFArray) and x.dtype is sf.sf16 and x.shape == (3,) and x.device == "host"
    assert x.codes.tolist() == [3277, -8192, 16384]
    assert sf.array(np.array([0.5, 0.25]), dtype="sf8").codes.tolist() == [64, 32]
    assert sf.array([[0.5], [0.25]], dtype=sf.sf16).shape == (2, 1)
    # dtype inferred from SF contents
    y = sf.array([sf.sf16(0.5), sf.sf8(0.25)])
    assert y.dtype is sf.sf16 and y.codes.tolist() == [16384, 8192]
    assert sf.array(sf.sf4(0.5)).dtype is sf.sf4 and sf.array(x).codes.tolist() == x.codes.tolist()
    assert sf.array(x, dtype=sf.sf16) is not x and sf.asarray(x) is x
    assert sf.asarray(x, dtype=sf.sf16) is x and sf.asarray(sf.array([0.5], dtype=sf.sf8), dtype=sf.sf16) is not None
    with pytest.raises(sf.SFTypeError):
        sf.array([0.5, 0.25])                       # no dtype, no SF values
    with pytest.raises(sf.SFTypeError):
        sf.array([sf.sf16(0.5), 0.25])             # mixed SF and host numbers
    with pytest.raises(sf.SFTypeError):
        sf.array(x, dtype=sf.sf8)                   # narrowing needs sf.cast
    assert sf.array(["0.5", "1/4"], dtype=sf.sf16).codes.tolist() == [16384, 8192]   # constructor only
    with pytest.raises(sf.SFTypeError):
        x + ["0.5", "0.5", "0.5"]
    with pytest.raises(sf.SFShapeError):
        sf.array([[1, 2], [3]], dtype=sf.sf16)
    # host values go through the policy, counted over the array
    with pytest.raises(sf.SFOverflowError, match="1 of 3"):
        sf.array([0.5, 1.0, 0.25], dtype=sf.sf16)
    with pytest.warns(sf.SFUnderflowWarning, match="2 of 3"):
        sf.array([1e-9, -1e-9, 0.25], dtype=sf.sf16)
    with pytest.raises(sf.SFInvalidError):
        sf.array([float("nan")], dtype=sf.sf16)
    # exact inputs: Fractions, Decimals, numpy float32, ints (0 only)
    assert sf.array([Fraction(1, 4), 0], dtype=sf.sf16).codes.tolist() == [8192, 0]
    assert sf.array(np.array([0.25], dtype=np.float32), dtype=sf.sf16).codes.tolist() == [8192]
    assert sf.array(np.array([0, 0]), dtype=sf.sf16).codes.tolist() == [0, 0]
    with pytest.raises(sf.SFOverflowError):
        sf.array(np.array([0, 1]), dtype=sf.sf16)


def test_zeros_full_ones():
    z = sf.zeros((2, 3), sf.sf8)
    assert z.shape == (2, 3) and z.dtype is sf.sf8 and not z.any()
    f = sf.full((2,), 0.25, sf.sf16)
    assert f.codes.tolist() == [8192, 8192]
    assert sf.full((2,), sf.sf8(0.5)).dtype is sf.sf8
    with pytest.raises(sf.SFOverflowError, match="sf.full"):
        sf.ones((2,), sf.sf16)
    with sf.errstate(over="ignore"):
        assert sf.ones((2,), sf.sf16).codes.tolist() == [32767, 32767]
    assert sf.full((2,), sf.info(sf.sf16).max, sf.sf16).codes.tolist() == [32767, 32767]


def test_codes_bits_roundtrip():
    for bits in sf.FORMATS:
        dt = dtype_of(bits)
        m = sf.info(dt).max_code
        codes = np.arange(-m, m + 1)
        x = sf.from_codes(codes, dt)
        assert np.array_equal(x.codes, codes)
        assert np.array_equal(sf.from_bits(x.bits, dt).codes, codes)
        assert x.bits.dtype == np.uint16
        # agrees with the scalar bit packing on every code
        assert all(int(x.bits[i]) == sf.from_code(int(codes[i]), dt).bits for i in range(0, len(codes), 97))
        assert np.array_equal(x.numpy(), codes / 2 ** (bits - 1))
    with pytest.raises(sf.SFInvalidError):
        sf.from_codes(np.array([32768]), sf.sf16)
    with pytest.raises(sf.SFInvalidError):
        sf.from_bits(np.array([1 << 16]), sf.sf16)
    with pytest.raises(sf.SFTypeError):
        sf.from_codes(np.array([0.5]), sf.sf16)
    x = sf.from_codes(np.array([1, 2]), sf.sf16)
    with pytest.raises(ValueError):
        x.codes[0] = 5                             # read-only view


def test_indexing_slicing_setitem_iteration():
    m = sf.array([[0.5, 0.25, 0.125], [-0.5, 0, 0.75]], dtype=sf.sf16)
    assert m[0, 1] == 0.25 and type(m[0, 1]) is sf.sf16
    assert m[1].codes.tolist() == [-16384, 0, 24576] and m[:, 0].shape == (2,)
    assert m[m > 0].codes.tolist() == [16384, 8192, 4096, 24576]
    assert m[np.array([1, 0])].shape == (2, 3) and m[-1, -1] == 0.75
    m[0, 0] = 0.75
    m[1] = [0.1, 0.2, 0.3]
    m[:, 2] = sf.sf8(0.5)
    assert m[0, 0] == 0.75 and m[1, 0] == sf.sf16(0.1) and m[:, 2].codes.tolist() == [16384, 16384]
    with pytest.raises(sf.SFOverflowError):
        m[0, 0] = 1
    with pytest.raises(sf.SFTypeError):
        m[0, 0] = "0.5"
    with pytest.raises(sf.SFTypeError):
        m[m]                                        # SF array as index
    with pytest.raises(IndexError):
        m[5, 0]
    assert [type(r) for r in m] == [sf.SFArray, sf.SFArray] and list(m[0])[0] == 0.75
    assert len(m) == 2 and m.size == 6 and m.ndim == 2 and m.T.shape == (3, 2)
    assert m.reshape(6).shape == (6,) and m.reshape((3, 2)).shape == (3, 2) and m.flatten().shape == (6,)
    assert m.tolist()[0][0] == 0.75 and float(m[0:1, 0:1]) == 0.75
    with pytest.raises(ValueError):
        bool(m)
    assert bool(m[0:1, 0:1]) and not bool(sf.zeros((1,)))
    with pytest.raises(TypeError):
        len(sf.from_codes(np.array(3), sf.sf16))
    assert repr(sf.array([0.5, -0.25], dtype=sf.sf8)) == "sfarray([ 0.5 , -0.25], dtype=sf8)"
    assert pickle.loads(pickle.dumps(m)).codes.tolist() == m.codes.tolist()
    c = m.copy()
    c[0, 0] = 0
    assert m[0, 0] == 0.75


# --- every Stage 1.0 op on every shape -------------------------------------------

BINARY = [("add", operator.add), ("sub", operator.sub), ("mul", operator.mul), ("div", operator.truediv)]


@pytest.mark.parametrize("bits", WIDTHS)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_binary_ops_match_scalars(bits, shape):
    dt = dtype_of(bits)
    for seed in range(3):
        ka = rand_codes(bits, shape, seed, small=seed == 2)
        kb = rand_codes(bits, shape, seed + 10, small=seed == 2)
        a, b = arr(bits, ka), arr(bits, kb)
        for name, op in BINARY:
            r, raised = run(op, a, b)
            if shape == ():
                assert isinstance(r, sf.SFScalar)
                sr, sraised = run(op, dt.from_code(int(ka)), dt.from_code(int(kb)))
                assert r.code == sr.code and raised is sraised
                continue
            assert isinstance(r, sf.SFArray) and r.shape == shape and r.dtype is dt
            scalar_errors = set()
            for idx in np.ndindex(*shape):
                sr, sraised = run(op, dt.from_code(int(ka[idx])), dt.from_code(int(kb[idx])))
                assert int(r.codes[idx]) == sr.code, (name, idx)
                if sraised:
                    scalar_errors.add(sraised)
            # the array raises exactly when some element would, with the first
            # error in catalogue order (invalid, divide, overflow, underflow)
            expected = next((EVENT_ERROR[k] for k in ("invalid", "divide", "overflow", "underflow")
                             if EVENT_ERROR[k] in scalar_errors), None)
            assert raised is expected, (name, raised, scalar_errors)


@pytest.mark.parametrize("bits", WIDTHS)
@pytest.mark.parametrize("shape", SHAPES, ids=str)
def test_unary_and_fma_match_scalars(bits, shape):
    dt = dtype_of(bits)
    ka, kb, kc = (rand_codes(bits, shape, s) for s in (1, 2, 3))
    a, b, c = (arr(bits, k) for k in (ka, kb, kc))
    checks = [
        (lambda v: -v, lambda s: -s),
        (abs, abs),
        (lambda v: v ** 2, lambda s: s ** 2),
        (lambda v: v ** 5, lambda s: s ** 5),
        (lambda v: sf.sqrt(abs(v)), lambda s: sf.sqrt(abs(s))),
        (lambda v: sf.sqrt(v), lambda s: sf.sqrt(s)),
    ]
    for arr_fn, sc_fn in checks:
        r, raised = run(arr_fn, a)
        sc = [run(sc_fn, dt.from_code(int(ka[idx]))) for idx in np.ndindex(*shape)] if shape else \
            [run(sc_fn, dt.from_code(int(ka)))]
        codes = np.asarray(r.codes if isinstance(r, sf.SFArray) else r.code).reshape(-1)
        assert codes.tolist() == [s.code for s, _ in sc]
        assert (raised is not None) == any(e for _, e in sc)
    r, raised = run(sf.fma, a, b, c)
    r2, raised2 = run(lambda: a * b + c)
    codes = np.asarray(r.codes if isinstance(r, sf.SFArray) else r.code)
    codes2 = np.asarray(r2.codes if isinstance(r2, sf.SFArray) else r2.code)
    # same bits always; on arrays the fused form reports overflow before an
    # underflow elsewhere in the array, the two-step form the other way round
    assert np.array_equal(codes, codes2) and (raised is None) == (raised2 is None)
    for idx in (np.ndindex(*shape) if shape else [()]):
        s, _ = run(sf.fma, dt.from_code(int(ka[idx])), dt.from_code(int(kb[idx])), dt.from_code(int(kc[idx])))
        assert int(codes[idx]) == s.code


@pytest.mark.parametrize("bits", WIDTHS)
def test_comparisons_match_scalars(bits):
    dt = dtype_of(bits)
    ka, kb = rand_codes(bits, (40,), 5), rand_codes(bits, (40,), 6)
    a, b = arr(bits, ka), arr(bits, kb)
    for op in (operator.lt, operator.le, operator.gt, operator.ge, operator.eq, operator.ne):
        got = op(a, b)
        assert got.dtype == bool and got.tolist() == [op(int(x), int(y)) for x, y in zip(ka, kb)]
        assert op(a, b.numpy()).tolist() == got.tolist()           # exact against floats
        assert op(a, dt.from_code(int(kb[0]))).tolist() == [op(int(x), int(kb[0])) for x in ka]
    # exact, never quantized
    x = sf.array([0.1, 0.5], dtype=sf.sf16)
    assert (x == 0.1).tolist() == [False, False] and (x > 0.1).tolist() == [True, True]
    assert (x < 1).all() and (x < 2 ** 40).all() and (x > -1).all()
    assert (x == np.array([0.1, 0.5])).tolist() == [False, True]
    assert (x == [Fraction(3277, 32768), Fraction(1, 2)]).tolist() == [True, True]
    assert (x == np.array([Fraction(3277, 32768), Fraction(1, 2)], dtype=object)).tolist() == [True, True]
    assert (x == float("nan")).tolist() == [False, False] and (x != float("nan")).tolist() == [True, True]
    assert (x < float("inf")).all() and not (x > float("inf")).any()
    assert (x == sf.array([0.1, 0.5], dtype=sf.sf8)).tolist() == [False, True]   # 0.1 differs by width
    with pytest.raises(sf.SFShapeError):
        x == sf.array([0.1, 0.2, 0.3], dtype=sf.sf16)
    assert (x == "0.1") is False and (x != "0.1") is True


# --- broadcasting, literals, promotion ------------------------------------------------

def test_broadcasting():
    m = sf.from_codes(np.arange(6).reshape(2, 3) * 100, sf.sf16)
    v = sf.from_codes(np.array([1, 2, 3]), sf.sf16)
    col = sf.from_codes(np.array([[10], [20]]), sf.sf16)
    assert (m + v).codes.tolist() == (m.codes + v.codes).tolist()
    assert (m + col).codes.tolist() == (m.codes + col.codes).tolist()
    assert (col + v).shape == (2, 3) and (v + m).shape == (2, 3)
    assert (m + sf.sf16(0.25)).codes.tolist() == (m.codes + 8192).tolist()
    assert (0.25 + m).codes.tolist() == (m.codes + 8192).tolist()
    with pytest.raises(sf.SFShapeError):
        m + sf.from_codes(np.array([1, 2]), sf.sf16)
    with pytest.raises(sf.SFShapeError):
        m + np.zeros((3, 2))


def test_literals_and_promotion():
    x = sf.array([0.5, 0.25], dtype=sf.sf16)
    assert (x + 0.25).codes.tolist() == [24576, 16384] and (0.25 + x).codes.tolist() == [24576, 16384]
    assert (np.float32(0.25) * x).codes.tolist() == [4096, 2048]
    assert (np.array([0.25, 0.5]) + x).codes.tolist() == [24576, 24576]
    assert (x + [0.25, 0.5]).codes.tolist() == [24576, 24576]
    with pytest.raises(sf.SFOverflowError, match="literal 1 is not representable"):
        x + 1
    with pytest.raises(sf.SFOverflowError, match="literal cast: overflow in 1 of 2"):
        x + np.array([0.0, 1.0])
    with pytest.warns(sf.SFUnderflowWarning):
        x + 1e-9
    # promotion to the wider type, both ways, scalars and arrays
    y8 = sf.array([0.25, 0.25], dtype=sf.sf8)
    assert type(x + y8) is sf.SFArray and (x + y8).dtype is sf.sf16 and (y8 + x).dtype is sf.sf16
    assert (x + y8).codes.tolist() == [24576, 16384]
    with sf.errstate(over="ignore"):
        assert (x + sf.array([0.5, 0.25], dtype=sf.sf8)).codes.tolist() == [32767, 16384]
    assert (sf.sf8(0.25) + x).dtype is sf.sf16 and (x + sf.sf4(0.25)).dtype is sf.sf16
    assert (sf.sf16(0.25) + y8).dtype is sf.sf16 and isinstance(sf.sf16(0.25) + y8, sf.SFArray)
    assert (y8 + sf.sf16(0.25)).codes.tolist() == [16384, 16384]
    # in-place forms
    z = sf.array([0.5, 0.25], dtype=sf.sf16)
    z += 0.25
    z *= y8
    assert z.codes.tolist() == [6144, 4096]
    # foreign operands
    for bad in ("0.5", None, object()):
        with pytest.raises(sf.SFTypeError):
            x + bad
        with pytest.raises(sf.SFTypeError):
            bad * x
    for expr in ("x // 2", "x % 2", "x & x", "~x", "x << 1", "int(x)", "x ** 0.5", "x ** -1",
                 "2 ** x", "round(x)", "[1, 2][x]"):
        with pytest.raises(sf.SFUnsupportedError):
            eval(expr, {"x": x})
    with pytest.raises(sf.SFOverflowError):
        x ** 0


def test_array_events_count_and_policy():
    x = sf.array([0.75, 0.25, -0.75], dtype=sf.sf16)
    with pytest.raises(sf.SFOverflowError, match="overflow in 2 of 3"):
        x + x
    with sf.errstate(over="warn"):
        with pytest.warns(sf.SFOverflowWarning, match="2 of 3"):
            assert (x + x).codes.tolist() == [32767, 16384, -32767]
    with sf.errstate(over="ignore"):
        assert (x + x).codes.tolist() == [32767, 16384, -32767]        # hardware value
    # catalogue order: invalid before divide before overflow before underflow
    with pytest.raises(sf.SFZeroDivisionError):
        sf.array([0.5, 0.5], dtype=sf.sf16) / sf.array([0.25, 0], dtype=sf.sf16)
    with sf.errstate(divide="ignore"):
        with pytest.raises(sf.SFOverflowError):
            sf.array([0.5, 0.5], dtype=sf.sf16) / sf.array([0.25, 0], dtype=sf.sf16)
    with sf.errstate(all="ignore"):
        r = sf.array([0.5, 0.5, 0], dtype=sf.sf16) / sf.array([0.25, 0, 0], dtype=sf.sf16)
        assert r.codes.tolist() == [32767, 32767, 0]


def test_cast_arrays_match_scalar_cast():
    rng = np.random.default_rng(7)
    codes = rng.integers(-32767, 32768, 500)
    x = sf.from_codes(codes, sf.sf16)
    for bits in (4, 8, 12, 15):
        dt = dtype_of(bits)
        with sf.errstate(all="ignore"):
            y = sf.cast(x, dt)
            expect = [sf.cast(sf.from_code(int(k), sf.sf16), dt).code for k in codes]
            assert x.astype(dt).codes.tolist() == expect
        assert y.dtype is dt and y.codes.tolist() == expect
        assert sf.cast(y, sf.sf16).codes.tolist() == (y.codes << (16 - bits)).tolist()
    with pytest.raises(sf.SFOverflowError):
        sf.cast(sf.array([0.999], dtype=sf.sf16), sf.sf4)
    with pytest.warns(sf.SFUnderflowWarning):
        sf.cast(sf.array([2 ** -15], dtype=sf.sf16), sf.sf8)
    assert sf.cast([0.5, 0.25], sf.sf8).codes.tolist() == [64, 32]
    assert sf.cast(np.array([0.5]), "sf16").codes.tolist() == [16384]
