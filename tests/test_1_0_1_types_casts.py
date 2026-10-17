"""Milestone 1.0.1: types and casts.

Unit: every code of every width round-trips; ties to even; ±max,
±(max + ½ LSB), ±½ LSB, -0.0, NaN, ±inf, ints.
Mix/regression: golden file of all codes (test_golden.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import sf
from conftest import ALL_WIDTHS, dtype_of, singles, width_params
from oracle import Fmt


# --- the format table in spec section 2 -----------------------------------

SPEC_TABLE = {   # bits: (lsb, max, codes)
    16: (2 ** -15, 0.999969482421875, 65535),
    15: (2 ** -14, 0.99993896484375, 32767),
    14: (2 ** -13, 0.9998779296875, 16383),
    13: (2 ** -12, 0.999755859375, 8191),
    12: (2 ** -11, 0.99951171875, 4095),
    11: (2 ** -10, 0.9990234375, 2047),
    10: (2 ** -9, 0.998046875, 1023),
    9: (2 ** -8, 0.99609375, 511),
    8: (2 ** -7, 0.9921875, 255),
    7: (2 ** -6, 0.984375, 127),
    6: (2 ** -5, 0.96875, 63),
    5: (2 ** -4, 0.9375, 31),
    4: (2 ** -3, 0.875, 15),
}


@pytest.mark.parametrize("bits", ALL_WIDTHS)
def test_info_matches_spec_table(bits):
    lsb, mx, codes = SPEC_TABLE[bits]
    i = sf.info(dtype_of(bits))
    assert i.bits == bits
    assert i.lsb == lsb
    assert i.max == mx and i.min == -mx
    assert i.codes == codes
    assert i.max_code == codes // 2
    assert i.exact_max == Fraction(mx) and i.exact_lsb == Fraction(lsb)
    assert sf.info(f"sf{bits}") == i == sf.info(bits) == sf.info(dtype_of(bits)(0))


def test_dtype_objects():
    assert sf.sf16 is sf.dtypes[-1] and sf.sf4 is sf.dtypes[0]
    assert len(sf.dtypes) == 13
    assert sf.dtype("sf12") is sf.sf12 and sf.dtype(sf.sf12(0)) is sf.sf12
    assert isinstance(sf.sf16(0.5), sf.SFScalar)
    assert sf.sf16(0.5).dtype is sf.sf16
    assert not hasattr(sf, "sf3") and not hasattr(sf, "sf17")
    with pytest.raises(sf.SFTypeError):
        sf.dtype("sf3")
    with pytest.raises(sf.SFTypeError):
        sf.dtype(17)
    with pytest.raises(sf.SFTypeError):
        sf.dtype(float)


# --- round trips ------------------------------------------------------------

@pytest.mark.parametrize("bits", width_params(ALL_WIDTHS))
def test_every_code_round_trips(bits):
    dt = dtype_of(bits)
    fmt = Fmt(bits)
    for k in singles(bits):
        x = sf.from_code(k, dt)
        assert x.code == k
        # value -> float -> construct -> same code (every value is exact in float64)
        assert dt(float(x)).code == k
        assert dt(x.exact).code == k
        assert dt(str(x.exact)).code == k
        assert dt(Decimal(float(x))).code == k
        # device bit pattern round trip, sign-magnitude
        assert sf.from_bits(x.bits, dt).code == k
        assert x.sign == (1 if k < 0 else 0)
        assert x.mantissa == abs(k)
        assert x.bits == (abs(k) | (x.sign << (bits - 1)))
        # exact value agrees with the oracle's value grid
        assert x.exact == k * fmt.lsb
        # widening to sf16 is exact; same-width copy is identity
        assert sf.sf16(x).exact == x.exact
        assert dt(x).code == k and sf.cast(x, dt).code == k


@pytest.mark.parametrize("bits", ALL_WIDTHS)
def test_negative_zero_is_canonical(bits):
    dt = dtype_of(bits)
    z = dt(-0.0)
    assert z.code == 0 and z.sign == 0 and z.bits == 0
    assert z == dt(0.0) and not z
    assert math.copysign(1.0, float(z)) == 1.0
    # the device may hand us a -0 pattern; it is folded to +0
    assert sf.from_bits(1 << (bits - 1), dt).code == 0
    assert (-z).code == 0 and abs(z).code == 0 and repr(-z) == f"sf{bits}(0.0)"


# --- rounding ----------------------------------------------------------------

@pytest.mark.parametrize("bits", width_params(ALL_WIDTHS))
def test_ties_to_even(bits):
    dt = dtype_of(bits)
    lsb = sf.info(dt).exact_lsb
    m = sf.info(dt).max_code
    with sf.errstate(under="ignore"):
        for k in singles(bits):
            if k == m:
                continue                       # k + ½ is the overflow case below
            tie = (k + Fraction(1, 2)) * lsb    # exactly halfway between k and k+1
            expect = k if k % 2 == 0 else k + 1  # nearest even code
            assert dt(tie).code == expect, f"tie at k={k}"
            assert dt(float(tie)).code == expect
            # just below / above the tie go to k / k+1
            assert dt(tie - lsb / 8).code == k
            assert dt(tie + lsb / 8).code == k + 1


@pytest.mark.parametrize("bits", ALL_WIDTHS)
def test_boundary_values(bits):
    dt = dtype_of(bits)
    i = sf.info(dt)
    lsb, mx, m = i.exact_lsb, i.exact_max, i.max_code
    assert dt(mx).code == m and dt(-mx).code == -m
    assert dt(float(mx)).code == m
    # ±(max + ½ LSB) is a tie between M (odd) and M+1 (even): rounds up -> overflow
    for v in (mx + lsb / 2, -(mx + lsb / 2), Fraction(1), Fraction(-1), 1.0, -1.0):
        with pytest.raises(sf.SFOverflowError):
            dt(v)
        with sf.errstate(over="ignore"):
            assert dt(v).code == (m if v > 0 else -m)         # saturation
    # the spec's cast overflow threshold: |v| >= 1 - 2^-N
    thr = Fraction(1) - Fraction(1, 2 ** bits)
    with pytest.raises(sf.SFOverflowError):
        dt(thr)
    assert dt(thr - lsb / 1024).code == m
    if bits == 16:
        assert thr == Fraction("0.9999847412109375")
    # ±½ LSB is a tie between 0 and 1: rounds to 0 -> underflow
    for v in (lsb / 2, -lsb / 2, Fraction(1, 2 ** bits), lsb / 4):
        with pytest.warns(sf.SFUnderflowWarning):
            assert dt(v).code == 0
        with sf.errstate(under="raise"):
            with pytest.raises(sf.SFUnderflowError):
                dt(v)
    assert dt(lsb / 2 + lsb / 1024).code == 1   # just above the tie
    assert dt(lsb).code == 1 and dt(-lsb).code == -1


@pytest.mark.parametrize("bits", ALL_WIDTHS)
def test_nan_inf_and_ints(bits):
    dt = dtype_of(bits)
    m = sf.info(dt).max_code
    for nan in (float("nan"), np.float32("nan"), Decimal("NaN")):
        with pytest.raises(sf.SFInvalidError):
            dt(nan)
        with sf.errstate(invalid="ignore"):
            assert dt(nan).code == 0
    for inf, sign in ((float("inf"), 1), (float("-inf"), -1), (np.float64("inf"), 1),
                      (Decimal("-Infinity"), -1)):
        with pytest.raises(sf.SFOverflowError):
            dt(inf)
        with sf.errstate(over="ignore"):
            assert dt(inf).code == sign * m
    # 0 is the only representable integer
    assert dt(0).code == 0 and dt(np.int32(0)).code == 0 and dt(False).code == 0
    for n in (1, -1, 2, 1000, True, np.int64(7), np.uint8(1)):
        with pytest.raises(sf.SFOverflowError):
            dt(n)
        with sf.errstate(over="ignore"):
            assert dt(n).code == (m if n > 0 else -m)


@pytest.mark.parametrize("bits", ALL_WIDTHS)
def test_cast_matches_oracle_on_random_floats(bits):
    dt = dtype_of(bits)
    fmt = Fmt(bits)
    rnd = np.random.default_rng(bits)
    vals = np.concatenate([
        rnd.uniform(-1.2, 1.2, 4000),
        rnd.uniform(-2 ** -(bits - 3), 2 ** -(bits - 3), 2000),  # around underflow
        rnd.standard_normal(2000) * 1e-3,
    ])
    with sf.errstate(all="ignore"):
        for v in vals.tolist() + [float(x) for x in rnd.standard_normal(500).astype(np.float32)]:
            r, ev = fmt.cast(v)
            x = dt(v)
            assert x.exact == r, v
            # the event the policy would see
            with sf.errstate(all="raise"):
                if ev is None:
                    dt(v)
                else:
                    with pytest.raises({"overflow": sf.SFOverflowError,
                                        "underflow": sf.SFUnderflowError,
                                        "invalid": sf.SFInvalidError}[ev]):
                        dt(v)


@settings(max_examples=300, deadline=None)
@given(v=st.floats(allow_nan=False, allow_infinity=False, width=64),
       bits=st.integers(4, 16))
def test_cast_matches_oracle_hypothesis(v, bits):
    dt = dtype_of(bits)
    r, ev = Fmt(bits).cast(v)
    with sf.errstate(all="ignore"):
        assert dt(v).exact == r


# --- numpy scalars, Decimal, Fraction, str --------------------------------------

def test_host_scalar_types():
    assert sf.sf16(np.float32(0.25)).code == 8192
    assert sf.sf16(np.float16(0.25)).code == 8192
    assert sf.sf16(np.float64(-0.5)).code == -16384
    assert sf.sf16(Fraction(1, 4)).code == 8192
    assert sf.sf16(Decimal("0.25")).code == 8192
    assert sf.sf16("0.25").code == 8192 and sf.sf16(" -1/4 ").code == -8192
    # 0.1 as a decimal string is quantized from exactly 1/10, not from the float
    assert sf.sf16("0.1").code == round(Fraction(1, 10) * 2 ** 15) == 3277
    assert sf.sf16(0.1).code == round(Fraction(0.1) * 2 ** 15) == 3277
    for bad in ("abc", None, [0.5], object(), 1j):
        with pytest.raises(sf.SFTypeError):
            sf.sf16(bad)


# --- casts between widths -------------------------------------------------------

def test_narrowing_needs_cast():
    x = sf.sf16(0.1)
    with pytest.raises(sf.SFTypeError):
        sf.sf8(x)
    y = sf.cast(x, sf.sf8)
    assert type(y) is sf.sf8
    assert y.exact == Fmt(8).round_half_even(x.exact)
    assert sf.cast(sf.sf8(0.5), sf.sf16).code == 0x4000     # exact widening
    assert sf.cast(0.5, "sf12").code == 1024                 # host scalar
    # narrowing can underflow / overflow and follows the policy
    with pytest.warns(sf.SFUnderflowWarning):
        assert sf.cast(sf.sf16(2 ** -15), sf.sf8).code == 0
    with pytest.raises(sf.SFOverflowError):
        sf.cast(sf.sf16(0.999), sf.sf4)                      # 0.999 rounds to 1.0 in sf4
    with sf.errstate(over="ignore"):
        assert sf.cast(sf.sf16(0.999), sf.sf4).code == 7


@pytest.mark.parametrize("bits", ALL_WIDTHS)
def test_narrowing_cast_matches_oracle(bits):
    narrow = Fmt(bits)
    dt = dtype_of(bits)
    rnd = np.random.default_rng(100 + bits)
    with sf.errstate(all="ignore"):
        for k in rnd.integers(-32767, 32768, 3000).tolist():
            x = sf.from_code(int(k), sf.sf16)
            expect, _ = narrow.cast(x.exact)
            assert sf.cast(x, dt).exact == expect


# --- from_code / from_bits validation ----------------------------------------------

def test_from_code_and_bits_validation():
    with pytest.raises(sf.SFInvalidError):
        sf.from_code(32768, sf.sf16)
    with pytest.raises(sf.SFInvalidError):
        sf.from_code(-32768, sf.sf16)
    with pytest.raises(sf.SFTypeError):
        sf.from_code(1.0, sf.sf16)
    with pytest.raises(sf.SFTypeError):
        sf.from_code(True, sf.sf16)
    with pytest.raises(sf.SFInvalidError):
        sf.from_bits(1 << 16, sf.sf16)
    with pytest.raises(sf.SFInvalidError):
        sf.from_bits(-1, sf.sf16)
    assert sf.from_bits(0x8001, sf.sf16).code == -1
    assert sf.from_bits(0x7FFF, sf.sf16).code == 32767
    assert sf.from_bits(0xFFFF, sf.sf16).code == -32767


# --- Python protocol surface -----------------------------------------------------------

def test_python_protocols():
    x = sf.sf16(0.75)
    assert repr(x) == "sf16(0.75)" and str(x) == "0.75" and f"{x:.1f}" == "0.8"
    assert float(x) == 0.75 and int(x) == 0 and int(sf.sf16(-0.9)) == 0
    assert math.trunc(x) == 0
    assert bool(x) and not sf.sf16(0)
    assert hash(x) == hash(0.75) == hash(Fraction(3, 4)) == hash(sf.sf4(0.75))
    assert pickle.loads(pickle.dumps(x)) == x and type(pickle.loads(pickle.dumps(x))) is sf.sf16
    assert sf.sf16(0.5).__doc__.startswith("Signed fixed point Q1.15")
    assert sf.sf16.__module__ == "sf" and sf.sf16.__name__ == "sf16"
    # immutable: no attribute assignment
    with pytest.raises(AttributeError):
        x.code = 3
    with pytest.raises(AttributeError):
        x.foo = 3
    assert isinstance(x, sf.SFScalar) and isinstance(x, sf.sf16) and not isinstance(x, sf.sf8)


# --- golden file -------------------------------------------------------------------------

def golden_table(bits: int) -> str:
    """Canonical text of every code: ``code<TAB>bits<TAB>value``."""
    dt = dtype_of(bits)
    m = sf.info(dt).max_code
    return "".join(f"{k}\t{sf.from_code(k, dt).bits:0{(bits + 3) // 4}x}\t{float(sf.from_code(k, dt))!r}\n"
                   for k in range(-m, m + 1))


@pytest.mark.parametrize("bits", ALL_WIDTHS)
def test_golden_codes(bits, request):
    """The golden file is the baseline for every later milestone.

    Full tables are stored for sf4..sf12; every width also has a SHA-256
    recorded in ``golden/checksums.json``.  Regenerate with
    ``python tools/make_golden.py`` only when the format definition changes.
    """
    here = request.config.rootpath / "tests" / "golden"
    table = golden_table(bits)
    digest = hashlib.sha256(table.encode()).hexdigest()
    checksums = json.loads((here / "checksums.json").read_text())
    assert checksums[f"sf{bits}"] == digest, f"sf{bits} code table changed"
    full = here / f"sf{bits}.tsv"
    if full.exists():
        assert full.read_text() == table
