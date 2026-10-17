"""Every code example in the spec that belongs to Stage 1.0, run as written."""
from __future__ import annotations

import pytest

import sf


def test_section_3_error_table_examples():
    with pytest.raises(sf.SFOverflowError):
        sf.sf16(0.75) + 0.5                         # raises SFOverflowError immediately
    with pytest.raises(sf.SFTypeError):
        sf.dtype("sf3")                             # bad dtype
    with pytest.raises(sf.SFOverflowError):
        sf.sf16(1)
    with pytest.warns(sf.SFUnderflowWarning):
        sf.sf16(1e-6)
    with pytest.warns(sf.SFUnderflowWarning):
        sf.sf16(0.005) * 0.005
    with pytest.raises(sf.SFZeroDivisionError):
        sf.sf16(0.5) / sf.sf16(0)
    with pytest.raises(sf.SFInvalidError):
        sf.sf16(float("nan"))
    with pytest.raises(sf.SFInvalidError):
        sf.sqrt(sf.sf16(-0.25))


def test_section_3_1_examples():
    with sf.errstate(over="ignore"):
        assert float(sf.sf16(0.75) + 0.5) == 0.999969482421875
    with sf.errstate(under="ignore"):
        assert (sf.sf16(0.005) * 0.005).code == 0
    assert sf.info(sf.sf16).max == 0.999969482421875
    # cast overflow threshold for sf16
    with pytest.raises(sf.SFOverflowError):
        sf.sf16(0.9999847412109375)
    assert sf.sf16(0.9999847412109374).code == 32767
    # a * b + c gives the same bits as sf.fma(a, b, c)
    a, b, c = sf.sf16(0.3), sf.sf16(0.7), sf.sf16(-0.1)
    assert (a * b + c).code == sf.fma(a, b, c).code
    # (0.9 + 0.9) - 0.9 = 0.09997 but 0.9 + (0.9 - 0.9) = 0.9
    with sf.errstate(over="ignore"):
        x = sf.sf16(0.9)
        # exactly (32767 - 29491) / 32768 = 0.0999755859375; the spec prints 0.09997
        assert ((x + x) - x).code == 32767 - 29491
        assert float((x + x) - x) == pytest.approx(0.09997, abs=1e-5)
        assert (x + (x - x)).code == 29491 and f"{float(x + (x - x)):.1f}" == "0.9"


def test_section_3_3_error_policy():
    with pytest.raises(sf.SFOverflowError):
        sf.sf16(0.75) + 0.5                         # 1.25 > sf16 max (0.999969482421875)
    with sf.errstate(over="ignore"):                # silent saturation, same bits as the GPU
        assert repr(sf.sf16(0.75) + 0.5) == "sf16(0.999969482421875)"
    old = sf.seterr(over="warn", under="ignore", divide="raise", invalid="raise")
    try:
        with pytest.warns(sf.SFOverflowWarning):
            sf.sf16(0.75) + 0.5
        assert (sf.sf16(0.005) * 0.005).code == 0  # no warning
    finally:
        sf.seterr(**old)


def test_section_5_scalar_examples():
    a, b = sf.sf16(0.5), sf.sf16(0.25)
    assert repr(a + b) == "sf16(0.75)"
    assert repr(a * b) == "sf16(0.125)"
    assert repr(b / a) == "sf16(0.5)"
    with pytest.raises(sf.SFOverflowError):
        a / b                                       # 2.0 is outside sf16
    with pytest.raises(sf.SFZeroDivisionError):
        a / sf.sf16(0)                              # caught on host
    with pytest.raises(sf.SFOverflowError):
        sf.sf16(1)                                  # 1 is not representable
    assert (a < 1) is True                          # comparisons are exact


def test_section_5_rules():
    x = sf.sf16(0.5)
    # Promotion
    assert type(x + sf.sf16(0.25)) is sf.sf16
    assert type(x + sf.sf8(0.25)) is sf.sf16
    with pytest.raises(sf.SFTypeError):
        sf.sf8(x)
    # Literals: x + 1 overflows at the literal; 0 is the only representable integer
    with pytest.raises(sf.SFOverflowError, match="literal"):
        x + 1
    assert (x + 0).code == x.code
    # Comparisons do not quantize
    assert x < 1.0 and sf.sf16(0.1) != 0.1
    # Integers
    assert int(x) == 0 and int(sf.sf16(-0.99)) == 0
    assert x.code == 16384 and sf.from_code(16384, sf.sf16) == x
    # Not supported
    for bad in (lambda: x // 1, lambda: x % 1, lambda: x & 1, lambda: x ** 0.5):
        with pytest.raises(sf.SFUnsupportedError):
            bad()


def test_section_1_0_1_api_surface():
    x = sf.sf16(-0.75)
    assert x.sign == 1 and x.mantissa == 24576 and x.code == -24576
    assert float(x) == -0.75
    assert sf.cast(x, sf.sf8).code == -96
    assert sf.info(sf.sf16).bits == 16 and sf.info(sf.sf16).lsb == 2 ** -15
    assert sf.info(sf.sf16).min == -sf.info(sf.sf16).max
    assert sf.sf4 is sf.dtypes[0] and sf.sf16 is sf.dtypes[-1]
    import numpy as np
    assert sf.sf16(np.float32(0.25)) == 0.25
    assert hasattr(sf, "errstate") and hasattr(sf, "seterr") and hasattr(sf, "geterr")
