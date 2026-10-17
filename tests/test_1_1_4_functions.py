"""Milestone 1.1.4: elementary functions.

Unit: domain errors; sqrt is monotonic and sqrt(x) >= x.
Mix/regression: sf.div(a, b, where=b != 0) never raises, while
sf.where(b != 0, a / b, 0) does: Python evaluates both branches.
"""
from __future__ import annotations

import numpy as np
import pytest

import sf
from conftest import dtype_of, run


@pytest.mark.parametrize("bits", [4, 8, 16])
def test_sqrt_arrays(bits):
    dt = dtype_of(bits)
    m = sf.info(dt).max_code
    x = sf.from_codes(np.arange(0, m + 1), dt)
    r = sf.sqrt(x)
    assert r.codes.tolist() == [sf.sqrt(sf.from_code(int(k), dt)).code for k in x.codes]
    assert np.all(np.diff(r.codes) >= 0)                      # monotonic
    assert (r >= x).all()                                      # sqrt(x) >= x on [0, 1)
    neg = sf.from_codes(np.array([-1, 4, -m]), dt)
    with pytest.raises(sf.SFInvalidError, match="invalid in 2 of 3"):
        sf.sqrt(neg)
    with sf.errstate(invalid="ignore"):
        assert sf.sqrt(neg).codes.tolist() == [0, sf.sqrt(sf.from_code(4, dt)).code, 0]
    assert sf.sqrt([sf.sf16(0.25)]).codes.tolist() == [16384]
    with pytest.raises(sf.SFTypeError):
        sf.sqrt(np.array([0.25]))


def test_pow_arrays():
    x = sf.array([0.5, -0.5, 0.9, 0], dtype=sf.sf16)
    assert (x ** 1).codes.tolist() == x.codes.tolist()
    assert (x ** 2).codes.tolist() == (x * x).codes.tolist()
    with sf.errstate(under="ignore"):
        assert (x ** 3).codes.tolist() == [(sf.from_code(int(k), sf.sf16) ** 3).code for k in x.codes]
        assert (x ** 10 ** 9).codes.tolist() == [0, 0, 0, 0]
    with pytest.warns(sf.SFUnderflowWarning):
        x ** 20
    for bad in (0.5, -1):
        with pytest.raises(sf.SFUnsupportedError):
            x ** bad
    with pytest.raises(sf.SFOverflowError):
        x ** 0
    assert (x ** np.int64(2)).codes.tolist() == (x ** 2).codes.tolist()


def test_clip_maximum_minimum():
    x = sf.array([0.5, -0.75, 0.1, 0.9], dtype=sf.sf16)
    c = sf.clip(x, -0.2, 0.2)
    assert c.codes.tolist() == [sf.sf16(0.2).code, sf.sf16(-0.2).code, sf.sf16(0.1).code, sf.sf16(0.2).code]
    assert sf.clip(x, sf.sf8(0), 0.5).codes.tolist() == [16384, 0, 3277, 16384]
    assert sf.clip(sf.sf16(0.9), 0, 0.5) == 0.5 and type(sf.clip(sf.sf16(0.9), 0, 0.5)) is sf.sf16
    with pytest.raises(sf.SFInvalidError):
        sf.clip(x, 0.5, -0.5)
    with pytest.raises(sf.SFOverflowError):
        sf.clip(x, -1, 1)                                      # bounds are literals
    assert sf.maximum(x, 0).codes.tolist() == [16384, 0, 3277, sf.sf16(0.9).code]
    assert sf.minimum(x, 0).codes.tolist() == [0, -24576, 0, 0]
    assert sf.maximum(x, sf.array([0.6, 0, 0, 0], dtype=sf.sf8)).dtype is sf.sf16
    assert sf.maximum(sf.sf16(0.5), sf.sf16(0.25)) == 0.5 and sf.minimum(sf.sf8(0.5), 0.25) == 0.25
    with pytest.raises(sf.SFShapeError):
        sf.maximum(x, sf.zeros((3,)))
    # selection ops never raise, even on ±max
    with sf.errstate(all="raise"):
        mx = sf.full((2,), sf.info(sf.sf16).max, sf.sf16)
        assert sf.maximum(mx, -mx).codes.tolist() == [32767, 32767]
        assert sf.clip(mx, -mx, mx).codes.tolist() == [32767, 32767]


def test_div_where_and_where():
    a = sf.array([0.1, 0.2, 0.3, 0.4], dtype=sf.sf16)
    b = sf.array([0.5, 0, 0.6, 0], dtype=sf.sf16)
    # the spec's example: sf.div(a, b, where=b != 0) never raises
    with sf.errstate(all="raise"):
        q = sf.div(a, b, where=b != 0)
    assert q.codes.tolist() == [(a[0] / b[0]).code, 0, (a[2] / b[2]).code, 0]
    # while sf.where(b != 0, a / b, 0) does: Python evaluates both branches
    with pytest.raises(sf.SFZeroDivisionError):
        sf.where(b != 0, a / b, 0)
    with sf.errstate(divide="ignore"):
        w = sf.where(b != 0, a / b, 0)
    assert w.codes.tolist() == q.codes.tolist()
    # mask broadcasting and scalar masks
    assert sf.div(a, 0.5, where=np.array([True, False, True, False])).codes.tolist() == \
        [(a[0] / 0.5).code, 0, (a[2] / 0.5).code, 0]
    assert sf.div(sf.sf16(0.25), sf.sf16(0), where=False) == 0
    with pytest.raises(sf.SFShapeError):
        sf.div(a, b, where=np.array([True, False]))
    # masked elements raise nothing even when they would overflow
    with sf.errstate(all="raise"):
        assert sf.div(sf.array([0.5, 0.5], dtype=sf.sf16), sf.array([0.25, 0.5], dtype=sf.sf16),
                      where=np.array([False, False])).codes.tolist() == [0, 0]
    # sf.where on arrays
    assert sf.where(a > 0.25, a, 0).codes.tolist() == [0, 0, a[2].code, a[3].code]
    assert sf.where(np.array([True, False, True, False]), a, b).codes.tolist() == \
        [a[0].code, b[1].code, a[2].code, b[3].code]
    assert sf.where(True, a, b).codes.tolist() == a.codes.tolist()
    assert sf.where(a[0] > 0, a[0], b[0]) == a[0] and type(sf.where(a[0] > 0, a[0], b[0])) is sf.sf16
    assert sf.where(a > 0.25, a, sf.sf8(0)).dtype is sf.sf16
    assert sf.where(np.array(True), sf.sf16(0.5), sf.sf16(0.25)) == 0.5
    with pytest.raises(sf.SFShapeError):
        sf.where(np.array([True, False]), a, b)


def test_functions_on_device_arrays():
    x = sf.array([0.5, -0.5, 0.25], dtype=sf.sf16, device="atreides")
    assert sf.sqrt(abs(x)).device == "atreides" and sf.sqrt(abs(x)).status.exact
    assert (x ** 2).device == "atreides" and (x ** 2).codes.tolist() == [8192, 8192, 2048]
    assert sf.clip(x, -0.2, 0.2).device == "atreides"
    assert sf.maximum(x, 0).device == "atreides" and sf.where(x > 0, x, 0).device == "atreides"
    assert sf.div(x, x, where=np.array([False, False, False])).device == "atreides"
    with pytest.raises(sf.SFInvalidError):
        sf.sqrt(x)
