"""Regression tests for the independent Stage 1.0 review findings."""
from __future__ import annotations

import asyncio
import copy
import pickle
import threading
import time

import numpy as np
import pytest

import sf
from conftest import ALL_WIDTHS, dtype_of


# P1: scalars are immutable and keep their hash --------------------------------

def test_scalar_is_immutable():
    x = sf.sf16(0.5)
    d = {x: "half"}
    for attempt in (lambda: setattr(x, "_code", 3),
                    lambda: delattr(x, "_code"),
                    lambda: setattr(x, "code", 3),
                    lambda: setattr(x, "anything", 3),
                    lambda: object.__setattr__(x, "__class__", sf.sf8) if False else setattr(x, "_fmt", None)):
        with pytest.raises(AttributeError):
            attempt()
    assert x.code == 16384 and hash(x) == hash(0.5) and d[sf.sf16(0.5)] == "half"
    # construction paths all go through the one-time set
    for y in (sf.sf16(0.25), sf.from_code(3, sf.sf16), sf.from_bits(3, sf.sf16),
              pickle.loads(pickle.dumps(x)), copy.copy(x), copy.deepcopy(x), -x, abs(x), x + x * 0):
        with pytest.raises(AttributeError):
            y._code = 1


# P1: errstate decorator under concurrency and on coroutines ---------------------

def test_decorator_is_safe_across_threads():
    @sf.errstate(over="ignore")
    def f():
        time.sleep(0.005)
        return sf.sf16(0.75) + 0.5

    errors, results = [], []

    def worker():
        try:
            results.append(f().code)
        except BaseException as e:            # noqa: BLE001
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == [] and results == [32767] * 16
    assert sf.geterr()["over"] == "raise"


def test_same_errstate_object_entered_from_several_threads():
    es = sf.errstate(over="ignore")
    seen = []

    def worker():
        with es:
            time.sleep(0.005)
            seen.append((sf.sf16(0.75) + 0.5).code)
        seen.append(sf.geterr()["over"])

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert seen.count(32767) == 8 and seen.count("raise") == 8


def test_decorator_wraps_coroutines():
    @sf.errstate(over="ignore")
    async def f():
        await asyncio.sleep(0)
        return sf.sf16(0.75) + 0.5

    async def main():
        r = await asyncio.gather(*(f() for _ in range(8)))
        return [x.code for x in r], sf.geterr()["over"]

    codes, after = asyncio.run(main())
    assert codes == [32767] * 8 and after == "raise"
    assert asyncio.iscoroutinefunction(f)


def test_decorator_rejects_generators():
    with pytest.raises(TypeError):
        @sf.errstate(over="ignore")
        def gen():
            yield 1

    with pytest.raises(TypeError):
        @sf.errstate(over="ignore")
        async def agen():
            yield 1


def test_exit_without_enter_is_an_error():
    es = sf.errstate(over="ignore")
    with pytest.raises(RuntimeError):
        es.__exit__(None, None, None)
    assert sf.geterr()["over"] == "raise"


# P2: foreign operator operands raise SFTypeError ---------------------------------

def test_foreign_operands_raise_sf_type_error():
    x = sf.sf16(0.5)
    for other in ("0.25", b"1", None, [0.25], (0.25,), {"a": 1}, {0.25}, object()):
        for expr in (lambda: x + other, lambda: other + x, lambda: x - other, lambda: other - x,
                     lambda: x * other, lambda: other * x, lambda: x / other, lambda: other / x):
            with pytest.raises(sf.SFTypeError):
                expr()
            with pytest.raises(TypeError):          # still a TypeError for generic handlers
                expr()
    # a third-party numeric type that implements the reflected op still gets it
    class Wrapper:
        def __init__(self, v):
            self.v = v

        def __radd__(self, other):
            return ("wrapped", float(other) + self.v)

        def __mul__(self, other):
            return ("wrapped", float(other) * self.v)

    assert x + Wrapper(1.0) == ("wrapped", 1.5)
    assert Wrapper(2.0) * x == ("wrapped", 1.0)
    with pytest.raises(sf.SFTypeError):
        x - Wrapper(1.0)                              # Wrapper has no __rsub__
    # Fraction/Decimal/numpy stay host numbers
    from decimal import Decimal
    from fractions import Fraction
    assert (x + Fraction(1, 4)).code == 0x6000 and (Decimal("0.25") + x).code == 0x6000


# P2: integer-like inputs ----------------------------------------------------------

def test_integer_like_inputs():
    x = sf.sf16(0.5)
    assert (x ** np.int64(2)).code == (x ** 2).code == 0x2000
    assert (x ** np.uint8(3)).code == 0x1000
    assert sf.from_code(np.int64(-3), sf.sf16).code == -3
    assert sf.from_code(np.int16(3), sf.sf4).code == 3
    assert sf.from_bits(np.uint16(0x8003), sf.sf16).code == -3
    assert sf.from_bits(np.int32(3), sf.sf16).code == 3
    assert type(sf.from_code(np.int64(3), sf.sf16).code) is int
    assert type(sf.from_bits(np.uint16(3), sf.sf16).code) is int

    class Idx:
        def __index__(self):
            return 2

    assert (x ** Idx()).code == 0x2000 and sf.from_code(Idx(), sf.sf16).code == 2
    for bad in (True, 2.0, np.float32(2), "2", None):
        with pytest.raises(sf.SFTypeError):
            sf.from_code(bad, sf.sf16)
        with pytest.raises(sf.SFTypeError):
            sf.from_bits(bad, sf.sf16)
        with pytest.raises(sf.SFUnsupportedError):
            x ** bad


# P2: large powers finish immediately ------------------------------------------------

def test_large_powers_are_bounded():
    t = time.perf_counter()
    assert (sf.sf16(0) ** (10 ** 9)).code == 0
    with sf.errstate(under="ignore"):
        assert (sf.sf16(0.5) ** (10 ** 9)).code == 0
        assert (sf.from_code(32767, sf.sf16) ** (10 ** 12)).code == 0
        assert (sf.from_code(-32767, sf.sf16) ** (10 ** 12 + 1)).code == 0
    assert time.perf_counter() - t < 2.0
    # the result is still the left-to-right chain where it is nonzero
    y = sf.from_code(32767, sf.sf16)
    with sf.errstate(under="ignore"):
        assert (y ** 100).code == 32767 - 99               # each step drops exactly one code
        chain = y
        for _ in range(99):
            chain = chain * y
        assert (y ** 100).code == chain.code
    with pytest.warns(sf.SFUnderflowWarning):
        sf.sf16(0.5) ** 16                                 # the step that reaches 0 warns


# P2: single-value coverage is complete on every width ---------------------------------

@pytest.mark.parametrize("bits", ALL_WIDTHS)
def test_every_single_code_is_covered(bits):
    from conftest import singles
    m = sf.info(dtype_of(bits)).max_code
    assert list(singles(bits)) == list(range(-m, m + 1))
