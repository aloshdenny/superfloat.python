"""Error policy (spec section 3.3) and the error catalogue (section 3.2)."""
from __future__ import annotations

import asyncio
import threading
import warnings

import pytest

import sf


def test_defaults():
    assert sf.geterr() == {"over": "raise", "under": "warn", "divide": "raise", "invalid": "raise"}


def test_seterr_returns_old_and_sets_new():
    old = sf.seterr(over="warn", under="ignore", divide="raise", invalid="raise")
    assert old == {"over": "raise", "under": "warn", "divide": "raise", "invalid": "raise"}
    assert sf.geterr() == {"over": "warn", "under": "ignore", "divide": "raise", "invalid": "raise"}
    with pytest.warns(sf.SFOverflowWarning):
        assert (sf.sf16(0.75) + 0.5).code == 32767
    assert (sf.sf16(0.005) * 0.005).code == 0            # silent underflow
    sf.seterr(**old)
    assert sf.geterr() == old
    assert sf.seterr(all="ignore") == old
    assert set(sf.geterr().values()) == {"ignore"}
    sf.seterr(all="raise", under="warn")
    assert sf.geterr()["under"] == "warn" and sf.geterr()["over"] == "raise"


def test_bad_keys_and_values():
    with pytest.raises(ValueError):
        sf.seterr(overflow="raise")
    with pytest.raises(ValueError):
        sf.seterr(over="explode")
    with pytest.raises(ValueError):
        sf.errstate(bogus="warn")


def test_errstate_context_restores():
    before = sf.geterr()
    with sf.errstate(over="ignore"):
        assert sf.geterr()["over"] == "ignore"
        assert (sf.sf16(0.75) + 0.5).code == 32767       # silent saturation, same bits as the GPU
        with sf.errstate(over="warn"):
            with pytest.warns(sf.SFOverflowWarning):
                sf.sf16(0.75) + 0.5
        assert sf.geterr()["over"] == "ignore"
    assert sf.geterr() == before
    # restored even when the block raises
    with pytest.raises(RuntimeError):
        with sf.errstate(all="ignore"):
            raise RuntimeError
    assert sf.geterr() == before


def test_errstate_as_decorator():
    @sf.errstate(over="ignore", under="ignore")
    def f():
        return sf.sf16(0.75) + 0.5, sf.sf16(0.001) * 0.001

    a, b = f()
    assert a.code == 32767 and b.code == 0
    with pytest.raises(sf.SFOverflowError):
        sf.sf16(0.75) + 0.5


def test_value_is_always_the_hardware_value():
    """The policy decides whether the host raises or warns; the value never changes."""
    cases = [
        (lambda: sf.sf16(0.75) + 0.5, 32767),
        (lambda: sf.sf16(-0.75) - 0.5, -32767),
        (lambda: sf.sf16(0.005) * 0.005, 0),
        (lambda: sf.sf16(0.5) / 0, 32767),
        (lambda: sf.sf16(0) / 0, 0),
        (lambda: sf.sqrt(sf.sf16(-0.25)), 0),
        (lambda: sf.sf16(1.5), 32767),
        (lambda: sf.sf16(float("nan")), 0),
        (lambda: sf.sf16(2 ** -17), 0),
    ]
    for fn, code in cases:
        with sf.errstate(all="ignore"):
            assert fn().code == code
        with sf.errstate(all="warn"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                assert fn().code == code
            assert len(w) == 1 and issubclass(w[0].category, sf.SFWarning)
        with sf.errstate(all="raise"):
            with pytest.raises(sf.SFError):
                fn()


def test_error_catalogue_types():
    """Spec section 3.2: Python type and base class per event."""
    with pytest.raises(sf.SFTypeError) as e:
        sf.sf8(sf.sf16(0.5))                 # narrowing mix
    assert isinstance(e.value, TypeError)
    with pytest.raises(sf.SFTypeError):
        sf.dtype("sf3")                      # bad dtype
    with pytest.raises(sf.SFOverflowError) as e:
        sf.sf16(1)
    assert isinstance(e.value, OverflowError)
    with pytest.raises(sf.SFOverflowError):
        sf.sf16(0.75) + 0.5
    with pytest.warns(sf.SFUnderflowWarning):
        sf.sf16(1e-6)
    with pytest.warns(sf.SFUnderflowWarning):
        sf.sf16(0.005) * 0.005
    with pytest.raises(sf.SFZeroDivisionError) as e:
        sf.sf16(0.5) / sf.sf16(0)
    assert isinstance(e.value, ZeroDivisionError)
    with pytest.raises(sf.SFInvalidError) as e:
        sf.sf16(float("nan"))
    assert isinstance(e.value, ValueError)
    with pytest.raises(sf.SFInvalidError):
        sf.sqrt(sf.sf16(-0.25))
    with pytest.raises(sf.SFUnsupportedError) as e:
        sf.sf16(0.5) // 2
    assert isinstance(e.value, NotImplementedError)
    assert issubclass(sf.SFShapeError, ValueError)
    assert issubclass(sf.SFLaunchError, RuntimeError)
    assert issubclass(sf.SFCompileError, RuntimeError)
    assert issubclass(sf.SFDeviceError, RuntimeError)
    assert issubclass(sf.SFDeviceOverflowWarning, sf.SFWarning)
    for cls in (sf.SFTypeError, sf.SFOverflowError, sf.SFZeroDivisionError,
                sf.SFInvalidError, sf.SFUnsupportedError):
        assert issubclass(cls, sf.SFError)
    for cls in (sf.SFUnderflowWarning, sf.SFOverflowWarning, sf.SFZeroDivisionWarning,
                sf.SFInvalidWarning):
        assert issubclass(cls, sf.SFWarning) and issubclass(cls, RuntimeWarning)


def test_errors_point_at_the_user_line():
    """Raised at the Python line that caused the event; warnings likewise."""
    try:
        sf.sf16(0.75) + 0.5
    except sf.SFOverflowError as e:
        tb = e.__traceback__
        assert tb.tb_frame.f_code.co_filename == __file__
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sf.sf16(0.005) * 0.005                               # noqa: this line
        line = w[0].lineno
    assert w[0].filename == __file__
    import inspect
    assert "noqa: this line" in inspect.getsource(test_errors_point_at_the_user_line).splitlines()[
        line - test_errors_point_at_the_user_line.__code__.co_firstlineno]


def test_messages_are_informative():
    with pytest.raises(sf.SFOverflowError, match=r"0\.75 \+ 0\.5 = 1\.25 is outside ±0\.999969482421875"):
        sf.sf16(0.75) + 0.5
    with pytest.raises(sf.SFOverflowError, match=r"1 is not representable in sf16 \(max 0\.999969482421875\)"):
        sf.sf16(1)
    with pytest.raises(sf.SFOverflowError, match=r"0\.5 / 0\.25 = 2\.0 is outside"):
        sf.sf16(0.5) / sf.sf16(0.25)
    with pytest.raises(sf.SFZeroDivisionError, match="division by zero"):
        sf.sf16(0.5) / sf.sf16(0)
    with pytest.raises(sf.SFInvalidError, match="nan"):
        sf.sf16(float("nan"))
    with pytest.raises(sf.SFTypeError, match=r"use sf\.cast\(x, sf\.sf8\)"):
        sf.sf8(sf.sf16(0.5))


def test_unsupported_operators():
    x, y = sf.sf16(0.5), sf.sf16(0.25)
    for expr in ("x // y", "x % y", "divmod(x, y)", "x << 1", "x >> 1", "x & y", "x | y",
                 "x ^ y", "~x", "x @ y", "1 // x", "2 % x", "round(x)", "x ** 0.5",
                 "x ** -1", "2 ** x", "[1, 2][x]", "math.floor(x)", "math.ceil(x)"):
        with pytest.raises(sf.SFUnsupportedError):
            eval(expr, {"x": x, "y": y, "math": __import__("math")})


def test_policy_is_context_local_across_threads():
    seen = {}

    def worker():
        sf.seterr(all="ignore")
        seen["thread"] = sf.geterr()["over"]

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    assert seen["thread"] == "ignore"
    assert sf.geterr()["over"] == "raise"      # another thread's seterr does not leak here


def test_policy_is_context_local_across_tasks():
    async def inner():
        with sf.errstate(over="ignore"):
            await asyncio.sleep(0)
            return sf.geterr()["over"]

    async def outer():
        with sf.errstate(over="warn"):
            r = await asyncio.gather(inner(), asyncio.sleep(0, result=sf.geterr()["over"]))
            return r, sf.geterr()["over"]

    (inner_policy, sibling), after = asyncio.run(outer())
    assert inner_policy == "ignore" and sibling == "warn" and after == "warn"
