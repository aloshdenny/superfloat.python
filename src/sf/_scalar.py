"""Scalar types sf4 .. sf16 and the Stage 1.0 operator surface (spec section 5).

Rules implemented here:

* **Promotion.**  sfN op sfN gives sfN; sfN op sfM gives the wider type
  (widening is exact).  Narrowing needs ``sf.cast``.
* **Literals.**  Python and NumPy scalars take the SF operand's type and are
  quantized first, so ``x + 1`` overflows at the literal.
* **Comparisons** do not quantize; they compare exact values.
* **Integers.**  ``int(x)`` truncates and is 0 for every value.  Use
  ``x.code`` for the raw integer code.
* **Not supported:** ``//``, ``%``, bitwise operators and non-integer ``**``
  raise ``SFUnsupportedError`` instead of silently returning floats.
"""
from __future__ import annotations

import math
import operator
from fractions import Fraction
from typing import Any

from . import _errstate, _reference as ref
from ._errors import SFInvalidError, SFTypeError, SFUnsupportedError
from ._format import FORMATS, SFFormat, widen_code

_NOT_SET = object()


def _fmt_value(fmt: SFFormat, k: int) -> str:
    return repr(fmt.to_float(k))


class SFScalar:
    """Base class of every sfN scalar.  Instances are immutable: the code is
    set exactly once at construction and any later assignment or deletion
    raises AttributeError, so equality and hash never change."""

    __slots__ = ("_code",)

    def __setattr__(self, name, value):
        if name == "_code" and not hasattr(self, "_code"):
            object.__setattr__(self, name, value)
            return
        raise AttributeError(f"{type(self).__name__} values are immutable")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} values are immutable")
    _fmt: SFFormat                      # set on each concrete subclass

    # Tell NumPy to defer to our reflected operators instead of building
    # object arrays, so ``np.float32(0.5) + sf.sf16(0.25)`` lands here.
    __array_ufunc__ = None

    # --- construction ------------------------------------------------------
    def __init__(self, value: Any = 0) -> None:
        fmt = self._fmt
        if isinstance(value, SFScalar):
            if value._fmt.bits > fmt.bits:
                raise SFTypeError(
                    f"cannot store {value._fmt.name} into {fmt.name} implicitly; "
                    f"use sf.cast(x, sf.{fmt.name})"
                )
            self._code = widen_code(value._code, value._fmt, fmt)
            return
        try:
            k, event = ref.cast(fmt, value)
        except TypeError as e:
            raise SFTypeError(str(e)) from None
        if event == "invalid":
            _errstate.handle(event, f"{value!r} is not a valid {fmt.name} input")
        elif event == "overflow":
            _errstate.handle(
                event, f"{value!r} is not representable in {fmt.name} "
                       f"(max {_fmt_value(fmt, fmt.max_code)})"
            )
        elif event == "underflow":
            _errstate.handle(event, f"{value!r} underflows to 0 in {fmt.name}")
        self._code = k

    @classmethod
    def from_code(cls, code: int) -> "SFScalar":
        """Build from a signed integer code in [-max_code, max_code].
        Any integer-like value (``operator.index``) is accepted; bool is not."""
        code = _as_index(code, "code")
        try:
            cls._fmt.check_code(code)
        except ValueError as e:
            raise SFInvalidError(str(e)) from None
        obj = object.__new__(cls)
        obj._code = code
        return obj

    @classmethod
    def from_bits(cls, bits: int) -> "SFScalar":
        """Build from the sign-magnitude bit pattern the device stores."""
        bits = _as_index(bits, "bits")
        try:
            return cls.from_code(cls._fmt.from_bits(bits))
        except TypeError as e:
            raise SFTypeError(str(e)) from None
        except ValueError as e:
            raise SFInvalidError(str(e)) from None

    # --- introspection -----------------------------------------------------
    @property
    def dtype(self) -> type:
        return type(self)

    @property
    def code(self) -> int:
        """Signed integer code k, value = k / 2^(N-1)."""
        return self._code

    @property
    def bits(self) -> int:
        """Sign-magnitude bit pattern (the device representation)."""
        return self._fmt.to_bits(self._code)

    @property
    def sign(self) -> int:
        """1 for negative values, else 0 (there is no negative zero)."""
        return 1 if self._code < 0 else 0

    @property
    def mantissa(self) -> int:
        """Magnitude code m in [0, 2^(N-1) - 1]."""
        return abs(self._code)

    @property
    def exact(self) -> Fraction:
        """The exact rational value."""
        return self._fmt.to_fraction(self._code)

    def __float__(self) -> float:
        return self._fmt.to_float(self._code)

    def __int__(self) -> int:
        return 0                                   # trunc toward zero of |x| < 1

    def __bool__(self) -> bool:
        return self._code != 0

    def __hash__(self) -> int:
        return hash(self.exact)                    # == hash(float(x)) == hash(Fraction)

    def __repr__(self) -> str:
        return f"{self._fmt.name}({_fmt_value(self._fmt, self._code)})"

    def __str__(self) -> str:
        return _fmt_value(self._fmt, self._code)

    def __format__(self, spec: str) -> str:
        return format(float(self), spec)

    def __reduce__(self):
        return (_rebuild, (self._fmt.bits, self._code))

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    # --- operand coercion --------------------------------------------------
    def _coerce(self, other: Any):
        """Return (fmt, k_self, k_other) or None if ``other`` is foreign."""
        fmt = self._fmt
        if isinstance(other, SFScalar):
            if other._fmt.bits > fmt.bits:
                return other._fmt, widen_code(self._code, fmt, other._fmt), other._code
            return fmt, self._code, widen_code(other._code, other._fmt, fmt)
        if isinstance(other, (str, bytes)):          # strings are constructor-only
            return None
        try:
            k, event = ref.cast(fmt, other)
        except TypeError:
            return None
        if event == "invalid":
            _errstate.handle(event, f"literal {other!r} is not a valid {fmt.name} input")
        elif event == "overflow":
            _errstate.handle(
                event, f"literal {other!r} is not representable in {fmt.name} "
                       f"(max {_fmt_value(fmt, fmt.max_code)})"
            )
        elif event == "underflow":
            _errstate.handle(event, f"literal {other!r} underflows to 0 in {fmt.name}")
        return fmt, self._code, k

    # --- arithmetic --------------------------------------------------------
    def _binary(self, other, op, sym, reflected=False):
        pair = self._coerce(other)
        if pair is None:
            return _reject(self, other, sym, reflected)
        fmt, ka, kb = pair
        if reflected:
            ka, kb = kb, ka
        k, event = op(fmt, ka, kb)
        if event is not None:
            _errstate.handle(event, _binary_message(fmt, ka, kb, sym, k, event))
        return _dtype_of(fmt).from_code(k)

    def __add__(self, other):
        return self._binary(other, ref.add, "+")

    def __radd__(self, other):
        return self._binary(other, ref.add, "+", reflected=True)

    def __sub__(self, other):
        return self._binary(other, ref.sub, "-")

    def __rsub__(self, other):
        return self._binary(other, ref.sub, "-", reflected=True)

    def __mul__(self, other):
        return self._binary(other, ref.mul, "*")

    def __rmul__(self, other):
        return self._binary(other, ref.mul, "*", reflected=True)

    def __truediv__(self, other):
        return self._binary(other, ref.div, "/")

    def __rtruediv__(self, other):
        return self._binary(other, ref.div, "/", reflected=True)

    def __neg__(self):
        return type(self).from_code(-self._code)

    def __pos__(self):
        return self

    def __abs__(self):
        return type(self).from_code(abs(self._code))

    def __pow__(self, n, mod=None):
        fmt = self._fmt
        if mod is not None:
            raise SFUnsupportedError("pow() with a modulus is not supported for SF values")
        try:
            n = _as_index(n, "exponent")
        except SFTypeError:
            raise SFUnsupportedError(
                f"{fmt.name} ** {n!r}: only integer exponents k >= 1 are supported"
            ) from None
        if n == 0:
            _errstate.handle("overflow", f"x ** 0 = 1 is not representable in {fmt.name}")
            return type(self).from_code(fmt.max_code)
        if n < 0:
            raise SFUnsupportedError(
                f"{fmt.name} ** {n}: negative exponents (reciprocals) overflow for "
                "every nonzero SF value and are not exposed"
            )
        k, event = ref.pow_int(fmt, self._code, n)
        if event is not None:
            _errstate.handle(event, f"{self!r} ** {n} underflows to 0 in {fmt.name}")
        return type(self).from_code(k)

    def __rpow__(self, base):
        raise SFUnsupportedError("an SF value cannot be used as an exponent")

    # --- comparisons: exact, no quantization -------------------------------
    def _cmp_value(self, other):
        if isinstance(other, SFScalar):
            return other.exact
        if isinstance(other, (str, bytes)):
            return NotImplemented
        try:
            return ref.to_exact(other)         # Fraction, or float for nan/inf
        except TypeError:
            return NotImplemented

    def __eq__(self, other):
        v = self._cmp_value(other)
        if v is NotImplemented:
            return NotImplemented
        if isinstance(v, float):               # nan or ±inf
            return False
        return self.exact == v

    def __ne__(self, other):
        r = self.__eq__(other)
        return r if r is NotImplemented else not r

    def _order(self, other, op):
        v = self._cmp_value(other)
        if v is NotImplemented:
            return NotImplemented
        if isinstance(v, float):
            if math.isnan(v):
                return False
            return op(0, v)                    # every SF value sits between ±inf
        return op(self.exact, v)

    def __lt__(self, other):
        return self._order(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._order(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._order(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._order(other, lambda a, b: a >= b)

    # --- explicitly unsupported -------------------------------------------
    def _unsupported(self, what):
        raise SFUnsupportedError(
            f"{what} is not supported for SF values (it would silently produce a "
            f"float or an integer); use x.code for integer arithmetic on codes"
        )

    def __floordiv__(self, o): self._unsupported("//")
    def __rfloordiv__(self, o): self._unsupported("//")
    def __mod__(self, o): self._unsupported("%")
    def __rmod__(self, o): self._unsupported("%")
    def __divmod__(self, o): self._unsupported("divmod()")
    def __rdivmod__(self, o): self._unsupported("divmod()")
    def __lshift__(self, o): self._unsupported("<<")
    def __rlshift__(self, o): self._unsupported("<<")
    def __rshift__(self, o): self._unsupported(">>")
    def __rrshift__(self, o): self._unsupported(">>")
    def __and__(self, o): self._unsupported("&")
    def __rand__(self, o): self._unsupported("&")
    def __or__(self, o): self._unsupported("|")
    def __ror__(self, o): self._unsupported("|")
    def __xor__(self, o): self._unsupported("^")
    def __rxor__(self, o): self._unsupported("^")
    def __invert__(self): self._unsupported("~")
    def __matmul__(self, o): self._unsupported("@ on scalars")
    def __rmatmul__(self, o): self._unsupported("@ on scalars")
    def __index__(self): self._unsupported("using an SF value as an index")
    def __round__(self, n=None): self._unsupported("round()")
    def __trunc__(self): return 0
    def __floor__(self): self._unsupported("math.floor()")
    def __ceil__(self): self._unsupported("math.ceil()")


_REFLECTED = {"+": "__radd__", "-": "__rsub__", "*": "__rmul__", "/": "__rtruediv__"}
_FORWARD = {"+": "__add__", "-": "__sub__", "*": "__mul__", "/": "__truediv__"}


def _reject(self, other, sym, reflected):
    """A foreign operand.  Give a third-party numeric type its chance via the
    Python protocol (NotImplemented) only if it defines the matching
    operator itself; anything else gets the catalogue's SFTypeError."""
    hook = _FORWARD[sym] if reflected else _REFLECTED[sym]
    if not isinstance(other, (str, bytes, type(None), list, tuple, dict, set, frozenset)) \
            and hasattr(type(other), hook):
        return NotImplemented
    a, b = (other, self) if reflected else (self, other)
    raise SFTypeError(
        f"unsupported operand type(s) for {sym}: {type(a).__name__!r} and "
        f"{type(b).__name__!r} (SF values combine with SF values and host numbers)"
    )


def _as_index(v, what: str) -> int:
    """Integer-like -> int via operator.index; bool and floats are rejected."""
    if isinstance(v, bool):
        raise SFTypeError(f"{what} must be an integer, not bool")
    try:
        return operator.index(v)
    except TypeError:
        raise SFTypeError(f"{what} must be an integer, got {type(v).__name__}") from None


def _binary_message(fmt, ka, kb, sym, k, event) -> str:
    a, b = _fmt_value(fmt, ka), _fmt_value(fmt, kb)
    if event == "divide":
        return f"{fmt.name}: {a} / {b}: division by zero (device value {_fmt_value(fmt, k)})"
    if event == "overflow":
        exact = _exact_result(fmt, ka, kb, sym)
        return (f"{fmt.name}: {a} {sym} {b} = {exact} is outside ±"
                f"{_fmt_value(fmt, fmt.max_code)} (saturates to {_fmt_value(fmt, k)})")
    if event == "underflow":
        return f"{fmt.name}: {a} {sym} {b} underflows to 0"
    return f"{fmt.name}: {a} {sym} {b}: {event}"


def _exact_result(fmt, ka, kb, sym) -> str:
    a, b = Fraction(ka, fmt.scale), Fraction(kb, fmt.scale)
    r = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[sym]
    f = float(r)
    return repr(f) if Fraction(f) == r else f"{r} (≈{f!r})"


def _rebuild(bits: int, code: int):
    return _dtype_of(FORMATS[bits]).from_code(code)


# --- concrete types --------------------------------------------------------

def _make_type(fmt: SFFormat) -> type:
    cls = type(fmt.name, (SFScalar,), {"__slots__": (), "_fmt": fmt,
                                        "__doc__": f"Signed fixed point Q1.{fmt.frac_bits}, "
                                                   f"{fmt.bits} bits, sign-magnitude."})
    cls.__module__ = "sf"
    cls.__qualname__ = fmt.name
    return cls


DTYPES: dict[int, type] = {bits: _make_type(fmt) for bits, fmt in FORMATS.items()}


def _dtype_of(fmt: SFFormat) -> type:
    return DTYPES[fmt.bits]


def resolve_dtype(dtype: Any) -> type:
    """Accept an sfN class, an sfN instance, an SFFormat, 'sf16' or 16."""
    if isinstance(dtype, type) and issubclass(dtype, SFScalar):
        return dtype
    if isinstance(dtype, SFScalar):
        return type(dtype)
    if isinstance(dtype, SFFormat):
        return DTYPES[dtype.bits]
    if isinstance(dtype, str):
        name = dtype.strip().lower()
        for cls in DTYPES.values():
            if cls._fmt.name == name:
                return cls
    if isinstance(dtype, int) and not isinstance(dtype, bool) and dtype in DTYPES:
        return DTYPES[dtype]
    raise SFTypeError(f"{dtype!r} is not an SF dtype (expected sf.sf4 .. sf.sf16)")
