"""Module-level functions: sf.cast, sf.info, sf.fma, sf.div, sf.sqrt,
sf.where, sf.from_code, sf.from_bits, sf.dtype."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from . import _errstate, _reference as ref
from ._errors import SFTypeError
from ._format import SFFormat, widen_code
from ._scalar import DTYPES, SFScalar, _binary_message, _dtype_of, _fmt_value, resolve_dtype


def dtype(spec: Any) -> type:
    """Resolve 'sf16', 16, sf.sf16 or an sf16 value to the sf16 class."""
    return resolve_dtype(spec)


@dataclass(frozen=True)
class SFInfo:
    """What ``sf.info(dtype)`` returns, modelled on ``numpy.finfo``."""

    dtype: type
    bits: int
    max: float
    min: float
    lsb: float
    max_code: int
    codes: int
    exact_max: Fraction
    exact_lsb: Fraction

    @property
    def eps(self) -> float:
        return self.lsb

    @property
    def name(self) -> str:
        return self.dtype.__name__

    def __repr__(self) -> str:
        return (f"SFInfo({self.name}: bits={self.bits}, lsb={self.lsb!r}, "
                f"max={self.max!r}, min={self.min!r}, codes={self.codes})")


def info(dt: Any) -> SFInfo:
    """Constants of a format, the way ``numpy.finfo`` works."""
    cls = resolve_dtype(dt)
    fmt: SFFormat = cls._fmt
    return SFInfo(
        dtype=cls, bits=fmt.bits,
        max=float(fmt.max), min=float(fmt.min), lsb=float(fmt.lsb),
        max_code=fmt.max_code, codes=fmt.codes,
        exact_max=fmt.max, exact_lsb=fmt.lsb,
    )


def from_code(code: int, dt: Any) -> SFScalar:
    """Inverse of ``x.code``."""
    return resolve_dtype(dt).from_code(code)


def from_bits(bits: int, dt: Any) -> SFScalar:
    """Inverse of ``x.bits`` (sign-magnitude device pattern)."""
    return resolve_dtype(dt).from_bits(bits)


def cast(x: Any, dt: Any) -> SFScalar:
    """Convert ``x`` to ``dt``.

    Widening is exact.  Narrowing re-quantizes the exact value with
    round-half-even and goes through the error policy (overflow,
    underflow).  Host scalars are quantized like the constructor.
    """
    cls = resolve_dtype(dt)
    if isinstance(x, SFScalar):
        if x._fmt.bits <= cls._fmt.bits:
            return cls.from_code(widen_code(x._code, x._fmt, cls._fmt))
        fmt = cls._fmt
        k, event = ref.cast_exact(fmt, x.exact)
        if event == "overflow":
            _errstate.handle(event, f"{x!r} is not representable in {fmt.name} "
                                    f"(max {_fmt_value(fmt, fmt.max_code)})")
        elif event == "underflow":
            _errstate.handle(event, f"{x!r} underflows to 0 in {fmt.name}")
        return cls.from_code(k)
    return cls(x)


def _operands(name: str, *xs: Any):
    """Promote mixed operands to the widest SF format among them.

    At least one operand must be an SF value (it fixes the type the host
    literals are quantized to), like the binary operators.
    """
    sf_ops = [x for x in xs if isinstance(x, SFScalar)]
    if not sf_ops:
        raise SFTypeError(f"sf.{name}: at least one argument must be an SF value")
    fmt = max((x._fmt for x in sf_ops), key=lambda f: f.bits)
    pivot = _dtype_of(fmt).from_code(0)
    codes = []
    for x in xs:
        pair = pivot._coerce(x)
        if pair is None:
            raise SFTypeError(f"sf.{name}: cannot use {type(x).__name__} as an SF operand")
        codes.append(pair[2])
    return fmt, codes


def fma(a: Any, b: Any, c: Any) -> SFScalar:
    """a * b + c, bit-identical to ``a * b + c`` (the product never
    overflows and the add is exact, so the compiler may fuse freely)."""
    fmt, (ka, kb, kc) = _operands("fma", a, b, c)
    k, event = ref.fma(fmt, ka, kb, kc)
    if event is not None:
        if event == "overflow":
            kp, _ = ref.mul(fmt, ka, kb)
            msg = _binary_message(fmt, kp, kc, "+", k, event)
        else:
            msg = _binary_message(fmt, ka, kb, "*", 0, event)
        _errstate.handle(event, f"fma: {msg}")
    return _dtype_of(fmt).from_code(k)


def div(a: Any, b: Any) -> SFScalar:
    """Same as ``a / b``.  (``where=`` arrives with arrays in Stage 1.1.)"""
    fmt, (ka, kb) = _operands("div", a, b)
    k, event = ref.div(fmt, ka, kb)
    if event is not None:
        _errstate.handle(event, _binary_message(fmt, ka, kb, "/", k, event))
    return _dtype_of(fmt).from_code(k)


def sqrt(a: SFScalar) -> SFScalar:
    """Square root, truncated toward zero.  Negative input is a domain
    error (invalid); the device value is 0."""
    if not isinstance(a, SFScalar):
        raise SFTypeError("sf.sqrt expects an SF value")
    fmt = a._fmt
    k, event = ref.sqrt(fmt, a._code)
    if event is not None:
        _errstate.handle(event, f"{fmt.name}: sqrt({a}) of a negative value "
                                f"(device value {_fmt_value(fmt, k)})")
    return type(a).from_code(k)


def where(cond: Any, x: Any, y: Any) -> Any:
    """``x if cond else y``.  Note that, unlike a conditional expression,
    Python evaluates both ``x`` and ``y`` before ``where`` runs."""
    return x if cond else y


__all__ = ["dtype", "info", "SFInfo", "from_code", "from_bits", "cast",
           "fma", "div", "sqrt", "where"]
