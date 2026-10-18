"""Module-level functions: constructors, casts, info, fma, div, sqrt, clip,
maximum, minimum, where, from_code, from_bits, dtype.  Every function
accepts SF scalars and SF arrays alike."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from . import _device, _errstate, _reference as ref, _vectorized as vec
from ._array import SFArray, _as_sfarray, _elementwise, _handle_events, _nested_to_array, _to_codes, _widest
from ._errors import SFInvalidError, SFShapeError, SFTypeError
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
    return SFInfo(dtype=cls, bits=fmt.bits, max=float(fmt.max), min=float(fmt.min),
                  lsb=float(fmt.lsb), max_code=fmt.max_code, codes=fmt.codes,
                  exact_max=fmt.max, exact_lsb=fmt.lsb)


# --- constructors ----------------------------------------------------------------------

def array(obj: Any, dtype: Any = None, device: str = _device.HOST) -> SFArray | SFScalar:
    """Build an SF array from (nested) lists, NumPy arrays, SF scalars or SF
    arrays.  Host numbers are quantized through the error policy."""
    return SFArray(obj, dtype, device)


def asarray(obj: Any, dtype: Any = None, device: str = _device.HOST) -> SFArray:
    """Like ``array`` but returns the input itself when it is already an SF
    array of the requested dtype on the requested device."""
    if isinstance(obj, SFArray) and (dtype is None or resolve_dtype(dtype) is obj.dtype) \
            and obj.device == device:
        return obj
    return SFArray(obj, dtype, device)


def zeros(shape, dtype: Any = "sf16", device: str = _device.HOST) -> SFArray:
    from .kernels._ir import in_kernel
    if in_kernel():
        from .kernels import _lang
        return _lang.zeros(shape, dtype)
    cls = resolve_dtype(dtype)
    out = SFArray._wrap(np.zeros(shape, dtype=np.int64), cls)
    return out if device == _device.HOST else out.to(device)


def full(shape, value: Any, dtype: Any = None, device: str = _device.HOST) -> SFArray:
    """An array filled with ``value`` (quantized through the error policy)."""
    from .kernels._ir import in_kernel
    if in_kernel():
        from .kernels import _lang
        return _lang.full(shape, value, dtype or "sf16")
    if dtype is None:
        if not isinstance(value, SFScalar):
            raise SFTypeError("sf.full needs dtype= unless value is an SF scalar")
        dtype = type(value)
    cls = resolve_dtype(dtype)
    k = int(_to_codes(value, cls._fmt, what="value"))
    out = SFArray._wrap(np.full(shape, k, dtype=np.int64), cls)
    return out if device == _device.HOST else out.to(device)


def ones(shape, dtype: Any = "sf16", device: str = _device.HOST) -> SFArray:
    """Filled with the literal 1.  1.0 is not representable in any SF format,
    so this follows the policy like any other literal: it raises
    SFOverflowError by default and saturates to max under over='ignore'.
    For the largest representable value use ``sf.full(shape, sf.info(dt).max, dt)``."""
    cls = resolve_dtype(dtype)
    _errstate.handle("overflow", f"sf.ones: 1 is not representable in {cls.__name__} "
                                 f"(max {_fmt_value(cls._fmt, cls._fmt.max_code)}); "
                                 f"use sf.full(shape, sf.info(sf.{cls.__name__}).max, sf.{cls.__name__})")
    return full(shape, cls._fmt.max, cls, device)


def from_codes(codes, dt: Any, device: str = _device.HOST) -> SFArray:
    """Array inverse of ``x.codes``."""
    cls = resolve_dtype(dt)
    k = np.asarray(codes)
    if k.dtype.kind not in "iu":
        raise SFTypeError("codes must be an integer array")
    # validate in the source type, before any narrowing conversion
    if k.size and (int(k.max()) > cls._fmt.max_code or int(k.min()) < -cls._fmt.max_code):
        raise SFInvalidError(f"codes outside {cls.__name__} range ±{cls._fmt.max_code}")
    k = k.astype(np.int64)
    out = SFArray._wrap(k, cls)
    return out if device == _device.HOST else out.to(device)


def from_code(code, dt: Any) -> SFScalar | SFArray:
    """Inverse of ``x.code`` (scalar) or ``x.codes`` (array)."""
    if isinstance(code, np.ndarray) and code.ndim:
        return from_codes(code, dt)
    return resolve_dtype(dt).from_code(code)


def from_bits(bits, dt: Any) -> SFScalar | SFArray:
    """Inverse of ``x.bits`` (sign-magnitude device pattern), scalar or array."""
    cls = resolve_dtype(dt)
    if isinstance(bits, np.ndarray) and bits.ndim:
        try:
            return SFArray._wrap(vec.from_bits(cls._fmt, bits), cls)
        except ValueError as e:
            raise SFInvalidError(str(e)) from None
    return cls.from_bits(bits)


# --- casts --------------------------------------------------------------------------------

def cast(x: Any, dt: Any):
    """Convert ``x`` to ``dt``.  Widening is exact.  Narrowing re-quantizes
    the exact value with round-half-even through the error policy.  Host
    scalars and arrays are quantized like the constructor."""
    cls = resolve_dtype(dt)
    fmt = cls._fmt
    if _traced(x):
        from .kernels import _lang
        return _lang.cast(x, cls)
    if isinstance(x, SFScalar):
        if x._fmt.bits <= fmt.bits:
            return cls.from_code(widen_code(x._code, x._fmt, fmt))
        k, event = ref.cast_exact(fmt, x.exact)
        if event == "overflow":
            _errstate.handle(event, f"{x!r} is not representable in {fmt.name} "
                                    f"(max {_fmt_value(fmt, fmt.max_code)})")
        elif event == "underflow":
            _errstate.handle(event, f"{x!r} underflows to 0 in {fmt.name}")
        return cls.from_code(k)
    if isinstance(x, SFArray):
        src = x.dtype._fmt
        if src.bits <= fmt.bits:
            return SFArray._wrap(x._codes << (fmt.bits - src.bits), cls, x.device)
        # narrowing: round half even on the exact value = (code * 2^dst) / 2^src
        shift = src.bits - fmt.bits
        num = x._codes
        den = 1 << shift
        q, r = np.divmod(num, den)                      # floor division
        twice = 2 * r
        k = q + ((twice > den) | ((twice == den) & (q % 2 == 1)))
        over = np.abs(k) > fmt.max_code
        k = np.clip(k, -fmt.max_code, fmt.max_code)
        under = (k == 0) & (num != 0)
        _handle_events({"overflow": over, "underflow": under}, fmt, "cast", x.size)
        return SFArray._wrap(k, cls, x.device)
    if isinstance(x, (list, tuple, np.ndarray)):
        return SFArray(x, cls)
    return cls(x)


# --- elementwise functions ----------------------------------------------------------------

def _any_array(*xs) -> bool:
    return any(isinstance(x, (SFArray, np.ndarray, list, tuple)) for x in xs)


def _scalar_operands(name: str, *xs: Any):
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


def _traced(*xs):
    from .kernels import is_traced
    return any(is_traced(x) for x in xs)


def fma(a: Any, b: Any, c: Any):
    """a * b + c, bit-identical to ``a * b + c``.  Scalars or arrays."""
    if _traced(a, b, c):
        from .kernels import _lang
        return _lang.fma(a, b, c)
    if _any_array(a, b, c):
        return _elementwise("fma", a, b, c)
    fmt, (ka, kb, kc) = _scalar_operands("fma", a, b, c)
    k, event = ref.fma(fmt, ka, kb, kc)
    if event is not None:
        if event == "overflow":
            kp, _ = ref.mul(fmt, ka, kb)
            msg = _binary_message(fmt, kp, kc, "+", k, event)
        else:
            msg = _binary_message(fmt, ka, kb, "*", 0, event)
        _errstate.handle(event, f"fma: {msg}")
    return _dtype_of(fmt).from_code(k)


def div(a: Any, b: Any, where: Any = None):
    """``a / b``.  With ``where=mask`` (arrays) the division only happens
    where the mask is True; other elements are 0 and raise nothing."""
    if _traced(a, b):
        from .kernels import _lang
        return _lang.where(where, a / b, 0) if where is not None else a / b
    if where is None:
        if _any_array(a, b):
            return _elementwise("div", a, b)
        fmt, (ka, kb) = _scalar_operands("div", a, b)
        k, event = ref.div(fmt, ka, kb)
        if event is not None:
            _errstate.handle(event, _binary_message(fmt, ka, kb, "/", k, event))
        return _dtype_of(fmt).from_code(k)
    fmt = _widest([a, b])
    dtype = _dtype_of(fmt)
    ka, kb = _to_codes(a, fmt), _to_codes(b, fmt)
    mask = np.asarray(where, dtype=bool)
    try:
        shape = np.broadcast_shapes(ka.shape, kb.shape, mask.shape)
    except ValueError:
        raise SFShapeError("sf.div: operands and mask could not be broadcast together") from None
    ka, kb, mask = (np.broadcast_to(v, shape) for v in (ka, kb, mask))
    # masked-out elements become 0 / 1: exactly 0, no event, so the ordinary
    # div kernel (host or device) does the right thing and the gate sees only
    # the live elements
    safe_a = np.where(mask, ka, 0)
    safe_b = np.where(mask, kb, 1)
    device = _device_of_any(a, b)
    from ._array import _run_heavy, _should_offload
    if device == _device.HOST and not _should_offload(int(np.prod(shape)) if shape else 1):
        k, events = vec.div(fmt, safe_a, safe_b)
        _handle_events(events, fmt, "/", int(np.count_nonzero(mask)))
        if k.ndim == 0:
            return dtype.from_code(int(k))
        return SFArray._wrap(k, dtype)
    from . import _config
    target = device if device != _device.HOST else _config.get("default_device")
    return _run_heavy("div", fmt, dtype, target, [safe_a, safe_b], shape, {"live": mask})


def _device_of_any(*xs):
    from ._array import _device_of
    return _device_of(xs)


def sqrt(a: Any):
    """Square root, truncated toward zero.  Negative input is a domain error
    (invalid); the device value is 0.  Scalars or arrays."""
    if _traced(a):
        from .kernels import _lang
        return _lang.sqrt(a)
    if isinstance(a, (SFArray, list, tuple)):
        return _elementwise("sqrt", _as_sfarray(a, "sqrt"))
    if not isinstance(a, SFScalar):
        raise SFTypeError("sf.sqrt expects an SF value or array")
    fmt = a._fmt
    k, event = ref.sqrt(fmt, a._code)
    if event is not None:
        _errstate.handle(event, f"{fmt.name}: sqrt({a}) of a negative value "
                                f"(device value {_fmt_value(fmt, k)})")
    return type(a).from_code(k)


def _select(name, pick, *xs):
    """maximum / minimum: exact selection on codes, no events."""
    fmt = _widest(xs)
    dtype = _dtype_of(fmt)
    codes = [_to_codes(x, fmt) for x in xs]
    try:
        r = pick(*codes)
    except ValueError:
        raise SFShapeError(f"sf.{name}: operands could not be broadcast together") from None
    if np.ndim(r) == 0:
        return dtype.from_code(int(r))
    return SFArray._wrap(r, dtype, _device_of_any(*xs))


def maximum(a: Any, b: Any):
    if _traced(a, b):
        from .kernels import _lang
        return _lang.maximum(a, b)
    return _select("maximum", np.maximum, a, b)


def minimum(a: Any, b: Any):
    if _traced(a, b):
        from .kernels import _lang
        return _lang.minimum(a, b)
    return _select("minimum", np.minimum, a, b)


def clip(x: Any, lo: Any, hi: Any):
    """Clamp to [lo, hi].  Exact on codes; lo and hi are quantized like literals."""
    if _traced(x, lo, hi):
        from .kernels import _lang
        return _lang.minimum(_lang.maximum(x, lo), hi)
    fmt = _widest([x, lo, hi])
    dtype = _dtype_of(fmt)
    kx, klo, khi = (_to_codes(v, fmt) for v in (x, lo, hi))
    if np.any(klo > khi):
        raise SFInvalidError("sf.clip: lo must not exceed hi")
    r = np.clip(kx, klo, khi)
    if np.ndim(r) == 0:
        return dtype.from_code(int(r))
    return SFArray._wrap(r, dtype, _device_of_any(x))


def where(cond: Any, x: Any, y: Any):
    """``x if cond else y``, elementwise for arrays.  Note that Python
    evaluates both ``x`` and ``y`` before ``where`` runs."""
    if _traced(cond, x, y):
        from .kernels import _lang
        return _lang.where(cond, x, y)
    if not _any_array(cond, x, y):
        return x if cond else y
    fmt = _widest([x, y])
    dtype = _dtype_of(fmt)
    kx, ky = _to_codes(x, fmt), _to_codes(y, fmt)
    c = np.asarray(cond, dtype=bool) if not isinstance(cond, SFArray) else cond._codes != 0
    try:
        r = np.where(c, kx, ky)
    except ValueError:
        raise SFShapeError("sf.where: operands could not be broadcast together") from None
    if r.ndim == 0:
        return dtype.from_code(int(r))
    return SFArray._wrap(r, dtype, _device_of_any(x, y))


__all__ = ["dtype", "info", "SFInfo", "array", "asarray", "zeros", "ones", "full",
           "from_codes", "from_code", "from_bits", "cast", "fma", "div", "sqrt",
           "maximum", "minimum", "clip", "where"]
