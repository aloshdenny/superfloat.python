"""SFArray: NumPy-style arrays of one SF dtype (spec section 6).

An SF array is an int64 NumPy array of signed codes plus a dtype and a
placement.  Ops on host arrays are *light*: they run the vectorised
reference directly and events are raised on the host, counted over the
array.  Ops with any operand on a device are *heavy*: they go through the
pre-flight gate, run on the target, and report events in ``result.status``.

Semantics are the Stage 1.0 scalar semantics applied element by element,
with NumPy broadcasting.  Every scalar rule (promotion, literal
quantization, exact comparisons, unsupported operators) carries over.
"""
from __future__ import annotations

import operator
from typing import Any

import numpy as np

from . import _config, _device, _errstate, _linalg, _preflight, _reference as ref, _vectorized as vec
from ._errors import SFDeviceOverflowWarning, SFInvalidError, SFShapeError, SFTypeError, SFUnsupportedError
from ._format import FORMATS, SFFormat, widen_code
from ._scalar import DTYPES, SFScalar, _as_index, _dtype_of, resolve_dtype
from ._status import SFStatus

_SYM = {"add": "+", "sub": "-", "mul": "*", "div": "/", "fma": "fma", "neg": "-", "abs": "abs",
        "sqrt": "sqrt", "pow": "**", "matmul": "@", "sum": "sum"}
_EVENT_ORDER = ("invalid", "divide", "overflow", "underflow")


# --- event handling for arrays ---------------------------------------------

def _handle_events(events: dict, fmt: SFFormat, opname: str, total: int) -> None:
    for kind in _EVENT_ORDER:
        mask = events.get(kind)
        if mask is None:
            continue
        n = int(np.count_nonzero(mask))
        if n:
            _errstate.handle(kind, f"{fmt.name} {opname}: {kind} in {n} of {total} element(s)")


# --- operand coercion --------------------------------------------------------

def _is_sf(x) -> bool:
    return isinstance(x, (SFArray, SFScalar))


def _fmt_of(x) -> SFFormat:
    return x._fmt if isinstance(x, SFScalar) else x.dtype._fmt


def _widest(xs) -> SFFormat:
    fmts = [_fmt_of(x) for x in xs if _is_sf(x)]
    if not fmts:
        raise SFTypeError("at least one operand must be an SF value or array")
    return max(fmts, key=lambda f: f.bits)


def _device_of(xs):
    devs = {x.device for x in xs if isinstance(x, SFArray) and x.device != _device.HOST}
    if len(devs) > 1:
        raise SFTypeError(f"operands live on different devices: {sorted(devs)}")
    return devs.pop() if devs else _device.HOST


def _to_codes(x, fmt: SFFormat, what: str = "literal") -> np.ndarray:
    """Any operand -> int64 code array in ``fmt`` (with literal quantization
    through the error policy).  Host arrays and lists are quantized; SF
    values are widened (never narrowed)."""
    if isinstance(x, SFArray):
        if x.dtype._fmt.bits > fmt.bits:
            raise SFTypeError(f"cannot store {x.dtype.__name__} into {fmt.name} implicitly; "
                              f"use sf.cast(x, sf.{fmt.name})")
        return x._codes << (fmt.bits - x.dtype._fmt.bits)
    if isinstance(x, SFScalar):
        if x._fmt.bits > fmt.bits:
            raise SFTypeError(f"cannot store {x._fmt.name} into {fmt.name} implicitly; "
                              f"use sf.cast(x, sf.{fmt.name})")
        return np.asarray(widen_code(x._code, x._fmt, fmt), dtype=np.int64)
    if isinstance(x, (str, bytes)):
        raise SFTypeError(f"cannot use {type(x).__name__} as an SF operand")
    if isinstance(x, (list, tuple)):
        x = _nested_to_array(x)
        if isinstance(x, SFArray):
            return _to_codes(x, fmt)
    if isinstance(x, np.ndarray):
        if x.dtype.kind in "USO" and what == "literal" and any(isinstance(v, (str, bytes)) for v in x.ravel()):
            raise SFTypeError("strings are accepted by sf.array(...) only, not as operator literals")
        try:
            codes, events = vec.cast_array(fmt, x)
        except TypeError as e:
            raise SFTypeError(str(e)) from None
        _handle_events(events, fmt, f"{what} cast", x.size)
        return codes
    # a host scalar
    try:
        k, event = ref.cast(fmt, x)
    except TypeError as e:
        raise SFTypeError(f"cannot use {type(x).__name__} as an SF operand") from None
    if event == "invalid":
        _errstate.handle(event, f"{what} {x!r} is not a valid {fmt.name} input")
    elif event == "overflow":
        _errstate.handle(event, f"{what} {x!r} is not representable in {fmt.name} "
                                f"(max {fmt.to_float(fmt.max_code)!r})")
    elif event == "underflow":
        _errstate.handle(event, f"{what} {x!r} underflows to 0 in {fmt.name}")
    return np.asarray(k, dtype=np.int64)


def _nested_to_array(obj):
    """A (nested) list: SF scalars / SF arrays become one SFArray of the widest
    dtype; plain numbers become a NumPy array (object dtype if needed)."""
    flat = []

    def walk(o):
        if isinstance(o, (list, tuple)):
            for i in o:
                walk(i)
        else:
            flat.append(o)
    walk(obj)
    if any(_is_sf(v) for v in flat):
        if not all(_is_sf(v) for v in flat):
            raise SFTypeError("a list of SF values cannot mix SF values with host numbers; "
                              "convert the numbers with sf.array(..., dtype=...) first")
        fmt = _widest(flat)
        codes = np.asarray(_nest(obj, lambda v: _to_codes(v, fmt).tolist()), dtype=np.int64)
        return SFArray._wrap(codes, _dtype_of(fmt))
    try:
        arr = np.asarray(obj)
    except ValueError as e:
        raise SFShapeError(str(e)) from None
    if arr.dtype.kind not in "fiub":
        arr = np.asarray(obj, dtype=object)
    return arr


def _nest(obj, leaf):
    if isinstance(obj, (list, tuple)):
        return [_nest(o, leaf) for o in obj]
    return leaf(obj)


# --- the array type ------------------------------------------------------------

class SFArray:
    """N-dimensional array of one SF dtype."""

    __slots__ = ("_codes", "_dtype", "_device", "_status", "_events")
    __array_ufunc__ = None          # NumPy defers to our reflected operators

    def __init__(self, obj: Any, dtype: Any = None, device: str = _device.HOST) -> None:
        if isinstance(obj, SFArray) and dtype is None:
            dtype = obj.dtype
        if isinstance(obj, SFScalar) and dtype is None:
            dtype = type(obj)
        if isinstance(obj, (list, tuple)) and dtype is None:
            arr = _nested_to_array(obj)
            if isinstance(arr, SFArray):
                dtype = arr.dtype
        if dtype is None:
            raise SFTypeError("sf.array needs dtype= unless the input already holds SF values")
        cls = resolve_dtype(dtype)
        fmt = cls._fmt
        if isinstance(obj, SFArray):
            codes = _to_codes(obj, fmt).copy()
        else:
            codes = np.array(_to_codes(obj, fmt, what="value"), dtype=np.int64, copy=True)
        self._codes = codes
        self._dtype = cls
        self._device = _device.HOST
        self._status = None
        self._events = None
        if device != _device.HOST:
            self._device = _device.get_device(device).name
            _device.get_device(device).check(fmt, codes.size)

    @classmethod
    def _wrap(cls, codes: np.ndarray, dtype, device: str = _device.HOST, status=None) -> "SFArray":
        obj = object.__new__(cls)
        obj._codes = np.asarray(codes, dtype=np.int64)
        obj._dtype = dtype
        obj._device = device
        obj._status = status
        obj._events = None
        return obj

    # --- introspection ---------------------------------------------------------
    @property
    def dtype(self):
        return self._dtype

    @property
    def device(self) -> str:
        return self._device

    @property
    def status(self):
        """Post-op status of the heavy op that produced this array (None for host results)."""
        return self._status

    @property
    def shape(self) -> tuple:
        return self._codes.shape

    @property
    def ndim(self) -> int:
        return self._codes.ndim

    @property
    def size(self) -> int:
        return self._codes.size

    @property
    def codes(self) -> np.ndarray:
        """Signed integer codes (read-only view)."""
        v = self._codes.view()
        v.flags.writeable = False
        return v

    @property
    def bits(self) -> np.ndarray:
        """Sign-magnitude device patterns, uint16."""
        return vec.to_bits(self._dtype._fmt, self._codes)

    def numpy(self) -> np.ndarray:
        """Read back to the host as float64 (every SF value is exact in float64)."""
        return vec.to_float(self._dtype._fmt, self._codes)

    def tolist(self):
        return self.numpy().tolist()

    def to(self, device: str) -> "SFArray":
        """Move to ``device`` (``"host"`` or a target name).  Copies."""
        if device == _device.HOST:
            return SFArray._wrap(self._codes.copy(), self._dtype)
        dev = _device.get_device(device)
        dev.check(self._dtype._fmt, self.size)
        return SFArray._wrap(self._codes.copy(), self._dtype, dev.name)

    def copy(self) -> "SFArray":
        return SFArray._wrap(self._codes.copy(), self._dtype, self._device, self._status)

    def reshape(self, *shape) -> "SFArray":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return SFArray._wrap(self._codes.reshape(shape), self._dtype, self._device)

    @property
    def T(self) -> "SFArray":
        return SFArray._wrap(self._codes.T, self._dtype, self._device)

    def transpose(self, *axes) -> "SFArray":
        return SFArray._wrap(self._codes.transpose(*axes) if axes else self._codes.T,
                             self._dtype, self._device)

    def flatten(self) -> "SFArray":
        return SFArray._wrap(self._codes.flatten(), self._dtype, self._device)

    def astype(self, dtype) -> "SFArray":
        from ._functions import cast
        return cast(self, dtype)

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d SF array")
        return self._codes.shape[0]

    def __iter__(self):
        if self.ndim == 0:
            raise TypeError("iteration over a 0-d SF array")
        for i in range(self._codes.shape[0]):
            yield self[i]

    def __bool__(self) -> bool:
        if self.size != 1:
            raise ValueError("the truth value of an SF array with more than one element is "
                             "ambiguous; use .any() or .all()")
        return bool(self._codes.reshape(()) != 0)

    def any(self) -> bool:
        return bool(np.any(self._codes))

    def all(self) -> bool:
        return bool(np.all(self._codes))

    def __repr__(self) -> str:
        body = np.array2string(self.numpy(), separator=", ", threshold=50)
        extra = f", device='{self._device}'" if self._device != _device.HOST else ""
        return f"sfarray({body}, dtype={self._dtype.__name__}{extra})"

    __str__ = __repr__

    def __reduce__(self):
        return (_rebuild, (self._codes, self._dtype._fmt.bits, self._device))

    # --- indexing ------------------------------------------------------------------
    def _index(self, idx):
        if isinstance(idx, SFArray):
            raise SFTypeError("SF arrays cannot be used as indices; use a boolean NumPy array")
        return idx

    def __getitem__(self, idx):
        try:
            sub = self._codes[self._index(idx)]
        except IndexError as e:
            raise IndexError(str(e)) from None
        if isinstance(sub, np.ndarray):
            return SFArray._wrap(sub, self._dtype, self._device)
        return self._dtype.from_code(int(sub))

    def __setitem__(self, idx, value) -> None:
        codes = _to_codes(value, self._dtype._fmt, what="value")
        self._codes[self._index(idx)] = codes

    # --- elementwise operators ---------------------------------------------------
    def __add__(self, o): return _elementwise("add", self, o)
    def __radd__(self, o): return _elementwise("add", o, self)
    def __sub__(self, o): return _elementwise("sub", self, o)
    def __rsub__(self, o): return _elementwise("sub", o, self)
    def __mul__(self, o): return _elementwise("mul", self, o)
    def __rmul__(self, o): return _elementwise("mul", o, self)
    def __truediv__(self, o): return _elementwise("div", self, o)
    def __rtruediv__(self, o): return _elementwise("div", o, self)
    def __neg__(self): return _elementwise("neg", self)
    def __pos__(self): return self
    def __abs__(self): return _elementwise("abs", self)
    def __matmul__(self, o): return matmul(self, o)
    def __rmatmul__(self, o): return matmul(o, self)

    def __pow__(self, n, mod=None):
        if mod is not None:
            raise SFUnsupportedError("pow() with a modulus is not supported for SF arrays")
        try:
            n = _as_index(n, "exponent")
        except SFTypeError:
            raise SFUnsupportedError(f"{self._dtype.__name__} ** {n!r}: only integer exponents "
                                     "k >= 1 are supported") from None
        if n == 0:
            fmt = self._dtype._fmt
            _errstate.handle("overflow", f"x ** 0 = 1 is not representable in {fmt.name}")
            return SFArray._wrap(np.full(self.shape, fmt.max_code, dtype=np.int64), self._dtype,
                                 self._device)
        if n < 0:
            raise SFUnsupportedError("negative exponents (reciprocals) overflow for every nonzero "
                                     "SF value and are not exposed")
        return _elementwise("pow", self, n=n)

    def __rpow__(self, base):
        raise SFUnsupportedError("an SF array cannot be used as an exponent")

    # --- comparisons: exact, elementwise, return bool ndarrays -------------------
    def _cmp(self, other, op):
        if isinstance(other, (SFArray, SFScalar)):
            fmt = _widest([self, other])
            a = _to_codes(self, fmt)
            b = _to_codes(other, fmt)
            try:
                return op(a, b)
            except ValueError as e:
                raise SFShapeError(str(e)) from None
        if isinstance(other, (str, bytes)):
            return NotImplemented
        if isinstance(other, (list, tuple)):
            other = _nested_to_array(other)
            if isinstance(other, SFArray):
                return self._cmp(other, op)
        if isinstance(other, np.ndarray):
            if other.dtype == object:
                vals = np.frompyfunc(lambda c, v: _scalar_cmp(self._dtype, int(c), v, op), 2, 1)
                return vals(self._codes, other).astype(bool)
            if other.dtype.kind in "fiub":
                return op(self.numpy(), other.astype(np.float64))
            return NotImplemented
        try:
            v = ref.to_exact(other)
        except TypeError:
            return NotImplemented
        if isinstance(v, float):
            if np.isnan(v):
                return np.zeros(self.shape, dtype=bool) if op is not operator.ne \
                    else np.ones(self.shape, dtype=bool)
            return op(np.zeros(self.shape), np.full(self.shape, v))
        return np.frompyfunc(lambda c: op(self._dtype._fmt.to_fraction(int(c)), v), 1, 1)(
            self._codes).astype(bool)

    def __eq__(self, o): return self._cmp(o, operator.eq)
    def __ne__(self, o): return self._cmp(o, operator.ne)
    def __lt__(self, o): return self._cmp(o, operator.lt)
    def __le__(self, o): return self._cmp(o, operator.le)
    def __gt__(self, o): return self._cmp(o, operator.gt)
    def __ge__(self, o): return self._cmp(o, operator.ge)
    __hash__ = None

    # --- reductions ------------------------------------------------------------------
    def sum(self, axis=None):
        return sum_(self, axis)

    def max(self, axis=None):
        return _minmax(self, axis, np.max)

    def min(self, axis=None):
        return _minmax(self, axis, np.min)

    def argmax(self, axis=None):
        return np.argmax(self._codes, axis=axis)

    def argmin(self, axis=None):
        return np.argmin(self._codes, axis=axis)

    # --- explicitly unsupported ---------------------------------------------------------
    def _unsupported(self, what):
        raise SFUnsupportedError(f"{what} is not supported for SF arrays; use .codes for "
                                 "integer arithmetic on codes")

    def __floordiv__(self, o): self._unsupported("//")
    def __rfloordiv__(self, o): self._unsupported("//")
    def __mod__(self, o): self._unsupported("%")
    def __rmod__(self, o): self._unsupported("%")
    def __and__(self, o): self._unsupported("&")
    def __or__(self, o): self._unsupported("|")
    def __xor__(self, o): self._unsupported("^")
    def __invert__(self): self._unsupported("~")
    def __lshift__(self, o): self._unsupported("<<")
    def __rshift__(self, o): self._unsupported(">>")
    def __int__(self): self._unsupported("int()")
    def __float__(self):
        if self.size != 1:
            raise TypeError("only size-1 SF arrays can be converted to float")
        return float(self._dtype._fmt.to_float(int(self._codes.reshape(()))))
    def __index__(self): self._unsupported("using an SF array as an index")
    def __round__(self, n=None): self._unsupported("round()")


def _scalar_cmp(dtype, code, v, op):
    x = dtype.from_code(code)
    return op(x, v)


def _rebuild(codes, bits, device):
    return SFArray._wrap(codes, DTYPES[bits], device)


# --- the elementwise engine (light and heavy) ----------------------------------------

def _result(codes, dtype, device=_device.HOST, status=None):
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim == 0 and device == _device.HOST:
        return dtype.from_code(int(codes))
    return SFArray._wrap(codes, dtype, device, status)


def _elementwise(op: str, *xs, **kw):
    """Apply a vectorised reference op to any mix of SF arrays, SF scalars and
    host operands, light or heavy depending on placement."""
    fmt = _widest(xs)
    dtype = _dtype_of(fmt)
    device = _device_of(xs)
    codes = [_to_codes(x, fmt) for x in xs]
    try:
        shape = np.broadcast_shapes(*(c.shape for c in codes))
    except ValueError:
        raise SFShapeError(f"operands could not be broadcast together: "
                           f"{[c.shape for c in codes]}") from None
    total = int(np.prod(shape)) if shape else 1
    opname = _SYM[op]
    if device == _device.HOST and not _should_offload(total):
        fn = {"abs": vec.abs_, "pow": vec.pow_int}.get(op) or getattr(vec, op)
        result, events = fn(fmt, *codes, **kw)
        _handle_events(events, fmt, opname, total)
        return _result(np.broadcast_to(result, shape).copy() if np.shape(result) != shape else result,
                       dtype)
    return _run_heavy(op, fmt, dtype, device if device != _device.HOST else _config.get("default_device"),
                      codes, shape, kw)


def _should_offload(total: int) -> bool:
    t = _config.get("offload_threshold")
    return t is not None and total > t


_REDUCTIONS = ("sum", "matmul")


# ops whose result can underflow (a nonzero product truncated to 0); the rest never do
_CAN_UNDERFLOW = ("mul", "fma", "matmul", "pow")


def _no_underflow_possible(op, fmt, codes) -> bool:
    """A nonzero product underflows only when |k_a k_b| < scale.  If the two
    smallest nonzero magnitudes already multiply to at least the scale, no
    product of the op can underflow: O(inputs), no launch needed."""
    if op not in ("mul", "fma", "matmul"):
        return False
    mins = []
    for c in codes[:2]:
        nz = np.abs(np.asarray(c))[np.asarray(c) != 0]
        if nz.size == 0:
            return True                                    # every product is exactly 0
        mins.append(int(nz.min()))
    return mins[0] * mins[1] >= fmt.scale


def _run_heavy(op, fmt, dtype, device, codes, out_shape, kw):
    dev = _device.get_device(device)
    dev.check(fmt, max(int(np.prod(out_shape)) if out_shape else 1,
                       *(c.size for c in codes)))
    verdict = _preflight.check(op, fmt, *codes, **kw)
    # known answers first: div0 / invalid are read off the inputs, exact, before launch
    opname = _SYM[op]
    if verdict.events.get("invalid"):
        _errstate.handle("invalid", f"{fmt.name} {opname} on {dev.name}: invalid in "
                                    f"{verdict.events['invalid']} element(s) (status.invalid)")
    if verdict.events.get("div0"):
        _errstate.handle("divide", f"{fmt.name} {opname} on {dev.name}: division by zero in "
                                   f"{verdict.events['div0']} element(s) (status.div0)")
    verdict = _preflight.decide(verdict, op, fmt, *codes, **kw)
    ops = getattr(dev, "supported_ops", None)
    if ops is not None and op not in ops:              # the target has no instruction for it
        raise SFUnsupportedError(f"{fmt.name} {opname} has no instruction on {dev.name} "
                                 f"(supported: {', '.join(sorted(ops))}); nothing was launched")
    kw = {k: v for k, v in kw.items() if k != "live"}        # the gate's mask never reaches a kernel
    result, flags = dev.run(op, fmt, *codes, **kw)
    result = np.asarray(result, dtype=np.int64)
    result = np.broadcast_to(result, out_shape).copy() if result.shape != tuple(out_shape) else result
    at_max = int(np.count_nonzero(np.abs(result) == fmt.max_code))
    can_under = op in _CAN_UNDERFLOW
    masks = None                                           # per-output event masks, when known exactly
    if dev.sticky_flags and flags is not None:
        masks = {k: np.broadcast_to(np.asarray(v, dtype=bool), out_shape) for k, v in flags.items()}
    elif _config.get("status") == "replay":
        _, events = _preflight.host_events(op, fmt, *codes, **kw)
        masks = {k: np.broadcast_to(np.asarray(v, dtype=bool), out_shape)
                 for k, v in events.items() if k in ("overflow", "underflow")}
    if can_under and masks is None and _no_underflow_possible(op, fmt, codes):
        can_under = False                                  # proved on the host: no product reaches 0
    if masks is not None:
        saturated, exact = int(np.count_nonzero(masks.get("overflow", False))), True
        underflow = int(np.count_nonzero(masks["underflow"])) if "underflow" in masks else \
            (None if can_under else 0)
    elif verdict.kind == "safe":
        saturated, exact = 0, True                         # proved on the host; no scan needed
        underflow = None if can_under else 0
    elif op in _REDUCTIONS:
        saturated, exact = None, False                     # a partial sum can saturate and come back
        underflow = None if can_under else 0
    else:
        saturated, exact = at_max, False                   # elementwise: a true upper bound
        underflow = None if can_under else 0
    status = SFStatus(saturated=saturated, div0=verdict.events.get("div0", 0),
                      invalid=verdict.events.get("invalid", 0), exact=exact, outputs_at_max=at_max,
                      underflow=underflow)
    _apply_status_policy(status, fmt, opname, dev)
    out = SFArray._wrap(result, dtype, dev.name, status)
    out._events = masks                                    # for callers that merge events per output
    return out


def _apply_status_policy(status: SFStatus, fmt: SFFormat, opname: str, dev) -> None:
    """Heavy ops apply the same policy after the op, from result.status.
    div0 and invalid were already handled before launch (they are exact).
    An exact saturation count honours the policy; an upper bound only warns
    (spec 3.4); an unknown count says nothing."""
    if status.saturated:
        _overflow_policy(status, fmt, opname, dev)
    if status.underflow:                                   # only ever an exact count
        _errstate.handle("underflow", f"{fmt.name} {opname} on {dev.name}: underflow in "
                                      f"{status.underflow} element(s) (status.underflow)")


def _overflow_policy(status: SFStatus, fmt: SFFormat, opname: str, dev) -> None:
    msg = (f"{fmt.name} {opname} on {dev.name}: saturation in "
           f"{'' if status.exact else '<='}{status.saturated} element(s) (status.saturated)")
    if status.exact:
        _errstate.handle("overflow", msg)
    elif _errstate.geterr()["over"] != "ignore":
        import warnings
        warnings.warn(msg, SFDeviceOverflowWarning, stacklevel=_errstate._user_stacklevel())


# --- reductions and linear algebra --------------------------------------------------------

def _as_sfarray(x, name: str) -> SFArray:
    if isinstance(x, SFArray):
        return x
    if isinstance(x, SFScalar):
        return SFArray._wrap(np.asarray(x._code), type(x))
    if isinstance(x, (list, tuple)):
        arr = _nested_to_array(x)
        if isinstance(arr, SFArray):
            return arr
    raise SFTypeError(f"sf.{name} expects SF arrays (got {type(x).__name__}); "
                      "convert host data with sf.asarray(x, dtype=...)")


def _traced(*xs):
    from .kernels import is_traced
    return any(is_traced(x) for x in xs)


def sum_(x, axis=None):
    """Saturating sum in the target's accumulation order (sequential)."""
    if _traced(x):
        from .kernels import _lang
        return _lang.sum(x, axis)
    x = _as_sfarray(x, "sum")
    fmt, dtype = x.dtype._fmt, x.dtype
    codes = x._codes
    if axis is None:
        codes, axis = codes.reshape(-1), 0
    elif axis < 0:
        axis += codes.ndim
    if not (0 <= axis < codes.ndim):
        raise SFShapeError(f"axis {axis} is out of bounds for an array of dimension {codes.ndim}")
    if x.device == _device.HOST and not _should_offload(codes.size):
        result, events = _linalg.sum_codes(fmt, codes, axis)
        _handle_events(events, fmt, "sum", int(np.prod(result.shape)) if result.shape else 1)
        return _result(result, dtype)
    out_shape = codes.shape[:axis] + codes.shape[axis + 1:]
    target = x.device if x.device != _device.HOST else _config.get("default_device")
    return _run_heavy("sum", fmt, dtype, target, [codes], out_shape, {"axis": axis})


def _minmax(x, axis, fn):
    x = _as_sfarray(x, "max")
    if x.size == 0:
        raise SFShapeError("zero-size array to reduction operation which has no identity")
    r = fn(x._codes, axis=axis)
    return _result(r, x.dtype, x.device if np.ndim(r) else _device.HOST)


def dot(a, b):
    """1-D · 1-D as a chain of fma in target order; higher ranks go to matmul."""
    if _traced(a, b):
        from .kernels import _lang
        return _lang.dot(a, b)
    a, b = _as_sfarray(a, "dot"), _as_sfarray(b, "dot")
    if a.ndim == 1 and b.ndim == 1:
        if a.shape != b.shape:
            raise SFShapeError(f"sf.dot: shapes {a.shape} and {b.shape} differ")
        fmt = _widest([a, b])
        ka, kb = _to_codes(a, fmt), _to_codes(b, fmt)
        device = _device_of([a, b])
        if device == _device.HOST and not _should_offload(ka.size):
            k, events = _linalg.dot_codes(fmt, ka, kb)
            first = events.pop("first")
            if first:                                # same order as the scalar fma chain
                _errstate.handle(first, f"{fmt.name} dot: {first} in the accumulation chain")
                events.pop(first, None)
            _handle_events(events, fmt, "dot", 1)
            return _dtype_of(fmt).from_code(int(k))
        target = device if device != _device.HOST else _config.get("default_device")
        r = _run_heavy("matmul", fmt, _dtype_of(fmt), target, [ka[None, :], kb[:, None]], (1, 1), {})
        return SFArray._wrap(r._codes.reshape(()), r.dtype, r.device, r.status)
    return matmul(a, b)


def matmul(a, b):
    """``a @ b`` with NumPy's rules for 1-D operands and batching.
    Shape errors are raised on the host before any launch."""
    if _traced(a, b):
        from .kernels import _lang
        return _lang.dot(a, b)
    a, b = _as_sfarray(a, "matmul"), _as_sfarray(b, "matmul")
    if a.ndim == 0 or b.ndim == 0:
        raise SFShapeError("matmul: operands must be at least 1-D (use * for scalars)")
    fmt = _widest([a, b])
    dtype = _dtype_of(fmt)
    ka, kb = _to_codes(a, fmt), _to_codes(b, fmt)
    squeeze_row = squeeze_col = False
    if ka.ndim == 1:
        ka, squeeze_row = ka[None, :], True
    if kb.ndim == 1:
        kb, squeeze_col = kb[:, None], True
    if ka.shape[-1] != kb.shape[-2]:
        raise SFShapeError(f"matmul: shapes {a.shape} and {b.shape} are not aligned "
                           f"({ka.shape[-1]} != {kb.shape[-2]})")
    try:
        batch = np.broadcast_shapes(ka.shape[:-2], kb.shape[:-2])
    except ValueError:
        raise SFShapeError(f"matmul: batch shapes {ka.shape[:-2]} and {kb.shape[:-2]} "
                           "cannot be broadcast") from None
    out_shape = batch + (ka.shape[-2], kb.shape[-1])
    device = _device_of([a, b])
    if device == _device.HOST and not _should_offload(int(np.prod(out_shape)) * ka.shape[-1]):
        result, events = _linalg.matmul_codes(fmt, ka, kb)
        _handle_events(events, fmt, "@", int(np.prod(out_shape)))
        out = SFArray._wrap(result, dtype)
    else:
        target = device if device != _device.HOST else _config.get("default_device")
        out = _run_heavy("matmul", fmt, dtype, target, [ka, kb], out_shape, {})
    codes = out._codes
    if squeeze_row:
        codes = codes[..., 0, :]
    if squeeze_col:
        codes = codes[..., 0] if not squeeze_row else codes[..., 0]
    if codes.ndim == 0 and out.device == _device.HOST:
        return dtype.from_code(int(codes))
    res = SFArray._wrap(codes, dtype, out.device, out.status)
    ev = getattr(out, "_events", None)
    if ev is not None:                                     # keep per-output masks, squeezed like the codes
        res._events = {k: np.asarray(v).reshape(out._codes.shape).reshape(codes.shape)
                       if v.size == codes.size else v for k, v in ev.items()}
    return res


def addmm(a, b, d):
    """``a @ b + d``: the matmul chain, then one saturating add of ``d``."""
    return matmul(a, b) + d


__all__ = ["SFArray", "sum_", "dot", "matmul", "addmm"]
