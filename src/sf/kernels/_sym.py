"""Symbolic values seen by kernel code while it is traced (spec 3.1, 3.2).

``Sym`` wraps an IR value and overloads the Python operators; ``Ptr`` is a
device buffer plus an offset vector; ``Shared`` is a tile scratchpad.
``bool(sym)`` raises ``SFCompileError``: a per-tile machine has one branch
decision per tile, so a per-thread ``if`` on data is never silently wrong.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .. import _reference as ref
from .._errors import SFCompileError, SFTypeError, SFUnsupportedError
from .._scalar import SFScalar
from ._ir import BOOL, INT, Builder, PtrType, SFType, Value, broadcast_shapes, builder

_CMP = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}


def _lockstep_error(what: str):
    return SFCompileError(
        f"{what} on a traced value: Atreides runs the threads of a tile in lockstep with one "
        "branch decision per tile, so a per-thread branch on data cannot be compiled. Use "
        "sf.where(mask, a, b) for a value, or 'with sf.predicate(mask):' around the statements "
        "whose stores should be masked; loops must run over Python ints or sf.constexpr arguments.")


class Sym:
    """A traced value of type sf<N>, int or bool with a static shape."""

    __slots__ = ("v",)

    def __init__(self, v: Value) -> None:
        self.v = v

    @property
    def type(self):
        return self.v.type

    @property
    def shape(self) -> tuple:
        return self.v.shape

    @property
    def ndim(self) -> int:
        return len(self.v.shape)

    @property
    def dtype(self):
        if isinstance(self.v.type, SFType):
            import sf
            return sf.dtypes[self.v.type.bits - 4]
        return self.v.type

    def __repr__(self):
        return f"Sym({self.v!r})"

    def __bool__(self):
        raise _lockstep_error("if / while / and / or / not")

    def __iter__(self):
        raise SFCompileError("iterating over a traced value is not supported; index it or "
                             "use sf.arange over a constexpr length")

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a 0-d traced value")
        return self.shape[0]

    def __index__(self):
        raise SFCompileError("a traced value cannot be used as a loop bound, range argument or "
                             "index: loops unroll at trace time, so the bound must be a Python int "
                             "or an sf.constexpr argument")

    def __hash__(self):
        return id(self)

    # --- arithmetic ---------------------------------------------------------------------------
    def __add__(self, o): return _binary("add", self, o)
    def __radd__(self, o): return _binary("add", o, self)
    def __sub__(self, o): return _binary("sub", self, o)
    def __rsub__(self, o): return _binary("sub", o, self)
    def __mul__(self, o): return _binary("mul", self, o)
    def __rmul__(self, o): return _binary("mul", o, self)
    def __truediv__(self, o): return _binary("div", self, o)
    def __rtruediv__(self, o): return _binary("div", o, self)
    def __floordiv__(self, o): return _binary("floordiv", self, o)
    def __rfloordiv__(self, o): return _binary("floordiv", o, self)
    def __mod__(self, o): return _binary("mod", self, o)
    def __rmod__(self, o): return _binary("mod", o, self)
    def __neg__(self): return _unary("neg", self)
    def __pos__(self): return self
    def __abs__(self): return _unary("abs", self)
    def __and__(self, o): return _binary("and", self, o)
    def __rand__(self, o): return _binary("and", o, self)
    def __or__(self, o): return _binary("or", self, o)
    def __ror__(self, o): return _binary("or", o, self)
    def __xor__(self, o): return _binary("xor", self, o)
    def __invert__(self): return _unary("not", self)
    def __lt__(self, o): return _compare("lt", self, o)
    def __le__(self, o): return _compare("le", self, o)
    def __gt__(self, o): return _compare("gt", self, o)
    def __ge__(self, o): return _compare("ge", self, o)
    def __eq__(self, o): return _compare("eq", self, o)
    def __ne__(self, o): return _compare("ne", self, o)

    def __pow__(self, n):
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise SFUnsupportedError("x ** k in a kernel needs a Python int k >= 1")
        out = self
        for _ in range(n - 1):
            out = out * self
        return out

    def __matmul__(self, o):
        from ._lang import dot
        return dot(self, o)

    # --- shape ------------------------------------------------------------------------------
    def __getitem__(self, idx):
        """``x[:, None]`` / ``x[None, :]`` expand dims; integer / slice indexing of a
        block value is not supported (blocks are whole tiles)."""
        if not isinstance(idx, tuple):
            idx = (idx,)
        shape = list(self.shape)
        out = []
        d = 0
        for it in idx:
            if it is None:
                out.append(1)
            elif isinstance(it, slice) and it == slice(None):
                if d >= len(shape):
                    raise SFCompileError("too many indices for a traced value")
                out.append(shape[d])
                d += 1
            elif it is Ellipsis:
                rest = len(shape) - d - sum(1 for j in idx if isinstance(j, slice))
                out.extend(shape[d:d + rest])
                d += rest
            else:
                raise SFCompileError("indexing a block value by position is not supported; "
                                     "use masks (sf.where) or pointer loads instead")
        out.extend(shape[d:])
        return reshape(self, tuple(out))

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, tuple(shape))

    @property
    def T(self):
        return _unary("transpose", self, shape=self.shape[::-1])

    def to(self, dtype):
        from ._lang import cast
        return cast(self, dtype)


class Ptr:
    """A device buffer plus an int offset vector.  ``p + offs`` and ``p[i]``."""

    __slots__ = ("base", "offs", "bits", "b")

    def __init__(self, base: Value, offs: Sym | None, bits: int) -> None:
        self.base = base
        self.offs = offs
        self.bits = bits

    @property
    def shape(self):
        return self.offs.shape if self.offs is not None else ()

    def _off(self, extra) -> Sym:
        extra = as_int(extra)
        return extra if self.offs is None else self.offs + extra

    def __add__(self, o):
        return Ptr(self.base, self._off(o), self.bits)

    __radd__ = __add__

    def __sub__(self, o):
        return Ptr(self.base, self._off(-as_int(o)), self.bits)

    def __getitem__(self, idx):
        from ._lang import load
        return load(self + idx if idx is not None else self)

    def __setitem__(self, idx, value):
        from ._lang import store
        store(self + idx if idx is not None else self, value)

    def __bool__(self):
        raise _lockstep_error("branching")

    def __repr__(self):
        return f"Ptr(%{self.base.id} + {self.offs!r})"


class Shared:
    """A tile scratchpad (``sf.shared``) indexed by lane vectors."""

    __slots__ = ("buf",)

    def __init__(self, buf) -> None:
        self.buf = buf

    @property
    def shape(self):
        return self.buf.shape

    def __getitem__(self, idx):
        from ._lang import shared_load
        return shared_load(self, idx)

    def __setitem__(self, idx, value):
        from ._lang import shared_store
        shared_store(self, idx, value)

    def __repr__(self):
        return f"Shared(@{self.buf.id} sf{self.buf.bits}{list(self.buf.shape)})"


# --- coercion ------------------------------------------------------------------------------

def is_traced(x) -> bool:
    return isinstance(x, (Sym, Ptr, Shared))


def const_int(value: int, shape=()) -> Sym:
    b = builder()
    return Sym(b.emit("const", [], {"value": int(value), "kind": "int"}, INT, shape))


def const_bool(value: bool, shape=()) -> Sym:
    b = builder()
    return Sym(b.emit("const", [], {"value": bool(value), "kind": "bool"}, BOOL, shape))


def const_sf(code: int, bits: int, shape=()) -> Sym:
    b = builder()
    return Sym(b.emit("const", [], {"value": int(code), "kind": "sf"}, SFType(bits), shape))


def as_int(x) -> Sym:
    if isinstance(x, Sym):
        if x.type is INT:
            return x
        if x.type is BOOL:
            return _unary("bool_to_int", x, type_=INT)
        raise SFTypeError(f"expected an int value, got {x.type}")
    if isinstance(x, (bool, np.bool_)):
        raise SFTypeError("expected an int, got bool")
    if isinstance(x, (int, np.integer)):
        return const_int(int(x))
    raise SFTypeError(f"expected an int, got {type(x).__name__}")


def as_sf(x, bits: int) -> Sym:
    """An SF operand: traced sf value (widened to ``bits`` if narrower),
    SF scalar, or host number (quantized like a literal, Stage 1 rules)."""
    if isinstance(x, Sym):
        if isinstance(x.type, SFType):
            if x.type.bits == bits:
                return x
            if x.type.bits < bits:
                return _unary("widen", x, type_=SFType(bits), attrs={"bits": bits})
            raise SFTypeError(f"cannot store sf{x.type.bits} into sf{bits} implicitly; use sf.cast")
        raise SFTypeError(f"expected an sf value, got {x.type}")
    if isinstance(x, SFScalar):
        if x._fmt.bits > bits:
            raise SFTypeError(f"cannot store {x._fmt.name} into sf{bits} implicitly; use sf.cast")
        return const_sf(x._code << (bits - x._fmt.bits), bits)
    if isinstance(x, (bool, np.bool_)):
        raise SFTypeError("expected an sf value, got bool")
    if isinstance(x, (int, float, np.integer, np.floating)):
        from .. import _errstate
        from .._format import FORMATS
        fmt = FORMATS[bits]
        k, event = ref.cast(fmt, x)
        if event:
            _errstate.handle(event, f"kernel literal {x!r}: {event} in {fmt.name}")
        return const_sf(k, bits)
    raise SFTypeError(f"cannot use {type(x).__name__} as an sf operand in a kernel")


def as_bool(x) -> Sym:
    if isinstance(x, Sym):
        if x.type is BOOL:
            return x
        raise SFTypeError(f"expected a bool mask, got {x.type}")
    if isinstance(x, (bool, np.bool_)):
        return const_bool(bool(x))
    raise SFTypeError(f"expected a bool mask, got {type(x).__name__}")


def _sf_bits(*xs) -> int | None:
    bits = [x.type.bits for x in xs if isinstance(x, Sym) and isinstance(x.type, SFType)]
    bits += [x._fmt.bits for x in xs if isinstance(x, SFScalar)]
    return max(bits) if bits else None


def _kind(x) -> str:
    if isinstance(x, Sym):
        return "sf" if isinstance(x.type, SFType) else "int" if x.type is INT else "bool"
    if isinstance(x, SFScalar):
        return "sf"
    if isinstance(x, (bool, np.bool_)):
        return "bool"
    if isinstance(x, (int, np.integer)):
        return "int"
    if isinstance(x, (float, np.floating)):
        return "sf"
    if isinstance(x, Ptr):
        return "ptr"
    raise SFTypeError(f"cannot use {type(x).__name__} in a kernel expression")


_INT_OPS = {"add": "iadd", "sub": "isub", "mul": "imul", "floordiv": "idiv", "mod": "imod"}
_BOOL_OPS = {"and": "and", "or": "or", "xor": "xor"}
_SF_OPS = {"add": "add", "sub": "sub", "mul": "mul", "div": "div"}


def _binary(name, a, b) -> Sym:
    if isinstance(a, Ptr) or isinstance(b, Ptr):
        if name == "add":
            return (a + b) if isinstance(a, Ptr) else (b + a)
        raise SFTypeError(f"'{name}' is not defined on pointers")
    b_ = builder()
    ka, kb = _kind(a), _kind(b)
    if name in _BOOL_OPS:
        x, y = as_bool(a), as_bool(b)
        shape = broadcast_shapes(x.shape, y.shape)
        return Sym(b_.emit(_BOOL_OPS[name], [x.v, y.v], {}, BOOL, shape))
    if ka == "sf" or kb == "sf":
        if name not in _SF_OPS:
            raise SFUnsupportedError(f"'{name}' is not supported on sf values (it would silently "
                                     "produce a float or an integer)")
        bits = _sf_bits(a, b)
        x, y = as_sf(a, bits), as_sf(b, bits)
        shape = broadcast_shapes(x.shape, y.shape)
        return Sym(b_.emit(_SF_OPS[name], [x.v, y.v], {}, SFType(bits), shape))
    if name in _INT_OPS:
        x, y = as_int(a), as_int(b)
        shape = broadcast_shapes(x.shape, y.shape)
        return Sym(b_.emit(_INT_OPS[name], [x.v, y.v], {}, INT, shape))
    if name == "div":
        raise SFUnsupportedError("'/' on int values: use '//'")
    raise SFTypeError(f"'{name}' is not defined for {ka} and {kb}")


def _unary(name, a: Sym, type_=None, attrs=None, shape=None) -> Sym:
    b_ = builder()
    if name == "not":
        x = as_bool(a)
        return Sym(b_.emit("not", [x.v], {}, BOOL, x.shape))
    if name in ("neg", "abs"):
        if _kind(a) == "int":
            return Sym(b_.emit("i" + name, [a.v], {}, INT, a.shape))
        x = as_sf(a, _sf_bits(a))
        return Sym(b_.emit(name, [x.v], {}, x.type, x.shape))
    return Sym(b_.emit(name, [a.v], attrs or {}, type_ or a.type, a.shape if shape is None else shape))


def _compare(op, a, b) -> Sym:
    b_ = builder()
    ka, kb = _kind(a), _kind(b)
    if ka == "sf" or kb == "sf":
        # exact comparison, never quantizing: a host number is compared as a value
        bits = _sf_bits(a, b)
        traced = (Sym, SFScalar)                               # anything else is a host literal
        x = as_sf(a, bits) if ka == "sf" and isinstance(a, traced) else None
        y = as_sf(b, bits) if kb == "sf" and isinstance(b, traced) else None
        if x is None or y is None:
            lit = b if x is not None else a
            return _compare_literal(op, x if x is not None else y, lit, flipped=x is None)
        shape = broadcast_shapes(x.shape, y.shape)
        return Sym(b_.emit("cmp", [x.v, y.v], {"op": op}, BOOL, shape))
    x, y = as_int(a), as_int(b)
    shape = broadcast_shapes(x.shape, y.shape)
    return Sym(b_.emit("icmp", [x.v, y.v], {"op": op}, BOOL, shape))


def _compare_literal(op, x: Sym, lit, flipped: bool) -> Sym:
    """sf value against a host number: exact, like Stage 1 (``x < 1`` is always True)."""
    from fractions import Fraction
    b_ = builder()
    if flipped:
        op = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le", "eq": "eq", "ne": "ne"}[op]
    v = ref.to_exact(lit)
    if isinstance(v, float):                                  # nan / inf
        if v != v:
            return const_bool(op == "ne", x.shape)
        r = {"lt": v > 0, "le": v > 0, "gt": v < 0, "ge": v < 0, "eq": False, "ne": True}[op]
        return const_bool(r, x.shape)
    fmt = x.type.fmt
    q = v * fmt.scale                                          # exact rational code position
    # compare integer code against rational q: code op q
    return Sym(b_.emit("cmp_lit", [x.v], {"op": op, "num": q.numerator, "den": q.denominator},
                       BOOL, x.shape))


def reshape(x: Sym, shape: tuple) -> Sym:
    if int(np.prod(shape)) != x.v.size:
        raise SFCompileError(f"cannot reshape {x.shape} to {shape}")
    if len(shape) > 3:
        raise SFCompileError("kernel values are at most 3-D")
    b_ = builder()
    return Sym(b_.emit("reshape", [x.v], {"shape": tuple(shape)}, x.type, tuple(shape)))
