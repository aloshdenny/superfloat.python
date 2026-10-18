"""SF IR: the small typed IR the DSL owns (spec section 11).

Python kernels are *traced* into this IR: the kernel function runs once with
symbolic values, every operation appends an op.  The IR is what the backends
consume (``_interp`` executes it on the host, ``_cgen`` emits C with the
``__sf16_*`` builtins).  Semantics live in the host reference, not here:
every arithmetic op has exactly the Stage 1 meaning.

Types: ``sf<N>`` (codes), ``int`` (indices and offsets, int64), ``bool``
(masks), ``ptr<sf<N>>`` (a device buffer plus an offset vector).  Every
value has a static shape; ops broadcast like NumPy.

Lockstep model (Atreides v1 runs the threads of a tile in lockstep with one
branch decision per tile): a Python ``if`` / ``while`` on a traced value is
rejected at trace time with ``SFCompileError``; data-dependent control flow
is written with ``sf.where`` or under ``sf.predicate(mask)``, which masks
side effects and events (if-conversion).  Loops over Python ints and
``sf.constexpr`` arguments unroll at trace time.
"""
from __future__ import annotations

import contextvars
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .._errors import SFCompileError, SFTypeError
from .._format import FORMATS, SFFormat

MAX_RANK = 3


# --- types -----------------------------------------------------------------------------------

@dataclass(frozen=True)
class SFType:
    bits: int

    @property
    def fmt(self) -> SFFormat:
        return FORMATS[self.bits]

    def __str__(self):
        return f"sf{self.bits}"


@dataclass(frozen=True)
class IntType:
    def __str__(self):
        return "int"


@dataclass(frozen=True)
class BoolType:
    def __str__(self):
        return "bool"


@dataclass(frozen=True)
class PtrType:
    bits: int

    def __str__(self):
        return f"ptr<sf{self.bits}>"


INT = IntType()
BOOL = BoolType()


# --- values and ops ----------------------------------------------------------------------------

@dataclass(eq=False)
class Value:
    id: int
    type: Any
    shape: tuple

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1

    def __repr__(self):
        return f"%{self.id}:{self.type}{list(self.shape)}"


@dataclass(eq=False)
class Op:
    name: str
    args: list                     # Values or Python constants
    attrs: dict
    result: Value | None
    pred: Value | None = None      # predicate mask in force (if-conversion)

    def __repr__(self):
        a = ", ".join(repr(x) for x in self.args)
        at = (" " + " ".join(f"{k}={v}" for k, v in self.attrs.items())) if self.attrs else ""
        p = f" pred={self.pred!r}" if self.pred is not None else ""
        r = f"{self.result!r} = " if self.result is not None else ""
        return f"{r}{self.name}({a}){at}{p}"


@dataclass
class Param:
    name: str
    kind: str                      # "ptr" | "sf" | "int" | "constexpr"
    type: Any = None
    value: Any = None              # constexpr value
    vid: int | None = None         # IR value id (ptr / sf / int params)


@dataclass
class SharedBuf:
    id: int
    bits: int
    shape: tuple

    @property
    def size(self) -> int:
        return int(np.prod(self.shape)) if self.shape else 1


@dataclass
class KernelIR:
    name: str
    style: str                              # "thread" | "block"
    params: list = field(default_factory=list)
    ops: list = field(default_factory=list)
    shared: list = field(default_factory=list)
    values: list = field(default_factory=list)
    constexpr: dict = field(default_factory=dict)
    block_dim: tuple = (1, 1, 1)            # thread kernels: lanes per block

    @property
    def ptr_params(self):
        return [p for p in self.params if p.kind == "ptr"]

    def dump(self) -> str:
        head = f"kernel {self.name} [{self.style}] (" + ", ".join(
            f"{p.name}: {p.kind}" + (f"={p.value}" if p.kind == "constexpr" else "")
            for p in self.params) + ")"
        lines = [head]
        for s in self.shared:
            lines.append(f"  shared @{s.id}: sf{s.bits}{list(s.shape)}")
        lines += ["  " + repr(op) for op in self.ops]
        return "\n".join(lines)


# --- builder (the trace context) ---------------------------------------------------------------

# ops whose result shape is set by the op itself, not by the lanes it runs on
_FIXED_SHAPE = frozenset({"param", "program_id", "thread_idx", "arange", "broadcast", "reshape",
                          "transpose", "sum", "max", "min", "dot"})


class Builder:
    def __init__(self, name: str, style: str) -> None:
        self.ir = KernelIR(name, style)
        self.pred_stack: list[Value] = []
        self._next = 0

    def value(self, type_, shape) -> Value:
        v = Value(self._next, type_, tuple(int(s) for s in shape))
        self._next += 1
        self.ir.values.append(v)
        return v

    def emit(self, name, args, attrs=None, type_=None, shape=()) -> Value | None:
        pred = self.current_pred()
        if type_ is not None and pred is not None and name not in _FIXED_SHAPE:
            # under a predicate every live lane holds its own copy of the value, so a
            # scalar computed inside ``with sf.predicate(mask)`` is one value per lane
            try:
                shape = broadcast_shapes(tuple(shape), pred.shape)
            except SFCompileError:
                raise SFCompileError(f"{name}: a value of shape {tuple(shape)} cannot be computed "
                                     f"under a predicate of shape {pred.shape}") from None
        result = self.value(type_, shape) if type_ is not None else None
        self.ir.ops.append(Op(name, list(args), dict(attrs or {}), result, pred))
        return result

    def current_pred(self) -> Value | None:
        return self.pred_stack[-1] if self.pred_stack else None

    def shared(self, bits, shape) -> SharedBuf:
        buf = SharedBuf(len(self.ir.shared), bits, tuple(shape))
        self.ir.shared.append(buf)
        return buf


_current: contextvars.ContextVar[Builder | None] = contextvars.ContextVar("sf_builder", default=None)


def builder() -> Builder:
    b = _current.get()
    if b is None:
        raise SFCompileError("this operation is only valid inside an @sf.kernel or @sf.jit "
                             "function while it is being traced")
    return b


def in_kernel() -> bool:
    return _current.get() is not None


class trace_context:
    def __init__(self, b: Builder) -> None:
        self.b = b

    def __enter__(self):
        self._tok = _current.set(self.b)
        return self.b

    def __exit__(self, *exc):
        _current.reset(self._tok)


def broadcast_shapes(*shapes) -> tuple:
    try:
        out = np.broadcast_shapes(*shapes)
    except ValueError:
        raise SFCompileError(f"shapes {list(shapes)} cannot be broadcast together") from None
    if len(out) > MAX_RANK:
        raise SFCompileError(f"kernel values are at most {MAX_RANK}-D, got {out}")
    return tuple(out)


def check_sf_bits(bits: int) -> None:
    if bits not in FORMATS:
        raise SFTypeError(f"unsupported SF width {bits}")
