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
``sf.constexpr`` arguments unroll at trace time; ``sf.fori_loop`` keeps a loop
with a constant trip count as one ``for`` op whose body is traced once.
"""
from __future__ import annotations

import contextlib
import contextvars
import itertools
import os
import sys
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
    bits: int                               # 0: a read-only int buffer (per-output shifts and other tables)

    def __str__(self):
        return "ptr<int>" if self.bits == 0 else f"ptr<sf{self.bits}>"


INT = IntType()
BOOL = BoolType()


# --- values and ops ----------------------------------------------------------------------------

@dataclass(eq=False)
class Value:
    id: int
    type: Any
    shape: tuple
    scope: int = 0                 # the loop body that made it (0: the kernel's top level)

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
    loc: tuple | None = None       # (file name, line, column or None) of the user's code that made the op
    body: "Loop | None" = None     # the ``for`` op's body; None for every other op

    def __repr__(self):
        a = ", ".join(repr(x) for x in self.args)
        at = (" " + " ".join(f"{k}={v}" for k, v in self.attrs.items())) if self.attrs else ""
        p = f" pred={self.pred!r}" if self.pred is not None else ""
        r = f"{self.result!r} = " if self.result is not None else ""
        return f"{r}{self.name}({a}){at}{p}"


@dataclass(eq=False)
class Loop:
    """The body of a ``for`` op (``scf.for`` in MLIR).

    The op's args are the initial values of the carried values.  Each trip
    sets ``iv`` to the trip's index and runs ``ops``; ``yields[i]`` becomes
    ``carried[i]`` for the next trip, and after the last one ``results[i]``.
    ``carried[i]`` and ``results[i]`` have the type and shape of ``args[i]``."""
    iv: Value
    carried: list
    ops: list
    yields: list
    results: list


def walk(ops):
    """Every op in ``ops``, loop bodies included, in program order."""
    for op in ops:
        yield op
        if op.body is not None:
            yield from walk(op.body.ops)


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

        def show(ops, ind):
            for op in ops:
                if op.body is None:
                    lines.append(ind + repr(op))
                    continue
                L = op.body
                lines.append(f"{ind}{', '.join(map(repr, L.results))} = for {L.iv!r} in range("
                             f"{op.attrs['start']}, {op.attrs['stop']}, {op.attrs['step']}) "
                             f"carry({', '.join(f'{c!r}={a!r}' for c, a in zip(L.carried, op.args))}) {{")
                show(L.ops, ind + "  ")
                lines.append(f"{ind}  yield {', '.join(map(repr, L.yields))}")
                lines.append(ind + "}")
        show(self.ops, "  ")
        return "\n".join(lines)

    def to_json(self) -> str:
        """The kernel as SF IR JSON, the compiler's input (see ``_irjson``)."""
        from ._irjson import dumps
        return dumps(self)

    @classmethod
    def from_json(cls, text: str) -> "KernelIR":
        """Read a kernel back from ``to_json`` text."""
        from ._irjson import loads
        return loads(text)


# --- builder (the trace context) ---------------------------------------------------------------

# ops whose result shape is set by the op itself, not by the lanes it runs on
_FIXED_SHAPE = frozenset({"param", "program_id", "thread_idx", "arange", "broadcast", "reshape",
                          "transpose", "sum", "max", "min", "dot", "dot_wide"})


_SF_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../sf
_KERNELS_DIR = os.path.join(_SF_DIR, "kernels")
_machinery: dict[str, bool] = {}       # co_filename -> is it sf's DSL machinery


def _is_machinery(path: str) -> bool:
    """True for the files between an op and the code that asked for it: sf/kernels/*
    and sf's private top-level modules (sf/_functions.py, sf/_array.py, sf/__init__.py,
    ...).  A kernel defined anywhere else, sf.torch included, keeps its own lines."""
    hit = _machinery.get(path)
    if hit is None:
        d, name = os.path.split(os.path.abspath(path))
        hit = _machinery[path] = d == _KERNELS_DIR or (d == _SF_DIR and name.startswith("_"))
    return hit


def _column(f) -> int | None:
    """1-based column where the expression running in frame ``f`` starts (Python 3.11+)."""
    try:
        pos = next(itertools.islice(f.f_code.co_positions(), f.f_lasti // 2, None))
    except (AttributeError, StopIteration):
        return None
    return pos[2] + 1 if pos[2] is not None else None


def _user_loc() -> tuple | None:
    """(file name, line, column) of the first caller that is not sf's DSL machinery: the user's code.

    Only the file's base name is kept, never its full path, so an exported IR
    names no folder (and so no user) and hashes the same on every machine."""
    f = sys._getframe(2)
    while f is not None:
        if not _is_machinery(f.f_code.co_filename):
            return (os.path.basename(f.f_code.co_filename), f.f_lineno, _column(f))
        f = f.f_back
    return None


class Builder:
    def __init__(self, name: str, style: str) -> None:
        self.ir = KernelIR(name, style)
        self.pred_stack: list[Value] = []
        self._next = 0
        self.block: list = self.ir.ops        # where emit appends: the kernel, or the loop body being traced
        self.scopes: list[int] = [0]          # the loop bodies being traced, innermost last
        self._next_scope = 1
        self.fixed_loc: tuple | None = None   # set while ``Kernel.trace`` emits the params
        self.default_loc: tuple | None = None  # the kernel's def: used if no user frame is found

    def value(self, type_, shape) -> Value:
        v = Value(self._next, type_, tuple(int(s) for s in shape), self.scopes[-1])
        self._next += 1
        self.ir.values.append(v)
        return v

    def emit(self, name, args, attrs=None, type_=None, shape=(), body=None) -> Value | None:
        pred = self.current_pred()
        for a in args:
            if isinstance(a, Value) and a.scope not in self.scopes:
                raise SFCompileError(
                    f"{name} uses a value made inside an sf.fori_loop body after the loop ended: "
                    "a value leaves the loop only by being carried (return it from the body)")
        if type_ is not None and pred is not None and name not in _FIXED_SHAPE:
            # under a predicate every live lane holds its own copy of the value, so a
            # scalar computed inside ``with sf.predicate(mask)`` is one value per lane
            try:
                shape = broadcast_shapes(tuple(shape), pred.shape)
            except SFCompileError:
                raise SFCompileError(f"{name}: a value of shape {tuple(shape)} cannot be computed "
                                     f"under a predicate of shape {pred.shape}") from None
        result = self.value(type_, shape) if type_ is not None else None
        loc = self.fixed_loc if self.fixed_loc is not None else (_user_loc() or self.default_loc)
        self.block.append(Op(name, list(args), dict(attrs or {}), result, pred, loc, body))
        return result

    @contextlib.contextmanager
    def loop_body(self, ops: list):
        """Trace into ``ops`` as a new loop body: values made here stay inside it."""
        outer = self.block
        self.block = ops
        self.scopes.append(self._next_scope)
        self._next_scope += 1
        try:
            yield
        finally:
            self.scopes.pop()
            self.block = outer

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
