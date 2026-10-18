"""The kernel language: what kernel code calls (spec 3.1 and 3.2).

Thread style (CUDA):  ``@sf.kernel``, ``sf.thread_idx(axis)``,
``sf.block_idx(axis)``, ``sf.block_dim(axis)``, ``sf.shared(shape, dtype)``,
``sf.barrier()``, ``sf.predicate(mask)``, pointer indexing ``p[i]``,
launch ``k[grid, block](...)``.

Block style (Triton):  ``@sf.jit``, ``sf.program_id(axis)``,
``sf.arange(0, BLOCK)``, ``sf.load(p + offs, mask=m)``,
``sf.store(p + offs, v, mask=m)``, ``sf.dot(a, b)`` (systolic array),
``sf.constexpr``, launch ``k[(grid,)](..., BLOCK=8)``.

Both styles trace to the same SF IR and run on the same backends; a thread
kernel is a block kernel whose lane vector is ``thread_idx``.
"""
from __future__ import annotations

import builtins
import contextlib
from typing import Any

import numpy as np

from .. import _reference as ref
from .._errors import SFCompileError, SFTypeError
from .._scalar import SFScalar
from ._ir import BOOL, INT, SFType, broadcast_shapes, builder, check_sf_bits, in_kernel
from ._sym import Ptr, Shared, Sym, _sf_bits, as_bool, as_int, as_sf, const_bool, const_int, is_traced


class constexpr:
    """Annotation for a kernel argument fixed at compile time (``BLOCK: sf.constexpr``)."""

    def __init__(self, value=None):
        self.value = value

    def __class_getitem__(cls, item):
        return cls


def _bits_of(dtype) -> int:
    import sf
    return sf.info(sf.dtype(dtype)).bits


# --- indices -----------------------------------------------------------------------------

def _axis(axis) -> int:
    if axis not in (0, 1, 2):
        raise SFCompileError("axis must be 0, 1 or 2")
    return int(axis)


def program_id(axis: int = 0) -> Sym:
    """Index of this program (block) along ``axis`` of the grid."""
    b = builder()
    return Sym(b.emit("program_id", [], {"axis": _axis(axis)}, INT, ()))


block_idx = program_id


def thread_idx(axis: int = 0) -> Sym:
    """Index of the lane along ``axis`` of the block, as a lane vector:
    axis 0 varies along the last dimension of the lane grid, axis 1 along
    the one before, axis 2 along the first (so the three broadcast)."""
    b = builder()
    if b.ir.style != "thread":
        raise SFCompileError("sf.thread_idx is for @sf.kernel (thread style); block kernels use "
                             "sf.program_id and sf.arange")
    ax = _axis(axis)
    n = b.ir.block_dim[ax]
    shape = [1] * 3
    shape[2 - ax] = n
    # trim leading 1s that correspond to unused block dims
    rank = builtins.max([i + 1 for i in range(3) if b.ir.block_dim[i] > 1] or [1])
    shape = tuple(shape[3 - rank:])
    return Sym(b.emit("thread_idx", [], {"axis": ax}, INT, shape))


def block_dim(axis: int = 0) -> int:
    """Lanes per block along ``axis`` (a Python int: fixed at launch)."""
    b = builder()
    if b.ir.style != "thread":
        raise SFCompileError("sf.block_dim is for @sf.kernel (thread style)")
    return int(b.ir.block_dim[_axis(axis)])


def arange(start: int, end: int) -> Sym:
    """``[start, end)`` as an int vector; both ends are compile-time ints."""
    b = builder()
    if isinstance(start, Sym) or isinstance(end, Sym):
        raise SFCompileError("sf.arange bounds must be Python ints or sf.constexpr arguments")
    start, end = int(start), int(end)
    if end < start:
        raise SFCompileError("sf.arange: end < start")
    return Sym(b.emit("arange", [], {"start": start, "end": end}, INT, (end - start,)))


def cdiv(a, b):
    """Ceiling division for grid sizes (host ints)."""
    if isinstance(a, Sym) or isinstance(b, Sym):
        return (as_int(a) + as_int(b) - 1) // as_int(b)
    return -(-int(a) // int(b))


# --- memory ------------------------------------------------------------------------------

def load(ptr: Ptr, mask=None, other: Any = 0) -> Sym:
    """Gather from a device buffer.  Masked-off lanes read ``other``."""
    if not isinstance(ptr, Ptr):
        raise SFTypeError("sf.load expects a pointer (a device array argument plus offsets)")
    b = builder()
    offs = ptr.offs if ptr.offs is not None else const_int(0)
    m = as_bool(mask) if mask is not None else const_bool(True)
    shape = broadcast_shapes(offs.shape, m.shape)
    oth = as_sf(other, ptr.bits)
    shape = broadcast_shapes(shape, oth.shape)
    return Sym(b.emit("load", [ptr.base, offs.v, m.v, oth.v], {}, SFType(ptr.bits), shape))


def store(ptr: Ptr, value, mask=None) -> None:
    """Scatter to a device buffer.  Masked-off lanes write nothing.  Lanes
    that hit the same address within one store are a race: the result is
    whichever lane the target writes last (the simulator: the highest lane)."""
    if not isinstance(ptr, Ptr):
        raise SFTypeError("sf.store expects a pointer (a device array argument plus offsets)")
    b = builder()
    offs = ptr.offs if ptr.offs is not None else const_int(0)
    v = as_sf(value, ptr.bits)
    m = as_bool(mask) if mask is not None else const_bool(True)
    broadcast_shapes(offs.shape, v.shape, m.shape)
    b.emit("store", [ptr.base, offs.v, v.v, m.v], {})


def shared(shape, dtype="sf16") -> Shared:
    """A tile scratchpad shared by the lanes of a block (``sf.shared``)."""
    b = builder()
    if isinstance(shape, int):
        shape = (shape,)
    shape = tuple(int(s) for s in shape)
    if any(isinstance(s, Sym) for s in shape):
        raise SFCompileError("sf.shared shape must be compile-time ints")
    bits = _bits_of(dtype)
    return Shared(b.shared(bits, shape))


def shared_load(buf: Shared, idx, mask=None) -> Sym:
    b = builder()
    i = as_int(idx)
    m = as_bool(mask) if mask is not None else const_bool(True)
    shape = broadcast_shapes(i.shape, m.shape)
    return Sym(b.emit("shared_load", [i.v, m.v], {"buf": buf.buf.id}, SFType(buf.buf.bits), shape))


def shared_store(buf: Shared, idx, value, mask=None) -> None:
    b = builder()
    i = as_int(idx)
    v = as_sf(value, buf.buf.bits)
    m = as_bool(mask) if mask is not None else const_bool(True)
    broadcast_shapes(i.shape, v.shape, m.shape)
    b.emit("shared_store", [i.v, v.v, m.v], {"buf": buf.buf.id})


def barrier() -> None:
    """Block-wide barrier.  Lockstep execution satisfies it trivially; it is
    recorded so a backend with real threads can honour it."""
    builder().emit("barrier", [], {})


@contextlib.contextmanager
def predicate(mask):
    """Predicated execution: stores, shared stores and numeric events inside
    the block are masked by ``mask`` (and any enclosing predicate).  This is
    how a per-thread ``if`` lowers on a lockstep machine."""
    b = builder()
    m = as_bool(mask)
    cur = b.current_pred()
    if cur is not None:
        m = Sym(b.emit("and", [cur, m.v], {}, BOOL, broadcast_shapes(cur.shape, m.shape)))
    b.pred_stack.append(m.v)
    try:
        yield
    finally:
        b.pred_stack.pop()


# --- arithmetic helpers ---------------------------------------------------------------------

def where(cond, x, y) -> Sym:
    b = builder()
    c = as_bool(cond)
    kx = "sf" if (isinstance(x, (SFScalar, float)) or (isinstance(x, Sym) and isinstance(x.type, SFType))) else None
    ky = "sf" if (isinstance(y, (SFScalar, float)) or (isinstance(y, Sym) and isinstance(y.type, SFType))) else None
    if kx == "sf" or ky == "sf":
        bits = _sf_bits(x, y)
        if bits is None:
            raise SFTypeError("sf.where: at least one branch must be an sf value")
        a, b2 = as_sf(x, bits), as_sf(y, bits)
        t = SFType(bits)
    elif isinstance(x, Sym) and x.type is BOOL or isinstance(x, bool):
        a, b2, t = as_bool(x), as_bool(y), BOOL
    else:
        a, b2, t = as_int(x), as_int(y), INT
    shape = broadcast_shapes(c.shape, a.shape, b2.shape)
    return Sym(b.emit("where", [c.v, a.v, b2.v], {}, t, shape))


def fma(a, b, c) -> Sym:
    bb = builder()
    bits = _sf_bits(a, b, c)
    if bits is None:
        raise SFTypeError("sf.fma: at least one operand must be an sf value")
    x, y, z = as_sf(a, bits), as_sf(b, bits), as_sf(c, bits)
    shape = broadcast_shapes(x.shape, y.shape, z.shape)
    return Sym(bb.emit("fma", [x.v, y.v, z.v], {}, SFType(bits), shape))


def sqrt(a) -> Sym:
    b = builder()
    x = as_sf(a, _sf_bits(a))
    return Sym(b.emit("sqrt", [x.v], {}, x.type, x.shape))


def _select(name, a, b) -> Sym:
    bb = builder()
    bits = _sf_bits(a, b)
    if bits is not None:
        x, y = as_sf(a, bits), as_sf(b, bits)
        t = SFType(bits)
    else:
        x, y = as_int(a), as_int(b)
        t = INT
    return Sym(bb.emit(name, [x.v, y.v], {}, t, broadcast_shapes(x.shape, y.shape)))


def maximum(a, b) -> Sym:
    return _select("maximum", a, b)


def minimum(a, b) -> Sym:
    return _select("minimum", a, b)


def dot(a, b) -> Sym:
    """Block matmul on the systolic array: the Stage 1.1 chain (per-product
    truncation, per-step saturation, sequential over K)."""
    bb = builder()
    bits = _sf_bits(a, b)
    if bits is None:
        raise SFTypeError("sf.dot: operands must be sf block values")
    x, y = as_sf(a, bits), as_sf(b, bits)
    if x.ndim != 2 or y.ndim != 2:
        raise SFCompileError(f"sf.dot expects 2-D blocks, got {x.shape} and {y.shape}")
    if x.shape[1] != y.shape[0]:
        raise SFCompileError(f"sf.dot: shapes {x.shape} and {y.shape} are not aligned")
    return Sym(bb.emit("dot", [x.v, y.v], {}, SFType(bits), (x.shape[0], y.shape[1])))


def sum(x, axis=None) -> Sym:                 # noqa: A001
    """Saturating sum along ``axis`` (or all), sequential from index 0."""
    bb = builder()
    v = as_sf(x, _sf_bits(x))
    if axis is None:
        v = v.reshape(v.v.size)
        axis = 0
    axis = int(axis)
    if axis < 0:
        axis += v.ndim
    if not 0 <= axis < v.ndim:
        raise SFCompileError("sf.sum: axis out of range")
    shape = v.shape[:axis] + v.shape[axis + 1:]
    return Sym(bb.emit("sum", [v.v], {"axis": axis}, v.type, shape))


def _reduce(name, x, axis) -> Sym:
    bb = builder()
    v = as_sf(x, _sf_bits(x)) if _sf_bits(x) is not None else as_int(x)
    if axis is None:
        v = v.reshape(v.v.size)
        axis = 0
    axis = int(axis)
    if axis < 0:
        axis += v.ndim
    shape = v.shape[:axis] + v.shape[axis + 1:]
    return Sym(bb.emit(name, [v.v], {"axis": axis}, v.type, shape))


def max(x, axis=None) -> Sym:                 # noqa: A001
    return _reduce("max", x, axis)


def min(x, axis=None) -> Sym:                 # noqa: A001
    return _reduce("min", x, axis)


def cast(x, dtype) -> Sym:
    """Change width: widening is exact; narrowing rounds half to even on the
    exact value and follows the error policy after the launch (events)."""
    bb = builder()
    bits = _bits_of(dtype)
    v = as_sf(x, _sf_bits(x))
    if v.type.bits == bits:
        return v
    if v.type.bits < bits:
        return as_sf(v, bits)
    return Sym(bb.emit("narrow", [v.v], {"bits": bits}, SFType(bits), v.shape))


def full(shape, value, dtype="sf16") -> Sym:
    """A block filled with one SF value (a literal quantized like Stage 1)."""
    bits = _bits_of(dtype)
    v = as_sf(value, bits)
    if isinstance(shape, int):
        shape = (shape,)
    bb = builder()
    return Sym(bb.emit("broadcast", [v.v], {"shape": tuple(shape)}, v.type, tuple(shape)))


def zeros(shape, dtype="sf16") -> Sym:
    return full(shape, 0, dtype)
