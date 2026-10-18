"""Interpreter backend: executes SF IR on the host with the bit-exact
reference, one program at a time, lanes vectorised with NumPy.

This is the *simulator* target's execution engine.  It is also the oracle
every other backend (``_cgen``) is checked against.  Events are exact: the
interpreter sees every saturation, underflow, divide-by-zero and domain
error, masked by the predicate in force, which is what a target with
per-tile sticky flags would report.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .. import _linalg, _vectorized as vec
from .._errors import SFDeviceError
from .._format import FORMATS
from ._ir import BOOL, INT, KernelIR, PtrType, SFType

_CMP = {"lt": np.less, "le": np.less_equal, "gt": np.greater, "ge": np.greater_equal,
        "eq": np.equal, "ne": np.not_equal}


@dataclass
class Counters:
    programs: int = 0
    ops: int = 0
    lane_ops: int = 0
    loads: int = 0
    stores: int = 0
    macs: int = 0
    saturated: int = 0
    underflow: int = 0
    div0: int = 0
    invalid: int = 0
    barriers: int = 0

    @property
    def cycles(self) -> int:
        """A simple cost model: one cycle per vector op of up to WIDTH lanes,
        plus the dot MACs on the systolic array, plus memory ops."""
        from ._runtime import WIDTH, SYSTOLIC
        return (int(math.ceil(self.lane_ops / WIDTH)) + int(math.ceil(self.macs / SYSTOLIC))
                + self.loads + self.stores + self.barriers)


def _trunc_div(num, den):
    q = np.abs(num) // np.abs(den)
    return np.where((num < 0) != (den < 0), -q, q)


class Interpreter:
    def __init__(self, ir: KernelIR, buffers: dict, scalars: dict, grid: tuple, limits) -> None:
        self.ir = ir
        self.buffers = buffers        # param name -> int64 numpy array (device memory)
        self.scalars = scalars        # param name -> int (sf code or int)
        self.grid = grid
        self.limits = limits
        self.counters = Counters()
        self.param_of_value = {p.vid: p for p in ir.params if p.kind != "constexpr"}

    # --- entry ---------------------------------------------------------------------------
    def run(self) -> Counters:
        gx, gy, gz = self.grid
        for pz in range(gz):
            for py in range(gy):
                for px in range(gx):
                    self._program((px, py, pz))
        return self.counters

    def _program(self, pid) -> None:
        self.counters.programs += 1
        env = {}
        shared = [np.zeros(s.shape, dtype=np.int64) for s in self.ir.shared]
        ir = self.ir
        for op in ir.ops:
            pred = env[op.pred.id] if op.pred is not None else None
            r = self._exec(op, env, shared, pid, pred)
            if op.result is not None:
                if np.shape(r) != op.result.shape:            # one copy per live lane
                    r = np.broadcast_to(r, op.result.shape).copy()
                env[op.result.id] = r
            self.counters.ops += 1

    # --- helpers -------------------------------------------------------------------------
    def _events(self, ev: dict, pred, shape) -> None:
        for name, key in (("overflow", "saturated"), ("underflow", "underflow"),
                          ("divide", "div0"), ("invalid", "invalid")):
            m = ev.get(name)
            if m is None:
                continue
            m = np.broadcast_to(m, shape)
            if pred is not None:
                m = m & np.broadcast_to(pred, shape)
            setattr(self.counters, key, getattr(self.counters, key) + int(np.count_nonzero(m)))

    def _fault(self, msg: str):
        raise SFDeviceError(f"device fault in kernel {self.ir.name!r}: {msg}")

    def _exec(self, op, env, shared, pid, pred):
        n = op.name
        a = [env[x.id] if hasattr(x, "id") else x for x in op.args]
        shape = op.result.shape if op.result is not None else ()
        lanes = int(np.prod(shape)) if shape else 1
        self.counters.lane_ops += lanes

        if n == "param":
            p = op.attrs["param"]
            if op.attrs["kind"] == "ptr":
                return None                                   # buffers are looked up by value id
            return np.int64(self.scalars[p])
        if n == "const":
            v = op.attrs["value"]
            return np.full(shape, v, dtype=bool if op.attrs["kind"] == "bool" else np.int64)
        if n == "program_id":
            return np.int64(pid[op.attrs["axis"]])
        if n == "thread_idx":
            ax = op.attrs["axis"]
            return np.arange(self.ir.block_dim[ax], dtype=np.int64).reshape(shape)
        if n == "arange":
            return np.arange(op.attrs["start"], op.attrs["end"], dtype=np.int64)
        if n == "broadcast":
            return np.broadcast_to(a[0], shape).copy()
        if n == "reshape":
            return np.asarray(a[0]).reshape(shape)
        if n == "transpose":
            return np.asarray(a[0]).T.copy()
        # ints
        if n == "iadd":
            return a[0] + a[1]
        if n == "isub":
            return a[0] - a[1]
        if n == "imul":
            return a[0] * a[1]
        if n == "idiv":
            if np.any(np.broadcast_to(a[1], shape) == 0):
                self._fault("integer division by zero")
            return np.floor_divide(a[0], a[1])
        if n == "imod":
            if np.any(np.broadcast_to(a[1], shape) == 0):
                self._fault("integer modulo by zero")
            return np.mod(a[0], a[1])
        if n == "ineg":
            return -a[0]
        if n == "iabs":
            return np.abs(a[0])
        if n == "icmp":
            return _CMP[op.attrs["op"]](a[0], a[1])
        if n == "bool_to_int":
            return np.asarray(a[0]).astype(np.int64)
        # bools
        if n == "and":
            return np.logical_and(a[0], a[1])
        if n == "or":
            return np.logical_or(a[0], a[1])
        if n == "xor":
            return np.logical_xor(a[0], a[1])
        if n == "not":
            return np.logical_not(a[0])
        if n == "where":
            return np.where(a[0], a[1], a[2])
        # sf arithmetic
        fmt = FORMATS[op.result.type.bits] if op.result is not None and isinstance(op.result.type, SFType) else None
        if n in ("add", "sub", "mul", "div"):
            r, ev = getattr(vec, n)(fmt, a[0], a[1])
            self._events(ev, pred, shape)
            return np.broadcast_to(r, shape).copy()
        if n == "fma":
            r, ev = vec.fma(fmt, a[0], a[1], a[2])
            self._events(ev, pred, shape)
            return np.broadcast_to(r, shape).copy()
        if n == "neg":
            return -a[0]
        if n == "abs":
            return np.abs(a[0])
        if n == "sqrt":
            r, ev = vec.sqrt(fmt, a[0])
            self._events(ev, pred, shape)
            return r
        if n == "widen":
            src = op.args[0].type.bits
            return np.asarray(a[0]) << (fmt.bits - src)
        if n == "narrow":
            src = op.args[0].type.bits
            shift = src - fmt.bits
            num = np.asarray(a[0])
            den = 1 << shift
            q, r = np.divmod(num, den)
            twice = 2 * r
            k = q + ((twice > den) | ((twice == den) & (q % 2 == 1)))
            over = np.abs(k) > fmt.max_code
            k = np.clip(k, -fmt.max_code, fmt.max_code)
            self._events({"overflow": over, "underflow": (k == 0) & (num != 0)}, pred, shape)
            return k
        if n == "cmp":
            return _CMP[op.attrs["op"]](a[0], a[1])
        if n == "cmp_lit":
            num, den = op.attrs["num"], op.attrs["den"]
            return _CMP[op.attrs["op"]](np.asarray(a[0]) * den, num)
        if n in ("maximum", "minimum"):
            return (np.maximum if n == "maximum" else np.minimum)(a[0], a[1])
        if n == "sum":
            r, ev = _linalg.sum_codes(fmt, a[0], op.attrs["axis"])
            self._events(ev, pred, shape)
            return r
        if n == "max":
            return np.max(a[0], axis=op.attrs["axis"])
        if n == "min":
            return np.min(a[0], axis=op.attrs["axis"])
        if n == "dot":
            r, ev = _linalg.matmul_codes(fmt, a[0], a[1])
            self._events(ev, pred, shape)
            self.counters.macs += int(a[0].shape[0] * a[0].shape[1] * a[1].shape[1])
            return r
        # memory
        if n == "load":
            buf = self._buffer(op.args[0])
            offs, mask, other = (np.broadcast_to(x, shape) for x in a[1:4])
            self.counters.loads += 1
            live = mask if pred is None else mask & np.broadcast_to(pred, shape)
            if live.any():
                o = offs[live]
                if o.size and (o.min() < 0 or o.max() >= buf.size):
                    self._fault(f"load out of bounds: offset {int(o.min()) if o.min() < 0 else int(o.max())} "
                                f"on a buffer of {buf.size} elements")
            out = np.array(np.broadcast_to(other, shape), dtype=np.int64)
            out[live] = buf.reshape(-1)[offs[live]]
            return out
        if n == "store":
            buf = self._buffer(op.args[0])
            bshape = np.broadcast_shapes(*(np.shape(x) for x in a[1:4]))
            offs, val, mask = (np.broadcast_to(x, bshape) for x in a[1:4])
            self.counters.stores += 1
            live = mask if pred is None else mask & np.broadcast_to(pred, bshape)
            if live.any():
                o = offs[live]
                if o.min() < 0 or o.max() >= buf.size:
                    self._fault(f"store out of bounds: offset {int(o.min()) if o.min() < 0 else int(o.max())} "
                                f"on a buffer of {buf.size} elements")
                buf.reshape(-1)[o] = val[live]
            return None
        if n == "shared_load":
            buf = shared[op.attrs["buf"]].reshape(-1)
            idx, mask = (np.broadcast_to(x, shape) for x in a[:2])
            live = mask if pred is None else mask & np.broadcast_to(pred, shape)
            if live.any() and (idx[live].min() < 0 or idx[live].max() >= buf.size):
                self._fault("shared memory load out of bounds")
            out = np.zeros(shape, dtype=np.int64)
            out[live] = buf[idx[live]]
            return out
        if n == "shared_store":
            buf = shared[op.attrs["buf"]].reshape(-1)
            bshape = np.broadcast_shapes(*(np.shape(x) for x in a[:3]))
            idx, val, mask = (np.broadcast_to(x, bshape) for x in a[:3])
            live = mask if pred is None else mask & np.broadcast_to(pred, bshape)
            if live.any():
                if idx[live].min() < 0 or idx[live].max() >= buf.size:
                    self._fault("shared memory store out of bounds")
                buf[idx[live]] = val[live]
            return None
        if n == "barrier":
            self.counters.barriers += 1
            return None
        raise SFDeviceError(f"interpreter: unknown op {n}")

    def _buffer(self, base_value) -> np.ndarray:
        p = self.param_of_value[base_value.id]
        return self.buffers[p.name]
