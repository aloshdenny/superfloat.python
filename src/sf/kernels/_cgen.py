"""C backend: SF IR -> C with the ``__sf16_*`` builtins (spec section 11).

This is the "SF IR -> C -> custom Clang -> LLVM -> Atreides" path with the
host C compiler standing in for the custom Clang and ``sf16_builtins.h``
standing in for the device builtins.  User code never sees C: the emitter
is internal, the semantics are the host reference's, and the backend is
correct only if it agrees bit for bit with the interpreter on the same IR
(the Stage 3 test suite runs every kernel through both).

Only sf16 is emitted (the device has only sf16 builtins); the interpreter
serves other widths.  Values are static-shape int64 arrays on the stack;
broadcasting is index arithmetic computed at emit time.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import tempfile
import threading

import numpy as np

from .._errors import SFCompileError, SFDeviceError, SFUnsupportedError
from ._interp import Counters
from ._ir import BOOL, INT, KernelIR, PtrType, SFType

_HERE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(tempfile.gettempdir(), "sf-kernels-c")
_lock = threading.Lock()
_loaded: dict[str, ctypes.CDLL] = {}

_CMP_C = {"lt": "<", "le": "<=", "gt": ">", "ge": ">=", "eq": "==", "ne": "!="}


def compiler() -> str | None:
    for cc in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if cc and shutil.which(cc):
            return cc
    return None


def _strides(shape):
    s, out = 1, []
    for d in reversed(shape):
        out.append(s)
        s *= d
    return list(reversed(out))


def _index(src_shape, dst_shape) -> str:
    """C expression mapping flat index ``e`` of ``dst_shape`` to the flat
    index of a value of ``src_shape`` broadcast to it."""
    if tuple(src_shape) == tuple(dst_shape):
        return "e"
    if not src_shape or int(np.prod(src_shape)) == 1:
        return "0"
    rank = len(dst_shape)
    src = (1,) * (rank - len(src_shape)) + tuple(src_shape)
    dst_str = _strides(dst_shape)
    src_str = _strides(src)
    terms = []
    for d in range(rank):
        if src[d] == 1:
            continue
        terms.append(f"((e / {dst_str[d]}) % {dst_shape[d]}) * {src_str[d]}")
    return " + ".join(terms) if terms else "0"


class Emitter:
    def __init__(self, ir: KernelIR) -> None:
        self.ir = ir
        self.lines: list[str] = []
        self.ptr_names = {}
        self.scalar_names = {}
        for p in ir.params:
            if p.kind == "ptr":
                self.ptr_names[p.vid] = p.name
            elif p.kind in ("sf", "int"):
                self.scalar_names[p.vid] = p.name

    def w(self, s: str) -> None:
        self.lines.append(s)

    def emit(self) -> str:
        ir = self.ir
        ptrs = [p for p in ir.params if p.kind == "ptr"]
        scalars = [p for p in ir.params if p.kind in ("sf", "int")]
        for p in ir.params:
            if p.kind in ("ptr", "sf") and p.type.bits != 16:
                raise SFUnsupportedError("the C backend emits sf16 only")
        for v in ir.values:
            if isinstance(v.type, SFType) and v.type.bits != 16:
                raise SFUnsupportedError("the C backend emits sf16 only")
        self.w('#include "sf16_builtins.h"')
        self.w("#include <string.h>")
        self.w("typedef struct { int64_t programs, ops, lane_ops, loads, stores, macs, "
               "sat, under, div0, invalid, barriers, fault; } sf_counters;")
        args = [f"int64_t *{p.name}, int64_t {p.name}__size" for p in ptrs]
        args += [f"int64_t {p.name}" for p in scalars]
        args += ["int64_t gx, int64_t gy, int64_t gz", "sf_counters *C"]
        self.w(f"void sf_kernel({', '.join(args)}) {{")
        for s in ir.shared:
            self.w(f"  static int64_t shared{s.id}[{max(s.size, 1)}];")
        self.w("  for (int64_t pz = 0; pz < gz; pz++) for (int64_t py = 0; py < gy; py++) "
               "for (int64_t px = 0; px < gx; px++) {")
        self.w("    C->programs++;")
        for s in ir.shared:
            self.w(f"    memset(shared{s.id}, 0, sizeof(shared{s.id}));")
        for op in ir.ops:
            self.op(op)
        self.w("  }")
        self.w("}")
        return "\n".join(self.lines) + "\n"

    def decl(self, v) -> str:
        c = "char" if v.type is BOOL else "int64_t"
        return f"{c} v{v.id}[{max(v.size, 1)}]"

    def op(self, op) -> None:
        n = op.name
        r = op.result
        w = self.w
        pred = f"v{op.pred.id}[{_index(op.pred.shape, r.shape if r is not None else op.pred.shape)}]" \
            if op.pred is not None else None
        if r is not None:
            w(f"    {self.decl(r)};")
            w(f"    C->ops++; C->lane_ops += {r.size};")
        else:
            w("    C->ops++;")

        def ev(e):
            # per-lane event block, counted only where the predicate holds
            if op.pred is None:
                return "&E"
            return f"({pred} ? &E : (sf16_events*)0)".replace("[e]", "[e]") if False else \
                f"(v{op.pred.id}[{_index(op.pred.shape, r.shape)}] ? &E : (sf16_events*)0)"

        def loop(body, shape=None):
            size = int(np.prod(shape)) if shape else (r.size if r is not None else 1)
            w(f"    for (int64_t e = 0; e < {size}; e++) {{ {body} }}")

        def a(i, shape=None):
            arg = op.args[i]
            if hasattr(arg, "id"):
                if arg.id in self.scalar_names:
                    return self.scalar_names[arg.id]
                return f"v{arg.id}[{_index(arg.shape, shape if shape is not None else r.shape)}]"
            return str(arg)

        if n == "param":
            if op.attrs["kind"] == "ptr":
                w(f"    (void)0;")
            else:
                w(f"    v{r.id}[0] = {op.attrs['param']};")
            return
        if n == "const":
            v = op.attrs["value"]
            loop(f"v{r.id}[e] = {int(v)};")
            return
        if n == "program_id":
            w(f"    v{r.id}[0] = {['px', 'py', 'pz'][op.attrs['axis']]};")
            return
        if n == "thread_idx":
            ax = op.attrs["axis"]
            dim = self.ir.block_dim[ax]
            # lane index along the axis of this value's position
            strides = _strides(r.shape)
            pos = len(r.shape) - 1 - ax
            if pos < 0 or r.shape[pos] != dim:
                loop(f"v{r.id}[e] = 0;")
            else:
                loop(f"v{r.id}[e] = (e / {strides[pos]}) % {dim};")
            return
        if n == "arange":
            loop(f"v{r.id}[e] = {op.attrs['start']} + e;")
            return
        if n == "reshape":
            loop(f"v{r.id}[e] = v{op.args[0].id}[e];")          # same flat layout, new shape
            return
        if n == "broadcast":
            loop(f"v{r.id}[e] = {a(0)};")
            return
        if n == "transpose":
            rows, cols = op.args[0].shape
            loop(f"v{r.id}[e] = v{op.args[0].id}[(e % {rows}) * {cols} + e / {rows}];")
            return
        if n in ("iadd", "isub", "imul"):
            sym = {"iadd": "+", "isub": "-", "imul": "*"}[n]
            loop(f"v{r.id}[e] = {a(0)} {sym} {a(1)};")
            return
        if n in ("idiv", "imod"):
            w(f"    for (int64_t e = 0; e < {r.size}; e++) {{ int64_t d = {a(1)}; if (d == 0) {{ C->fault = 1; d = 1; }}"
              f" int64_t q = {a(0)} / d; int64_t m = {a(0)} % d; if (m != 0 && ((m < 0) != (d < 0))) {{ q--; m += d; }}"
              f" v{r.id}[e] = {'q' if n == 'idiv' else 'm'}; }}")
            return
        if n == "ineg":
            loop(f"v{r.id}[e] = -{a(0)};")
            return
        if n == "iabs":
            loop(f"v{r.id}[e] = {a(0)} < 0 ? -{a(0)} : {a(0)};")
            return
        if n in ("icmp", "cmp"):
            loop(f"v{r.id}[e] = {a(0)} {_CMP_C[op.attrs['op']]} {a(1)};")
            return
        if n == "cmp_lit":
            loop(f"v{r.id}[e] = {a(0)} * {op.attrs['den']}LL {_CMP_C[op.attrs['op']]} {op.attrs['num']}LL;")
            return
        if n == "bool_to_int":
            loop(f"v{r.id}[e] = {a(0)} ? 1 : 0;")
            return
        if n in ("and", "or", "xor"):
            sym = {"and": "&&", "or": "||", "xor": "!="}[n]
            loop(f"v{r.id}[e] = ({a(0)} != 0) {sym} ({a(1)} != 0);")
            return
        if n == "not":
            loop(f"v{r.id}[e] = !{a(0)};")
            return
        if n == "where":
            loop(f"v{r.id}[e] = {a(0)} ? {a(1)} : {a(2)};")
            return
        if n in ("add", "sub", "mul", "div"):
            loop(f"v{r.id}[e] = __sf16_{n}({a(0)}, {a(1)}, {ev(0)});")
            return
        if n == "fma":
            loop(f"v{r.id}[e] = __sf16_fma({a(0)}, {a(1)}, {a(2)}, {ev(0)});")
            return
        if n == "neg":
            loop(f"v{r.id}[e] = __sf16_neg({a(0)});")
            return
        if n == "abs":
            loop(f"v{r.id}[e] = __sf16_abs({a(0)});")
            return
        if n == "sqrt":
            loop(f"v{r.id}[e] = __sf16_sqrt({a(0)}, {ev(0)});")
            return
        if n in ("widen", "narrow"):
            raise SFUnsupportedError("the C backend emits sf16 only (no width changes)")
        if n in ("maximum", "minimum"):
            sym = ">" if n == "maximum" else "<"
            loop(f"int64_t x = {a(0)}, y = {a(1)}; v{r.id}[e] = x {sym} y ? x : y;")
            return
        if n == "sum":
            self.reduce(op, "sum")
            return
        if n in ("max", "min"):
            self.reduce(op, n)
            return
        if n == "dot":
            x, y = op.args
            M, K = x.shape
            N = y.shape[1]
            w(f"    for (int64_t i = 0; i < {M}; i++) for (int64_t j = 0; j < {N}; j++) {{ int64_t acc = 0; "
              f"sf16_events *pe = {'&E' if op.pred is None else f'(v{op.pred.id}[{_index(op.pred.shape, r.shape)}] ? &E : (sf16_events*)0)'.replace('[e]', '[i * ' + str(N) + ' + j]')};"
              f" sf16_events le = {{0, 0, 0, 0}};"
              f" for (int64_t k = 0; k < {K}; k++) acc = __sf16_fma(v{x.id}[i * {K} + k], v{y.id}[k * {N} + j], acc, &le);"
              f" v{r.id}[i * {N} + j] = acc; sf16_fold(pe, &le); }} C->macs += {M * K * N};")
            return
        if n == "load":
            base, offs, mask, other = op.args
            name = self.ptr_names[base.id]
            live = f"v{mask.id}[{_index(mask.shape, r.shape)}]"
            if op.pred is not None:
                live = f"({live} && v{op.pred.id}[{_index(op.pred.shape, r.shape)}])"
            w(f"    C->loads++;")
            w(f"    for (int64_t e = 0; e < {r.size}; e++) {{ if ({live}) {{ int64_t o = v{offs.id}[{_index(offs.shape, r.shape)}];"
              f" if (o < 0 || o >= {name}__size) {{ C->fault = 2; v{r.id}[e] = 0; }} else v{r.id}[e] = {name}[o]; }}"
              f" else v{r.id}[e] = v{other.id}[{_index(other.shape, r.shape)}]; }}")
            return
        if n == "store":
            base, offs, val, mask = op.args
            name = self.ptr_names[base.id]
            shape = tuple(np.broadcast_shapes(offs.shape, val.shape, mask.shape))
            size = int(np.prod(shape)) if shape else 1
            live = f"v{mask.id}[{_index(mask.shape, shape)}]"
            if op.pred is not None:
                live = f"({live} && v{op.pred.id}[{_index(op.pred.shape, shape)}])"
            w(f"    C->stores++;")
            w(f"    for (int64_t e = 0; e < {size}; e++) {{ if ({live}) {{ int64_t o = v{offs.id}[{_index(offs.shape, shape)}];"
              f" if (o < 0 || o >= {name}__size) C->fault = 3; else {name}[o] = v{val.id}[{_index(val.shape, shape)}]; }} }}")
            return
        if n == "shared_load":
            idx, mask = op.args
            buf = self.ir.shared[op.attrs["buf"]]
            live = f"v{mask.id}[{_index(mask.shape, r.shape)}]"
            if op.pred is not None:
                live = f"({live} && v{op.pred.id}[{_index(op.pred.shape, r.shape)}])"
            w(f"    for (int64_t e = 0; e < {r.size}; e++) {{ if ({live}) {{ int64_t o = v{idx.id}[{_index(idx.shape, r.shape)}];"
              f" if (o < 0 || o >= {buf.size}) {{ C->fault = 4; v{r.id}[e] = 0; }} else v{r.id}[e] = shared{buf.id}[o]; }}"
              f" else v{r.id}[e] = 0; }}")
            return
        if n == "shared_store":
            idx, val, mask = op.args
            buf = self.ir.shared[op.attrs["buf"]]
            shape = tuple(np.broadcast_shapes(idx.shape, val.shape, mask.shape))
            size = int(np.prod(shape)) if shape else 1
            live = f"v{mask.id}[{_index(mask.shape, shape)}]"
            if op.pred is not None:
                live = f"({live} && v{op.pred.id}[{_index(op.pred.shape, shape)}])"
            # lockstep semantics: every lane reads before any lane writes
            w(f"    {{ static int64_t tmp[{max(size, 1)}]; char lv[{max(size, 1)}];"
              f" for (int64_t e = 0; e < {size}; e++) {{ lv[e] = {live}; tmp[e] = v{val.id}[{_index(val.shape, shape)}]; }}"
              f" for (int64_t e = 0; e < {size}; e++) {{ if (lv[e]) {{ int64_t o = v{idx.id}[{_index(idx.shape, shape)}];"
              f" if (o < 0 || o >= {buf.size}) C->fault = 5; else shared{buf.id}[o] = tmp[e]; }} }} }}")
            return
        if n == "barrier":
            w("    C->barriers++;")
            return
        raise SFCompileError(f"C backend: unknown op {n}")

    def reduce(self, op, kind) -> None:
        r = op.result
        x = op.args[0]
        axis = op.attrs["axis"]
        shape = x.shape
        outer = int(np.prod(shape[:axis])) if shape[:axis] else 1
        n = shape[axis]
        inner = int(np.prod(shape[axis + 1:])) if shape[axis + 1:] else 1
        pe = "&E" if op.pred is None else f"(v{op.pred.id}[{_index(op.pred.shape, r.shape)}] ? &E : (sf16_events*)0)"
        if kind == "sum":
            step = f"acc = __sf16_add(acc, v{x.id}[o * {n * inner} + k * {inner} + i], &le);"
            init = "0"
        elif kind == "max":
            step = f"{{ int64_t t = v{x.id}[o * {n * inner} + k * {inner} + i]; if (t > acc) acc = t; }}"
            init = f"v{x.id}[o * {n * inner} + i]"
        else:
            step = f"{{ int64_t t = v{x.id}[o * {n * inner} + k * {inner} + i]; if (t < acc) acc = t; }}"
            init = f"v{x.id}[o * {n * inner} + i]"
        # events are counted once per output, like the Stage 1.1 reductions and the interpreter
        self.w(f"    for (int64_t o = 0; o < {outer}; o++) for (int64_t i = 0; i < {inner}; i++) {{ int64_t e = o * {inner} + i; (void)e;"
               f" sf16_events *pe = {pe}; sf16_events le = {{0, 0, 0, 0}}; int64_t acc = {init};"
               f" for (int64_t k = 0; k < {n}; k++) {step} v{r.id}[e] = acc; sf16_fold(pe, &le); }}")


def emit_c(ir: KernelIR) -> str:
    src = Emitter(ir).emit()
    # the event block is shared by every op of a program; declare it once
    src = src.replace("    C->programs++;", "    C->programs++; sf16_events E = {0, 0, 0, 0};", 1)
    src = src.replace("  }\n}\n", "    C->sat += E.sat; C->under += E.under; C->div0 += E.div0; C->invalid += E.invalid;\n  }\n}\n")
    return src


class _CCounters(ctypes.Structure):
    _fields_ = [(f, ctypes.c_int64) for f in ("programs", "ops", "lane_ops", "loads", "stores", "macs",
                                              "sat", "under", "div0", "invalid", "barriers", "fault")]


def build(ir: KernelIR) -> ctypes.CDLL:
    cc = compiler()
    if cc is None:
        raise SFUnsupportedError("the C kernel backend needs a C compiler (cc, gcc or clang) on PATH")
    src = emit_c(ir)
    key = hashlib.sha256(src.encode()).hexdigest()[:24]
    with _lock:
        lib = _loaded.get(key)
        if lib is not None:
            return lib
        os.makedirs(_CACHE_DIR, exist_ok=True)
        c_path = os.path.join(_CACHE_DIR, f"{key}.c")
        so_path = os.path.join(_CACHE_DIR, f"{key}.so")
        if not os.path.exists(so_path):
            with open(c_path, "w") as f:
                f.write(src)
            cmd = [cc, "-O2", "-shared", "-fPIC", "-std=c99", f"-I{_HERE}", c_path, "-o", so_path, "-lm"]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise SFCompileError(f"C backend: {cc} failed\n{proc.stderr}\n--- IR ---\n{ir.dump()}")
        lib = ctypes.CDLL(so_path)
        _loaded[key] = lib
        return lib


def run_c(ir: KernelIR, buffers: dict, scalars: dict, grid) -> Counters:
    lib = build(ir)
    # same order as the emitted signature: every pointer (with its size), then every scalar
    args = []
    keep = []
    for p in ir.params:
        if p.kind == "ptr":
            buf = np.ascontiguousarray(buffers[p.name].reshape(-1), dtype=np.int64)
            keep.append(buf)
            args += [buf.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), ctypes.c_int64(buf.size)]
    for p in ir.params:
        if p.kind in ("sf", "int"):
            args.append(ctypes.c_int64(int(scalars[p.name])))
    args += [ctypes.c_int64(grid[0]), ctypes.c_int64(grid[1]), ctypes.c_int64(grid[2])]
    cnt = _CCounters()
    lib.sf_kernel(*args, ctypes.byref(cnt))
    # write back buffers that had to be copied to become contiguous
    for p, buf in zip([p for p in ir.params if p.kind == "ptr"], keep):
        dst = buffers[p.name]
        if dst.ctypes.data != buf.ctypes.data:
            dst.reshape(-1)[:] = buf
    if cnt.fault:
        raise SFDeviceError(f"device fault in kernel {ir.name!r}: "
                            + {1: "integer division by zero", 2: "load out of bounds", 3: "store out of bounds",
                               4: "shared memory load out of bounds", 5: "shared memory store out of bounds"}[cnt.fault])
    return Counters(programs=cnt.programs, ops=cnt.ops, lane_ops=cnt.lane_ops, loads=cnt.loads,
                    stores=cnt.stores, macs=cnt.macs, saturated=cnt.sat, underflow=cnt.under,
                    div0=cnt.div0, invalid=cnt.invalid, barriers=cnt.barriers)
