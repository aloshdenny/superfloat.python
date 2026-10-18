"""Runtime (spec 3.0): targets, allocation, copies, launch, synchronize,
status readback, kernel cache, autotuning and profiler counters.

Targets for kernels:

* ``"atreides-sim"``: the simulator.  Executes SF IR with the interpreter
  (or the C backend under ``sf.config(kernel_backend="c")``), reports exact
  events (it has sticky flags), and enforces the launch limits of Atreides
  v1 before any launch.
* ``"atreides-hw"``: the hardware target.  Same limits and interface; its
  ``launch`` raises ``SFDeviceError`` until a driver is attached with
  ``HardwareAtreides.attach(driver)``.

Both also serve the Stage 1.1 built-in ops, so arrays can live on them.
"""
from __future__ import annotations

import inspect
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .. import _config, _device, _errstate
from .._array import SFArray
from .._errors import SFCompileError, SFDeviceError, SFLaunchError, SFTypeError, SFUnsupportedError
from .._scalar import SFScalar
from .._status import SFStatus
from . import _ir
from ._interp import Counters, Interpreter
from ._ir import INT, Builder, KernelIR, Param, PtrType, SFType, trace_context
from ._sym import Ptr, Sym

WIDTH = 256          # lanes per vector op in the cost model
SYSTOLIC = 4096      # MACs per cycle on the systolic array in the cost model


@dataclass(frozen=True)
class Limits:
    max_block: int = 1024            # lanes per block (thread kernels)
    max_tile: int = 16384            # elements per block value (block kernels)
    max_grid: int = 65535            # programs per grid axis
    max_shared: int = 32768          # sf16 elements of scratchpad per block
    max_elements: int = 1 << 28      # elements of any one buffer


@dataclass
class LaunchResult:
    kernel: str
    grid: tuple
    block: tuple
    device: str
    status: SFStatus
    counters: Counters
    backend: str
    config: dict = field(default_factory=dict)

    @property
    def cycles(self) -> int:
        return self.counters.cycles

    def __repr__(self):
        return (f"LaunchResult({self.kernel!r} on {self.device}, grid={self.grid}, "
                f"block={self.block}, backend={self.backend}, {self.status}, cycles={self.cycles})")


# --- targets -----------------------------------------------------------------------------------

class KernelDevice(_device.MockAtreides):
    """A target that runs user kernels (spec 3.0) as well as the built-ins."""

    limits = Limits()
    sticky_flags = True
    supported_bits = (16,)

    def __init__(self) -> None:
        super().__init__()
        self.kernel_launches = 0
        self.last: LaunchResult | None = None

    # Stage 1.1 built-ins: with sticky flags the runtime gets exact counts
    def run(self, op, fmt, *operands, **kw):
        self.launches += 1
        codes, events = _device._KERNELS[op](fmt, *operands, **kw)
        return np.asarray(codes, dtype=np.int64), {k: np.asarray(v, dtype=bool)
                                                   for k, v in events.items() if k != "first"}

    def check_launch(self, ir: KernelIR, grid, block, buffers) -> None:
        L = self.limits
        for g in grid:
            if not (1 <= g <= L.max_grid):
                raise SFLaunchError(f"grid {grid} exceeds the {self.name} limit of {L.max_grid} per axis")
        lanes = int(np.prod(block))
        if ir.style == "thread" and lanes > L.max_block:
            raise SFLaunchError(f"block {block} has {lanes} lanes; the {self.name} limit is {L.max_block}")
        biggest = max((v.size for v in ir.values), default=0)
        if biggest > L.max_tile:
            raise SFLaunchError(f"a block value has {biggest} elements; the {self.name} limit is {L.max_tile}")
        sh = sum(s.size for s in ir.shared)
        if sh > L.max_shared:
            raise SFLaunchError(f"{sh} elements of shared memory; the {self.name} limit is {L.max_shared}")
        for name, buf in buffers.items():
            if buf.size > L.max_elements:
                raise SFLaunchError(f"buffer {name!r} has {buf.size} elements; the limit is {L.max_elements}")

    def launch(self, ir, grid, block, buffers, scalars) -> Counters:
        raise NotImplementedError

    # allocation and copies (3.0)
    def alloc(self, shape, dtype="sf16") -> SFArray:
        import sf
        return sf.zeros(shape, dtype, device=self.name)

    def upload(self, host) -> SFArray:
        import sf
        if isinstance(host, SFArray):
            return host.to(self.name)
        return sf.asarray(host, dtype=sf.sf16, device=self.name)

    def download(self, arr: SFArray) -> SFArray:
        return arr.to(_device.HOST)

    def synchronize(self) -> LaunchResult | None:
        return self.last


class SimulatorAtreides(KernelDevice):
    name = "atreides-sim"

    def launch(self, ir, grid, block, buffers, scalars) -> Counters:
        self.kernel_launches += 1
        backend = _config.get("kernel_backend")
        if backend == "interp":
            return Interpreter(ir, buffers, scalars, grid, self.limits).run()
        if backend == "c":
            from ._cgen import run_c
            return run_c(ir, buffers, scalars, grid)
        raise SFUnsupportedError(f"unknown kernel backend {backend!r}")


class HardwareAtreides(KernelDevice):
    name = "atreides-hw"
    _driver = None

    @classmethod
    def attach(cls, driver) -> None:
        """Attach a driver object with ``launch(ir, grid, block, buffers, scalars) -> Counters``."""
        cls._driver = driver

    def launch(self, ir, grid, block, buffers, scalars) -> Counters:
        if self._driver is None:
            raise SFDeviceError("no Atreides hardware is attached (HardwareAtreides.attach(driver))")
        self.kernel_launches += 1
        return self._driver.launch(ir, grid, block, buffers, scalars)


_device.register_device(SimulatorAtreides())
_device.register_device(HardwareAtreides())


def synchronize(device="atreides-sim") -> LaunchResult | None:
    """Wait for the device (the simulator is synchronous) and return the
    last launch's result."""
    dev = _device.get_device(device)
    if not isinstance(dev, KernelDevice):
        raise SFUnsupportedError(f"{dev.name} does not run kernels")
    return dev.synchronize()


# --- kernels ------------------------------------------------------------------------------------

_cache_lock = threading.Lock()


class Kernel:
    """A traced kernel.  ``kernel[grid, block](*args, **constexpr)`` launches it."""

    def __init__(self, fn, style: str) -> None:
        self.fn = fn
        self.style = style
        self.name = fn.__name__
        self.sig = inspect.signature(fn)
        self._constexpr = tuple(n for n, p in self.sig.parameters.items()
                                if _is_constexpr_annotation(p.annotation)
                                or isinstance(p.default, _constexpr_cls()))
        self.cache: dict = {}
        self.autotune_configs = None
        self.autotune_key = ()
        self.tuned: dict = {}

    def __repr__(self):
        return f"<sf {'kernel' if self.style == 'thread' else 'jit'} {self.name}>"

    def __getitem__(self, spec):
        return Launcher(self, spec)

    def __call__(self, *a, **k):
        raise SFCompileError(f"launch {self.name} with {self.name}[grid, block](...) "
                             f"(thread style) or {self.name}[(grid,)](..., BLOCK=...) (block style)")

    # --- tracing ------------------------------------------------------------------------
    def constexpr_names(self):
        return self._constexpr

    def trace(self, args, constexpr, block) -> KernelIR:
        b = Builder(self.name, self.style)
        b.ir.block_dim = tuple(block)
        b.ir.constexpr = dict(constexpr)
        call_args = []
        given = dict(args)
        with trace_context(b):
            for name in self.sig.parameters:               # signature order, configs included
                if name in constexpr:
                    p = Param(name, "constexpr", value=constexpr[name])
                    b.ir.params.append(p)
                    call_args.append(constexpr[name])
                    continue
                if name not in given:
                    raise SFTypeError(f"{self.name}: missing a required argument: {name!r}")
                arg = given[name]
                if isinstance(arg, SFArray):
                    v = b.emit("param", [], {"param": name, "kind": "ptr"}, PtrType(arg.dtype._fmt.bits), ())
                    b.ir.params.append(Param(name, "ptr", PtrType(arg.dtype._fmt.bits), vid=v.id))
                    call_args.append(Ptr(v, None, arg.dtype._fmt.bits))
                elif isinstance(arg, SFScalar):
                    v = b.emit("param", [], {"param": name, "kind": "sf"}, SFType(arg._fmt.bits), ())
                    b.ir.params.append(Param(name, "sf", SFType(arg._fmt.bits), vid=v.id))
                    call_args.append(Sym(v))
                else:
                    v = b.emit("param", [], {"param": name, "kind": "int"}, INT, ())
                    b.ir.params.append(Param(name, "int", INT, vid=v.id))
                    call_args.append(Sym(v))
            out = self.fn(*call_args)
            if out is not None:
                raise SFCompileError("a kernel must not return a value; write results with "
                                     "sf.store or pointer indexing")
        return b.ir

    def compiled(self, args, constexpr, block) -> KernelIR:
        key = (tuple(sorted(constexpr.items())), tuple(block),
               tuple((n, _arg_kind(a)) for n, a in args))
        with _cache_lock:
            ir = self.cache.get(key)
        if ir is None:
            ir = self.trace(args, constexpr, block)
            with _cache_lock:
                self.cache[key] = ir
        return ir


def _constexpr_cls():
    from ._lang import constexpr
    return constexpr


def _is_constexpr_annotation(ann) -> bool:
    """``BLOCK: sf.constexpr`` as an object, or as the string a module with
    ``from __future__ import annotations`` leaves behind."""
    if ann is _constexpr_cls():
        return True
    return isinstance(ann, str) and ann.split(".")[-1].strip("'\"") == "constexpr"


def _arg_kind(a):
    if isinstance(a, SFArray):
        return ("ptr", a.dtype._fmt.bits)
    if isinstance(a, SFScalar):
        return ("sf", a._fmt.bits)
    return ("int",)


def _norm_grid(spec, what: str) -> tuple:
    if isinstance(spec, int):
        spec = (spec,)
    spec = tuple(int(s) for s in spec)
    if not 1 <= len(spec) <= 3 or any(s < 1 for s in spec):
        raise SFLaunchError(f"{what} must be 1 to 3 positive ints, got {spec}")
    return spec + (1,) * (3 - len(spec))


class Launcher:
    def __init__(self, kernel: Kernel, spec) -> None:
        self.kernel = kernel
        if kernel.style == "thread":
            if not (isinstance(spec, tuple) and len(spec) == 2):
                raise SFLaunchError(f"{kernel.name}[grid, block](...): a thread kernel needs a grid and a block")
            self.grid, self.block = _norm_grid(spec[0], "grid"), _norm_grid(spec[1], "block")
        elif callable(spec):
            self.grid, self.block = spec, (1, 1, 1)               # autotune: grid depends on the config
        else:
            self.grid, self.block = _norm_grid(spec, "grid"), (1, 1, 1)

    def __call__(self, *args, **kwargs) -> LaunchResult:
        k = self.kernel
        bound = _bind(k, args, kwargs)
        constexpr = {n: v for n, v in bound if n in k.constexpr_names()}
        for n, v in constexpr.items():
            if isinstance(v, (SFArray, SFScalar, Sym, Ptr)) or isinstance(v, bool):
                raise SFCompileError(f"constexpr argument {n!r} must be a Python int (got {type(v).__name__})")
        if k.autotune_configs is not None:
            return _autotuned(k, self, bound, constexpr)
        if callable(self.grid):
            raise SFLaunchError("a callable grid needs an @sf.autotune kernel")
        return _launch(k, self.grid, self.block, bound, constexpr)


def _bind(k: Kernel, args, kwargs):
    # constexpr names that the autotune configs supply need not be passed at the launch
    tuned = set()
    if k.autotune_configs is not None:
        for cfg in k.autotune_configs:
            tuned.update(cfg)
    try:
        bound = k.sig.bind_partial(*args, **kwargs)
    except TypeError as e:
        raise SFTypeError(f"{k.name}: {e}") from None
    bound.apply_defaults()
    missing = [n for n, p in k.sig.parameters.items()
               if n not in bound.arguments and n not in tuned and p.default is p.empty]
    if missing:
        raise SFTypeError(f"{k.name}: missing a required argument: {missing[0]!r}")
    out = []
    for name, val in bound.arguments.items():
        if isinstance(val, _constexpr_cls()):
            val = val.value
        if name in tuned and name in kwargs:
            raise SFTypeError(f"{k.name}: {name!r} is chosen by sf.autotune, do not pass it")
        out.append((name, val))
    return out


def _resolve_device(bound):
    arrays = [v for _, v in bound if isinstance(v, SFArray)]
    for _, v in bound:
        if isinstance(v, (float, bool)):
            raise SFTypeError("kernel arguments must be device arrays, SF scalars or Python ints "
                              f"(got {type(v).__name__}); quantize floats with sf.sf16(x)")
        if isinstance(v, SFArray) and v.device == _device.HOST:
            raise SFTypeError("kernel arguments must live on the device: use x.to('atreides-sim') "
                              "or sf.asarray(..., device='atreides-sim')")
    devs = {a.device for a in arrays}
    if len(devs) > 1:
        raise SFTypeError(f"kernel arguments live on different devices: {sorted(devs)}")
    name = devs.pop() if devs else _config.get("kernel_device")
    dev = _device.get_device(name)
    if not isinstance(dev, KernelDevice):
        raise SFUnsupportedError(f"{dev.name} does not run user kernels; use 'atreides-sim'")
    return dev


def _launch(k: Kernel, grid, block, bound, constexpr, config_tag=None) -> LaunchResult:
    dev = _resolve_device(bound)
    ir = k.compiled(bound, constexpr, block)
    for p in ir.params:
        if p.kind in ("ptr", "sf") and p.type.bits not in dev.supported_bits:
            raise SFUnsupportedError(f"sf{p.type.bits} is not supported on {dev.name}")
    buffers = {n: v._codes for n, v in bound if isinstance(v, SFArray)}
    scalars = {n: (v._code if isinstance(v, SFScalar) else int(v)) for n, v in bound
               if not isinstance(v, SFArray) and n not in constexpr}
    dev.check_launch(ir, grid, block, buffers)                 # before launch (3.0 gate)
    t0 = time.perf_counter()
    counters = dev.launch(ir, grid, block, buffers, scalars)
    counters_wall = time.perf_counter() - t0
    status = SFStatus(saturated=counters.saturated, div0=counters.div0, invalid=counters.invalid,
                      exact=True, outputs_at_max=0)
    res = LaunchResult(k.name, grid, block, dev.name, status, counters,
                       _config.get("kernel_backend"), config=dict(constexpr))
    res.wall = counters_wall
    dev.last = res
    _apply_policy(status, k.name, dev)
    return res


def _apply_policy(status: SFStatus, name: str, dev) -> None:
    if status.invalid:
        _errstate.handle("invalid", f"kernel {name!r} on {dev.name}: invalid in {status.invalid} "
                                    "lane(s) (status.invalid)")
    if status.div0:
        _errstate.handle("divide", f"kernel {name!r} on {dev.name}: division by zero in "
                                   f"{status.div0} lane(s) (status.div0)")
    if status.saturated:
        _errstate.handle("overflow", f"kernel {name!r} on {dev.name}: saturation in "
                                     f"{status.saturated} lane(s) (status.saturated)")


# --- autotune (3.3) -------------------------------------------------------------------------------

def _autotuned(k: Kernel, launcher: Launcher, bound, constexpr) -> LaunchResult:
    keyvals = tuple((n, v if not isinstance(v, SFArray) else v.shape) for n, v in bound if n in k.autotune_key)
    best = k.tuned.get(keyvals)
    grid_fn = launcher.grid
    if best is None:
        results = []
        for cfg in k.autotune_configs:
            ce = dict(constexpr)
            ce.update(cfg)
            grid = grid_fn(ce) if callable(grid_fn) else grid_fn
            grid = _norm_grid(grid, "grid")
            # a tuning run must not change the data a later run sees: work on copies
            snap = [(n, v.copy() if isinstance(v, SFArray) else v) for n, v in bound]
            with _errstate.errstate(all="ignore"):
                r = _launch(k, grid, launcher.block, snap, ce, cfg)
            results.append((r.cycles, r.wall, cfg))
        results.sort(key=lambda t: (t[0], t[1]))
        best = results[0][2]
        k.tuned[keyvals] = best
    ce = dict(constexpr)
    ce.update(best)
    grid = grid_fn(ce) if callable(grid_fn) else grid_fn
    return _launch(k, _norm_grid(grid, "grid"), launcher.block, bound, ce, best)


def autotune(configs, key=()):
    """Pick the constexpr configuration with the fewest simulated cycles
    (ties broken by wall time) for each distinct value of ``key`` arguments.
    Tuning changes speed, never bits: every config is run through the same
    bit-exact backend.  With autotune the launch grid may be a function of
    the chosen config: ``k[lambda cfg: (sf.cdiv(n, cfg["BLOCK"]),)](...)``."""
    configs = [dict(c) for c in configs]
    if not configs:
        raise SFCompileError("autotune needs at least one config")

    def deco(k):
        if not isinstance(k, Kernel):
            raise SFTypeError("sf.autotune decorates an @sf.jit or @sf.kernel function")
        k.autotune_configs = configs
        k.autotune_key = tuple(key)
        return k
    return deco


def kernel(fn) -> Kernel:
    """CUDA-style thread kernel: ``k[grid, block](...)``."""
    return Kernel(fn, "thread")


def jit(fn) -> Kernel:
    """Triton-style block kernel: ``k[(grid,)](..., BLOCK=8)``."""
    return Kernel(fn, "block")
