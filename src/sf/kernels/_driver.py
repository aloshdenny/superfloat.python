"""The Atreides driver: from a compiled kernel file to what the chip sees.

The compiler (``superfloat-compile``, Kartik's) turns SF IR JSON into a kernel
file: the program words plus a manifest (parameter-block layout, launch rule,
memory rules).  This module does the Python side of a launch:

1. ``compile_kernel``: run the compiler as a subprocess, with a disk cache.
2. ``read_kernel``: check a kernel file and read its manifest.
3. ``plan_launch``: lay out the chip's data memory (parameter block, then the
   buffers), pack codes into sign-magnitude words, and split the threads into
   launches of at most ``max_blocks_per_launch`` blocks, each with its block
   offset in the parameter block.
4. ``read_back``: unpack the output buffers from the memory after ``done``.

A ``LaunchPlan`` is exactly what the chip-level testbench takes: the program
as a list of 16-bit words and the data memory as ``{address: word}``
(compiler_v4.8.1.md, section 6), plus the thread count and block offset of
each launch.  Running it (testbench, ISA model or board) is a separate step.

Kernel file format: "atreides-kernel" 0.3-draft (Kartik, 2026-11-22; 0.1 and
0.2 are still read, any other version is refused).  From his answers:
* the compiler is called as described in ``compile_kernel``;
* ints and int tables are 16-bit two's complement words (int tables:
  ``dtype "int16"``); sf values below sf16 are left-aligned in their word;
* ``lanes_per_program`` (BLOCK as written in Python) is what the user's grid
  is checked against, ``threads_per_program`` (the compiler's choice) is what
  launches are planned from; a launch holds whole programs;
* 2-D grids run as row launches with both indices written to the parameter
  block (his draft, to fix once gemm compiles);
* the compiler always writes ``pointers_compared_with_cmp``; when missing,
  the driver assumes true (no buffer crosses 0x8000).
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field

import numpy as np

from .. import _cache
from .._array import SFArray
from .._errors import SFCompileError, SFLaunchError, SFUnsupportedError
from .._format import FORMATS
from .._scalar import SFScalar
from ._ir import KernelIR

KERNEL_FORMAT = "atreides-kernel"
KERNEL_FORMAT_VERSIONS = ("0.1", "0.2", "0.3")   # accepted major.minor (any suffix, e.g. "-draft")
DEFAULT_TARGET = "atreides-v4.9.2"
PROGRAM_WORDS = 512                        # program memory, words
WORD = 1 << 16
_CMP_SPLIT = 0x8000                        # signed CMP on pointers: a buffer must not cross this

_CACHE_DIR: str | None = None              # None: sf's private per-user cache; tests may set it
_versions: dict[tuple, str] = {}
_lock = threading.Lock()


# --- reading a kernel file ----------------------------------------------------------------------

def _int(x) -> int:
    return int(x, 0) if isinstance(x, str) else int(x)


def read_kernel(k: dict) -> dict:
    """Check a kernel file and return it.  Raises ``SFCompileError`` on a bad one."""
    def bad(msg):
        raise SFCompileError(f"kernel file {k.get('kernel', '?')!r}: {msg}")

    if k.get("format") != KERNEL_FORMAT:
        bad(f"format is {k.get('format')!r}, not {KERNEL_FORMAT!r}")
    if ".".join(str(k.get("format_version", "")).split("-")[0].split(".")[:2]) not in KERNEL_FORMAT_VERSIONS:
        bad(f"format_version {k.get('format_version')!r} is not one this driver reads {KERNEL_FORMAT_VERSIONS}")
    words = k.get("program", {}).get("words")
    if not isinstance(words, list) or not words:
        bad("no program words")
    if len(words) > PROGRAM_WORDS:
        bad(f"{len(words)} program words; the chip holds at most {PROGRAM_WORDS}")
    if any(not isinstance(w, int) or not 0 <= w < WORD for w in words):
        bad("every program word must be an int from 0 to 0xFFFF")
    slots = k.get("params", {}).get("slots", [])
    offsets = [s["offset"] for s in slots]
    if len(set(offsets)) != len(offsets):
        bad("two parameter slots share an offset")
    if sum(s["kind"] == "block_offset" for s in slots) > 1:
        bad("more than one block_offset slot")
    reserved = k.get("memory", {}).get("reserved_words")
    if reserved is not None and slots:
        base = _int(k.get("params", {}).get("base_address", 0))
        last = base + max(offsets)
        if _int(reserved[0]) > base or _int(reserved[1]) < last:
            bad(f"reserved_words {reserved} does not cover the parameter slots (last at word {last})")
    launch = k.get("launch", {})
    axes = int(launch.get("grid_axes", 1))
    for key in ("threads_per_block", "max_blocks_per_launch") + (("total_threads",) if axes == 1 else ("axes",)):
        if key not in launch:
            bad(f"launch has no {key!r}")
    names = {s["name"] for s in slots} | {p["name"] for p in k.get("params", {}).get("not_in_block", [])}
    if axes == 1:
        total = launch["total_threads"]
        if isinstance(total, str) and total not in names:
            bad(f"launch total_threads names {total!r}, which is not one of the kernel's arguments")
    elif axes == 2:
        if len(launch["axes"]) != 2:
            bad("a 2-D launch needs 2 entries in launch.axes")
        for a in launch["axes"]:
            if a.get("param_slot") not in offsets:
                bad(f"launch axis writes slot {a.get('param_slot')!r}, which is not a parameter slot")
    else:
        bad(f"grid_axes {axes}: the driver runs 1-D and 2-D grids")
    if launch["threads_per_block"] * launch["max_blocks_per_launch"] > 255:
        bad("one launch can run at most 255 threads (an 8-bit count)")
    lanes = launch.get("lanes_per_program")
    tpp = launch.get("threads_per_program", lanes)
    if lanes is not None and (int(lanes) < 1 or int(tpp) < 1 or int(lanes) % int(tpp)):
        bad(f"lanes_per_program {lanes} must be a whole number of threads_per_program {tpp}")
    if tpp is not None and int(tpp) > launch["threads_per_block"] * launch["max_blocks_per_launch"]:
        bad(f"threads_per_program {tpp} does not fit in one launch")
    return k


# --- the plan of a launch -----------------------------------------------------------------------

@dataclass
class Buffer:
    name: str
    base: int                              # first word address
    length: int                            # words
    dtype: str                             # "sf16", ..., or "int16" for a read-only int table
    access: str                            # "read", "write" or "readwrite"


@dataclass
class Launch:
    threads: int                           # 1..255, the count driven on bus_in[7:0]
    block_offset: int                      # index of this launch's first block in the whole grid
    writes: dict = field(default_factory=dict)    # {address: word} to write before this launch


@dataclass
class LaunchPlan:
    kernel: str
    program: list                          # 16-bit program words, loaded at address 0
    memory: dict                           # {address: word}: the data memory before the first launch
    buffers: dict                          # {param name: Buffer}
    launches: list                         # [Launch], run in order
    target: str | None = None
    grid: tuple = ()                       # programs per axis the chip covers; the user's grid must match
    lanes: int = 0                         # lanes per program (BLOCK as written in Python)


@dataclass
class SFCodes:
    """An sf buffer as raw codes: what the runtime hands a driver."""
    codes: np.ndarray
    bits: int


@dataclass
class SFCode:
    """An sf scalar as a raw code."""
    code: int
    bits: int


def _norm_arg(v):
    """SFArray / SFScalar to raw codes; everything else unchanged."""
    if isinstance(v, SFArray):
        return SFCodes(np.asarray(v.codes, dtype=np.int64), v.dtype._fmt.bits)
    if isinstance(v, SFScalar):
        return SFCode(int(v.code), v._fmt.bits)
    return v


def _sf_word(code: int, bits: int) -> int:
    """Signed code -> the 16-bit word in memory: sign-magnitude, left-aligned."""
    return FORMATS[bits].to_bits(int(code)) << (16 - bits)


def _sf_code(word: int, bits: int) -> int:
    return FORMATS[bits].from_bits(int(word) >> (16 - bits))


def _int_word(v: int, what: str) -> int:
    """An int as the chip's 16-bit integer (two's complement)."""
    v = int(v)
    if not -(1 << 15) <= v < (1 << 15):
        raise SFLaunchError(f"{what} = {v} does not fit the chip's 16-bit integers")
    return v & 0xFFFF


def _dtype_bits(dtype: str) -> int:
    if not dtype.startswith("sf"):
        raise SFLaunchError(f"unknown data type {dtype!r} in the kernel file")
    return int(dtype[2:])


def plan_launch(kernel: dict, args: dict, constexpr: dict | None = None,
                default_lanes: int | None = None) -> LaunchPlan:
    """Lay out memory and split launches for one call of ``kernel``.

    ``args`` maps each parameter name to its value: an ``SFArray`` (or
    ``SFCodes``) for an sf pointer, a NumPy int array for an int table
    (``ptr<int>``), an ``SFScalar`` (or ``SFCode``) for an sf scalar, an int
    for an int.

    ``constexpr`` (the kernel's constexpr values) is for size expressions such
    as ``cdiv(M, BM)`` in a 2-D launch.  ``default_lanes`` is the lanes per
    program when the file does not say (the block size for thread kernels);
    otherwise ``threads_per_block``.
    """
    args = {n: _norm_arg(v) for n, v in args.items()}
    k = read_kernel(kernel)
    params, launch, mem_rules = k["params"], k["launch"], k.get("memory", {})
    base = _int(params.get("base_address", 0))
    slots = sorted(params.get("slots", []), key=lambda s: s["offset"])
    below = _int(mem_rules.get("buffers_below", 0xFFC0))
    no_cross = bool(mem_rules.get("pointers_compared_with_cmp", True))   # our choice: strict when missing
    last_slot = base + (slots[-1]["offset"] if slots else -1)        # slots are sorted by offset
    reserved = mem_rules.get("reserved_words", [base, last_slot])
    nxt = max(_int(reserved[1]), last_slot) + 1                       # buffers start after both

    known = {s["name"] for s in slots} | {p["name"] for p in params.get("not_in_block", [])}
    missing = sorted(n for n in known if n not in args and not n.startswith("__"))
    if missing:
        raise SFLaunchError(f"{k['kernel']}: missing arguments {missing}")

    memory: dict[int, int] = {}
    buffers: dict[str, Buffer] = {}
    offset_slot = None
    axis_slots = {a["param_slot"] for a in launch.get("axes", [])} if launch.get("grid_axes", 1) == 2 else set()
    for s in slots:
        addr = base + s["offset"]
        kind, name = s["kind"], s["name"]
        if kind == "block_offset":
            offset_slot = addr
            continue
        if s["offset"] in axis_slots:                   # written per launch by the 2-D launch rule
            continue
        v = args[name]
        if kind == "ptr":
            if isinstance(v, SFCodes):
                data = np.asarray(v.codes, dtype=np.int64).reshape(-1)
                bits = v.bits
                dtype, encoding = f"sf{bits}", "sign_magnitude"
                words = [_sf_word(c, bits) for c in data]
            else:
                data = np.asarray(v).reshape(-1)
                if not np.issubdtype(data.dtype, np.integer):
                    raise SFLaunchError(f"{name}: a pointer argument is an SFArray or an int array")
                dtype, encoding = "int16", "twos_complement"
                words = [_int_word(x, f"{name}[{i}]") for i, x in enumerate(data)]
            if s.get("dtype") not in (None, dtype):
                raise SFLaunchError(f"{name}: the kernel wants {s['dtype']}, got {dtype}")
            if s.get("encoding") not in (None, encoding):
                raise SFLaunchError(f"{name}: the kernel file says {s['encoding']} for {dtype}; "
                                    f"this driver writes {encoding}")
            n = len(words)
            if no_cross and nxt < _CMP_SPLIT < nxt + n:
                nxt = _CMP_SPLIT                       # signed pointer compares: stay in one half
            if nxt + n > below:
                raise SFLaunchError(f"{k['kernel']}: buffer {name} ({n} words) does not fit below "
                                    f"{below:#06x}; the data memory is full at {nxt:#06x}")
            access = s.get("access", "read")
            if access != "write":
                memory.update(zip(range(nxt, nxt + n), words))
            buffers[name] = Buffer(name, nxt, n, dtype, access)
            memory[addr] = nxt
            nxt += n
        elif kind == "sf":
            if not isinstance(v, SFCode):
                raise SFLaunchError(f"{name}: an sf parameter takes an SFScalar")
            memory[addr] = _sf_word(v.code, v.bits)
        elif kind == "int":
            memory[addr] = _int_word(v, name)
        else:
            raise SFLaunchError(f"{k['kernel']}: unknown parameter kind {kind!r} in slot {s['offset']}")

    tpb, max_blocks = launch["threads_per_block"], launch["max_blocks_per_launch"]
    lanes = int(launch.get("lanes_per_program", default_lanes or tpb))
    tpp = int(launch.get("threads_per_program", lanes))     # the compiler's choice; lanes is the user's BLOCK
    per_launch = (tpb * max_blocks) // tpp                  # whole programs in one launch
    if per_launch < 1 or lanes % tpp:
        raise SFLaunchError(f"{k['kernel']}: {lanes} lanes on {tpp} threads per program does not fit a launch")
    launches = []
    if int(launch.get("grid_axes", 1)) == 1:
        size = launch["total_threads"]                      # lanes in all (axpy: n)
        size = int(args[size]) if isinstance(size, str) else int(size)
        if size < 1:
            raise SFLaunchError(f"{k['kernel']}: {size} threads; a launch needs at least 1")
        grid = (-(-size // lanes),)
        total = -(-size * tpp // lanes)                     # chip threads: the tail program may be short
        if offset_slot is None and total > per_launch * tpp:
            raise SFLaunchError(f"{k['kernel']}: {total} threads need several launches, but the kernel "
                                f"has no block_offset slot")
        for start in range(0, total, per_launch * tpp):
            first_block = start // tpb
            writes = {offset_slot: _int_word(first_block, "block offset")} if offset_slot is not None else {}
            launches.append(Launch(min(per_launch * tpp, total - start), first_block, writes))
    else:
        # 2-D (Kartik's draft, to fix once gemm compiles): the chip has only a 1-D block index, so run
        # row launches: at most per_launch programs along axis 0, one axis-1 index each, and write the
        # first axis-0 program index and the axis-1 index into their parameter slots
        env = {**{n: v for n, v in args.items() if isinstance(v, (int, np.integer))}, **(constexpr or {})}
        ax0, ax1 = launch["axes"]
        grid = (_size(ax0["size"], env, k["kernel"]), _size(ax1["size"], env, k["kernel"]))
        a0, a1 = base + ax0["param_slot"], base + ax1["param_slot"]
        for y in range(grid[1]):
            for x0 in range(0, grid[0], per_launch):
                count = min(per_launch, grid[0] - x0)
                launches.append(Launch(count * tpp, x0, {a0: _int_word(x0, "axis 0 index"),
                                                         a1: _int_word(y, "axis 1 index")}))
    return LaunchPlan(k["kernel"], list(k["program"]["words"]), memory, buffers, launches,
                      k.get("target", {}).get("isa"), grid, lanes)


_SIZE_FUNCS = {"cdiv": lambda a, b: -(-a // b)}


def _size(expr, env: dict, kernel: str) -> int:
    """A launch size: an int, an argument name, or an expression such as ``cdiv(M, BM)``
    (ints, names, + - * //, cdiv).  Parsed, never eval'd."""
    import ast
    import operator as op
    if isinstance(expr, int):
        return expr
    bins = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.FloorDiv: op.floordiv}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, int):
            return n.value
        if isinstance(n, ast.Name) and n.id in env:
            return int(env[n.id])
        if isinstance(n, ast.BinOp) and type(n.op) in bins:
            return bins[type(n.op)](ev(n.left), ev(n.right))
        if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _SIZE_FUNCS
                and not n.keywords):
            return _SIZE_FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise SFLaunchError(f"{kernel}: cannot work out the launch size {expr!r} "
                            f"(unknown name or form; known names: {sorted(env)})")
    try:
        v = ev(ast.parse(str(expr), mode="eval").body)
    except SyntaxError:
        raise SFLaunchError(f"{kernel}: launch size {expr!r} is not an expression") from None
    if v < 1:
        raise SFLaunchError(f"{kernel}: launch size {expr!r} is {v}; it must be at least 1")
    return v


def read_back(plan: LaunchPlan, memory: dict) -> dict:
    """{name: codes} for every buffer the kernel writes, from the memory after ``done``."""
    out = {}
    for b in plan.buffers.values():
        if b.access == "read":
            continue
        words = [memory.get(a, 0) for a in range(b.base, b.base + b.length)]
        if b.dtype == "int16":
            out[b.name] = np.array([w - WORD if w & 0x8000 else w for w in words], dtype=np.int64)
        else:
            bits = _dtype_bits(b.dtype)
            out[b.name] = np.array([_sf_code(w, bits) for w in words], dtype=np.int64)
    return out


# --- calling the compiler -----------------------------------------------------------------------

def _command(compiler) -> list:
    if compiler is None:
        compiler = os.environ.get("SF_COMPILER", "superfloat-compile")
    cmd = shlex.split(compiler) if isinstance(compiler, str) else list(compiler)
    if not cmd or (shutil.which(cmd[0]) is None and not os.path.exists(cmd[0])):
        raise SFUnsupportedError(f"the Atreides compiler {cmd[0] if cmd else compiler!r} is not installed "
                                 f"(set SF_COMPILER to its command)")
    return cmd


def compiler_version(compiler=None) -> str:
    """``superfloat-compile --version``, asked once per command."""
    cmd = _command(compiler)
    key = tuple(cmd)
    with _lock:
        if key in _versions:
            return _versions[key]
    proc = subprocess.run([*cmd, "--version"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SFCompileError(f"{cmd[0]} --version failed:\n{proc.stderr}")
    with _lock:
        _versions[key] = proc.stdout.strip()
    return _versions[key]


def cache_key(ir_json: str, version: str, argv) -> str:
    """The SF IR JSON, the compiler's version line and the exact argument list passed to it
    (Kartik: hash the arguments as given, so no option can be left out of the key)."""
    parts = [ir_json, version, json.dumps(list(argv))]               # order kept: "-x 1 -y 2" != "-x 2 -y 1"
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def is_dirty(version: str) -> bool:
    """``superfloat-compile <semver> (<git hash>-dirty)``: a build with uncommitted changes.
    Two different dirty builds can print the same line, so their output is never cached."""
    return version.rstrip().endswith("-dirty)")


def compile_kernel(ir: KernelIR | str, target: str = DEFAULT_TARGET, options=(), compiler=None) -> dict:
    """The kernel file for ``ir``, from the disk cache or a run of the compiler.

    The interface (Kartik, 2026-11-22): ``superfloat-compile --target=<target>
    [-O0|-O1|-O2]`` reads SF IR JSON on stdin, writes the kernel file JSON on
    stdout, and on failure prints ``file:line: error: ...`` on stderr and
    exits non-zero.  The file must be for the target asked for.  Debug options
    (``--emit=...``) are refused: their output is not a kernel file.  A dirty
    compiler build (version ending in ``-dirty``) is never cached.
    """
    ir_json = ir if isinstance(ir, str) else ir.to_json()
    options = list(options)
    debug = [o for o in options if str(o).startswith("--emit")]
    if debug:
        raise SFUnsupportedError(f"compiler debug options {debug} are not for the driver; run "
                                 f"superfloat-compile by hand to see that output")
    argv = [f"--target={target}", *options]
    cmd = _command(compiler)
    version = compiler_version(cmd)
    cacheable = not is_dirty(version)
    key = cache_key(ir_json, version, argv)
    cache = _cache.private_dir("kernels-atreides") if _CACHE_DIR is None else _cache.ensure_private(_CACHE_DIR)
    path = os.path.join(cache, f"{key[:32]}.json")
    if cacheable and os.path.exists(path):
        _cache.check_file(path)
        with open(path) as f:
            return read_kernel(json.load(f))
    proc = subprocess.run([*cmd, *argv], input=ir_json, capture_output=True, text=True)
    if proc.returncode != 0:
        name = json.loads(ir_json).get("kernel", "?")
        raise SFCompileError(f"{cmd[0]} failed on kernel {name!r} (exit {proc.returncode}):\n{proc.stderr}")
    try:
        k = json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise SFCompileError(f"{cmd[0]} wrote something that is not JSON: {e}") from None
    read_kernel(k)
    got = (k.get("target") or {}).get("isa")
    if got != target:
        raise SFCompileError(f"{cmd[0]} was asked for target {target!r} but wrote a kernel for {got!r}")
    if not cacheable:
        return k
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix=f".{key[:12]}-", dir=cache)
    try:                                            # publish atomically: no half-written cache file
        with os.fdopen(fd, "w") as f:
            json.dump(k, f, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return k


# --- the driver HardwareAtreides calls ---------------------------------------------------------

def check_grid(plan: LaunchPlan, grid) -> int:
    """Refuse a launch whose grid is not the grid of programs the chip will run.

    The chip's thread count comes from the kernel file (for axpy, ``n``
    lanes in programs of ``lanes_per_program``), not from the grid the user
    wrote, so any other grid would give other results than atreides-sim.
    Returns the number of programs.
    """
    def trim(g):
        g = tuple(int(x) for x in g)
        while len(g) > 1 and g[-1] == 1:
            g = g[:-1]
        return g
    if trim(grid) != trim(plan.grid):
        raise SFLaunchError(f"{plan.kernel}: the grid {tuple(grid)} does not match the chip's grid of programs "
                            f"{plan.grid} ({plan.lanes} lanes per program); atreides-sim would compute "
                            f"something else")
    return int(np.prod(plan.grid))


class AtreidesDriver:
    """The object ``HardwareAtreides.attach`` takes: compile, plan, run, read back.

    ``runner(plan) -> memory`` executes a ``LaunchPlan`` and returns the data
    memory after the last launch (``{address: word}``).  It is the one part
    that changes with what runs the program: a toy model in the tests, then
    Kartik's ISA model, Alosh's testbench, or the board.

    The launch grid is not sent to the compiler: the kernel file's launch rule
    (``total_threads``) sets the thread count.  The chip has no event flags,
    so the returned counters leave every event unknown (``None``);
    ``sf.config(status="replay")`` recomputes them on the host and checks the
    output bits.
    """

    def __init__(self, runner, target: str = DEFAULT_TARGET, options=(), compiler=None) -> None:
        self.runner = runner
        self.target = target
        self.options = tuple(options)
        self.compiler = compiler
        self.last_plan: LaunchPlan | None = None

    def launch(self, ir, grid, block, buffers, scalars):
        from ._interp import Counters
        kernel = compile_kernel(ir, self.target, self.options, self.compiler)
        args = {}
        for p in ir.params:
            if p.kind == "constexpr":
                continue
            if p.kind == "ptr":
                b = buffers[p.name]
                args[p.name] = SFCodes(b, p.type.bits) if p.type.bits else np.asarray(b, dtype=np.int64)
            elif p.kind == "sf":
                args[p.name] = SFCode(scalars[p.name], p.type.bits)
            else:
                args[p.name] = scalars[p.name]
        lanes = int(np.prod(block)) if ir.style == "thread" else None
        plan = plan_launch(kernel, args, ir.constexpr, lanes)
        programs = check_grid(plan, grid)
        self.last_plan = plan
        memory = self.runner(plan)
        for name, codes in read_back(plan, memory).items():
            buffers[name][...] = codes.reshape(buffers[name].shape)
        # programs as the simulator counts them (one per grid point); chip launches: len(plan.launches)
        return Counters(programs=programs, saturated=None, underflow=None,
                        div0=None, invalid=None)
