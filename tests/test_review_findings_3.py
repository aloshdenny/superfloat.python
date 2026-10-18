"""Regressions from the independent Stage 3 review (2026-10-18).

1. exact comparison against tiny / huge host literals (int64 overflow in cmp_lit)
2. writes through a transposed or strided device view silently lost
3. the C backend kept running after a device fault
4. a kernel parameter named like an emitter local broke the generated C
5. the shared-memory limit was enforced in elements, not bytes
plus the two code-inspection risks: the C build cache key, static shared scratch.
"""
from __future__ import annotations

import threading
from fractions import Fraction

import numpy as np
import pytest

import sf
from sf import _reference as ref
from sf.kernels import _cgen

DEV = "atreides-sim"
BACKENDS = ["interp"] + (["c"] if _cgen.compiler() else [])


@pytest.fixture
def dev():
    return sf.device(DEV)


@pytest.fixture(params=BACKENDS)
def backend(request):
    with sf.config(kernel_backend=request.param):
        yield request.param


# --- 1. literal comparisons are exact for every host number ---------------------------------------

LITERALS = [0.0001, 1e-20, -1e-20, 1e20, -1e20, 1e300, -1e300, 0.1, -0.1, 0.5, 1, -1, 2, 0,
            Fraction(1, 3), Fraction(-32767, 32768), Fraction(32767, 32768), 32767 / 32768,
            0.999, -0.999, 1e-5, 3, 1.5, -2.5, 0.25, np.float32(0.3), np.int64(1)]
OPS = {"lt": lambda a, b: a < b, "le": lambda a, b: a <= b, "gt": lambda a, b: a > b,
       "ge": lambda a, b: a >= b, "eq": lambda a, b: a == b, "ne": lambda a, b: a != b}


@pytest.mark.parametrize("opname", sorted(OPS))
def test_literal_comparisons_are_exact(dev, backend, opname):
    cmp = OPS[opname]
    codes = np.array([-32767, -32766, -3277, -1, 0, 1, 3, 3277, 16384, 32766, 32767])
    x = dev.upload(sf.from_codes(codes, sf.sf16))
    for lit in LITERALS:
        @sf.jit
        def k(x_ptr, out_ptr, BLOCK: sf.constexpr):
            offs = sf.arange(0, BLOCK)
            v = sf.load(x_ptr + offs)
            m = cmp(v, lit)
            sf.store(out_ptr + offs, sf.where(m, sf.sf16(0.5), sf.sf16(-0.5)))
        out = dev.alloc(len(codes))
        k[(1,)](x, out, BLOCK=len(codes))
        got = out.codes > 0
        exact = [cmp(Fraction(int(c), 32768), ref.to_exact(lit)) for c in codes]
        assert got.tolist() == exact, (opname, lit)
        # and the same answer the host array gives
        assert got.tolist() == cmp(x.to("host"), lit).tolist(), (opname, lit)


def test_literal_comparison_with_the_literal_on_the_left(dev, backend):
    @sf.jit
    def k(x_ptr, out_ptr, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        v = sf.load(x_ptr + offs)
        sf.store(out_ptr + offs, sf.where((1e-20 < v) & (0.75 >= v), sf.sf16(0.5), sf.sf16(-0.5)))

    x = dev.upload(np.array([0.5, 0.0, 0.75, 0.8, -0.1]))
    out = dev.alloc(5)
    k[(1,)](x, out, BLOCK=5)
    assert (out.codes > 0).tolist() == [True, False, True, False, False]


def test_nan_and_inf_literals(dev, backend):
    @sf.jit
    def k(x_ptr, out_ptr, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        v = sf.load(x_ptr + offs)
        a = v < float("inf")
        b = v > float("-inf")
        c = v == float("nan")
        d = v != float("nan")
        sf.store(out_ptr + offs, sf.where(a & b & ~c & d, sf.sf16(0.5), sf.sf16(-0.5)))

    x = dev.upload(np.array([0.5, -0.5, 0.0]))
    out = dev.alloc(3)
    k[(1,)](x, out, BLOCK=3)
    assert (out.codes > 0).all()


@pytest.mark.parametrize("bits", [4, 8, 12, 16])
def test_literal_thresholds_follow_the_format(bits):
    """The lowered threshold is the exact integer the Fraction rule gives, for every width
    the interpreter serves."""
    from sf.kernels._sym import _compare_literal, Sym
    from sf.kernels._ir import Builder, trace_context, SFType
    fmt = sf.dtype(f"sf{bits}")._fmt
    b = Builder("t", "block")
    with trace_context(b):
        v = Sym(b.emit("const", [], {"value": 0, "kind": "sf"}, SFType(bits), ()))
        for lit in LITERALS:
            for op in OPS:
                r = _compare_literal(op, v, lit, flipped=False)
                last = b.ir.ops[-1]
                q = ref.to_exact(lit) * fmt.scale
                if last.name == "cmp_lit" and last.result is r.v:
                    t = last.attrs["value"]
                    assert -fmt.max_code <= t <= fmt.max_code
                    for k in range(-fmt.max_code, fmt.max_code + 1):
                        assert OPS[op](k, t) == OPS[op](Fraction(k), q), (bits, lit, op, k)
                else:                                   # folded to a constant
                    c = last.attrs["value"]
                    for k in (-fmt.max_code, 0, fmt.max_code):
                        assert OPS[op](Fraction(k), q) == c, (bits, lit, op, k)


# --- 2. non-contiguous views are refused before launch ----------------------------------------------

@sf.jit
def fill(x_ptr, n, BLOCK: sf.constexpr):
    offs = sf.arange(0, BLOCK)
    sf.store(x_ptr + offs, sf.full((BLOCK,), 0.5), mask=offs < n)


@pytest.mark.parametrize("view", ["T", "strided", "transpose3"])
def test_non_contiguous_pointer_arguments_are_rejected(dev, backend, view):
    base = dev.alloc((4, 6)) if view != "transpose3" else dev.alloc((2, 3, 4))
    bad = {"T": lambda b: b.T, "strided": lambda b: b[:, ::2], "transpose3": lambda b: b.transpose(2, 0, 1)}[view](base)
    n0 = dev.kernel_launches
    with pytest.raises(sf.SFLaunchError, match="non-contiguous"):
        fill[(1,)](bad, bad.size, BLOCK=32)
    assert dev.kernel_launches == n0 and not base.any()           # nothing launched, nothing written
    # the documented way: a contiguous copy, then copy back
    c = bad.copy()
    fill[(1,)](c, c.size, BLOCK=32)
    assert c.numpy().tolist() == np.full(bad.shape, 0.5).tolist()


def test_contiguous_slices_and_reads_still_work(dev, backend):
    base = dev.alloc((4, 6))
    head = base[:2]                                                 # contiguous: a plain view
    fill[(1,)](head, head.size, BLOCK=16)
    assert base.numpy()[:2].tolist() == np.full((2, 6), 0.5).tolist() and not base[2:].any()


# --- 3. a fault stops the launch; the faulting op writes nothing ----------------------------------

@sf.jit
def two_stores(x_ptr, BLOCK: sf.constexpr):
    offs = sf.arange(0, BLOCK)
    sf.store(x_ptr + offs + 100, sf.full((BLOCK,), 0.5))             # out of bounds
    sf.store(x_ptr + offs, sf.full((BLOCK,), 0.25))                  # must never run


@sf.jit
def partial(x_ptr, BLOCK: sf.constexpr):
    offs = sf.arange(0, BLOCK)
    sf.store(x_ptr + offs, sf.full((BLOCK,), 0.125))                 # before the fault: visible
    sf.store(x_ptr + offs * 3, sf.full((BLOCK,), 0.5))               # some lanes out of bounds


@sf.kernel
def shared_fault(x, out, SH: sf.constexpr):
    i = sf.thread_idx(0)
    tile = sf.shared((SH,))
    tile[i] = x[i]                                                   # i >= SH faults
    sf.barrier()
    out[i] = tile[0]


def test_fault_stops_the_kernel(dev, backend):
    x = dev.alloc(8)
    with pytest.raises(sf.SFDeviceError, match="store out of bounds"):
        two_stores[(1,)](x, BLOCK=8)
    assert not x.any()
    with pytest.raises(sf.SFDeviceError, match="store out of bounds"):
        partial[(1,)](x, BLOCK=8)
    assert x.numpy().tolist() == [0.125] * 8                        # earlier op visible, faulting op wrote nothing
    x = dev.alloc(16)
    out = dev.alloc(16)
    with pytest.raises(sf.SFDeviceError, match="shared memory store out of bounds"):
        shared_fault[(1,), (16,)](x, out, SH=8)
    assert not out.any()


def test_fault_in_a_multi_program_grid(dev, backend):
    @sf.jit
    def k(x_ptr, n, BLOCK: sf.constexpr):
        offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
        sf.store(x_ptr + offs, sf.full((BLOCK,), 0.5))               # the last program runs off the end

    x = dev.alloc(10)
    with pytest.raises(sf.SFDeviceError, match="out of bounds"):
        k[(3,)](x, 10, BLOCK=4)
    assert x.numpy().tolist() == [0.5] * 8 + [0.0] * 2             # programs 0 and 1 completed, 2 wrote nothing


def test_fault_leaves_both_backends_in_the_same_state(dev):
    if not _cgen.compiler():
        pytest.skip("no C compiler")
    states = {}
    for be in ("interp", "c"):
        x = dev.alloc(8)
        with sf.config(kernel_backend=be), pytest.raises(sf.SFDeviceError):
            partial[(1,)](x, BLOCK=8)
        states[be] = x.codes.tolist()
    assert states["interp"] == states["c"]


# --- 4. parameter names never reach the generated C ----------------------------------------------

@pytest.mark.skipif(not _cgen.compiler(), reason="no C compiler")
@pytest.mark.parametrize("names", [("o", "e", "i"), ("C", "gx", "px"), ("E", "acc", "pe"),
                                   ("tmp", "lv", "d"), ("x", "y", "k"), ("arg0", "arg1", "arg2"),
                                   ("shared0", "sf_kernel", "memset")])
def test_parameter_names_are_mangled(dev, names):
    src = f'''
@sf.jit
def kern({names[0]}, {names[1]}, {names[2]}, BLOCK: sf.constexpr):
    offs = sf.arange(0, BLOCK)
    v = sf.load({names[0]} + offs)
    sf.store({names[1]} + offs, v + {names[2]})
'''
    ns = {"sf": sf}
    exec(src, ns)
    kern = ns["kern"]
    x = dev.upload(np.full(4, 0.25))
    outs = {}
    for be in ("interp", "c"):
        out = dev.alloc(4)
        with sf.config(kernel_backend=be):
            kern[(1,)](x, out, sf.sf16(0.5), BLOCK=4)
        outs[be] = out.codes.tolist()
    assert outs["interp"] == outs["c"] == [24576] * 4
    ir = next(iter(kern.cache.values()))
    c = _cgen.emit_c(ir)
    sig = c.split("void sf_kernel(")[1].split(")")[0]
    for n in names:
        assert n not in sig.replace("int64_t", "").split("arg")[0] or n.startswith("arg")
    assert "arg0" in sig and "arg1" in sig and "arg2" in sig


# --- 5. the shared-memory limit is in bytes --------------------------------------------------------

def test_shared_limit_is_bytes(dev, backend):
    @sf.kernel
    def k(x, out, SH: sf.constexpr):
        i = sf.thread_idx(0)
        tile = sf.shared((SH,))
        tile[i] = x[i]
        sf.barrier()
        out[i] = tile[i]

    x = dev.upload(np.full(16, 0.25))
    out = dev.alloc(16)
    n0 = dev.kernel_launches
    with pytest.raises(sf.SFLaunchError, match="40000 bytes of shared memory"):
        k[(1,), (16,)](x, out, SH=20000)                            # 40 KiB of sf16: over 32 KiB
    assert dev.kernel_launches == n0
    k[(1,), (16,)](x, out, SH=16384)                                # exactly 32 KiB: allowed
    assert out.numpy().tolist() == [0.25] * 16
    assert dev.limits.max_shared == 32768


# --- risks: build cache key, shared scratch ------------------------------------------------------------

@pytest.mark.skipif(not _cgen.compiler(), reason="no C compiler")
def test_build_cache_key_covers_header_compiler_and_flags(monkeypatch):
    import hashlib

    @sf.jit
    def k(x_ptr, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        sf.store(x_ptr + offs, sf.load(x_ptr + offs))

    dev = sf.device(DEV)
    x = dev.alloc(4)
    with sf.config(kernel_backend="c"):
        k[(1,)](x, BLOCK=4)
    ir = next(iter(k.cache.values()))
    src = _cgen.emit_c(ir)
    cc = _cgen.compiler()

    def key():
        return hashlib.sha256("\0".join([src, _cgen._header_text(), cc, _cgen._toolchain(cc),
                                         " ".join(_cgen._FLAGS)]).encode()).hexdigest()[:24]

    k0 = key()
    monkeypatch.setattr(_cgen, "_header_text", lambda: "/* revised builtins */")
    assert key() != k0
    monkeypatch.undo()
    monkeypatch.setattr(_cgen, "_FLAGS", ["-O0", "-shared", "-fPIC", "-std=c99"])
    assert key() != k0
    monkeypatch.undo()
    monkeypatch.setitem(_cgen._toolchains, cc, "another compiler 1.0")
    assert key() != k0
    monkeypatch.undo()
    assert key() == k0
    # and the key really is what build() uses: the same source, header and flags hit the cache
    with _cgen._lock:
        assert k0 in _cgen._loaded


@pytest.mark.skipif(not _cgen.compiler(), reason="no C compiler")
def test_concurrent_c_launches_do_not_share_shared_memory(dev):
    @sf.kernel
    def k(x, out, n, BLOCK: sf.constexpr):
        i = sf.thread_idx(0)
        tile = sf.shared((BLOCK,))
        tile[i] = x[i]
        sf.barrier()
        acc = sf.sf16(0)
        with sf.predicate(i == 0):
            for j in range(BLOCK):
                acc = acc + tile[j]
            out[i] = acc

    rng = np.random.default_rng(0)
    inputs = [rng.uniform(0.001, 0.01, 64) for _ in range(8)]      # no underflow on the cast
    refs = [sf.array(v, sf.sf16).sum().code for v in inputs]
    xs = [dev.upload(v) for v in inputs]
    outs = [dev.alloc(1) for _ in inputs]
    with sf.config(kernel_backend="c"):
        k[(1,), (64,)](xs[0], outs[0], 64, BLOCK=64)                # compile once
    assert "static" not in _cgen.emit_c(next(iter(k.cache.values())))
    errors = []

    def worker(t):
        try:
            with sf.config(kernel_backend="c"):
                for _ in range(200):
                    k[(1,), (64,)](xs[t], outs[t], 64, BLOCK=64)
                    assert int(outs[t].codes[0]) == refs[t]
        except Exception as e:                                       # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(8)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors
