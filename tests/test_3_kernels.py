"""Stage 3: GPU programming.

3.0 runtime: copies round-trip bit for bit; limits raise SFLaunchError before launch.
3.1 thread kernels: hand-written axpy, reduction and matmul match the Stage 1.1 built-ins.
3.2 block kernels: the 3.1 kernels rewritten as block kernels give identical bits.
3.3 performance: tuning changes speed, never bits.
Section 11: every backend (interpreter, C) passes the same bit-exact suite.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

import sf
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


def rand(shape, lo, hi, seed):
    return np.random.default_rng(seed).uniform(lo, hi, shape)


# --- 3.0 runtime --------------------------------------------------------------------------------

def test_targets_and_copies(dev):
    assert "atreides-sim" in sf.devices() and "atreides-hw" in sf.devices()
    assert isinstance(dev, sf.SimulatorAtreides) and dev.sticky_flags and dev.supported_bits == (16,)
    codes = np.random.default_rng(0).integers(-32767, 32768, (7, 9))
    host = sf.from_codes(codes, sf.sf16)
    d = dev.upload(host)
    assert d.device == DEV and np.array_equal(d.codes, codes)
    back = dev.download(d)
    assert back.device == "host" and np.array_equal(back.codes, codes)        # bit for bit
    assert np.array_equal(dev.upload(codes / 32768.0).codes, codes)           # from float too
    z = dev.alloc((3, 4))
    assert z.device == DEV and z.shape == (3, 4) and not z.any()
    assert sf.synchronize(DEV) is dev.last
    with pytest.raises(sf.SFUnsupportedError):
        dev.alloc((2,), sf.sf8)
    with pytest.raises(sf.SFUnsupportedError):
        sf.synchronize("atreides")                                            # no kernels there


def test_hardware_target_is_a_stub_until_attached(dev):
    hw = sf.device("atreides-hw")
    assert isinstance(hw, sf.HardwareAtreides)

    @sf.jit
    def k(x, n, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        sf.store(x + offs, sf.load(x + offs, mask=offs < n), mask=offs < n)

    x = sf.zeros((4,), device="atreides-hw")
    with pytest.raises(sf.SFDeviceError, match="no Atreides hardware"):
        k[(1,)](x, 4, BLOCK=4)

    class Driver:
        def __init__(self):
            self.calls = 0

        def launch(self, ir, grid, block, buffers, scalars):
            self.calls += 1
            from sf.kernels._interp import Interpreter
            return Interpreter(ir, buffers, scalars, grid, hw.limits).run()

    drv = Driver()
    sf.HardwareAtreides.attach(drv)
    try:
        r = k[(1,)](x, 4, BLOCK=4)
        assert drv.calls == 1 and r.device == "atreides-hw"
    finally:
        sf.HardwareAtreides.attach(None)


def test_launch_limits_raise_before_launch(dev):
    @sf.kernel
    def t(x, n):
        i = sf.thread_idx(0)
        with sf.predicate(i < n):
            x[i] = x[i]

    @sf.jit
    def big_tile(x, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        sf.store(x + offs, sf.load(x + offs))

    @sf.kernel
    def big_shared(x, SH: sf.constexpr):
        tile = sf.shared((SH,))
        tile[sf.thread_idx(0)] = x[sf.thread_idx(0)]

    x = dev.alloc(16)
    n0 = dev.kernel_launches
    with pytest.raises(sf.SFLaunchError, match="lanes"):
        t[(1,), (2048,)](x, 16)
    with pytest.raises(sf.SFLaunchError, match="per axis"):
        t[(70000,), (16,)](x, 16)
    with pytest.raises(sf.SFLaunchError, match="block value"):
        big_tile[(1,)](x, BLOCK=32768)
    with pytest.raises(sf.SFLaunchError, match="shared memory"):
        big_shared[(1,), (16,)](x, SH=65536)
    with pytest.raises(sf.SFLaunchError):
        t[(0,), (16,)](x, 16)
    with pytest.raises(sf.SFLaunchError, match="needs a grid and a block"):
        t[(1,)](x, 16)
    assert dev.kernel_launches == n0                                          # nothing launched
    r = t[(1,), (16,)](x, 16)
    assert dev.kernel_launches == n0 + 1 and r.grid == (1, 1, 1) and r.block == (16, 1, 1)


def test_argument_rules(dev):
    @sf.kernel
    def t(x, a, n):
        i = sf.thread_idx(0)
        with sf.predicate(i < n):
            x[i] = x[i] * a

    x = dev.alloc(4)
    with pytest.raises(sf.SFTypeError, match="live on the device"):
        t[(1,), (4,)](x.to("host"), sf.sf16(0.5), 4)
    with pytest.raises(sf.SFTypeError, match="quantize floats"):
        t[(1,), (4,)](x, 0.5, 4)
    with pytest.raises(sf.SFTypeError, match="different devices"):
        t[(1,), (4,)](x, sf.sf16(0.5), sf.zeros((4,), device="atreides-hw"))
    with pytest.raises(sf.SFTypeError):
        t[(1,), (4,)](x, sf.sf16(0.5))                                        # missing argument
    with pytest.raises(sf.SFCompileError, match="launch"):
        t(x, sf.sf16(0.5), 4)
    with pytest.raises(sf.SFUnsupportedError, match="sf8"):
        t[(1,), (4,)](x, sf.sf8(0.5), 4)
    with pytest.raises(sf.SFCompileError, match="must not return"):
        @sf.jit
        def ret(x, BLOCK: sf.constexpr):
            return sf.load(x + sf.arange(0, BLOCK))
        ret[(1,)](x, BLOCK=4)


# --- 3.1 thread kernels ---------------------------------------------------------------------------

@sf.kernel
def axpy_t(a, x, y, out, n):
    i = sf.block_idx(0) * sf.block_dim(0) + sf.thread_idx(0)
    with sf.predicate(i < n):
        out[i] = sf.fma(a, x[i], y[i])


@sf.kernel
def sum_t(x, out, n, BLOCK: sf.constexpr):
    tid = sf.thread_idx(0)
    tile = sf.shared((BLOCK,))
    with sf.predicate(tid < n):
        tile[tid] = x[tid]
    sf.barrier()
    acc = sf.sf16(0)
    with sf.predicate(tid == 0):                  # one lane accumulates (and counts events)
        for k in range(BLOCK):                    # sequential: the target's accumulation order
            acc = acc + tile[k]
        out[tid] = acc


@sf.kernel
def matmul_t(a, b, c, M, N, K: sf.constexpr):
    row = sf.block_idx(1) * sf.block_dim(1) + sf.thread_idx(1)
    col = sf.block_idx(0) * sf.block_dim(0) + sf.thread_idx(0)
    live = (row < M) & (col < N)
    acc = sf.sf16(0)
    for k in range(K):
        acc = sf.fma(sf.load(a + (row * K + k), mask=live), sf.load(b + (k * N + col), mask=live), acc)
    with sf.predicate(live):
        c[row * N + col] = acc


# --- 3.2 block kernels ---------------------------------------------------------------------------

@sf.jit
def axpy_b(a, x_ptr, y_ptr, out_ptr, n, BLOCK: sf.constexpr):
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    mask = offs < n
    x = sf.load(x_ptr + offs, mask=mask)
    y = sf.load(y_ptr + offs, mask=mask)
    sf.store(out_ptr + offs, sf.fma(a, x, y), mask=mask)


@sf.jit
def sum_b(x_ptr, out_ptr, n, BLOCK: sf.constexpr):
    offs = sf.arange(0, BLOCK)
    x = sf.load(x_ptr + offs, mask=offs < n)
    sf.store(out_ptr, sf.sum(x))


@sf.jit
def matmul_b(a_ptr, b_ptr, c_ptr, M, N, K: sf.constexpr, BM: sf.constexpr, BN: sf.constexpr,
             BK: sf.constexpr):
    pm, pn = sf.program_id(0), sf.program_id(1)
    rm = pm * BM + sf.arange(0, BM)
    rn = pn * BN + sf.arange(0, BN)
    acc = sf.zeros((BM, BN))
    for k0 in range(0, K, BK):
        rk = k0 + sf.arange(0, BK)
        a = sf.load(a_ptr + rm[:, None] * K + rk[None, :], mask=(rm[:, None] < M) & (rk[None, :] < K))
        b = sf.load(b_ptr + rk[:, None] * N + rn[None, :], mask=(rk[:, None] < K) & (rn[None, :] < N))
        acc = acc + sf.dot(a, b)
    sf.store(c_ptr + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


@pytest.mark.parametrize("n", [1, 7, 64, 333])
def test_axpy_matches_builtin(dev, backend, n):
    x, y = dev.upload(rand(n, -0.9, 0.9, n)), dev.upload(rand(n, -0.9, 0.9, n + 1))
    a = sf.sf16(0.75)
    with sf.errstate(all="ignore"):
        ref = sf.fma(a, x.to("host"), y.to("host"))
        out_t, out_b = dev.alloc(n), dev.alloc(n)
        rt = axpy_t[(sf.cdiv(n, 32),), (32,)](a, x, y, out_t, n)
        rb = axpy_b[(sf.cdiv(n, 32),)](a, x, y, out_b, n, BLOCK=32)
    assert np.array_equal(out_t.codes, ref.codes) and np.array_equal(out_b.codes, ref.codes)
    assert rt.status.exact and rb.status.exact and rt.status == rb.status
    # events are counted exactly, as on a flagged target, and only on live lanes
    expect_sat = int(np.count_nonzero((a * x.to("host")).codes + y.to("host").codes > 32767)
                     + np.count_nonzero((a * x.to("host")).codes + y.to("host").codes < -32767))
    assert rt.status.saturated == expect_sat == rb.status.saturated
    assert rt.backend == backend == rb.backend


@pytest.mark.parametrize("n", [1, 5, 50, 64])
def test_reduction_matches_builtin(dev, backend, n):
    x = dev.upload(rand(n, -0.3, 0.3, 100 + n))
    with sf.errstate(all="ignore"):
        ref = x.to("host").sum()
        out_t, out_b = dev.alloc(1), dev.alloc(1)
        rt = sum_t[(1,), (64,)](x, out_t, n, BLOCK=64)
        rb = sum_b[(1,)](x, out_b, n, BLOCK=64)
    assert int(out_t.codes[0]) == ref.code == int(out_b.codes[0])
    # status counts results that saturated: the hand-written loop has one result per add,
    # the sf.sum reduction has one result, counted once, like the Stage 1.1 built-in
    assert rb.status.saturated == (1 if rt.status.saturated else 0)


@pytest.mark.parametrize("shape", [(1, 1, 1), (4, 4, 4), (5, 7, 6), (9, 11, 8), (3, 2, 16)])
def test_matmul_matches_builtin(dev, backend, shape):
    M, N, K = shape
    A = dev.upload(rand((M, K), -0.4, 0.4, M)); B = dev.upload(rand((K, N), -0.4, 0.4, N))
    with sf.errstate(all="ignore"):
        ref = A.to("host") @ B.to("host")
        Ct, Cb = dev.alloc((M, N)), dev.alloc((M, N))
        rt = matmul_t[(sf.cdiv(N, 4), sf.cdiv(M, 4)), (4, 4)](A, B, Ct, M, N, K=K)
        rb = matmul_b[(sf.cdiv(M, 4), sf.cdiv(N, 4))](A, B, Cb, M, N, K=K, BM=4, BN=4, BK=min(K, 8))
    assert np.array_equal(Ct.codes, ref.codes) and np.array_equal(Cb.codes, ref.codes)
    assert rb.counters.macs > 0 and rt.counters.macs == 0           # dot uses the systolic array


def test_thread_and_block_kernels_give_identical_bits(dev, backend):
    """The 3.2 gate: the 3.1 kernels rewritten as block kernels give identical bits,
    including on inputs that saturate."""
    n = 200
    x, y = dev.upload(rand(n, -0.99, 0.99, 7)), dev.upload(rand(n, -0.99, 0.99, 8))
    with sf.errstate(all="ignore"):
        o1, o2 = dev.alloc(n), dev.alloc(n)
        axpy_t[(sf.cdiv(n, 64),), (64,)](sf.sf16(0.9), x, y, o1, n)
        axpy_b[(sf.cdiv(n, 64),)](sf.sf16(0.9), x, y, o2, n, BLOCK=64)
        assert np.array_equal(o1.codes, o2.codes)
        big = dev.upload(rand(64, -0.9, 0.9, 9))
        s1, s2 = dev.alloc(1), dev.alloc(1)
        sum_t[(1,), (64,)](big, s1, 64, BLOCK=64)
        sum_b[(1,)](big, s2, 64, BLOCK=64)
        assert s1.codes[0] == s2.codes[0] == big.to("host").sum().code


# --- lockstep and predication ------------------------------------------------------------------

def test_data_dependent_branches_are_rejected(dev):
    x = dev.alloc(8)

    @sf.kernel
    def if_on_data(x, n):
        i = sf.thread_idx(0)
        if i < n:
            x[i] = x[i]

    @sf.kernel
    def while_on_data(x, n):
        i = sf.thread_idx(0)
        while i < n:
            i = i + 1

    @sf.kernel
    def bool_and(x, n):
        i = sf.thread_idx(0)
        m = (i < n) and (i > 0)
        x[i] = x[i]

    @sf.jit
    def range_on_data(x, n, BLOCK: sf.constexpr):
        for k in range(n):
            pass

    for bad in (if_on_data, while_on_data, bool_and):
        with pytest.raises(sf.SFCompileError, match="lockstep"):
            bad[(1,), (8,)](x, 8)
    with pytest.raises(sf.SFCompileError, match="loop bound"):
        range_on_data[(1,)](x, 8, BLOCK=8)
    assert sf.device(DEV).kernel_launches >= 0


def test_predicate_masks_stores_and_events(dev, backend):
    @sf.kernel
    def k(x, out, n):
        i = sf.thread_idx(0)
        with sf.predicate(i < n):
            out[i] = x[i] + x[i]                    # saturates for large x, but only live lanes count
        with sf.predicate(i >= n):
            out[i] = sf.sf16(0.25)                  # the other lanes take the other branch

    x = dev.upload(np.full(8, 0.9))
    out = dev.alloc(16)
    with sf.errstate(over="ignore"):
        r = k[(1,), (16,)](x, out, 8)
    assert r.status.saturated == 8                 # the x[i] loads on lanes 8..15 are masked: no fault
    assert out.numpy().tolist() == [32767 / 32768] * 8 + [0.25] * 8
    # nested predicates AND together
    @sf.kernel
    def nested(out):
        i = sf.thread_idx(0)
        with sf.predicate(i < 6):
            with sf.predicate(i >= 2):
                out[i] = sf.sf16(0.5)
    o = dev.alloc(8)
    nested[(1,), (8,)](o)
    assert o.numpy().tolist() == [0, 0, 0.5, 0.5, 0.5, 0.5, 0, 0]


def test_where_and_masked_loads(dev, backend):
    @sf.jit
    def relu(x_ptr, out_ptr, n, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        x = sf.load(x_ptr + offs, mask=offs < n, other=0.25)      # masked lanes read `other`
        y = sf.where(x > 0, x, 0)
        sf.store(out_ptr + offs, y + sf.maximum(x, -0.1) - sf.minimum(x, 0.1), mask=offs < n)

    v = rand(6, -0.5, 0.5, 3)
    x = dev.upload(v)
    out = dev.alloc(6)
    with sf.errstate(all="ignore"):
        relu[(1,)](x, out, 6, BLOCK=8)
        h = x.to("host")
        ref = sf.where(h > 0, h, 0) + sf.maximum(h, -0.1) - sf.minimum(h, 0.1)
    assert np.array_equal(out.codes, ref.codes)


def test_out_of_bounds_is_a_device_fault(dev, backend):
    @sf.jit
    def k(x, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        sf.store(x + offs, sf.load(x + offs))

    x = dev.alloc(4)
    with pytest.raises(sf.SFDeviceError, match="out of bounds"):
        k[(1,)](x, BLOCK=8)

    @sf.kernel
    def div0(x, out):
        i = sf.thread_idx(0)
        out[i // (i - i)] = x[i]                     # integer division by zero in the index

    with pytest.raises(sf.SFDeviceError, match="division by zero"):
        div0[(1,), (4,)](x, x)
    with pytest.raises(sf.SFUnsupportedError, match="floordiv"):
        @sf.kernel
        def bad(x):
            x[0] = x[0] // 2                         # no integer division on sf values
        bad[(1,), (1,)](x)


def test_status_follows_the_policy(dev, backend):
    @sf.jit
    def k(x, out, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        v = sf.load(x + offs)
        sf.store(out + offs, v / v + sf.sqrt(v))

    x = dev.upload(np.array([0.5, 0.0, -0.25, 0.1]))
    out = dev.alloc(4)
    with pytest.raises(sf.SFInvalidError, match="status.invalid"):       # catalogue order
        k[(1,)](x, out, BLOCK=4)
    with sf.errstate(invalid="ignore"):
        with pytest.raises(sf.SFZeroDivisionError):
            k[(1,)](x, out, BLOCK=4)
    with sf.errstate(invalid="ignore", divide="ignore"):
        with pytest.raises(sf.SFOverflowError):                         # x / x overflows
            k[(1,)](x, out, BLOCK=4)
    with sf.errstate(all="ignore"):
        r = k[(1,)](x, out, BLOCK=4)
    # v / v overflows for the three non-zero inputs, and two of those then saturate again in
    # the + sqrt(v); 0 / 0 is a div0, sqrt(-0.25) is invalid
    assert r.status == sf.SFStatus(saturated=5, div0=1, invalid=1, exact=True, outputs_at_max=0)
    h = x.to("host")
    with sf.errstate(all="ignore"):
        assert np.array_equal(out.codes, (h / h + sf.sqrt(h)).codes)
    assert r.status.ok is False


def test_literals_and_comparisons_follow_stage_1(dev, backend):
    @sf.jit
    def k(x, out, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        v = sf.load(x + offs)
        lt1 = v < 1                                 # always True: exact comparison
        eq = v == 0.1                               # never equal to the float 0.1
        sf.store(out + offs, sf.where(lt1 & ~eq, v + 0.25, 0))

    x = dev.upload(np.array([0.1, 0.5, -0.5]))
    out = dev.alloc(3)
    k[(1,)](x, out, BLOCK=3)
    assert np.array_equal(out.codes, (x.to("host") + 0.25).codes)

    @sf.jit
    def bad_literal(x, BLOCK: sf.constexpr):
        sf.store(x + sf.arange(0, BLOCK), sf.load(x + sf.arange(0, BLOCK)) + 1)

    with pytest.raises(sf.SFOverflowError, match="kernel literal 1"):
        bad_literal[(1,)](x, BLOCK=3)


# --- section 11: backends agree ---------------------------------------------------------------------

@pytest.mark.skipif(not _cgen.compiler(), reason="no C compiler")
def test_c_backend_is_bit_identical_to_the_interpreter(dev):
    """The same IR through SF IR -> C -> cc and through the interpreter: same
    bits and the same events, on every Stage 3 kernel, including saturating
    inputs, shared memory, predication and dot."""
    rng = np.random.default_rng(42)
    kernels = []
    n = 150
    x, y = dev.upload(rng.uniform(-0.99, 0.99, n)), dev.upload(rng.uniform(-0.99, 0.99, n))
    kernels.append(lambda out: axpy_t[(sf.cdiv(n, 32),), (32,)](sf.sf16(0.95), x, y, out, n))
    kernels.append(lambda out: axpy_b[(sf.cdiv(n, 32),)](sf.sf16(0.95), x, y, out, n, BLOCK=32))
    kernels.append(lambda out: sum_t[(1,), (64,)](x, out, 64, BLOCK=64))
    kernels.append(lambda out: sum_b[(1,)](x, out, 64, BLOCK=64))
    M, N, K = 6, 10, 8
    A, B = dev.upload(rng.uniform(-0.9, 0.9, (M, K))), dev.upload(rng.uniform(-0.9, 0.9, (K, N)))
    kernels.append(lambda out: matmul_t[(sf.cdiv(N, 4), sf.cdiv(M, 4)), (4, 4)](A, B, out, M, N, K=K))
    kernels.append(lambda out: matmul_b[(sf.cdiv(M, 4), sf.cdiv(N, 4))](A, B, out, M, N, K=K, BM=4, BN=4, BK=8))
    for launch in kernels:
        results = {}
        for be in ("interp", "c"):
            out = dev.alloc(n)
            with sf.config(kernel_backend=be), sf.errstate(all="ignore"):
                r = launch(out)
            results[be] = (out.codes.copy(), r.status, r.counters.saturated, r.counters.underflow)
        assert np.array_equal(results["interp"][0], results["c"][0])
        assert results["interp"][1:] == results["c"][1:]


def test_c_backend_emits_the_builtins(dev):
    @sf.jit
    def k(x, out, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        v = sf.load(x + offs)
        sf.store(out + offs, sf.fma(v, v, v) / (v - 0.5) + sf.sqrt(v))

    x = dev.upload(np.full(4, 0.25))
    with sf.errstate(all="ignore"):
        k[(1,)](x, dev.alloc(4), BLOCK=4)                               # trace (interp)
    ir = next(iter(k.cache.values()))
    src = _cgen.emit_c(ir)
    for builtin in ("__sf16_fma", "__sf16_sub", "__sf16_div", "__sf16_add", "__sf16_sqrt"):
        assert builtin in src
    assert "float" not in src.replace("sf16_builtins", "")               # integer codes only
    assert "kernel k [block]" in ir.dump() and "dot" not in ir.dump()


# --- 3.3 performance --------------------------------------------------------------------------------

def test_kernel_cache_and_profiler(dev):
    @sf.jit
    def k(x, out, n, BLOCK: sf.constexpr):
        offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
        m = offs < n
        sf.store(out + offs, sf.load(x + offs, mask=m) * 0.5, mask=m)

    x = dev.upload(rand(100, -0.5, 0.5, 1))
    out = dev.alloc(100)
    assert len(k.cache) == 0
    with sf.errstate(under="ignore"):
        r1 = k[(sf.cdiv(100, 32),)](x, out, 100, BLOCK=32)
        r2 = k[(sf.cdiv(100, 32),)](x, out, 100, BLOCK=32)
        assert len(k.cache) == 1                                        # traced once
        r3 = k[(sf.cdiv(100, 64),)](x, out, 100, BLOCK=64)
        assert len(k.cache) == 2                                        # a new constexpr: new entry
    c = r1.counters
    assert c.programs == 4 and c.loads == 4 and c.stores == 4 and c.ops > 0 and c.lane_ops > 0
    assert r1.cycles == r2.cycles and r3.cycles != r1.cycles and r1.cycles > 0
    assert "cycles=" in repr(r1) and r1.config == {"BLOCK": 32}


def test_autotune_changes_speed_never_bits(dev, backend):
    @sf.autotune(configs=[{"BLOCK": 8}, {"BLOCK": 32}, {"BLOCK": 128}], key=["n"])
    @sf.jit
    def scale(x, out, n, BLOCK: sf.constexpr):
        offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
        m = offs < n
        sf.store(out + offs, sf.load(x + offs, mask=m) + sf.load(x + offs, mask=m), mask=m)

    n = 300
    x = dev.upload(rand(n, -0.9, 0.9, 5))
    with sf.errstate(over="ignore"):
        ref = (x.to("host") + x.to("host")).codes
        outs = {}
        for cfg in ({"BLOCK": 8}, {"BLOCK": 32}, {"BLOCK": 128}):
            out = dev.alloc(n)
            # the untuned kernel body, launched directly with each config
            plain = sf.jit(scale.fn)
            r = plain[(sf.cdiv(n, cfg["BLOCK"]),)](x, out, n, **cfg)
            outs[cfg["BLOCK"]] = (out.codes.copy(), r.cycles)
        assert all(np.array_equal(c, ref) for c, _ in outs.values())      # never bits
        assert len({cyc for _, cyc in outs.values()}) > 1                 # but speed
        out = dev.alloc(n)
        r = scale[lambda cfg: (sf.cdiv(n, cfg["BLOCK"]),)](x, out, n)
    assert np.array_equal(out.codes, ref)
    assert r.config["BLOCK"] == min(outs, key=lambda b: outs[b][1])       # the fastest config won
    assert scale.tuned[(("n", n),)] == r.config
    # a different key value tunes again; the same one does not
    before = sf.device(DEV).kernel_launches
    with sf.errstate(over="ignore"):
        scale[lambda cfg: (sf.cdiv(n, cfg["BLOCK"]),)](x, out, n)
    assert sf.device(DEV).kernel_launches == before + 1
    with pytest.raises(sf.SFLaunchError, match="callable grid"):
        sf.jit(scale.fn)[lambda cfg: (1,)](x, out, n, BLOCK=8)


def test_hidden_module_level_sf_functions_dispatch_in_kernels(dev):
    """sf.sum, sf.dot, sf.where, sf.fma, sf.sqrt, sf.maximum, sf.minimum, sf.clip,
    sf.cast, sf.zeros, sf.full, sf.max, sf.min work on traced values."""
    @sf.jit
    def k(x, out, BLOCK: sf.constexpr):
        offs = sf.arange(0, BLOCK)
        v = sf.load(x + offs)
        t = sf.clip(v, -0.2, 0.2) + sf.full((BLOCK,), 0.1) + sf.zeros((BLOCK,))
        m2 = v.reshape(2, BLOCK // 2)
        s = sf.sum(m2, axis=1)                                   # (2,)
        mx = sf.max(m2, axis=1) - sf.min(m2, axis=1)
        sf.store(out + sf.arange(0, 2), s + mx)
        sf.store(out + 2 + offs, t)

    x = dev.upload(rand(8, -0.5, 0.5, 11))
    out = dev.alloc(10)
    with sf.errstate(all="ignore"):
        k[(1,)](x, out, BLOCK=8)
        h = x.to("host")
        m2 = h.reshape(2, 4)
        exp_s = (m2.sum(axis=1) + (sf.max(m2, axis=1) - sf.min(m2, axis=1))).codes
        exp_t = (sf.clip(h, -0.2, 0.2) + 0.1).codes
    assert out.codes[:2].tolist() == exp_s.tolist() and out.codes[2:].tolist() == exp_t.tolist()
