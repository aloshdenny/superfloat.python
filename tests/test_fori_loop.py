"""sf.fori_loop: a loop kept as one `for` op (spec 1.11.0, SF IR format_version 2).

The rule every test here leans on: a for op gives exactly the bits and events of
the same loop unrolled by Python, on the interpreter and on the C backend.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import sf
from sf.kernels._ir import KernelIR, walk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from check_sfir import check  # noqa: E402

DEV = "atreides-sim"


@pytest.fixture(params=["interp", "c"])
def backend(request):
    with sf.config(kernel_backend=request.param):
        yield request.param


def _ir(kernel) -> KernelIR:
    """The IR of the kernel's last launch (each test clears the cache before it launches)."""
    (ir,) = kernel.cache.values()
    return ir


@pytest.fixture(autouse=True)
def _fresh_cache():
    for k in (rows_loop, rows_unrolled, fib_swap, nested, no_trips, masked_loop):
        k.cache.clear()


def _codes(seed, n, lo=-0.9, hi=0.9, dtype=sf.sf16):
    with sf.errstate(all="ignore"):
        return sf.asarray(np.random.default_rng(seed).uniform(lo, hi, n), dtype=dtype, device=DEV)


# --- the same loop twice: as a for op, and unrolled by Python -----------------------------------

@sf.jit
def rows_loop(x_ptr, out_ptr, K: sf.constexpr, BLOCK: sf.constexpr, DT: sf.constexpr):
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    acc = sf.fori_loop(0, K, lambda k, acc: acc + sf.load(x_ptr + offs * K + k), sf.zeros((BLOCK,), DT))
    sf.store(out_ptr + offs, acc)


@sf.jit
def rows_unrolled(x_ptr, out_ptr, K: sf.constexpr, BLOCK: sf.constexpr, DT: sf.constexpr):
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    acc = sf.zeros((BLOCK,), DT)
    for k in range(K):
        acc = acc + sf.load(x_ptr + offs * K + k)
    sf.store(out_ptr + offs, acc)


def _both(x, n, K, BLOCK, dtype="sf16"):
    out = []
    for kern in (rows_loop, rows_unrolled):
        kern.cache.clear()
        o = sf.zeros(n, dtype, device=DEV)
        with sf.errstate(all="ignore"):
            r = kern[(n // BLOCK,)](x, o, K=K, BLOCK=BLOCK, DT=dtype)
        out.append((o.codes.copy(), r.status))
    return out


@pytest.mark.parametrize("lo,hi", [(-0.2, 0.2), (0.3, 0.9)])          # the second saturates often
def test_same_bits_and_events_as_the_unrolled_loop(backend, lo, hi):
    x = _codes(1, 16 * 12, lo, hi)
    (a, sa), (b, sb) = _both(x, 16, 12, 8)
    assert np.array_equal(a, b)
    assert sa == sb
    if hi == 0.9:
        assert sa.saturated > 0


def test_the_file_holds_the_loop_once():
    x = _codes(3, 8 * 40)
    _both(x, 8, 40, 8)
    looped, unrolled = _ir(rows_loop), _ir(rows_unrolled)
    assert len(list(walk(looped.ops))) < 25 < len(unrolled.ops)
    assert json.loads(looped.to_json())["format_version"] == 2
    assert json.loads(unrolled.to_json())["format_version"] == 1      # no loop: still version 1
    assert "= for %" in looped.dump()


# --- carries, nesting, steps --------------------------------------------------------------------

@sf.jit
def fib_swap(out_ptr, T: sf.constexpr):
    # a two-value carry that swaps: (a, b) -> (b, a + b); the C backend copies through temporaries
    one = sf.full((1,), 1 / 64)
    a, b = sf.fori_loop(0, T, lambda t, ab: (ab[1], ab[0] + ab[1]), (one, one))
    sf.store(out_ptr + sf.arange(0, 1), a)
    sf.store(out_ptr + 1 + sf.arange(0, 1), b)


def test_tuple_carry_with_a_swap(backend):
    o = sf.zeros(2, "sf16", device=DEV)
    fib_swap[(1,)](o, T=7)
    f = [1, 1]
    for _ in range(7):
        f = [f[1], f[0] + f[1]]
    assert list(o.codes) == [c * 512 for c in f]                    # 1/64 is code 512


@sf.jit
def nested(x_ptr, out_ptr, R: sf.constexpr, C: sf.constexpr):
    def row(r, total):
        def col(c, s):
            return s + sf.load(x_ptr + (r * C + c) + sf.arange(0, 1))
        return sf.fori_loop(0, C, col, total)
    total = sf.fori_loop(R - 1, -1, row, sf.zeros((1,)), step=-1)    # rows backwards
    count = sf.fori_loop(0, R * C, lambda i, n: n + 1, sf.program_id(0) * 0)   # an int carry
    sf.store(out_ptr + sf.arange(0, 1), total)
    sf.store(out_ptr + 1 + sf.arange(0, 1), sf.where(count == R * C, total, -total))


def test_nested_loops_negative_step_and_int_carry(backend):
    x = _codes(4, 12, -0.05, 0.05)
    o = sf.zeros(2, "sf16", device=DEV)
    nested[(1,)](x, o, R=3, C=4)
    with sf.errstate(all="ignore"):
        ref = sf.zeros(1, "sf16")
        for r in (2, 1, 0):
            for c in range(4):
                ref = ref + x.to("host")[r * 4 + c: r * 4 + c + 1]
    assert list(o.codes) == [ref.codes[0], ref.codes[0]]
    fors = [op for op in walk(_ir(nested).ops) if op.name == "for"]
    assert [f.attrs["start"] for f in fors] == [2, 0, 0] and fors[0].attrs["step"] == -1


@sf.jit
def no_trips(out_ptr, T: sf.constexpr):
    v = sf.fori_loop(0, T, lambda t, v: v + 1 / 8, sf.full((1,), 1 / 4))
    sf.store(out_ptr + sf.arange(0, 1), v)


def test_zero_trips_give_the_initial_value_and_no_loop():
    o = sf.zeros(1, "sf16", device=DEV)
    no_trips[(1,)](o, T=0)
    assert float(o.to("host")[0]) == 0.25
    assert not any(op.name == "for" for op in walk(_ir(no_trips).ops))


@sf.jit
def masked_loop(x_ptr, out_ptr, n, K: sf.constexpr, BLOCK: sf.constexpr):
    offs = sf.arange(0, BLOCK)
    with sf.predicate(offs < n):
        acc = sf.fori_loop(0, K, lambda k, acc: acc + sf.load(x_ptr + offs * K + k), sf.zeros((BLOCK,)))
        sf.store(out_ptr + offs, acc)


def test_a_predicate_stays_in_force_inside_the_body(backend):
    x = _codes(5, 8 * 3)
    o = sf.full(8, 0.5, "sf16", device=DEV)
    with sf.errstate(all="ignore"):
        masked_loop[(1,)](x, o, 5, K=3, BLOCK=8)
        ref = sf.zeros(5, "sf16")
        for k in range(3):
            ref = ref + x.to("host")[np.arange(5) * 3 + k]
    assert list(o.codes[:5]) == list(ref.codes) and list(o.codes[5:]) == [16384] * 3


def test_targets_refuse_ops_inside_a_loop_body_too():
    @sf.jit
    def divides(x_ptr, out_ptr, K: sf.constexpr):
        i = sf.arange(0, 4)
        v = sf.fori_loop(0, K, lambda k, v: v / (v + 0.5), sf.load(x_ptr + i))
        sf.store(out_ptr + i, v)
    with pytest.raises(sf.SFUnsupportedError, match="divide|div"):
        divides[(1,)](_codes(6, 4), sf.zeros(4, "sf16", device=DEV), K=2)


# --- errors at trace time -----------------------------------------------------------------------

def _trace_error(body, init=None, **kw):
    @sf.jit
    def k(x_ptr, out_ptr):
        i = sf.arange(0, 4)
        start = kw.get("start", 0)
        v = sf.fori_loop(start() if callable(start) else start, 3, body(x_ptr, i),
                         init(x_ptr, i) if init else sf.zeros((4,)), step=kw.get("step", 1))
        sf.store(out_ptr + i, v if not isinstance(v, tuple) else v[0])
    with pytest.raises(sf.SFCompileError) as e:
        k[(1,)](_codes(7, 4), sf.zeros(4, "sf16", device=DEV))
    return str(e.value)


def test_a_body_value_used_after_the_loop_is_refused():
    leaked = []

    @sf.jit
    def k(x_ptr, out_ptr):
        i = sf.arange(0, 4)

        def body(t, v):
            leaked.append(v + 1 / 8)
            return leaked[-1]
        sf.fori_loop(0, 3, body, sf.zeros((4,)))
        sf.store(out_ptr + i, leaked[-1])
    with pytest.raises(sf.SFCompileError, match="only by being carried"):
        k[(1,)](_codes(8, 4), sf.zeros(4, "sf16", device=DEV))


def test_the_body_must_keep_types_and_shapes():
    assert "keep its type and shape" in _trace_error(lambda p, i: lambda t, v: sf.zeros((2,)))
    assert "keep its type and shape" in _trace_error(lambda p, i: lambda t, v: i)
    assert "must return 1 carried value" in _trace_error(lambda p, i: lambda t, v: (v, v))


def test_bounds_are_compile_time_ints():
    assert "start must be a Python int" in _trace_error(lambda p, i: lambda t, v: v, start=lambda: sf.program_id(0))
    assert "step must not be 0" in _trace_error(lambda p, i: lambda t, v: v, step=0)
    assert "start must be a Python int" in _trace_error(lambda p, i: lambda t, v: v, start=1.5)


def test_pointers_and_python_numbers_are_not_carried():
    assert "carry the offsets" in _trace_error(lambda p, i: lambda t, v: v, init=lambda p, i: p + i)
    assert "not a traced" in _trace_error(lambda p, i: lambda t, v: v, init=lambda p, i: 0.5)


# --- SF IR JSON and the checker -----------------------------------------------------------------

def _looped_json() -> dict:
    x = _codes(9, 8 * 4)
    _both(x, 8, 4, 8)
    return json.loads(_ir(rows_loop).to_json())


def test_json_round_trip_and_checker():
    d = _looped_json()
    text = json.dumps(d, sort_keys=True, indent=1) + "\n"
    back = KernelIR.from_json(text)
    assert back.to_json() == text
    assert check(d) == []
    (f,) = [o for o in d["ops"] if o["op"] == "for"]
    assert set(f) == {"id", "op", "args", "attrs", "type", "shape", "pred", "loc", "body"}
    assert set(f["body"]) == {"iv", "carried", "ops", "yield", "results"}


def _broken(change) -> dict:
    d = _looped_json()
    f = next(o for o in d["ops"] if o["op"] == "for")
    change(d, f)
    return d


@pytest.mark.parametrize("change,reader,checker", [
    (lambda d, f: d.update(format_version=1), "needs format_version 2", "needs format_version 2"),
    (lambda d, f: f["attrs"].update(step=0), "step", "step is 0"),
    (lambda d, f: f["body"].pop("yield"), "missing fields", "body is not"),
    (lambda d, f: f["body"]["yield"].append(f["body"]["iv"]), "need 1 ids", "need 1 ids"),
    (lambda d, f: f["body"].update({"yield": [f["body"]["iv"]]}), "yield", "yield"),   # int into sf16
    (lambda d, f: d["ops"][-1]["args"].__setitem__(-2, f["body"]["ops"][-1]["id"]),
     "made inside a loop body", "made inside a loop body"),                          # a body value used after
    (lambda d, f: f.update(body=None) or f.pop("body"), "missing fields", "fields"),
])
def test_reader_and_checker_refuse_a_broken_loop(change, reader, checker):
    d = _broken(change)
    with pytest.raises(sf.SFCompileError, match=reader):
        KernelIR.from_json(json.dumps(d))
    errs = check(d)
    assert errs and any(checker in e for e in errs), errs


def test_body_field_only_on_for_ops():
    d = _looped_json()
    d["ops"][0]["body"] = {}
    with pytest.raises(sf.SFCompileError, match="unknown fields"):
        KernelIR.from_json(json.dumps(d))
    assert check(copy.deepcopy(d))



@sf.jit
def pred_carry(x_ptr, o_ptr, N: sf.constexpr, LOOP: sf.constexpr):
    i = sf.arange(0, N)
    x = sf.load(x_ptr + i)
    c = sf.program_id(0) * 0                           # a scalar, made outside the predicate
    with sf.predicate(x > 0):
        if LOOP:
            c = sf.fori_loop(0, 3, lambda t, v: v + 1, c)
        else:
            for _ in range(3):
                c = c + 1
        sf.store(o_ptr + i, sf.where(c == 3, x, x - x))


def test_a_scalar_carry_under_a_predicate_is_widened_like_the_unrolled_loop(backend):
    """Under sf.predicate every computed value is one per lane, so the carry widens, as when unrolled."""
    outs = []
    for loop in (1, 0):
        pred_carry.cache.clear()
        x = sf.from_codes(np.arange(-4, 4) * 1000, sf.sf16, device=DEV)
        o = sf.zeros(8, "sf16", device=DEV)
        pred_carry[(1,)](x, o, N=8, LOOP=loop)
        outs.append(o.codes.copy())
    assert np.array_equal(*outs)


def test_a_for_op_under_a_predicate_has_no_pred_of_its_own():
    pred_carry.cache.clear()
    pred_carry[(1,)](sf.zeros(8, "sf16", device=DEV), sf.zeros(8, "sf16", device=DEV), N=8, LOOP=1)
    ir = _ir(pred_carry)
    (op,) = [o for o in ir.ops if o.name == "for"]
    assert op.pred is None and all(o.pred is not None for o in op.body.ops)
    d = json.loads(ir.to_json())
    assert check(d) == []
    f = next(o for o in d["ops"] if o["op"] == "for")
    f["pred"] = f["body"]["ops"][0]["pred"]
    with pytest.raises(sf.SFCompileError, match="pred"):
        KernelIR.from_json(json.dumps(d))
    assert any("no pred" in e for e in check(d))
