"""SF IR JSON: source lines, writing, reading, and the round trip.

The whole suite can also run with every traced kernel sent through JSON and
back before it runs: ``SF_IR_ROUNDTRIP=1 pytest`` (see conftest.py).  Every
bit-exact kernel test then checks the round trip too.
"""
from __future__ import annotations

import importlib.util
import json
import os

import numpy as np
import pytest

import sf
from sf.kernels import _irjson
from sf.kernels._ir import KernelIR
from sf.kernels._runtime import Kernel

DEV = "atreides-sim"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _line(marker: str) -> int:
    """The line number in this file of the line ending with ``# <marker>``."""
    for i, text in enumerate(open(__file__).read().splitlines(), start=1):
        if text.rstrip().endswith(f"# {marker}"):
            return i
    raise AssertionError(marker)


@sf.jit
def axpy(a, x_ptr, y_ptr, out_ptr, n, BLOCK: sf.constexpr):                          # L-def
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)                            # L-offs
    mask = offs < n                                                                  # L-mask
    xv = sf.load(x_ptr + offs, mask=mask)                                            # L-loadx
    yv = sf.load(y_ptr + offs, mask=mask)
    sf.store(out_ptr + offs, sf.fma(a, xv, yv), mask=mask)                           # L-store


@sf.jit
def every_op(x_ptr, y_ptr, out_ptr, n, BLOCK: sf.constexpr):
    """The 11 ops no other kernel test uses: abs, bool_to_int, cmp, iabs, imod,
    ineg, narrow, neg, or, transpose, xor."""
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    m = offs < n
    x = sf.load(x_ptr + offs, mask=m)
    y = sf.load(y_ptr + offs, mask=m)
    big = x > y                                    # cmp
    either = big | m                               # or
    one = big ^ m                                  # xor
    i = abs(-offs % 3 - 1)                         # ineg, imod, iabs
    i = i + sf.where(one, 1, 0) + (either * 1)     # bool_to_int
    v = abs(-x) + y                                # neg, abs
    t = v.reshape((BLOCK // 4, 4)).T.T.reshape((BLOCK,))   # transpose
    t = sf.where(i > 1, t, -t)
    t = sf.cast(sf.cast(t, "sf8"), "sf16")                 # narrow (and widen back)
    sf.store(out_ptr + offs, t, mask=m)


def _traced(kernel, *args, **cx) -> KernelIR:
    """Launch on the simulator and return the IR it traced."""
    kernel.cache.clear()
    grid = (sf.cdiv(args[-1], cx["BLOCK"]),)
    with sf.errstate(all="ignore"):
        kernel[grid](*args, **cx)
    (ir,) = kernel.cache.values()
    return ir


def _axpy_args(n=20):
    x = sf.asarray(np.linspace(-0.9, 0.9, n), dtype=sf.sf16, device=DEV)
    y = sf.asarray(np.linspace(0.1, 0.4, n), dtype=sf.sf16, device=DEV)
    out = sf.zeros(n, "sf16", device=DEV)
    return sf.sf16(0.75), x, y, out, n


def _every_args(n=20):
    x = sf.asarray(np.linspace(-0.9, 0.9, n), dtype=sf.sf16, device=DEV)
    y = sf.asarray(np.linspace(0.3, -0.3, n), dtype=sf.sf16, device=DEV)
    out = sf.zeros(n, "sf16", device=DEV)
    return x, y, out, n


# --- source lines ------------------------------------------------------------------------------

def test_ops_carry_the_users_line():
    ir = _traced(axpy, *_axpy_args(), BLOCK=8)
    by_name = {}
    for op in ir.ops:
        by_name.setdefault(op.name, op.loc)
        assert op.loc is not None and op.loc[0] == os.path.basename(__file__)
    assert by_name["program_id"][1] == _line("L-offs")
    assert by_name["icmp"][1] == _line("L-mask")
    assert by_name["load"][1] == _line("L-loadx")
    assert by_name["store"][1] == _line("L-store")


def test_param_ops_point_at_the_def_line():
    ir = _traced(axpy, *_axpy_args(), BLOCK=8)
    params = [op for op in ir.ops if op.name == "param"]
    assert len(params) == 5
    assert all(op.loc == (os.path.basename(__file__), _line("L-def"), 1) for op in params)


def test_loc_has_the_column():
    ir = _traced(axpy, *_axpy_args(), BLOCK=8)
    text = open(__file__).read().splitlines()
    for name, marker, code in [("load", "L-loadx", "sf.load("), ("store", "L-store", "sf.store(")]:
        op = next(o for o in ir.ops if o.name == name)
        assert op.loc[1] == _line(marker)
        assert op.loc[2] == text[op.loc[1] - 1].index(code) + 1      # 1-based, where the call starts


def test_dump_is_unchanged_by_locations():
    ir = _traced(axpy, *_axpy_args(), BLOCK=8)
    assert os.path.basename(__file__) not in ir.dump()


def test_every_op_has_a_loc_even_without_a_user_frame(monkeypatch):
    from sf.kernels import _ir
    monkeypatch.setattr(_ir, "_user_loc", lambda: None)
    ir = _traced(axpy, *_axpy_args(), BLOCK=8)
    assert all(op.loc == (os.path.basename(__file__), _line("L-def"), 1) for op in ir.ops)
    d = json.loads(ir.to_json())
    d["ops"][5]["loc"] = None
    with pytest.raises(sf.SFCompileError, match="has no loc"):
        _irjson.from_dict(d)


def test_loc_skips_only_sf_machinery():
    from sf.kernels import _ir
    sf_dir = _ir._SF_DIR
    assert _ir._is_machinery(os.path.join(sf_dir, "kernels", "_lang.py"))
    assert _ir._is_machinery(os.path.join(sf_dir, "_functions.py"))
    assert _ir._is_machinery(os.path.join(sf_dir, "__init__.py"))
    # a kernel written inside sf (as sf.torch's will be) keeps its own line
    assert not _ir._is_machinery(os.path.join(sf_dir, "torch", "_offload.py"))
    assert not _ir._is_machinery(__file__)


# --- writing ----------------------------------------------------------------------------------

def test_json_fields():
    d = json.loads(_traced(axpy, *_axpy_args(), BLOCK=8).to_json())
    assert d["format"] == "sfir" and d["format_version"] == 1
    assert d["kernel"] == "axpy" and d["style"] == "block" and d["block"] == [1, 1, 1]
    assert d["constexpr"] == {"BLOCK": 8} and "grid" not in d
    assert [p["name"] for p in d["params"]] == ["a", "x_ptr", "y_ptr", "out_ptr", "n"]   # no BLOCK here
    assert d["params"][1] == {"name": "x_ptr", "kind": "ptr", "type": "ptr<sf16>", "id": 1}
    fma = next(o for o in d["ops"] if o["op"] == "fma")
    assert fma["args"] == [0, 12, 14] and fma["type"] == "sf16" and fma["shape"] == [8]
    store = d["ops"][-1]
    assert store["op"] == "store" and store["id"] is None and store["type"] is None
    consts = [o for o in d["ops"] if o["op"] == "const" and o["attrs"]["kind"] == "sf"]
    assert consts and all(isinstance(o["attrs"]["value"], int) for o in consts)


def test_same_kernel_same_text_and_no_paths():
    a = _traced(axpy, *_axpy_args(), BLOCK=8).to_json()
    again = Kernel(axpy.fn, "block")                         # a fresh trace of the same function
    b = _traced(again, *_axpy_args(), BLOCK=8).to_json()
    assert a == b
    assert os.sep not in "".join(o["loc"]["file"] for o in json.loads(a)["ops"])
    assert ROOT not in a


def test_spec_version_matches_the_spec():
    spec = importlib.util.spec_from_file_location("spec_ops", os.path.join(ROOT, "spec", "ops.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert _irjson.SPEC_VERSION == mod.SPEC_VERSION


def test_every_op_kernel_covers_the_missing_ops():
    ir = _traced(every_op, *_every_args(), BLOCK=8)
    names = {op.name for op in ir.ops}
    assert {"abs", "bool_to_int", "cmp", "iabs", "imod", "ineg", "narrow", "neg", "or",
            "transpose", "xor"} <= names


def test_constexpr_is_written_in_one_place():
    text = _traced(axpy, *_axpy_args(), BLOCK=8).to_json()
    d = json.loads(text)
    assert d["constexpr"] == {"BLOCK": 8}
    assert all(p["kind"] in ("ptr", "sf", "int") for p in d["params"])
    assert text.count('"BLOCK"') == 1


@pytest.mark.parametrize("kernel,args", [(axpy, _axpy_args), (every_op, _every_args)])
def test_pointer_params_have_the_id_loads_and_stores_use(kernel, args):
    d = json.loads(_traced(kernel, *args(), BLOCK=8).to_json())
    param_ops = {o["id"]: o for o in d["ops"] if o["op"] == "param"}
    ptrs = {p["id"]: p for p in d["params"] if p["kind"] == "ptr"}
    assert ptrs
    for i, p in ptrs.items():                        # each pointer param is a real value
        assert param_ops[i]["attrs"] == {"param": p["name"], "kind": "ptr"}
        assert param_ops[i]["type"] == p["type"] and p["type"].startswith("ptr<")
    mem = [o for o in d["ops"] if o["op"] in ("load", "store")]
    assert mem
    for o in mem:                                    # and loads / stores name it as their first argument
        assert o["args"][0] in ptrs
    assert {o["args"][0] for o in mem} == set(ptrs)  # every buffer is used


# --- reading and the round trip ---------------------------------------------------------------

@pytest.mark.parametrize("kernel,args", [(axpy, _axpy_args), (every_op, _every_args)])
def test_round_trip_text(kernel, args):
    ir = _traced(kernel, *args(), BLOCK=8)
    back = KernelIR.from_json(ir.to_json())
    # JSON sorts keys, so compare attributes as dicts (their print order may differ)
    for a, b in zip(back.ops, ir.ops, strict=True):
        assert (a.name, a.attrs, repr(a.result), repr(a.pred)) == (b.name, b.attrs, repr(b.result), repr(b.pred))
        assert [repr(x) for x in a.args] == [repr(x) for x in b.args]
    assert back.shared == ir.shared and back.block_dim == ir.block_dim and back.constexpr == ir.constexpr
    assert back.to_json() == ir.to_json()
    assert [op.loc for op in back.ops] == [op.loc for op in ir.ops]
    # constexpr params come back after the runtime ones (only the map is in the file)
    assert ([p.__dict__ for p in back.params if p.kind != "constexpr"]
            == [p.__dict__ for p in ir.params if p.kind != "constexpr"])
    assert ({p.name: p.value for p in back.params if p.kind == "constexpr"}
            == {p.name: p.value for p in ir.params if p.kind == "constexpr"})


@pytest.mark.parametrize("kernel,args", [(axpy, _axpy_args), (every_op, _every_args)])
def test_round_trip_same_bits(kernel, args, monkeypatch):
    """A kernel read back from JSON computes exactly the same codes."""
    def outputs(a):
        return [x.codes.copy() for x in a if hasattr(x, "codes")]

    first = args()
    _traced(kernel, *first, BLOCK=8)
    orig = Kernel.trace
    monkeypatch.setattr(Kernel, "trace",
                        lambda self, *a, **k: KernelIR.from_json(orig(self, *a, **k).to_json()))
    second = args()
    _traced(kernel, *second, BLOCK=8)
    for p, q in zip(outputs(first), outputs(second)):
        assert np.array_equal(p, q)


def test_rejects_bad_files():
    d = json.loads(_traced(axpy, *_axpy_args(), BLOCK=8).to_json())
    for key, value, msg in [("format", "atreides-kernel", "format"),
                            ("format_version", 3, "format_version"),
                            ("spec_version", "2.0.0", "major")]:
        with pytest.raises(sf.SFCompileError, match=msg):
            _irjson.from_dict({**d, key: value})
    bad = json.loads(json.dumps(d))
    bad["ops"][7]["args"] = [99, 6]
    with pytest.raises(sf.SFCompileError, match="value 99"):
        _irjson.from_dict(bad)
    with pytest.raises(sf.SFCompileError, match="unknown type"):
        _irjson._parse_type("float32")
    bad = json.loads(json.dumps(d))
    bad["params"].append({"name": "BLOCK", "kind": "constexpr", "value": 8})
    with pytest.raises(sf.SFCompileError, match="top-level constexpr map"):
        _irjson.from_dict(bad)
    for change, msg in [(lambda x: x.update(grid=[1, 1, 1]), r"the file has unknown fields \['grid'\]"),
                        (lambda x: x.pop("shared"), r"missing fields \['shared'\]"),
                        (lambda x: x["ops"][3].update(note="hi"), r"op 3 has unknown fields \['note'\]"),
                        (lambda x: x["ops"][3].update(op="frobnicate"), "not an SF IR op"),
                        (lambda x: x["params"][0].update(dtype="sf16"), "unknown fields")]:
        bad = json.loads(json.dumps(d))
        change(bad)
        with pytest.raises(sf.SFCompileError, match=msg):
            _irjson.from_dict(bad)
    bad = json.loads(json.dumps(d))
    bad["params"][1]["id"] = 77
    with pytest.raises(sf.SFCompileError, match="no param op made"):
        _irjson.from_dict(bad)



def test_reader_knows_exactly_the_spec_ops():
    spec = json.loads(open(os.path.join(ROOT, "spec", "ops.json")).read())
    assert _irjson.OPS == {o["name"] for o in spec["ops"]}
    assert _irjson.FIELDS == set(spec["sfir"]["fields"]) and _irjson.OP_FIELDS == set(spec["sfir"]["op_fields"])
