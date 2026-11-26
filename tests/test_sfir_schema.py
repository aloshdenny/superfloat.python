"""SF IR JSON against the spec: every op sf writes has exactly the args, attributes
and result type that its `ir` entry in spec/ops.json allows (tools/check_sfir.py).

``SF_IR_ROUNDTRIP=1 pytest`` also runs this check on every kernel the whole suite
traces (see conftest.py).
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from check_sfir import check, load_spec  # noqa: E402
from test_sfir_json import _axpy_args, _every_args, _traced, axpy, every_op  # noqa: E402

SPEC = load_spec()
EXAMPLES = sorted((ROOT / "examples" / "sfir").glob("*/kernel.sfir.json"))


def _file(kernel, args):
    return json.loads(_traced(kernel, *args(), BLOCK=8).to_json())


@pytest.mark.parametrize("kernel,args", [(axpy, _axpy_args), (every_op, _every_args)])
def test_traced_kernels_pass(kernel, args):
    assert check(_file(kernel, args), SPEC) == []


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.parent.name)
def test_examples_pass(path):
    assert check(json.loads(path.read_text()), SPEC) == []


def test_every_spec_op_has_an_ir_entry():
    for o in SPEC["ops"]:
        ir = o["ir"]
        assert set(ir) == {"args", "attrs", "result"} and set(ir["attrs"]) == set(o["attrs"])
        assert all(t is None or t in SPEC["sfir"]["types"] for t in ir["result"])


def test_sfir_section_matches_the_writer():
    from sf.kernels import _irjson
    d = _file(axpy, _axpy_args)
    assert set(d) == set(SPEC["sfir"]["fields"])
    assert all(set(o) == set(SPEC["sfir"]["op_fields"]) for o in d["ops"])
    assert SPEC["sfir"]["format_version"] == _irjson.FORMAT_VERSION and SPEC["sfir"]["format"] == _irjson.FORMAT


def _broken(change):
    d = copy.deepcopy(_file(axpy, _axpy_args))
    change(d)
    return check(d, SPEC)


def _op(d, name):
    return next(o for o in d["ops"] if o["op"] == name)


@pytest.mark.parametrize("change,msg", [
    (lambda d: _op(d, "arange").update(op="frobnicate"), "not an op"),
    (lambda d: _op(d, "arange")["attrs"].update(step=1), "attrs"),
    (lambda d: _op(d, "program_id")["attrs"].update(axis=3), "axis=3"),
    (lambda d: _op(d, "fma")["args"].pop(), "2 args"),
    (lambda d: _op(d, "fma").update(type="int"), "result type 'int'"),
    (lambda d: _op(d, "fma").update(type="float32"), "float32"),
    (lambda d: _op(d, "fma")["args"].__setitem__(0, 999), "value 999"),
    (lambda d: _op(d, "store").update(type="sf16"), "no result"),
    (lambda d: _op(d, "fma").update(loc={"file": "/home/me/axpy.py", "line": 3, "col": 1}), "a path"),
    (lambda d: d["params"].append({"name": "BLOCK", "kind": "constexpr", "value": 8}), "constexpr map"),
    (lambda d: d["params"][1].update(id=0), "no matching param op"),
    (lambda d: d.update(grid=[3, 1, 1]), "top-level fields"),
])
def test_checker_finds_broken_files(change, msg):
    errs = _broken(change)
    assert any(msg in e for e in errs), errs


def test_one_bad_op_gives_one_error():
    errs = _broken(lambda d: _op(d, "arange").update(comment="x"))    # later ops use its id
    assert len(errs) == 1 and "fields" in errs[0], errs
