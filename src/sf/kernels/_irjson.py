"""SF IR as JSON: the file the compiler reads.

``to_dict`` / ``dumps`` write a traced ``KernelIR``; ``from_dict`` / ``loads``
read one back.  The layout (format "sfir", version 1):

    {"format": "sfir", "format_version": 1, "spec_version": "1.10.1",
     "kernel": "axpy_b", "style": "block", "block": [1, 1, 1],
     "constexpr": {"BLOCK": 8},
     "params": [{"name": "a", "kind": "sf", "type": "sf16", "id": 0},
                {"name": "x_ptr", "kind": "ptr", "type": "ptr<sf16>", "id": 1}, ...],
     "shared": [{"id": 0, "type": "sf16", "shape": [8, 8]}],
     "ops": [{"id": 15, "op": "fma", "args": [0, 12, 14], "attrs": {},
              "type": "sf16", "shape": [8], "pred": null,
              "loc": {"file": "axpy.py", "line": 8, "col": 5}}, ...]}

Rules:

* ``params`` lists the runtime arguments only (kinds ptr, sf, int), in
  signature order.  Constexpr values are only in the top-level
  ``constexpr`` map.
* Every param, pointers included, has a ``param`` op whose result id is the
  param's ``id``.  For a pointer that value has type ``ptr<...>`` and is the
  first argument of every ``load`` and ``store`` on that buffer.
* ``args`` are the ids of earlier values, in the operand order of
  ``spec/ops.json``.  Ops with no result (``store``, ``shared_store``,
  ``barrier``) have ``id``, ``type`` and ``shape`` null.
* ``pred`` is the id of the mask in force, or null.
* ``loc`` is the file's base name, line and 1-based column (null on Python
  3.10), never a full path.  Every op has one, including ops sf adds itself:
  they point at the user's code that caused them.  Param ops point at the
  kernel's ``def``.
* SF constants are integer codes (``const`` with ``kind`` "sf"), never floats.
* Types are ``sf<N>``, ``int`` (int64 here, 16 bits on the chip), ``bool``,
  ``ptr<sf<N>>`` and ``ptr<int>`` (a read-only int table, e.g. per-output
  shifts).
* Tuple attributes (``shape`` on ``broadcast`` / ``reshape``) are lists in
  JSON and tuples again after reading.
* The grid is not in the file: it changes with every launch, and the file is
  hashed as the compile cache key.

The same kernel always gives the same text (sorted keys, no timestamps, no
paths), so ``dumps`` is safe to hash.

Per op, the number of args, each attribute's JSON type and the possible result
types are in spec/ops.json (each op's ``ir`` entry; the file layout is its
``sfir`` section).  ``tools/check_sfir.py FILE`` checks a file against them.

Planned for format_version 2: a ``for`` op (constant trip count, nested body)
so loops are not unrolled in the file.
"""
from __future__ import annotations

import json
import re

import numpy as np

from .._errors import SFCompileError
from ._ir import BOOL, INT, KernelIR, Op, Param, PtrType, SFType, SharedBuf, Value

FORMAT = "sfir"
FORMAT_VERSION = 1
SPEC_VERSION = "1.10.1"          # must equal spec/ops.py (checked by tests/test_sfir_json.py)

# every op the reader accepts (equal to spec/ops.json's ops; tests/test_sfir_json.py checks it)
OPS = frozenset({
    "abs", "add", "and", "arange", "barrier", "bool_to_int", "broadcast", "cmp",
    "cmp_lit", "const", "div", "dot", "dot_wide", "fma", "iabs", "iadd",
    "icmp", "idiv", "imod", "imul", "ineg", "isub", "load", "max",
    "maximum", "min", "minimum", "mul", "narrow", "neg", "not", "or",
    "param", "program_id", "reshape", "shared_load", "shared_store", "sqrt", "store", "sub",
    "sum", "thread_idx", "transpose", "where", "widen", "xor",
})
FIELDS = frozenset({"format", "format_version", "spec_version", "kernel", "style", "block", "constexpr",
                    "params", "shared", "ops"})
OP_FIELDS = frozenset({"id", "op", "args", "attrs", "type", "shape", "pred", "loc"})
PARAM_FIELDS = frozenset({"name", "kind", "type", "id"})
LOC_FIELDS = frozenset({"file", "line", "col"})

_PTR = re.compile(r"ptr<(sf(\d+)|int)>")
_SF = re.compile(r"sf(\d+)")


# --- writing ----------------------------------------------------------------------------------

def _type_str(t) -> str | None:
    return None if t is None else str(t)


def _plain(x):
    """A JSON-ready copy of an attribute or constexpr value."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (tuple, list)):
        return [_plain(v) for v in x]
    if x is None or isinstance(x, str):
        return x
    raise SFCompileError(f"SF IR JSON: cannot write a value of type {type(x).__name__}: {x!r}")


def _arg_id(op: Op, a) -> int:
    if not isinstance(a, Value):
        raise SFCompileError(f"SF IR JSON: op {op.name} has a non-value argument {a!r}")
    return a.id


def to_dict(ir: KernelIR) -> dict:
    """The JSON-ready dict for a traced kernel."""
    ops = []
    for op in ir.ops:
        r = op.result
        ops.append({
            "id": r.id if r is not None else None,
            "op": op.name,
            "args": [_arg_id(op, a) for a in op.args],
            "attrs": {k: _plain(v) for k, v in op.attrs.items()},
            "type": _type_str(r.type) if r is not None else None,
            "shape": list(r.shape) if r is not None else None,
            "pred": op.pred.id if op.pred is not None else None,
            "loc": {"file": op.loc[0], "line": op.loc[1], "col": op.loc[2]} if op.loc is not None else None,
        })
    # constexpr values live only in the top-level map, not here (one place, Kartik's review)
    params = [{"name": p.name, "kind": p.kind, "type": _type_str(p.type), "id": p.vid}
              for p in ir.params if p.kind != "constexpr"]
    return {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        "spec_version": SPEC_VERSION,
        "kernel": ir.name,
        "style": ir.style,
        "block": [int(b) for b in ir.block_dim],
        "constexpr": {k: _plain(v) for k, v in ir.constexpr.items()},
        "params": params,
        "shared": [{"id": s.id, "type": f"sf{s.bits}", "shape": list(s.shape)} for s in ir.shared],
        "ops": ops,
    }


def dumps(ir: KernelIR) -> str:
    """The kernel as JSON text: the same kernel always gives the same text."""
    return json.dumps(to_dict(ir), sort_keys=True, indent=1) + "\n"


# --- reading ----------------------------------------------------------------------------------

def _parse_type(s: str | None):
    if s is None:
        return None
    if s == "int":
        return INT
    if s == "bool":
        return BOOL
    m = _PTR.fullmatch(s)
    if m:
        return PtrType(int(m.group(2)) if m.group(2) else 0)
    m = _SF.fullmatch(s)
    if m:
        return SFType(int(m.group(1)))
    raise SFCompileError(f"SF IR JSON: unknown type {s!r}")


def _tuples(x):
    """JSON lists back to the tuples the IR uses."""
    return tuple(_tuples(v) for v in x) if isinstance(x, list) else x


def _exact_fields(obj: dict, want: frozenset, what: str) -> None:
    """Unknown or missing fields are errors: a reader never guesses (Kartik, 2026-11-22)."""
    got = set(obj)
    if got != want:
        extra, missing = sorted(got - want), sorted(want - got)
        raise SFCompileError(f"SF IR JSON: {what} has " + "; ".join(
            x for x in (f"unknown fields {extra}" if extra else "", f"missing fields {missing}" if missing else "")
            if x))


def from_dict(d: dict) -> KernelIR:
    """Rebuild a ``KernelIR`` from ``to_dict`` output."""
    if d.get("format") != FORMAT:
        raise SFCompileError(f"SF IR JSON: format is {d.get('format')!r}, not {FORMAT!r}")
    if d.get("format_version") != FORMAT_VERSION:
        raise SFCompileError(f"SF IR JSON: format_version {d.get('format_version')!r} is not "
                             f"{FORMAT_VERSION}, the one this sf reads")
    major = str(d.get("spec_version", "")).split(".")[0]
    if major != SPEC_VERSION.split(".")[0]:
        raise SFCompileError(f"SF IR JSON: spec_version {d.get('spec_version')!r} has a different major "
                             f"version than this sf's {SPEC_VERSION}")
    _exact_fields(d, FIELDS, "the file")
    ir = KernelIR(d["kernel"], d["style"])
    ir.block_dim = tuple(d["block"])
    ir.constexpr = {k: _tuples(v) for k, v in d["constexpr"].items()}
    values: dict[int, Value] = {}

    def ref(i, what):
        if i not in values:
            raise SFCompileError(f"SF IR JSON: {what} refers to value {i}, which no earlier op made")
        return values[i]

    for n, o in enumerate(d["ops"]):
        _exact_fields(o, OP_FIELDS, f"op {n}")
        if o["op"] not in OPS:
            raise SFCompileError(f"SF IR JSON: op {n} is {o['op']!r}, which is not an SF IR op")
        args = [ref(i, o["op"]) for i in o["args"]]
        pred = ref(o["pred"], o["op"]) if o["pred"] is not None else None
        result = None
        if o["id"] is not None:
            if o["id"] in values:
                raise SFCompileError(f"SF IR JSON: value {o['id']} is defined twice")
            result = Value(o["id"], _parse_type(o["type"]), tuple(o["shape"]))
            values[o["id"]] = result
        loc = None
        if o["loc"] is not None:
            _exact_fields(o["loc"], LOC_FIELDS, f"op {n} loc")
            loc = (o["loc"]["file"], o["loc"]["line"], o["loc"]["col"])
        attrs = {k: _tuples(v) for k, v in o["attrs"].items()}
        ir.ops.append(Op(o["op"], args, attrs, result, pred, loc))
    ir.values = [values[i] for i in sorted(values)]
    for p in d["params"]:
        if p.get("kind") == "constexpr":
            raise SFCompileError(f"SF IR JSON: param {p.get('name')!r} has kind 'constexpr'; constexpr "
                                 "values belong in the top-level constexpr map")
        _exact_fields(p, PARAM_FIELDS, f"param {p.get('name')!r}")
        if p["kind"] not in ("ptr", "sf", "int"):
            raise SFCompileError(f"SF IR JSON: param {p['name']!r} has kind {p['kind']!r}; constexpr "
                                 "values belong in the top-level constexpr map")
        if p["id"] not in values:
            raise SFCompileError(f"SF IR JSON: param {p['name']!r} has id {p['id']}, which no param op made")
        ir.params.append(Param(p["name"], p["kind"], _parse_type(p["type"]), vid=p["id"]))
    for name, value in ir.constexpr.items():       # after the runtime params (signature order is not kept)
        ir.params.append(Param(name, "constexpr", value=value))
    for s in d["shared"]:
        bits = _parse_type(s["type"]).bits
        ir.shared.append(SharedBuf(s["id"], bits, tuple(s["shape"])))
    return ir


def loads(text: str) -> KernelIR:
    return from_dict(json.loads(text))
