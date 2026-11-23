"""Check an SF IR JSON file against spec/ops.json (its `sfir` section and each op's `ir` entry).

    python tools/check_sfir.py FILE.sfir.json [...]   # prints the problems, exit 1 if any

This is what an importer (superfloat-compile) is expected to check: the file
layout, the type names, each op's arg count, attributes and result type, and
that every id refers to an earlier result (inside a for op's body, also the
body's own values; outside it, not those).  It does not check shapes or
semantics; the reference interpreter does that.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1] / "spec" / "ops.json"


def load_spec(path=SPEC) -> dict:
    return json.loads(Path(path).read_text())


def _type_name(t: str, types: dict) -> str | None:
    """The spec type name ("sfN", "ptr<int>", ...) that a concrete type string matches."""
    for name, pattern in types.items():
        if re.fullmatch(pattern.strip("^$"), t):
            return name
    return None


def _attr_ok(v, rule: dict) -> bool:
    if "enum" in rule:
        return v in rule["enum"]
    t = rule["type"]
    if v is None:
        return bool(rule.get("null"))
    if t == "string":
        return isinstance(v, str)
    if t == "int[]":
        return isinstance(v, list) and all(isinstance(x, int) and not isinstance(x, bool) for x in v)
    if t == "int|bool":
        return isinstance(v, int)
    if t == "int":
        return (isinstance(v, int) and not isinstance(v, bool)
                and rule.get("min", v) <= v <= rule.get("max", v))
    raise ValueError(f"unknown attribute type {t!r} in the spec")


def _check_ops(ops_list, seen, where, version, ops, sfir, types, defined, errs) -> None:
    """Check one list of ops (the kernel's, or a for op's body); ``seen``: the ids they may use."""
    for n, o in enumerate(ops_list):
        here = f"op {n}{where} ({o.get('op')})"
        want = set(sfir["op_fields"]) | ({"body"} if o.get("op") == "for" else set())
        if set(o) != want:
            errs.append(f"{here}: fields {sorted(set(o) ^ want)} differ from the spec")
            if isinstance(o.get("id"), int):          # still defined: later uses are not new errors
                defined[o["id"]] = (o.get("type"), o.get("shape"))
                seen.add(o["id"])
            continue
        rule = ops.get(o["op"])
        if rule is None:
            errs.append(f"{here}: not an op in spec/ops.json")
            continue
        lo, hi = rule["args"]
        if not lo <= len(o["args"]) <= hi:
            errs.append(f"{here}: {len(o['args'])} args, the spec allows {lo} to {hi}")
        for a in o["args"] + ([o["pred"]] if o["pred"] is not None else []):
            if a not in seen:
                errs.append(f"{here}: refers to value {a}, which no earlier op made"
                            + (" (it was made inside a loop body)" if a in defined else ""))
        if set(o["attrs"]) != set(rule["attrs"]):
            errs.append(f"{here}: attrs {sorted(o['attrs'])}, the spec has {sorted(rule['attrs'])}")
        for k, v in o["attrs"].items():
            if k in rule["attrs"] and not _attr_ok(v, rule["attrs"][k]):
                errs.append(f"{here}: attr {k}={v!r} does not match {rule['attrs'][k]}")
        if o["id"] is None:
            if None not in rule["result"] or o["type"] is not None or o["shape"] is not None:
                errs.append(f"{here}: no result, but the spec gives it one (or type / shape is set)")
        else:
            name = _type_name(o["type"], types) if isinstance(o["type"], str) else None
            if name not in rule["result"]:
                errs.append(f"{here}: result type {o['type']!r}, the spec allows {rule['result']}")
            if o["id"] in defined:
                errs.append(f"{here}: value {o['id']} is defined twice")
            if not (isinstance(o["shape"], list) and len(o["shape"]) <= 3):
                errs.append(f"{here}: shape {o['shape']!r} is not a list of rank <= 3")
            defined[o["id"]] = (o["type"], o["shape"])
            seen.add(o["id"])
        loc = o["loc"]
        if not isinstance(loc, dict) or set(loc) != {"file", "line", "col"}:
            errs.append(f"{here}: loc {loc!r} is not {{file, line, col}}")
        elif "/" in loc["file"] or "\\" in loc["file"]:
            errs.append(f"{here}: loc file {loc['file']!r} is a path, not a base name")
        if o["op"] == "for":
            _check_for(o, here, seen, version, ops, sfir, types, defined, errs)


def _check_for(o, here, seen, version, ops, sfir, types, defined, errs) -> None:
    if version < 2:
        errs.append(f"{here}: a for op needs format_version 2")
    if o["attrs"].get("step") == 0:
        errs.append(f"{here}: step is 0")
    if o.get("pred") is not None:
        errs.append(f"{here}: a for op has no pred (its body ops carry any predicate)")
    B = o["body"]
    if not isinstance(B, dict) or set(B) != set(sfir["for_body"]):
        errs.append(f"{here}: body is not {{{', '.join(sorted(sfir['for_body']))}}}")
        return
    n = len(o["args"])
    if not all(isinstance(B[k], list) and len(B[k]) == n for k in ("carried", "yield", "results")):
        errs.append(f"{here}: {n} args, so carried, yield and results need {n} ids each")
        return
    init = [defined.get(a, (None, None)) for a in o["args"]]
    inner = set(seen)

    def define(i, ts, what):
        if isinstance(i, bool) or not isinstance(i, int):
            errs.append(f"{here}: {what} id {i!r} is not an int")
            return
        if i in defined:
            errs.append(f"{here}: value {i} is defined twice")
        defined[i] = ts
        return i

    for i in [define(B["iv"], ("int", []), "iv")] + [define(c, ts, "carried") for c, ts in zip(B["carried"], init)]:
        if i is not None:
            inner.add(i)
    _check_ops(B["ops"], inner, f" in the body of {here}", version, ops, sfir, types, defined, errs)
    for y, ts in zip(B["yield"], init):
        if y not in inner:
            errs.append(f"{here}: yield {y}, which the body cannot see")
        elif defined.get(y) != ts:
            errs.append(f"{here}: yield {y} is {defined.get(y)}, the carried value is {ts}")
    for r, ts in zip(B["results"], init):
        if define(r, ts, "results") is not None:
            seen.add(r)


def check(d: dict, spec: dict | None = None) -> list[str]:
    """Every problem found in the SF IR dict ``d`` (empty when it passes)."""
    spec = spec or load_spec()
    sfir, ops = spec["sfir"], {o["name"]: o["ir"] for o in spec["ops"]}
    types = sfir["types"]
    errs = []
    version = d.get("format_version")
    if d.get("format") != sfir["format"] or isinstance(version, bool) or version not in sfir["format_versions"]:
        return [f"not SF IR format {sfir['format']} version {' or '.join(map(str, sfir['format_versions']))}"]
    if set(d) != set(sfir["fields"]):
        errs.append(f"top-level fields {sorted(set(d) ^ set(sfir['fields']))} differ from the spec")
    if str(d.get("spec_version", "")).split(".")[0] != spec["spec_version"].split(".")[0]:
        errs.append(f"spec_version {d.get('spec_version')!r}: different major version from {spec['spec_version']}")
    defined: dict[int, tuple] = {}         # every value in the file: (type, shape)
    if d.get("ops") is not None:
        _check_ops(d["ops"], set(), "", version, ops, sfir, types, defined, errs)
    param_ids = {o.get("id"): o for o in d.get("ops", []) if o.get("op") == "param"}
    for p in d.get("params", []):
        po = param_ids.get(p.get("id"))
        if p.get("kind") not in ("ptr", "sf", "int"):
            errs.append(f"param {p.get('name')!r}: kind {p.get('kind')!r} (constexpr belongs in the constexpr map)")
        elif po is None or po["attrs"] != {"param": p["name"], "kind": p["kind"]} or po["type"] != p.get("type"):
            errs.append(f"param {p.get('name')!r}: no matching param op with id {p.get('id')}")
    for s in d.get("shared", []):
        if _type_name(s.get("type", ""), types) != "sfN":
            errs.append(f"shared buffer {s.get('id')}: type {s.get('type')!r} is not sfN")
    return errs


def main(argv) -> int:
    spec = load_spec()
    bad = 0
    for f in argv:
        errs = check(json.loads(Path(f).read_text()), spec)
        bad += bool(errs)
        print(f"{f}: " + ("ok" if not errs else f"{len(errs)} problem(s)\n  " + "\n  ".join(errs)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
