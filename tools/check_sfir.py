"""Check an SF IR JSON file against spec/ops.json (its `sfir` section and each op's `ir` entry).

    python tools/check_sfir.py FILE.sfir.json [...]   # prints the problems, exit 1 if any

This is what an importer (superfloat-compile) is expected to check: the file
layout, the type names, each op's arg count, attributes and result type, and
that every id refers to an earlier result.  It does not check shapes or
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


def check(d: dict, spec: dict | None = None) -> list[str]:
    """Every problem found in the SF IR dict ``d`` (empty when it passes)."""
    spec = spec or load_spec()
    sfir, ops = spec["sfir"], {o["name"]: o["ir"] for o in spec["ops"]}
    types = sfir["types"]
    errs = []
    if d.get("format") != sfir["format"] or d.get("format_version") != sfir["format_version"]:
        return [f"not SF IR format {sfir['format']} version {sfir['format_version']}"]
    if set(d) != set(sfir["fields"]):
        errs.append(f"top-level fields {sorted(set(d) ^ set(sfir['fields']))} differ from the spec")
    if str(d.get("spec_version", "")).split(".")[0] != spec["spec_version"].split(".")[0]:
        errs.append(f"spec_version {d.get('spec_version')!r}: different major version from {spec['spec_version']}")
    defined: dict[int, str] = {}
    for n, o in enumerate(d.get("ops", [])):
        where = f"op {n} ({o.get('op')})"
        if set(o) != set(sfir["op_fields"]):
            errs.append(f"{where}: fields {sorted(set(o) ^ set(sfir['op_fields']))} differ from the spec")
            if isinstance(o.get("id"), int):          # still defined: later uses are not new errors
                defined[o["id"]] = o.get("type")
            continue
        rule = ops.get(o["op"])
        if rule is None:
            errs.append(f"{where}: not an op in spec/ops.json")
            continue
        lo, hi = rule["args"]
        if not lo <= len(o["args"]) <= hi:
            errs.append(f"{where}: {len(o['args'])} args, the spec allows {lo} to {hi}")
        for a in o["args"] + ([o["pred"]] if o["pred"] is not None else []):
            if a not in defined:
                errs.append(f"{where}: refers to value {a}, which no earlier op made")
        if set(o["attrs"]) != set(rule["attrs"]):
            errs.append(f"{where}: attrs {sorted(o['attrs'])}, the spec has {sorted(rule['attrs'])}")
        for k, v in o["attrs"].items():
            if k in rule["attrs"] and not _attr_ok(v, rule["attrs"][k]):
                errs.append(f"{where}: attr {k}={v!r} does not match {rule['attrs'][k]}")
        if o["id"] is None:
            if None not in rule["result"] or o["type"] is not None or o["shape"] is not None:
                errs.append(f"{where}: no result, but the spec gives it one (or type / shape is set)")
        else:
            name = _type_name(o["type"], types) if isinstance(o["type"], str) else None
            if name not in rule["result"]:
                errs.append(f"{where}: result type {o['type']!r}, the spec allows {rule['result']}")
            if o["id"] in defined:
                errs.append(f"{where}: value {o['id']} is defined twice")
            if not (isinstance(o["shape"], list) and len(o["shape"]) <= 3):
                errs.append(f"{where}: shape {o['shape']!r} is not a list of rank <= 3")
            defined[o["id"]] = o["type"]
        if o["loc"] is not None and ("/" in o["loc"]["file"] or "\\" in o["loc"]["file"]):
            errs.append(f"{where}: loc file {o['loc']['file']!r} is a path, not a base name")
    param_ids = {o["id"]: o for o in d.get("ops", []) if o.get("op") == "param"}
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
