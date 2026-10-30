"""Generate the lowering spec outputs from spec/ops.py.

    python tools/gen_spec.py          # write spec/ops.json, spec/golden/*.json, website/compiler/*.mdx
    python tools/gen_spec.py --check  # fail if any generated file is out of date

Golden vectors come from the exact-fraction scalar oracle (sf._reference), so
they are independent of the vectorized kernels the interpreter and C backend use.
"""
from __future__ import annotations

import json
import random
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from sf import _reference as ref          # noqa: E402
from sf._format import FORMATS            # noqa: E402
from spec.ops import (CHANGELOG, EVENTS, GLOBAL_RULES, OPS, PREDICATION, PUBLIC,  # noqa: E402
                      SPEC_VERSION, TYPES)

WIDTHS = (16, 8, 4)


# --- golden vectors ------------------------------------------------------------------------------

def _edge(fmt):
    M = fmt.max_code
    h = 1 << (fmt.bits - 2)                       # code of 0.5
    s = {0, 1, -1, 2, -2, M, -M, M - 1, -(M - 1), h, -h, h + 1, -(h + 1), h - 1}
    return sorted(k for k in s if -M <= k <= M)


def _rand(fmt, n, seed):
    r = random.Random(seed)
    return [r.randint(-fmt.max_code, fmt.max_code) for _ in range(n)]


def _ev(e):
    return e if e else None


def golden_for(name: str) -> dict | None:
    cases = []
    for bits in WIDTHS:
        f = FORMATS[bits]
        E = _edge(f)
        R = _rand(f, 24, bits)
        pairs = [(a, b) for a in E for b in E] + list(zip(R, _rand(f, 24, bits + 100)))
        if name in ("add", "sub", "mul", "div"):
            fn = getattr(ref, name)
            for a, b in pairs:
                k, e = fn(f, a, b)
                cases.append({"bits": bits, "in": [a, b], "out": k, "event": _ev(e)})
        elif name == "fma":
            for (a, b), c in zip(pairs, E * len(pairs)):
                k, e = ref.fma(f, a, b, c)
                cases.append({"bits": bits, "in": [a, b, c], "out": k, "event": _ev(e)})
        elif name in ("neg", "abs", "sqrt"):
            fn = {"neg": ref.neg, "abs": ref.abs_, "sqrt": ref.sqrt}[name]
            for a in E + R:
                k, e = fn(f, a)
                cases.append({"bits": bits, "in": [a], "out": k, "event": _ev(e)})
        elif name in ("narrow", "widen"):
            for dst in WIDTHS:
                if (name == "narrow" and dst >= bits) or (name == "widen" and dst <= bits):
                    continue
                g = FORMATS[dst]
                for a in E + R:
                    if name == "widen":
                        k, e = a << (dst - bits), None
                    else:
                        k, e = ref.cast_exact(g, Fraction(a, 1 << (bits - 1)))
                    cases.append({"bits": bits, "to_bits": dst, "in": [a], "out": k, "event": _ev(e)})
        elif name == "sum":
            r = random.Random(bits)
            vecs = [[f.max_code, 1, -1], [1, -1, f.max_code], [f.max_code, f.max_code, -f.max_code]]
            vecs += [[r.randint(-f.max_code, f.max_code) for _ in range(r.randint(1, 9))] for _ in range(20)]
            for v in vecs:
                acc, events = 0, set()
                for x in v:
                    acc, e = ref.add(f, acc, x)
                    if e:
                        events.add(e)
                cases.append({"bits": bits, "in": [v], "out": acc, "events": sorted(events)})
        elif name == "dot":
            r = random.Random(bits + 7)
            for _ in range(20):
                n = r.randint(1, 9)
                a = [r.randint(-f.max_code, f.max_code) for _ in range(n)]
                b = [r.randint(-f.max_code, f.max_code) for _ in range(n)]
                acc, events = 0, set()
                for x, y in zip(a, b):
                    acc, e = ref.fma(f, x, y, acc)
                    if e:
                        events.add(e)
                cases.append({"bits": bits, "in": [a, b], "out": acc, "events": sorted(events),
                              "note": "one (row, column) of dot: k in increasing order"})
        else:
            return None
    return {"op": name, "spec_version": SPEC_VERSION,
            "format": "codes are signed ints k, value k / 2^(bits-1); event is the single event of the scalar op "
                      "(null if none); events lists every event of a chain",
            "cases": cases}


# --- pages ---------------------------------------------------------------------------------------

def _esc(s: str) -> str:
    return str(s).replace("|", "\\|").replace("<", "&lt;").replace("{", "&#123;").replace("}", "&#125;")


def _code(s: str) -> str:
    return "`" + str(s).replace("|", "\\|") + "`"


def page_overview() -> str:
    L = ["---", 'title: "Lowering spec"',
         'description: "Everything the compiler needs to lower every Superfloat op: types, events, global rules, and where the machine-readable spec lives."',
         "---", "", f"**Spec version {SPEC_VERSION}.** Generated from `spec/ops.py` by `tools/gen_spec.py`; do not edit by hand.", "",
         "This section is written for the compiler (MLIR) side. The Python reference is the ground truth: if the compiled "
         "code gives different bits from the reference on any input, the compiler is wrong (or the spec is, and we fix it).", "",
         "## What you get", "",
         "| File | Use |", "|---|---|",
         "| `spec/ops.json` | Every IR op as JSON: operands, attributes, result, shape rule, semantics, events, memory, faults, predication, v1 hardware unit, suggested MLIR op and traits, legal rewrites. Generate your op definitions or tests from it. |",
         "| `spec/golden/<op>.json` | Conformance vectors per op at sf16, sf8 and sf4 (edge codes and random), from the exact-fraction oracle. Run your lowered op on `in`, compare `out` and the events bit for bit. |",
         "| [IR ops](/compiler/ops) | The same content as readable pages. |",
         "| [Python API map](/compiler/python-api) | Which IR op each public Python function becomes, and where it runs. |",
         "| [Versioning](/compiler/versioning) | How versions change and the changelog. |", "",
         "## Global rules", ""]
    L += [f"{i}. {_esc(r)}" for i, r in enumerate(GLOBAL_RULES, 1)]
    L += ["", "## Types", "", "| Type | Meaning |", "|---|---|"]
    L += [f"| {_code(k)} | {_esc(v)} |" for k, v in TYPES.items()]
    L += ["", "## Events", "", "Catalogue order (the order the host applies the policy): invalid, divide, overflow, underflow.", "",
          "| Event | Meaning and result value |", "|---|---|"]
    L += [f"| {_code(k)} | {_esc(v)} |" for k, v in EVENTS.items()]
    L += ["", "Events are per lane flags. The kernel returns a count per event (live lanes only). On v1 there is no flag register, "
          "so the runtime may replay the kernel on the host to get exact counts (`config(status=\"replay\")`).", "",
          "## Predication classes", "", "| Class | Rule |", "|---|---|"]
    L += [f"| {_code(k)} | {_esc(v)} |" for k, v in PREDICATION.items()]
    L += ["", "## Golden vector format", "", "```json",
          '{"op": "add", "cases": [{"bits": 16, "in": [32767, 1], "out": 32767, "event": "overflow"}]}',
          "```", "", "`narrow` and `widen` cases also carry `to_bits`. `sum` and `dot` cases carry `events` (a list) instead of `event`.", ""]
    return "\n".join(L)


CATS = [("sf", "SF arithmetic"), ("cast", "Casts"), ("reduction", "Reductions"), ("contraction", "Contraction"),
        ("select", "Selection"), ("memory", "Memory"), ("sync", "Synchronisation"), ("index", "Indices and parameters"),
        ("shape", "Shape"), ("int", "Integer"), ("bool", "Boolean")]


def page_ops() -> str:
    L = ["---", 'title: "IR ops"', f'description: "Every SF IR op, one section each. Spec version {SPEC_VERSION}."', "---", "",
         "Each op lists its operands, attributes, result, shape rule, exact semantics on integer codes, events, memory "
         "effects, faults, predication, the v1 hardware unit, a suggested MLIR op with traits, and the rewrites that are and "
         "are not legal. Ops with golden vectors say so.", ""]
    for cat, title in CATS:
        ops = [o for o in OPS if o["category"] == cat]
        if not ops:
            continue
        L += [f"## {title}", ""]
        for o in ops:
            L += [f"### {o['name']}", "", f"`{o['signature']}` · version {o['version']}", "", "| | |", "|---|---|"]
            L.append(f"| Operands | {_esc(', '.join(o['operands']) or 'none')} |")
            if o["attrs"]:
                L.append("| Attributes | " + _esc("; ".join(f"{k}: {v}" for k, v in o["attrs"].items())) + " |")
            L.append(f"| Result | {_esc(o['result'])} |")
            L.append(f"| Shape | {_esc(o['shape'])} |")
            L.append(f"| Semantics | {_esc(o['semantics'])} |")
            L.append(f"| Events | {_esc(', '.join(o['events']) or 'none')} |")
            L.append(f"| Memory | {_esc(o['memory'])} |")
            if o["faults"]:
                L.append(f"| Faults | {_esc(' '.join(o['faults']))} |")
            L.append(f"| Predication | {o['predication']} |")
            L.append(f"| v1 hardware | {_esc(o['hw_v1'])} |")
            L.append(f"| MLIR | {_esc(o['mlir']['name'])}; traits: {_esc(', '.join(o['mlir']['traits']) or 'none')} |")
            L.append(f"| From Python | {_esc(', '.join(o['python']))} |")
            if (ROOT / "spec" / "golden" / f"{o['name']}.json").exists() or golden_for(o["name"]) is not None:
                L.append(f"| Golden vectors | `spec/golden/{o['name']}.json` |")
            L.append("")
            if o["rewrites"]:
                L += ["**Rewrites:**", ""] + [f"- {_esc(r)}" for r in o["rewrites"]] + [""]
            if o["notes"]:
                L += ["**Notes:**", ""] + [f"- {_esc(n)}" for n in o["notes"]] + [""]
    return "\n".join(L)


def page_api() -> str:
    L = ["---", 'title: "Python API map"', 'description: "Which IR op each public Python function becomes, and where it runs."', "---", "",
         "`device`: a heavy op, runs on the target when an operand is on a device and returns a status. `host`: always runs "
         "on the host. `trace`: only inside `@sf.jit` / `@sf.kernel`, becomes IR.", "",
         "| Python | Runs | Lowers to | Notes |", "|---|---|---|---|"]
    L += [f"| {_code(n)} | {_esc(p)} | {_esc(t)} | {_esc(x)} |" for n, p, t, x in PUBLIC]
    L.append("")
    return "\n".join(L)


def page_versioning() -> str:
    L = ["---", 'title: "Versioning"', 'description: "How the spec version changes, and what changed in each version."', "---", "",
         f"Current version: **{SPEC_VERSION}** (MAJOR.MINOR.PATCH).", "",
         "| Change | Bump | Example |", "|---|---|---|",
         "| Add an op or a function, or change what one does (its contract: bits, events, shape, memory) | minor | 1.1 to 1.2 |",
         "| Edit an existing entry without changing the contract (clearer text, a new edge case written down, a fix in the docs) | patch | 1.1.1 to 1.1.2 |",
         "| Change the whole numeric model (for example the v4.8.1 wide accumulator) | major | 1.x to 2.0 |", "",
         "Each op carries the version in which its contract last changed. If your lowering was written against an op at "
         "version 1.1 and the op now says 1.2, look at it again. A patch bump never needs a compiler change.", "",
         "## Changelog", "", "| Version | Date | Change |", "|---|---|---|"]
    L += [f"| {v} | {d} | {_esc(c)} |" for v, d, c in CHANGELOG]
    L.append("")
    return "\n".join(L)


def outputs() -> dict[Path, str]:
    out = {ROOT / "spec" / "ops.json": json.dumps(
        {"spec_version": SPEC_VERSION, "changelog": CHANGELOG, "global_rules": GLOBAL_RULES, "types": TYPES,
         "events": EVENTS, "predication": PREDICATION, "ops": OPS,
         "python_api": [dict(zip(("python", "runs", "lowers_to", "notes"), p)) for p in PUBLIC]}, indent=1) + "\n"}
    for o in OPS:
        g = golden_for(o["name"])
        if g is not None:
            out[ROOT / "spec" / "golden" / f"{o['name']}.json"] = json.dumps(g, separators=(",", ":")) + "\n"
    w = ROOT / "website" / "compiler"
    out[w / "overview.mdx"] = page_overview()
    out[w / "ops.mdx"] = page_ops()
    out[w / "python-api.mdx"] = page_api()
    out[w / "versioning.mdx"] = page_versioning()
    return out


def main(argv):
    check = "--check" in argv
    stale = []
    for p, text in outputs().items():
        if check:
            if not p.exists() or p.read_text() != text:
                stale.append(str(p.relative_to(ROOT)))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text)
    if stale:
        print("out of date, run python tools/gen_spec.py:\n  " + "\n  ".join(stale))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
