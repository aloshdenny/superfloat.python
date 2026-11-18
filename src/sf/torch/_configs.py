"""The configuration chooser of Alosh's golden file (``sf_golden_v4.9.2.json``, ``heuristic.configuration``):
for a layer's shape and the user's preferences, the ranked configurations (operand width, packed or not, redo
policy) with their cycles, redo rate and bits kept.  ``tests/test_configs.py`` reproduces all 56 cases exactly.

What is read from the file, as given:

* cycles per MAC on target v4.9.2, for one token (GEMV) or many (GEMM), plain 16-bit words or weights packed
  (sf8, sf10, sf12): every cycle count in the 56 cases is ``round(MACs * c)`` with the ``tbl2`` redo policy
  costing its lower redo rate on top, ``round(MACs * c * 1.066)``;
* the redo rate of ``tbl2``, [0.066, 0.323], and of ``l2`` (redo never: every output popped at l2), 0;
* bits kept: the operand's bits minus 4.2 for ``tbl2``, minus h(K) and then 4.2 for ``l2`` (h from the exponent
  rule; in that order, so the floats match the file's).

What is inferred from the 56 cases (no rule is written in the file; each holds on every case):

* the LM head and grouped convs run at sf16, plain words, whatever the width limits;
* ``l2`` has no measured evidence; ``tbl2`` is measured at every width for ``model="other"`` and from sf12 up
  for ``"llm"`` and ``"any"`` (below that, unmeasured);
* a configuration keeping under 1 bit is lossy;
* ``deterministic_cycles`` needs a redo rate of 0;
* speed ranks by cycles, then bits kept; accuracy by bits kept, then cycles; the top 3 are returned.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .. import _heuristic

TARGETS = {
    "v4.9.2": {
        "cycles_per_mac": {("gemv", None): 9.075, ("gemv", 12): 7.581, ("gemv", 10): 6.639, ("gemv", 8): 5.706,
                           ("gemm", None): 2.867, ("gemm", 12): 2.684, ("gemm", 10): 2.539, ("gemm", 8): 2.399},
        "relaunch": {"tbl2": (0.066, 0.323), "l2": (0.0, 0.0)},
        "tbl2_bits_lost": 4.2,
        "launch": "8x32",
        "measured_from": {"llm": 12, "any": 12, "other": 8},
    },
}
KINDS = ("linear", "conv", "grouped", "head")


@dataclass(frozen=True)
class LayerDesc:
    M: int              # tokens (rows of the activation)
    N: int              # outputs
    K: int              # reduction length
    kind: str = "linear"


@dataclass(frozen=True)
class Preferences:
    priority: str = "speed"             # "speed" or "accuracy"
    max_operand_bits: int = 16
    min_operand_bits: int = 8
    deterministic_cycles: bool = False
    max_relaunch: float = 1.0
    allow_lossy: bool = False
    allow_unmeasured: bool = False
    model: str = "any"                  # "llm", "other" or "any"
    target: str = "v4.9.2"


def _candidates(desc: LayerDesc, pref: Preferences):
    t = TARGETS[pref.target]
    packable = {w for (_, w) in t["cycles_per_mac"] if w is not None}
    if desc.kind in ("head", "grouped"):
        shapes = [(16, False)]
    else:
        widths = range(pref.min_operand_bits, pref.max_operand_bits + 1)
        shapes = [(w, False) for w in widths] + [(w, True) for w in widths if w in packable]
    regime = "gemv" if desc.M == 1 else "gemm"
    macs = desc.M * desc.N * desc.K
    h = _heuristic.h_for(desc.K)
    for w, packed in shapes:
        c = t["cycles_per_mac"][(regime, w if packed else None)]
        for policy in ("tbl2", "l2"):
            lo, hi = t["relaunch"][policy]
            kept = (w - (h if policy == "l2" else 0)) - t["tbl2_bits_lost"]
            if policy == "l2":
                evidence = "unmeasured"
            else:
                evidence = "ok" if w >= t["measured_from"][pref.model] else "unmeasured"
            yield {"config": {"accumulate": "wide", "operand_bits": w, "policy": policy, "launch": t["launch"],
                              "packed": packed, "accumulator": pref.target, "headroom": "none"},
                   "cycles": round(macs * c * (1 + lo)), "relaunch": [lo, hi], "bits_kept": kept,
                   "evidence": evidence}


def choose_config(desc: LayerDesc | dict, pref: Preferences | dict | None = None, top: int = 3, **kw) -> list:
    """The ranked configurations for one layer, best first (at most ``top``), as the golden file writes them:
    ``{"config": {...}, "cycles", "relaunch": [low, high], "bits_kept", "evidence"}``.

    ``desc``: ``LayerDesc`` or ``{"M", "N", "K", "kind"}``; ``pref``: ``Preferences``, a dict of its fields, or
    the fields as keywords."""
    if isinstance(desc, dict):
        desc = LayerDesc(**desc)
    if pref is None:
        pref = Preferences(**kw)
    elif isinstance(pref, dict):
        pref = Preferences(**{**pref, **kw})
    if desc.kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {desc.kind!r}")
    if pref.priority not in ("speed", "accuracy"):
        raise ValueError(f"priority must be 'speed' or 'accuracy', got {pref.priority!r}")
    if pref.model not in TARGETS[pref.target]["measured_from"]:
        raise ValueError(f"model must be 'llm', 'other' or 'any', got {pref.model!r}")
    keep = []
    for c in _candidates(desc, pref):
        lo, hi = c["relaunch"]
        if pref.deterministic_cycles and hi > 0:
            continue
        if hi > pref.max_relaunch:
            continue
        if c["evidence"] != "ok" and not pref.allow_unmeasured:
            continue
        if c["bits_kept"] < 1 and not pref.allow_lossy:
            continue
        keep.append(c)
    if pref.priority == "speed":
        keep.sort(key=lambda c: (c["cycles"], -c["bits_kept"]))
    else:
        keep.sort(key=lambda c: (-c["bits_kept"], c["cycles"]))
    return keep[:top]


@dataclass
class ConfigPlan:
    layers: dict                # name -> the chosen configuration (choose_config's first entry), or None
    descs: dict                 # name -> LayerDesc
    pref: Preferences

    @property
    def cycles(self) -> int:
        return sum(c["cycles"] for c in self.layers.values() if c)

    def widths(self) -> dict:
        return {n: c["config"]["operand_bits"] for n, c in self.layers.items() if c}

    def convert_kwargs(self) -> dict:
        """``sft.convert`` arguments: each layer at its chosen operand width.  Packing and the redo policy are
        decided below sf (Alosh); convert runs the width."""
        return {"widths": {n: f"sf{b}" for n, b in self.widths().items()}, "keep_sf16": False}

    def __repr__(self):
        per: dict = {}
        for b in self.widths().values():
            per[b] = per.get(b, 0) + 1
        none = sum(1 for c in self.layers.values() if not c)
        return (f"ConfigPlan ({self.pref.priority}, model={self.pref.model}, target {self.pref.target}): "
                + ", ".join(f"sf{b} x {k}" for b, k in sorted(per.items()))
                + f"; {self.cycles:,} cycles" + (f"; {none} layer(s) with no admissible configuration" if none
                                                 else ""))


def plan_configs(module, /, tokens: int = 1, pref: Preferences | dict | None = None, **kw) -> ConfigPlan:
    """``choose_config`` for every matmul layer of ``module`` (``tokens`` rows: 1 for decode, the prompt length
    for prefill); the LM head is kind ``"head"``.  Preferences as keywords, ``model="llm"`` among them."""
    from ._widths import matmul_layers
    if pref is None:
        pref = Preferences(**kw)
    elif isinstance(pref, dict):
        pref = Preferences(**{**pref, **kw})
    descs, layers = {}, {}
    for name, (fi, fo, kind) in matmul_layers(module).items():
        d = LayerDesc(M=tokens, N=fo, K=fi, kind="head" if kind == "lm_head" else "linear")
        descs[name] = d
        r = choose_config(d, pref, top=1)
        layers[name] = r[0] if r else None
    return ConfigPlan(layers, descs, pref)


def preferences_dict(pref: Preferences) -> dict:
    return asdict(pref)
