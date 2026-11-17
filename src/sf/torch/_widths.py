"""A width per layer: measure how much each layer's width costs in accuracy, read what it costs on the chip from
Alosh's cost table, and pick the cheapest widths that keep the loss within a budget.

    >>> sens = sft.layer_sensitivity(model, evaluate, dtype="sf8")          # one evaluation per layer
    >>> plan = sft.choose_widths(model, sens, max_loss=0.02, widths=(8, 10, 12, 16))
    >>> fast = sft.convert(model, "sf8", accumulate="wide", **plan.convert_kwargs())

``evaluate(model) -> float`` is any loss where lower is better (``log`` of perplexity on a few windows works well:
then ``max_loss=0.02`` is about +2% perplexity).  The cost per MAC is read from ``cost_table_v0.json`` (chip
v4.9.2), never derived here; ``sft.cost_table(path)`` loads a newer version.

The model behind the choice is additive: a layer at width w adds ``loss(w)`` on its own, and the layers' losses
sum.  Measured on Qwen2.5-0.5B the sum is off by up to a factor of 1.5 either way, so ``fit_widths`` checks the plan
with ``evaluate`` and tightens or loosens the budget until the measured loss is within it.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
from torch import nn

METRICS = ("energy_130nm", "energy_2nm", "cycles", "bus_words")
_DEFAULT = Path(__file__).with_name("data") / "cost_table_v0.json"


class CostTable:
    """Alosh's cost per width and layer kind (``cost_table_v0.json``).  Read as given: nothing is derived here."""

    def __init__(self, data: dict, source: str = ""):
        self.meta = data.get("meta", {})
        self.version = self.meta.get("version", "?")
        self.source = source
        self.rows = {(r["width"], r["kind"]): r for r in data["rows"]}
        self.widths = sorted({w for w, _ in self.rows})

    @classmethod
    def load(cls, path=None) -> "CostTable":
        p = Path(path) if path is not None else _DEFAULT
        return cls(json.loads(p.read_text()), str(p))

    def per_mac(self, width: int, kind: str = "linear", metric: str = "energy_130nm", K: int | None = None,
                packed: bool = True) -> float:
        """Cost of one MAC of a ``kind`` layer at ``width``.  ``cycles`` / ``bus_words``: the better of plain 16-bit
        words and weights packed, where the table has packed (even widths; Alosh: treat packed as achievable); the
        ``by_K`` entry nearest to ``K`` (768 when not given)."""
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}, got {metric!r}")
        r = self.rows.get((width, kind))
        if r is None:
            raise ValueError(f"cost table {self.version} has no entry for sf{width} {kind}")
        if metric.startswith("energy"):
            node = metric.split("_", 1)[1]
            return float(r["energy_pj_per_mac"][node])
        key = "cycles_per_mac" if metric == "cycles" else "bus_words_per_mac"

        def pick(by_k, flat):
            if by_k:
                ks = sorted(by_k, key=lambda k: (abs(int(k) - (K or 768)), -int(k)))
                return float(by_k[ks[0]][key])
            return float(flat[key])
        plain = pick(r.get("by_K"), r)
        pk = (r.get("packed") or {}).get("weights_packed") if packed else None
        return min(plain, pick(pk, r)) if pk else plain       # packing is optional: the better of the two


def cost_table(path=None) -> CostTable:
    """The cost table: the packaged v0 (chip v4.9.2) or the file at ``path``."""
    return CostTable.load(path)


def matmul_layers(model: nn.Module) -> dict:
    """{name: (in_features, out_features, kind)} for every layer ``convert`` would run as a matmul: ``nn.Linear``,
    Hugging Face ``Conv1D``, SF layers.  ``kind`` is ``"lm_head"`` for the LM head, else ``"linear"``."""
    from ._modules import SFHFConv1D, SFLinear, find_lm_heads, is_hf_conv1d
    heads = find_lm_heads(model)
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Linear, SFLinear)):
            fi, fo = m.in_features, m.out_features
        elif isinstance(m, SFHFConv1D) or is_hf_conv1d(m):
            w = m.weight
            fi, fo = int(w.shape[0]), int(w.shape[1])
        else:
            continue
        out[name] = (int(fi), int(fo), "lm_head" if id(m) in heads else "linear")
    return out


def _get_parent(model, name):
    *path, last = name.split(".")
    parent = model
    for p in path:
        parent = getattr(parent, p)
    return parent, last


@torch.no_grad()
def layer_sensitivity(model: nn.Module, evaluate, dtype="sf8", names=None, *, accumulate: str = "wide",
                      progress=None) -> dict:
    """How much ``evaluate`` rises when one layer alone runs at ``dtype`` and every other layer stays as it is
    (float): {name: loss - baseline}, plus ``"_baseline"``.  One evaluation per layer; the model is put back."""
    from ._modules import convert
    import sf
    bits = sf.info(sf.dtype(dtype)).bits
    layers = matmul_layers(model)
    names = list(layers) if names is None else list(names)
    base = float(evaluate(model))
    out = {"_baseline": base, "_bits": bits}
    for i, nm in enumerate(names):
        parent, last = _get_parent(model, nm)
        orig = getattr(parent, last)
        sfl = convert(nn.Sequential(orig), dtype=dtype, accumulate=accumulate, keep_sf16=False, attention=False,
                      inplace=True)[0]
        sfl.sf_name = nm
        setattr(parent, last, sfl)
        try:
            out[nm] = float(evaluate(model)) - base
        finally:
            setattr(parent, last, orig)
        if progress:
            progress(i + 1, len(names), nm, out[nm])
    return out


def _curves(sensitivity: dict, widths) -> tuple[dict, set]:
    """{name: {bits: loss}} for every width in ``widths``.  A width not measured is estimated conservatively: the
    loss of the nearest measured narrower width (widening is assumed to buy nothing until measured), and 0 at
    sf16 and above (sf16 is the reference: the whole Qwen2.5-0.5B at sf16 is +0.006% perplexity).  The 4x-per-bit
    rule (the error's variance per bit) was tried and is wrong where it matters: on Qwen2.5-0.5B the three most
    sensitive layers lose 1.1 to 1.3x less at sf10 than at sf8, not 16x less.  Negative measurements (noise)
    count as 0.  Returns the curves and the estimated (name, bits) points."""
    measured: dict = {}
    default_bits = sensitivity.get("_bits")
    for k, v in sensitivity.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict):
            measured[k] = {int(b): max(0.0, float(x)) for b, x in v.items()}
        else:
            if default_bits is None:
                raise ValueError("sensitivity: give {name: {bits: loss}}, or the dict from layer_sensitivity")
            measured[k] = {int(default_bits): max(0.0, float(v))}
    curves, est = {}, set()
    for nm, pts in measured.items():
        c = {}
        for w in widths:
            if w in pts:
                c[w] = pts[w]
                continue
            narrower = [m for m in pts if m < w]
            if w >= 16:
                c[w] = 0.0
            elif narrower:
                c[w] = pts[max(narrower)]
            else:
                c[w] = max(pts.values())
            est.add((nm, w))
        for a, b in zip(widths, widths[1:]):            # wider is never worse
            c[b] = min(c[b], c[a])
        curves[nm] = c
    return curves, est


@dataclass
class WidthPlan:
    widths: dict                    # name -> bits
    predicted_loss: float           # additive estimate
    cost: float                     # metric units per token (MACs per token x cost per MAC)
    cost_narrowest: float           # every layer at the narrowest width allowed
    cost_widest: float              # every layer at the widest width allowed
    metric: str
    phase: str
    table_version: str
    max_loss: float
    estimated: list = field(default_factory=list)   # layers placed at a width not measured for them
    macs: dict = field(default_factory=dict)        # name -> MACs per token
    measured_loss: float | None = None              # set by fit_widths

    def convert_kwargs(self) -> dict:
        """Arguments for ``sft.convert``: every layer at its chosen width (``keep_sf16`` off: the plan decides)."""
        return {"widths": {n: f"sf{b}" for n, b in self.widths.items()}, "keep_sf16": False}

    def share(self) -> dict:
        """Share of MACs at each width."""
        tot = sum(self.macs.values()) or 1
        out: dict = {}
        for n, b in self.widths.items():
            out[b] = out.get(b, 0) + self.macs.get(n, 0) / tot
        return dict(sorted(out.items()))

    def __repr__(self):
        per: dict = {}
        for b in self.widths.values():
            per[b] = per.get(b, 0) + 1
        span = self.cost_widest - self.cost_narrowest
        where = 100 * (self.cost - self.cost_narrowest) / span if span else 0.0
        lines = [f"WidthPlan ({self.metric}, {self.phase}, cost table {self.table_version}): "
                 + ", ".join(f"sf{b} x {k}" for b, k in sorted(per.items())),
                 f"  MACs: " + ", ".join(f"sf{b} {100 * s:.1f}%" for b, s in self.share().items()),
                 f"  cost {self.cost:.4g} per token ({where:.0f}% of the way from all-narrowest "
                 f"{self.cost_narrowest:.4g} to all-widest {self.cost_widest:.4g})",
                 f"  loss: predicted {self.predicted_loss:.4g} (budget {self.max_loss:.4g})"
                 + (f", measured {self.measured_loss:.4g}" if self.measured_loss is not None else "")]
        if self.estimated:
            lines.append(f"  {len(self.estimated)} layer(s) at a width not measured for them (assumed no better "
                         f"than the next narrower measured width)")
        return "\n".join(lines)


def choose_widths(model: nn.Module, sensitivity: dict, max_loss: float, widths=(8, 16), *,
                  metric: str = "energy_130nm", phase: str = "prefill", table: CostTable | None = None,
                  keep_sf16="auto") -> WidthPlan:
    """The cheapest width per layer whose summed loss stays within ``max_loss``.

    ``sensitivity``: from ``layer_sensitivity`` ({name: loss at its dtype}), or {name: {bits: loss}} with several
    widths measured.  ``widths``: the widths allowed.  ``metric``: what "cheap" means, one of ``METRICS``, read from
    ``table`` (default: the packaged cost table v0) per MAC times each layer's MACs per token; ``phase``:
    ``"prefill"`` (GEMM rows ``linear`` / ``lm_head``) or ``"decode"`` (one token, the ``decode_gemv`` rows).  ``keep_sf16="auto"``
    pins the LM head at the widest width, as ``convert`` does; ``False`` lets the chooser decide it too.

    Greedy on the best loss saved per unit of cost: start every layer at the narrowest width, then widen the layer
    whose next step saves the most loss per cost until the predicted loss is within budget.  A layer with no
    sensitivity entry stays at the widest width.
    """
    if phase not in ("prefill", "decode"):
        raise ValueError(f"phase must be 'prefill' or 'decode', got {phase!r}")
    table = table or CostTable.load()
    widths = sorted({int(w) for w in widths})
    layers = matmul_layers(model)
    curves, est = _curves(sensitivity, widths)
    macs = {n: fi * fo for n, (fi, fo, _) in layers.items()}

    def cost(n, w):
        fi, fo, kind = layers[n]
        return macs[n] * table.per_mac(w, "decode_gemv" if phase == "decode" else kind, metric, K=fi)

    lo, hi = widths[0], widths[-1]
    choice = {}
    for n in layers:
        pinned = keep_sf16 == "auto" and layers[n][2] == "lm_head"
        choice[n] = hi if (n not in curves or pinned) else lo
    loss = lambda: sum(curves[n][choice[n]] for n in curves)          # noqa: E731
    total = loss()
    while total > max_loss:
        best = None
        for n, c in curves.items():
            w0 = choice[n]
            for w in widths:
                if w <= w0:
                    continue
                gain = c[w0] - c[w]
                if gain <= 0:
                    continue
                dc = cost(n, w) - cost(n, w0)
                score = math.inf if dc <= 0 else gain / dc
                if best is None or score > best[0]:
                    best = (score, n, w, gain)
        if best is None:
            break
        _, n, w, gain = best
        choice[n] = w
        total -= gain
    return WidthPlan(widths=choice, predicted_loss=loss(), cost=sum(cost(n, w) for n, w in choice.items()),
                     cost_narrowest=sum(cost(n, lo) for n in layers), cost_widest=sum(cost(n, hi) for n in layers),
                     metric=metric, phase=phase, table_version=table.version, max_loss=max_loss,
                     estimated=sorted(n for n, w in choice.items() if (n, w) in est and w != hi), macs=macs)


@torch.no_grad()
def fit_widths(model: nn.Module, evaluate, sensitivity: dict, max_loss: float, widths=(8, 16), *, dtype=None,
               metric: str = "energy_130nm", phase: str = "prefill", table: CostTable | None = None, keep_sf16="auto",
               accumulate: str = "wide", rounds: int = 4, progress=None, **convert_kw) -> WidthPlan:
    """``choose_widths``, then check: convert a copy with the plan, measure ``evaluate``, and if the measured loss
    misses the budget scale the budget by predicted / measured and choose again (``rounds`` times at most).
    Returns the cheapest plan whose measured loss was within ``max_loss`` (or the closest one, if none was)."""
    from ._modules import convert
    base = sensitivity.get("_baseline")
    if base is None:
        base = float(evaluate(model))
    lo = f"sf{min(widths)}" if dtype is None else dtype
    budget, best, closest = max_loss, None, None
    for _ in range(rounds):
        plan = choose_widths(model, sensitivity, budget, widths, metric=metric, phase=phase, table=table,
                             keep_sf16=keep_sf16)
        m = convert(model, lo, accumulate=accumulate, **plan.convert_kwargs(), **convert_kw)
        plan.measured_loss = float(evaluate(m)) - base
        del m
        if progress:
            progress(plan)
        if plan.measured_loss <= max_loss and (best is None or plan.cost < best.cost):
            best = plan
        if closest is None or abs(plan.measured_loss - max_loss) < abs(closest.measured_loss - max_loss):
            closest = plan
        if plan.measured_loss <= 0 or plan.predicted_loss <= 0:
            break
        nb = budget * max_loss / plan.measured_loss
        if abs(nb - budget) < 0.02 * budget or (best is not None and plan.measured_loss <= max_loss
                                                 and plan.measured_loss > 0.9 * max_loss):
            break
        budget = nb
    return best or closest
