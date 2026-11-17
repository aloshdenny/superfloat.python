"""A width per layer (DSL 0.15.0): the cost table is read, not derived; the chooser is the cheapest additive plan
within budget; convert takes the plan."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import sf.torch as sft  # noqa: E402
from torch import nn  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def test_cost_table_is_alosh_s_v0():
    t = sft.cost_table()
    raw = json.loads((ROOT / "tests" / "data" / "cost_table_v0.json").read_text())
    assert t.version == "v0" and t.meta == raw["meta"] and t.widths == list(range(4, 17))
    assert t.per_mac(16, "linear") == 12.85 and t.per_mac(8, "linear") == 5.75 and t.per_mac(4, "linear") == 4.34
    assert t.per_mac(8, "linear", "energy_2nm") == 0.109
    assert t.per_mac(8, "linear", "cycles") == 1.6379                          # packed (even width)
    assert t.per_mac(16, "linear", "cycles") == 2.1161                         # plain beats packed at sf16
    assert t.per_mac(9, "linear", "cycles") == t.per_mac(9, "linear", "cycles", packed=False)
    assert t.per_mac(8, "decode_gemv", "cycles") == 5.272
    with pytest.raises(ValueError, match="metric"):
        t.per_mac(8, "linear", "joules")


def _mlp():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8))


def test_chooser_spends_the_budget_where_it_is_cheapest():
    m = _mlp()
    sens = {"_bits": 8, "0": 0.010, "2": 0.010, "4": 0.001}          # "2" is 4x the MACs of "0"
    p = sft.choose_widths(m, sens, max_loss=0.012, keep_sf16=False)
    assert p.widths == {"0": 16, "2": 8, "4": 8}                       # same loss saved, a quarter of the cost
    assert abs(p.predicted_loss - 0.011) < 1e-6 and p.estimated == []    # sf16: 4^-8 of sf8's
    assert p.cost_narrowest < p.cost < p.cost_widest
    p0 = sft.choose_widths(m, sens, max_loss=1.0, keep_sf16=False)
    assert set(p0.widths.values()) == {8} and p0.cost == p0.cost_narrowest
    p1 = sft.choose_widths(m, sens, max_loss=0.0, keep_sf16=False)
    assert set(p1.widths.values()) == {16}
    assert "WidthPlan" in repr(p) and abs(sum(p.share().values()) - 1) < 1e-12


def test_intermediate_widths_and_estimates():
    m = _mlp()
    sens = {"0": {8: 0.02, 12: 0.001}, "2": {8: 0.02}, "4": {8: -0.003}}
    p = sft.choose_widths(m, sens, max_loss=0.01, widths=(8, 10, 12, 16), keep_sf16=False)
    assert p.widths["4"] == 8                                           # negative = noise = free
    assert p.predicted_loss <= 0.01 and p.widths["2"] > 8
    assert "2" in p.estimated                                          # its sf10/sf12 points are extrapolated
    pd = sft.choose_widths(m, sens, max_loss=0.01, widths=(8, 10, 12, 16), phase="decode", metric="cycles",
                           keep_sf16=False)
    assert pd.phase == "decode" and pd.predicted_loss <= 0.01


def test_convert_takes_a_width_per_layer():
    m = _mlp()
    x = torch.randn(4, 16)
    c = sft.convert(m, "sf8", accumulate="wide", widths={"2": "sf16", "4": "sf12"})
    assert sft.conversion_report(c).widths == {"0": 8, "2": 16, "4": 12}
    with torch.no_grad():
        assert torch.isfinite(c(x)).all()
    with pytest.raises(ValueError, match="widths: no module"):
        sft.convert(m, "sf8", widths={"nope": "sf16"})


def test_sensitivity_and_fit_end_to_end():
    m = _mlp()
    torch.manual_seed(1)
    x = torch.randn(256, 16)
    y = m(x).detach()

    def evaluate(model):
        with torch.no_grad():
            return float(((model(x) - y) ** 2).mean() / (y ** 2).mean())

    sens = sft.layer_sensitivity(m, evaluate, "sf6")
    assert sens["_bits"] == 6 and set(sens) - {"_baseline", "_bits"} == {"0", "2", "4"}
    assert all(sens[k] > 0 for k in ("0", "2", "4"))
    assert isinstance(m[0], nn.Linear)                                  # put back
    full = evaluate(sft.convert(m, "sf6", accumulate="wide", keep_sf16=False))
    plan = sft.fit_widths(m, evaluate, sens, max_loss=full / 3, widths=(6, 16), keep_sf16=False)
    assert plan.measured_loss is not None and plan.measured_loss <= full / 3
    assert 6 in plan.widths.values() or plan.cost == plan.cost_widest
