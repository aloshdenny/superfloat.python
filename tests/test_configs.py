"""The configuration chooser: Alosh's 56 cases (``sf_golden_v4.9.2.json``, ``heuristic.configuration``), exactly."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import sf.torch as sft  # noqa: E402
from torch import nn  # noqa: E402

CASES = json.loads((Path(__file__).parent / "data" / "sf_golden_v4.9.2.json").read_text())["heuristic"]["configuration"]


def test_there_are_56():
    assert len(CASES) == 56


@pytest.mark.parametrize("i", range(len(CASES)), ids=lambda i: f"{CASES[i]['shape']}-{i % 8}")
def test_every_case_exactly(i):
    c = CASES[i]
    assert sft.choose_config(c["desc"], c["pref"]) == c["ranked"]


def test_keywords_and_errors():
    d = sft.LayerDesc(M=1, N=6144, K=2048)
    assert sft.choose_config(d, model="other")[0]["config"]["operand_bits"] == 8
    assert sft.choose_config(d, model="llm")[0]["config"]["operand_bits"] == 12
    lossy = sft.choose_config(d, model="other", deterministic_cycles=True, allow_unmeasured=True, allow_lossy=True)
    assert lossy[0]["config"]["operand_bits"] == 8 and lossy[0]["bits_kept"] < 1     # l2 at sf8 keeps 0.8 bits
    assert sft.choose_config(d, model="llm", max_operand_bits=8) == []               # nothing measured for an LLM
    with pytest.raises(ValueError, match="kind"):
        sft.choose_config({"M": 1, "N": 8, "K": 8, "kind": "attention"})
    with pytest.raises(ValueError, match="priority"):
        sft.choose_config(d, priority="cheap")


def test_plan_for_a_model():
    torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(2048, 6144), nn.ReLU(), nn.Linear(6144, 2048))
    p = sft.plan_configs(m, tokens=1, model="llm")
    assert p.widths() == {"0": 12, "2": 12}
    assert p.cycles == 2 * sft.choose_config({"M": 1, "N": 6144, "K": 2048}, model="llm")[0]["cycles"]
    assert p.convert_kwargs()["widths"] == {"0": "sf12", "2": "sf12"} and "ConfigPlan" in repr(p)
    c = sft.convert(m, "sf16", accumulate="wide", **p.convert_kwargs())
    assert sft.conversion_report(c).widths == {"0": 12, "2": 12}
