"""sf.torch wide mode (Alosh's build order, items 2, 5 and 6).

``accumulate="wide"`` runs a layer on the MX engine with the wide-mode
shift rule (``sf._heuristic``): one power-of-two exponent per token and per
weight channel, ``eo = l2 - h``, a redo at ``l2`` where an output saturates.
Pinned end to end by the 13 matmul cases and the ``h`` table of the
heuristic section in ``sf_golden_v4.9.2.json``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sf  # noqa: E402
import sf.kernels  # noqa: E402,F401  (registers atreides-sim)
import sf.torch as sft  # noqa: E402
from sf import _heuristic  # noqa: E402
from sf._format import FORMATS  # noqa: E402
from sf.torch import _core  # noqa: E402

GOLDEN = json.loads((Path(__file__).parent / "data" / "sf_golden_v4.9.2.json").read_text())["heuristic"]
CASES = GOLDEN["matmul"]


# --- the rule --------------------------------------------------------------------------------

def test_h_table_matches_the_golden_file():
    for row in GOLDEN["exponent_rule"]:
        assert row["margin"] == _heuristic.MARGIN
        assert _heuristic.h_for(row["K"]) == row["h"], row


def test_pow2_exp():
    assert _heuristic.pow2_exp([0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.9]).tolist() == [0, 0, 0, 1, 1, 2, 2]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_quantize_and_plan_match_the_golden_steps(case):
    fmt = FORMATS[case["bits"]]
    x, w = np.array(case["x"]), np.array(case["w"])
    qa, ea, _ = _heuristic.quantize_pow2(fmt, x, -1)
    qw, ew, _ = _heuristic.quantize_pow2(fmt, w, -1)
    assert qa.tolist() == case["a_codes"] and ea.tolist() == case["a_exp"]
    assert qw.T.tolist() == case["w_codes_KxN"] and ew.tolist() == case["w_exp"]
    shift, _, h = _heuristic.plan(fmt, qa, ea, qw.T, ew)
    assert shift.tolist() == case["shift"]
    assert h == _heuristic.h_for(case["K"])


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_wide_linear_gives_the_golden_final_values(case):
    r = _core.wide_linear(np.array(case["x"]), np.array(case["w"]), None, case["bits"])
    assert np.array_equal(r.y, np.array(case["final_value"]))
    assert r.redone == sum(map(sum, case["pop_saturated"]))
    assert r.acc_clipped == 0 and r.guard_raised == 0 and r.saturated == 0


@pytest.mark.parametrize("case", [c for c in CASES if "outlier" in c["id"] and c["bits"] == 16],
                         ids=lambda c: c["id"])
def test_a_redo_on_the_device_gives_the_host_bits(case):                # atreides runs sf16 (item 7: more widths)
    x, w = np.array(case["x"]), np.array(case["w"])
    with sf.config(preflight="ignore"):
        d = _core.wide_linear(x, w, None, case["bits"], runner=_core.DeviceRunner("atreides"))
    assert np.array_equal(d.y, np.array(case["final_value"])) and d.redone == 1
    assert d.flushed is None                                   # the chip cannot report it


def test_the_guard_raises_a_token_exponent_on_v481():
    """512 terms of 0.81 total 415: over v4.8.1's +-32, under v4.9.2's +-512.
    Codes are scaled per token, so a token is far from the limit when its
    norm is small against its largest value, not when its values are small."""
    x = np.full((2, 512), 0.9)
    x[1, 1:] = 0.0                                             # one term: bound 0.9 x 20.4, under 32
    w = np.full((3, 512), 0.9)
    r481 = _core.wide_linear(x, w, None, 16, mx="v4.8.1")
    r492 = _core.wide_linear(x, w, None, 16, mx="v4.9.2")
    assert r481.guard_raised == 1 and r481.acc_clipped == 0
    assert r492.guard_raised == 0 and r492.acc_clipped == 0
    np.testing.assert_allclose(r481.y, x @ w.T, rtol=2e-3)
    np.testing.assert_allclose(r492.y, x @ w.T, rtol=2e-4)


def test_the_guard_also_protects_v492():
    """1024 terms of 0.81 total 829, over +-512: the guard raises the token
    and nothing clips, where an unguarded accumulator would."""
    x, w = np.full((1, 1024), 0.9), np.full((2, 1024), 0.9)
    r = _core.wide_linear(x, w, None, 16)
    assert r.guard_raised == 1 and r.acc_clipped == 0
    np.testing.assert_allclose(r.y, x @ w.T, rtol=2e-4)


# --- the plugin --------------------------------------------------------------------------------

def _mlp():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(64, 32), torch.nn.GELU(), torch.nn.Linear(32, 8)).eval()


def test_sft_linear_wide_is_the_golden_case():
    c = next(c for c in CASES if c["id"] == "v4.9.2_sf16_k64_outlier")
    x, w = torch.tensor(c["x"], dtype=torch.float64), torch.tensor(c["w"], dtype=torch.float64)
    y = sft.linear(x, w, None, accumulate="wide", name="golden.wide")
    assert torch.equal(y, torch.tensor(c["final_value"], dtype=torch.float64))
    st = sft.last_status()["golden.wide"]
    assert st.accumulate == "wide" and st.redone == 1 and st.h == 1
    assert (st.shift_min, st.shift_max) == (-6, -4) and st.flushed == 0 and st.acc_clipped == 0


def test_convert_wide_is_more_accurate_than_the_chain_and_reports_status():
    m, x = _mlp(), torch.randn(5, 64)
    ref = m(x)
    with sft.capture_status() as sw:
        yw = sft.convert(m, accumulate="wide")(x)
    with sft.capture_status() as sc:
        yc = sft.convert(m)(x)
    assert (yw - ref).abs().max() < (yc - ref).abs().max() / 4
    assert all(s.accumulate == "wide" and s.h is not None and s.headroom_bits == 0 for s in sw.values())
    assert all(s.accumulate == "chain" and s.h is None for s in sc.values())
    assert "accumulate=wide" in repr(sft.convert(m, accumulate="wide"))


def test_chain_is_unchanged_by_default():
    m, x = _mlp(), torch.randn(5, 64)
    a = sft.convert(m)(x)
    b = sft.convert(m, accumulate="chain")(x)
    assert torch.equal(a, b)


def test_autocast_wide_equals_convert_wide():
    m, x = _mlp(), torch.randn(5, 64)
    with sft.autocast(accumulate="wide"):
        ya = m(x)
    assert torch.equal(ya, sft.convert(m, accumulate="wide")(x))


def test_hf_conv1d_and_conv2d_take_wide_mode():
    from sf.torch import SFHFConv1D
    conv = torch.nn.Conv2d(3, 4, 3, padding=1).eval()
    x = torch.randn(1, 3, 6, 6)
    y = sft.convert(torch.nn.Sequential(conv), accumulate="wide")(x)
    assert (y - conv(x)).abs().max() < 1e-3

    class Conv1D(torch.nn.Module):                            # Hugging Face's layout: weight (in, out)
        def __init__(self, nf, nx):
            super().__init__()
            self.nf, self.nx = nf, nx
            self.weight = torch.nn.Parameter(torch.randn(nx, nf) * 0.1)
            self.bias = torch.nn.Parameter(torch.zeros(nf))

        def forward(self, x):
            return x @ self.weight + self.bias

    c1 = Conv1D(6, 16)
    x1 = torch.randn(4, 16)
    out = sft.convert(torch.nn.Sequential(c1), accumulate="wide")
    assert isinstance(out[0], SFHFConv1D) and (out(x1) - c1(x1)).abs().max() < 1e-3


def test_sft_matmul_wide_batched_and_1d():
    torch.manual_seed(1)
    a, b = torch.randn(2, 4, 16), torch.randn(2, 16, 3)
    y = sft.matmul(a, b, accumulate="wide", name="mm.wide")
    assert y.shape == (2, 4, 3) and (y - a @ b).abs().max() < 1e-3
    assert sft.last_status()["mm.wide"].op == "matmul"
    v = sft.matmul(a[0, 0], b[0], accumulate="wide")
    assert v.shape == (3,) and torch.allclose(v, a[0, 0] @ b[0], atol=1e-3)


def test_an_explicit_shift_overrides_the_rule():
    torch.manual_seed(2)
    x, w = torch.rand(3, 64) * 0.5, torch.rand(8, 64) * 0.5
    sft.linear(x, w, accumulate="wide", shift=-4, name="fixed")
    st = sft.last_status()["fixed"]
    assert (st.shift_min, st.shift_max, st.h, st.redone) == (-4, -4, None, 0)


def test_gradients_flow_straight_through():
    m = _mlp().train()
    mw = sft.convert(m, accumulate="wide")
    x = torch.randn(5, 64, requires_grad=True)
    mw(x).sum().backward()
    assert x.grad is not None and mw[0].weight.grad.shape == (32, 64) and mw[0].bias.grad is not None
    a = torch.randn(4, 16, requires_grad=True)
    b = torch.randn(16, 3, requires_grad=True)
    sft.matmul(a, b, accumulate="wide").sum().backward()
    assert a.grad.shape == a.shape and b.grad.shape == b.shape


@pytest.mark.parametrize("device", ["atreides", "atreides-sim"])
def test_compile_backend_runs_wide_on_the_target_with_the_host_bits(device):
    m, x = _mlp(), torch.randn(5, 64)
    mw = sft.convert(m, accumulate="wide")
    with torch.no_grad():
        want = mw(x)
        torch._dynamo.reset()
        with sf.config(default_device=device, preflight="ignore"):
            got = torch.compile(mw, backend="superfloat")(x)
    assert torch.equal(got, want)
    rep = sft.compile_report()
    assert {op for _, op, _, _ in rep.on_device} == {"linear_wide"}
    st = sft.last_status(mw)["0"]
    assert st.placement == device and st.flushed is None and st.acc_clipped == 0


def test_a_target_without_mx_keeps_wide_layers_on_the_host():
    class NoMX(sf.MockAtreides):
        name = "no-mx-torch"
        mx = False

    sf.register_device(NoMX())
    try:
        mw = sft.convert(_mlp(), accumulate="wide")
        x = torch.randn(2, 64)
        with torch.no_grad():
            want = mw(x)
            torch._dynamo.reset()
            with sf.config(default_device="no-mx-torch"):
                got = torch.compile(mw, backend="superfloat")(x)
        assert torch.equal(got, want)
        assert all("no MX engine" in why for *_, why in sft.compile_report().host_emulated)
    finally:
        sf._device._registry.pop("no-mx-torch", None)


def test_bad_settings_are_refused():
    m = _mlp()
    with pytest.raises(ValueError, match="accumulate must be one of"):
        sft.convert(m, accumulate="fast")
    with pytest.raises(ValueError, match="calibration must be one of"):
        sft.convert(m, accumulate="wide", calibration="learned")
    with pytest.raises(ValueError, match="only applies"):
        sft.convert(m, shift=-2)
    with pytest.raises(ValueError, match="between -32 and 31"):
        sft.convert(m, accumulate="wide", shift=40)
