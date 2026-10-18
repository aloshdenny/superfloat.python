"""Stage 2.2: ``torch.compile(model, backend="superfloat")``.

TorchDynamo hands over FX graphs; SF ops run on Atreides, the rest fall back
to the CPU.  Gate: compiled output equals emulated output bit for bit; op
coverage report per model."""
from __future__ import annotations

import warnings

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

import sf                       # noqa: E402
import sf.torch as sft          # noqa: E402

DEV = "atreides"


def launches():
    return sf.get_device(DEV).launches


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


class Attention(nn.Module):
    """Single-head attention: SF projections and SF matmuls, float softmax."""

    def __init__(self, d=16):
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.d = d

    def forward(self, x):
        q, k, v = self.q(x), self.k(x), self.v(x)
        scores = sft.matmul(q, k.transpose(-1, -2), name="attn.qk") / self.d ** 0.5
        w = torch.softmax(scores, dim=-1)
        return self.o(sft.matmul(w, v, name="attn.wv"))


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 4, 3, padding=1)
        self.fc = nn.Linear(4 * 4 * 4, 3)

    def forward(self, x):
        x = F.max_pool2d(torch.relu(self.c1(x)), 2)
        return self.fc(x.flatten(1))


MODELS = {
    "mlp": (lambda: nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 6)), (16, 12)),
    "cnn": (CNN, (5, 1, 8, 8)),
    "attention": (Attention, (2, 7, 16)),
}


@pytest.mark.parametrize("arch", list(MODELS))
@pytest.mark.parametrize("headroom", ["bound", "none"])
def test_compiled_equals_emulated_bit_for_bit(arch, headroom):
    make, shape = MODELS[arch]
    torch.manual_seed(0)
    model = sft.convert(make(), headroom=headroom)
    x = torch.randn(*shape) * 3
    with torch.no_grad():
        eager = model(x)
        eager_status = sft.last_status(model)
        compiled = torch.compile(model, backend="superfloat")
        n0 = launches()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", sf.SFPreflightWarning)
            out = compiled(x)
        device_status = sft.last_status(model)
    assert torch.equal(out, eager)                                    # the 2.2 gate
    assert launches() > n0                                             # it really ran on the target
    for name, st in device_status.items():
        assert st.placement == DEV
        if headroom == "bound":
            # host-side scaling proved the bound, so the device pre-flight
            # proves it too: an exact zero without a scan
            assert st.saturated == 0 and st.exact, (name, st)
        else:
            e = eager_status[name]
            assert st.saturated is None or (st.saturated == 0 and e.saturated == 0)
        assert st.act_scale == eager_status[name].act_scale


def test_coverage_report():
    torch.manual_seed(0)
    model = sft.convert(Attention())
    compiled = torch.compile(model, backend="superfloat")
    with torch.no_grad():
        compiled(torch.randn(2, 5, 16))
    rep = sft.compile_report()
    layers = sorted(name for _, _, name, _ in rep.on_device)
    assert layers == ["attn.qk", "attn.wv", "k", "o", "q", "v"]
    assert {op for _, op, _, _ in rep.on_device} == {"linear", "matmul"}
    assert rep.unconverted == [] and rep.host_emulated == [] and rep.coverage == 1.0
    assert any("softmax" in t for _, t in rep.cpu)
    text = repr(rep)
    assert "6 SF op(s) on device" in text and "coverage 100%" in text
    assert sft.compile_reports()[-1] is rep


def test_unconverted_model_is_reported_not_silently_changed():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
    compiled = torch.compile(model, backend="superfloat")
    x = torch.randn(3, 8)
    n0 = launches()
    with torch.no_grad():
        out = compiled(x)
        assert torch.equal(out, model(x))                             # float semantics kept
    assert launches() == n0
    rep = sft.compile_report()
    assert rep.on_device == [] and len(rep.unconverted) == 2 and rep.coverage == 0.0
    assert "not converted" in repr(rep)


def test_unsupported_width_is_host_emulated():
    torch.manual_seed(0)
    model = sft.convert(nn.Sequential(nn.Linear(6, 6), nn.Tanh(), nn.Linear(6, 2)), dtype="sf8")
    x = torch.randn(4, 6)
    n0 = launches()
    with torch.no_grad():
        eager = model(x)
        out = torch.compile(model, backend="superfloat")(x)
    assert torch.equal(out, eager) and launches() == n0               # Atreides v1 is sf16 only
    rep = sft.compile_report()
    assert len(rep.host_emulated) == 2 and rep.on_device == []
    assert all("sf8 is not supported on atreides" in why for *_, why in rep.host_emulated)
    assert all(s.placement == "host" for s in sft.last_status(model).values())


def test_mixed_widths_split_between_device_and_host():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(6, 6), nn.ReLU(), nn.Linear(6, 2))
    model = sft.convert(model)
    model[2] = sft.SFLinear.from_float(nn.Linear(6, 2), "head8", sft.SFConfig(bits=8))
    with torch.no_grad():
        eager = model(torch.ones(2, 6))
        out = torch.compile(model, backend="superfloat")(torch.ones(2, 6))
    assert torch.equal(out, eager)
    st = sft.last_status(model)
    assert st["0"].placement == DEV and st["head8"].placement == "host"


def test_backend_is_inference_only():
    model = sft.convert(nn.Linear(4, 2))
    compiled = torch.compile(model, backend="superfloat")
    with pytest.raises(sf.SFUnsupportedError, match="inference-only"):
        compiled(torch.randn(3, 4))
    with torch.no_grad():
        compiled(torch.randn(3, 4))
    # inputs that do not require grad and frozen parameters are fine too
    for p in model.parameters():
        p.requires_grad_(False)
    torch._dynamo.reset()
    torch.compile(model, backend="superfloat")(torch.randn(3, 4))


def test_preflight_still_guards_the_compiled_path():
    """A layer the host can prove saturates everywhere is refused before launch,
    even inside a compiled model."""
    torch.manual_seed(0)
    lin = nn.Linear(64, 2, bias=False)
    with torch.no_grad():
        lin.weight.fill_(1.0)
    model = sft.convert(lin, headroom="none")
    compiled = torch.compile(model, backend="superfloat")
    n0 = launches()
    with torch.no_grad(), sf.config(suspect="verify"):
        with pytest.raises(sf.SFOverflowError, match="pre-flight"):
            compiled(torch.ones(3, 64))
    assert launches() == n0
    # the default bound policy makes the same layer safe
    safe = sft.convert(lin)
    ok = torch.compile(safe, backend="superfloat")
    with torch.no_grad():
        ok(torch.ones(3, 64))
    (st,) = sft.last_status(safe).values()
    assert st.saturated == 0 and st.exact and st.placement == DEV
