"""Regression tests for the independent Stage 2 review."""
from __future__ import annotations

import asyncio
import threading
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import sf                       # noqa: E402
import sf.torch as sft          # noqa: E402

F64 = torch.float64


# P1: NaN and inf go through the SF error policy, on every path --------------------

_CONV = sft.SFConv2d(nn.Conv2d(1, 1, 1), "c", sft.SFConfig())


def _paths():
    w = torch.ones(2, 2)
    return {
        "input": lambda t: sft.linear(t.reshape(1, 2), w, None),
        "weight": lambda t: sft.linear(torch.ones(1, 2), t.reshape(2, 1).expand(2, 2).clone(), None),
        "bias": lambda t: sft.linear(torch.ones(1, 2), w, t),
        "matmul_left": lambda t: sft.matmul(t.reshape(1, 2), w),
        "matmul_right": lambda t: sft.matmul(torch.ones(1, 2), t.reshape(2, 1).expand(2, 2).clone()),
        "conv_input": lambda t: _CONV(t.reshape(1, 1, 1, 2)),
        "quantize": lambda t: sft.quantize(t, scale="dynamic"),
    }


@pytest.mark.parametrize("path", list(_paths()))
def test_nan_raises_by_default_and_follows_the_policy(path):
    fn = _paths()[path]
    t = torch.tensor([0.5, float("nan")])
    with pytest.raises(sf.SFInvalidError):
        fn(t)
    with sf.errstate(invalid="warn"):
        with pytest.warns(sf.SFInvalidWarning):
            out = fn(t)
    assert torch.isfinite(out).all()
    with sf.errstate(invalid="ignore"):
        out2 = fn(t)
    assert torch.equal(out, out2)                              # NaN behaves as 0 (device value)
    # and under ignore the NaN element really acted as 0
    with sf.errstate(all="ignore"):
        zero = fn(torch.tensor([0.5, 0.0]))
    assert torch.equal(out2, zero)


@pytest.mark.parametrize("path", [p for p in _paths() if p != "quantize"])
def test_inf_is_an_overflow_event(path):
    fn = _paths()[path]
    t = torch.tensor([0.5, float("inf")])
    with pytest.raises(sf.SFOverflowError):
        fn(t)
    with sf.errstate(over="warn"):
        with pytest.warns(sf.SFOverflowWarning):
            out = fn(t)
    assert torch.isfinite(out).all()
    with sf.errstate(over="ignore"):
        assert torch.equal(fn(t), out)
        # the inf saturates to the largest magnitude of the tensor at the chosen scale
        assert torch.equal(fn(t), fn(torch.tensor([0.5, 0.5])))
        assert torch.isfinite(fn(torch.tensor([float("-inf"), float("inf")]))).all()


def test_compiled_path_agrees_on_nan():
    torch._dynamo.reset()
    model = sft.convert(nn.Linear(2, 2))
    compiled = torch.compile(model, backend="superfloat")
    x = torch.tensor([[0.5, float("nan")]])
    with torch.no_grad():
        with pytest.raises(sf.SFInvalidError):
            compiled(x)
        with sf.errstate(invalid="ignore"):
            assert torch.equal(compiled(x), model(x))


# P1: conversion preserves parameters, requires_grad and training state -----------------

def test_frozen_and_eval_state_preserved():
    lin = nn.Linear(3, 2)
    lin.weight.requires_grad_(False)
    lin.eval()
    c = sft.convert(lin)
    assert c.weight.requires_grad is False and c.bias.requires_grad is True
    assert c.training is False
    conv = nn.Conv2d(1, 2, 3).eval()
    for p in conv.parameters():
        p.requires_grad_(False)
    cc = sft.convert(conv)
    assert cc.training is False and not any(p.requires_grad for p in cc.parameters())
    # a model in train mode stays in train mode
    assert sft.convert(nn.Sequential(nn.Linear(2, 2)).train())[0].training is True


def test_inplace_conversion_keeps_the_optimizer_connected():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    sft.convert(model, inplace=True)
    assert isinstance(model[0], sft.SFLinear)
    opt_params = [q for g in opt.param_groups for q in g["params"]]
    assert all(any(p is q for q in opt_params) for p in model.parameters())
    before = [p.detach().clone() for p in model.parameters()]
    opt.zero_grad()
    model(torch.randn(8, 4)).pow(2).sum().backward()
    opt.step()
    assert all(not torch.equal(b, p.detach()) for b, p in zip(before, model.parameters()))
    # a non-inplace conversion deep-copies, so the original keeps its own parameters
    orig = nn.Linear(2, 2)
    conv = sft.convert(orig)
    assert conv.weight is not orig.weight and torch.equal(conv.weight, orig.weight)
    # conv2d too
    cnn = nn.Sequential(nn.Conv2d(1, 2, 3))
    w = cnn[0].weight
    sft.convert(cnn, inplace=True)
    assert cnn[0].weight is w


# P1: shared layers are converted once and every alias points at the result ----------

def test_shared_layers():
    shared = nn.Linear(3, 3)

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = shared
            self.b = shared
            self.inner = nn.Sequential(shared, nn.ReLU())

        def forward(self, x):
            return self.inner(self.b(self.a(x)))

    m = sft.convert(M())
    assert isinstance(m.a, sft.SFLinear) and m.a is m.b is m.inner[0]
    assert m.a.weight is m.b.weight and len(list(m.parameters())) == 2
    rep = sft.conversion_report(m)
    assert [n for n, _ in rep.converted] == ["a"] and sorted(a for a, _ in rep.shared) == ["b", "inner.0"]
    assert "shared module, converted once" in repr(rep)
    x = torch.randn(2, 3)
    with torch.no_grad():
        y = m(x)
        ref = torch.relu(sft.linear(sft.linear(sft.linear(x, shared.weight, shared.bias),
                                               shared.weight, shared.bias), shared.weight, shared.bias))
    assert torch.equal(y, ref)
    # a shared container is visited once
    block = nn.Sequential(nn.Linear(2, 2))
    outer = nn.Sequential(block, block)
    o = sft.convert(outer)
    assert o[0] is o[1] and isinstance(o[0][0], sft.SFLinear)


# P2: a standalone Conv2d --------------------------------------------------------------------

@torch.no_grad()
def test_root_conv2d():
    conv = nn.Conv2d(1, 2, 3, padding=1)
    c = sft.convert(conv)
    assert isinstance(c, sft.SFConv2d) and sft.conversion_report(c).converted == [("", "Conv2d")]
    x = torch.randn(1, 1, 5, 5)
    assert float((c(x) - conv(x)).abs().max()) < 0.05 * float(conv(x).abs().max())
    g = sft.convert(nn.Conv2d(2, 2, 3, groups=2))
    assert isinstance(g, nn.Conv2d) and not isinstance(g, sft.SFConv2d)
    assert sft.conversion_report(g).skipped[0][2].startswith("groups=2")


# P2: extreme but valid float64 values --------------------------------------------------------

def test_extreme_float64_values_do_not_crash():
    z = torch.zeros(1, 2, dtype=F64)
    y = sft.linear(z, torch.zeros(1, 2, dtype=F64), torch.tensor([1e308], dtype=F64), name="e1")
    assert y.tolist() == [[1e308]] and sft.last_status()["e1"].bias_saturated == 0
    y = sft.linear(torch.ones(1, 2, dtype=F64) * 1e300, torch.ones(1, 2, dtype=F64) * 1e300,
                   torch.tensor([-1e300], dtype=F64), name="e2")
    assert torch.isinf(y).all()                               # 2e600 is not a float64: inf, like float
    assert sft.last_status()["e2"].saturated == 0
    y = sft.linear(torch.ones(1, 2, dtype=F64) * 1e-300, torch.ones(1, 2, dtype=F64),
                   torch.tensor([1e-300], dtype=F64), name="e3")
    assert abs(float(y) - 3e-300) < 1e-303
    # a bias no finite scale can encode next to tiny weights is counted, not crashed on
    y = sft.linear(torch.ones(1, 2, dtype=F64), torch.ones(1, 2, dtype=F64) * 1e-300,
                   torch.tensor([1e308], dtype=F64), name="e4")
    assert torch.isfinite(y).all() and sft.last_status()["e4"].bias_saturated == 1
    big = torch.full((2, 2), 1.7e308, dtype=F64)
    assert torch.isfinite(sft.matmul(big, torch.eye(2, dtype=F64))).all()
    with sf.errstate(all="ignore"):
        assert torch.isfinite(sft.quantize(big, scale="dynamic")).all()


# readiness: request-scoped status -------------------------------------------------------------

def test_capture_status_is_per_thread_and_per_task():
    a = sft.convert(nn.Sequential(nn.Linear(4, 4)))
    b = sft.convert(nn.Sequential(nn.Linear(4, 4)))        # same layer name "0"
    results = {"a": set(), "b": set()}

    def run(model, key, scale):
        for _ in range(30):
            with sft.capture_status() as st:
                model(torch.full((3, 4), scale))
            results[key].add(round(st["0"].act_scale, 6))

    threads = [threading.Thread(target=run, args=(a, "a", 0.5)),
               threading.Thread(target=run, args=(b, "b", 50.0))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results["a"]) == 1 and len(results["b"]) == 1 and results["a"] != results["b"]

    @sft.capture_status()
    async def serve(model, x):
        await asyncio.sleep(0)
        return model(x)

    async def main():
        await asyncio.gather(serve(a, torch.full((1, 4), 0.5)), serve(b, torch.full((1, 4), 50.0)))

    asyncio.run(main())
    assert list(serve.last) == ["0"]
    with sft.capture_status() as outer:
        a(torch.ones(1, 4))
        with sft.capture_status() as inner:
            b(torch.ones(1, 4) * 50)
    # the outer capture sees both calls (last one wins for the shared name), the inner only its own
    assert set(outer) == {"0"} and set(inner) == {"0"}
    assert outer["0"] is inner["0"]
