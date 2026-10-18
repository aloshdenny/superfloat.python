"""Stage 2.1: model conversion.  ``sft.convert`` swaps Linear and Conv2d for
SF versions with a scale policy; ``sft.autocast`` does the same for a block
of code.  Gate (spec): converted model accuracy matches the baselines.  The
research baselines are not in this repository; here the gate is measured
against the float model on a real task (scikit-learn digits)."""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")
nn = torch.nn
F = torch.nn.functional

import sf                       # noqa: E402
import sf.torch as sft          # noqa: E402


# --- an independent recomputation through the Stage 1 public API -----------------------

def stage1_linear(x, W, b, dt, wscale, a):
    """What an SF linear must compute for activation scale ``a``, written with
    nothing but Stage 1 objects (sf.asarray, @, +)."""
    m = sf.info(dt).max
    W = np.asarray(W, dtype=np.float64)
    if wscale == "per_tensor":
        amax = np.abs(W).max()
        s_w = np.full(W.shape[0], amax / m if amax > 0 else 1.0)
    else:
        amax = np.abs(W).max(axis=1)
        s_w = np.where(amax > 0, amax / m, 1.0)
    with sf.errstate(all="ignore"):
        X = sf.asarray(np.asarray(x, dtype=np.float64) / a, dtype=dt)
        Wq = sf.asarray(W / s_w[:, None], dtype=dt)
        acc = X @ Wq.T
        if b is not None:
            acc = acc + sf.asarray(np.asarray(b, dtype=np.float64) / (a * s_w), dtype=dt)
    return acc.numpy() * a * s_w[None, :]


def stage1_saturations(x, W, b, dt, wscale, a):
    """Outputs whose chain or bias add saturated, via the host reference."""
    from sf import _linalg, _vectorized as vec
    fmt = dt._fmt
    m = sf.info(dt).max
    W = np.asarray(W, dtype=np.float64)
    if wscale == "per_tensor":
        s_w = np.full(W.shape[0], np.abs(W).max() / m)
    else:
        s_w = np.abs(W).max(axis=1) / m
    kx, _ = vec.cast_array(fmt, np.asarray(x, dtype=np.float64) / a)
    kw, _ = vec.cast_array(fmt, W / s_w[:, None])
    acc, ev = _linalg.matmul_codes(fmt, kx, kw.T)
    over = ev["overflow"]
    if b is not None:
        kb, _ = vec.cast_array(fmt, np.asarray(b, dtype=np.float64) / (a * s_w))
        _, ev2 = vec.add(fmt, acc, kb[None, :])
        over = over | ev2["overflow"]
    return int(over.sum())


@pytest.mark.parametrize("wscale", ["per_tensor", "per_channel"])
@pytest.mark.parametrize("headroom", ["bound", "none"])
@pytest.mark.parametrize("dtype", ["sf16", "sf12", "sf8"])
def test_sf_linear_matches_stage_1(wscale, headroom, dtype):
    dt = sf.dtype(dtype)
    torch.manual_seed(3)
    for trial in range(6):
        K, N = [(5, 3), (64, 16), (1, 1), (300, 4), (17, 33), (8, 8)][trial]
        lin = nn.Linear(K, N, bias=trial % 2 == 0)
        x = torch.randn(7, K) * (10 ** (trial - 2))
        m = sft.SFLinear.from_float(lin, f"t{trial}", sft.SFConfig(sf.info(dt).bits, wscale, headroom))
        y = m(x)
        st = sft.last_status()[f"t{trial}"]
        b = None if lin.bias is None else lin.bias.detach().numpy()
        ref = stage1_linear(x.numpy(), lin.weight.detach().numpy(), b, dt, wscale, st.act_scale)
        assert y.dtype == torch.float32
        assert np.array_equal(y.detach().double().numpy(), ref.astype(np.float32).astype(np.float64)), trial
        assert st.placement == "host" and st.exact and st.dtype == dtype and st.outputs == 7 * N
        assert st.saturated == stage1_saturations(x.numpy(), lin.weight.detach().numpy(), b, dt,
                                                  wscale, st.act_scale)
        amax = float(x.abs().max())
        a0 = amax / sf.info(dt).max
        assert st.act_scale >= a0 * (1 - 1e-12)
        assert math.isclose(st.headroom_bits, math.log2(st.act_scale / a0), rel_tol=1e-9, abs_tol=1e-12)
        if headroom == "bound":
            assert st.saturated == 0 and st.bias_saturated == 0      # provably, every time
        else:
            assert st.headroom_bits == 0


def test_bound_policy_never_saturates_on_adversarial_inputs():
    """Worst cases for the accumulation chain: every product at full scale
    and the same sign, a long reduction, and a bias far larger than the
    products.  The bound policy must give an exact zero every time."""
    cases = [
        (torch.ones(4, 512), torch.ones(3, 512), torch.zeros(3)),
        (torch.ones(2, 64) * 7.0, torch.ones(2, 64) * -3.0, torch.tensor([1e4, -1e4])),
        (torch.randn(8, 128).abs(), torch.randn(5, 128).abs(), None),
        (torch.zeros(3, 16), torch.randn(4, 16), torch.randn(4)),            # all-zero input
        (torch.randn(6, 32), torch.zeros(2, 32), torch.tensor([0.5, -0.5])),  # all-zero weight
    ]
    for i, (x, W, b) in enumerate(cases):
        for wscale in ("per_tensor", "per_channel"):
            y = sft.linear(x, W, b, scale=wscale, name=f"adv{i}")
            st = sft.last_status()[f"adv{i}"]
            assert st.saturated == 0 and st.bias_saturated == 0 and st.exact, (i, wscale, st)
            ref = F.linear(x, W, b)
            tol = 2.0 ** -(15 - st.headroom_bits - 1) * (float(ref.abs().max()) + 1) * x.shape[1]
            assert float((y - ref).abs().max()) <= tol, (i, wscale)
    # the same inputs without headroom saturate, and the count is exact
    y = sft.linear(torch.ones(4, 512), torch.ones(3, 512), None, headroom="none", name="adv_none")
    st = sft.last_status()["adv_none"]
    assert st.saturated == 12 and st.headroom_bits == 0
    m = sf.info(sf.sf16).max
    s_w = 1.0 / m                                                             # weights are all 1
    assert torch.allclose(y, torch.full_like(y, m * st.act_scale * s_w))      # saturated at max
    assert float(y.max()) < 2.0 < 512.0                                       # the true answer is 512


def test_status_fields_and_filtering():
    a = sft.convert(nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2)))
    b = sft.convert(nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 3)), headroom="none")
    a(torch.randn(2, 4))
    sa = sft.last_status(a)
    assert list(sa) == ["0", "2"] and all(s.placement == "host" for s in sa.values())
    b(torch.randn(2, 4) * 50)
    sb = sft.last_status(b)
    assert sb["2"].outputs == 6                                      # b's entries replaced a's
    assert "SFLayerStatus('0', linear on host, sf16" in repr(sb["0"])
    sft.reset_status()
    assert sft.last_status() == {}
    st = sft.SFLayerStatus("x", "linear", "atreides", "sf16", 4, None, False, None, 0, 0, 0, 1.0)
    assert st.ok is None and "unknown" in repr(st)


# --- conv ------------------------------------------------------------------------------------

@torch.no_grad()
def test_conv2d_is_unfold_plus_sf_linear():
    torch.manual_seed(1)
    for kw in [dict(kernel_size=3, padding=1), dict(kernel_size=(2, 3), stride=2, padding=(1, 0)),
               dict(kernel_size=3, dilation=2, bias=False), dict(kernel_size=1)]:
        conv = nn.Conv2d(3, 5, **kw)
        x = torch.randn(2, 3, 9, 10)
        sc = sft.SFConv2d(conv, "c", sft.SFConfig())
        y = sc(x)
        assert y.shape == conv(x).shape
        cols = F.unfold(x, sc.kernel_size, dilation=sc.dilation, padding=sc.padding, stride=sc.stride)
        y2 = sft.linear(cols.transpose(1, 2), conv.weight.reshape(5, -1), conv.bias, name="c2")
        assert torch.equal(y, y2.transpose(1, 2).reshape(y.shape))
        assert float((y - conv(x)).abs().max()) < 0.02 * float(conv(x).abs().max())
        with sft.autocast(sf.sf16):
            y3 = conv(x)
        assert torch.equal(y, y3)
    with pytest.raises(ValueError):
        sft.SFConv2d(nn.Conv2d(4, 4, 3, groups=2), "g", sft.SFConfig())


# --- convert ------------------------------------------------------------------------------------

def test_convert_replaces_and_reports():
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = nn.Sequential(nn.Conv2d(1, 4, 3, padding=1), nn.ReLU(),
                                       nn.Conv2d(4, 4, 3, groups=2, padding=1),
                                       nn.Conv2d(4, 4, 3, padding="same"))
            self.rnn = nn.LSTM(4, 4)
            self.head = nn.Linear(4, 2)

    net = Net()
    out = sft.convert(net, dtype=sf.sf12, scale="per_channel")
    assert isinstance(out.block[0], sft.SFConv2d) and isinstance(out.head, sft.SFLinear)
    assert isinstance(out.block[2], nn.Conv2d) and isinstance(out.block[3], nn.Conv2d)
    assert isinstance(net.head, nn.Linear)                       # the original is untouched
    assert out.head.sf == sft.SFConfig(12, "per_channel", "bound") and out.head.sf_name == "head"
    rep = sft.conversion_report(out)
    assert [n for n, _ in rep.converted] == ["block.0", "head"]
    skipped = {n: why for n, _, why in rep.skipped}
    assert "groups=2" in skipped["block.2"] and "'same'" in skipped["block.3"] and "rnn" in skipped
    assert "2 converted, 3 skipped" in repr(rep)
    again = sft.convert(out)                                     # idempotent
    assert all("already SF" in k for _, k in sft.conversion_report(again).converted)
    lin = sft.convert(nn.Linear(3, 2))
    assert isinstance(lin, sft.SFLinear)
    with pytest.raises(ValueError):
        sft.conversion_report(net)
    with pytest.raises(ValueError):
        sft.convert(net, scale="per_row")
    with pytest.raises(ValueError):
        sft.convert(net, headroom="some")
    with pytest.raises(sf.SFTypeError):
        sft.convert(net, dtype="sf3")
    inplace = sft.convert(net, inplace=True)
    assert inplace is net and isinstance(net.head, sft.SFLinear)
    assert "dtype=sf16" in repr(net.head) and "headroom=bound" in repr(net.head)


def test_gradients_are_straight_through():
    torch.manual_seed(2)
    lin = nn.Linear(6, 3)
    sl = sft.SFLinear.from_float(lin, "g", sft.SFConfig())
    x = torch.randn(4, 6, requires_grad=True)
    sl(x).pow(2).sum().backward()
    # the STE gradient is the float gradient of x @ W_q^T + b at the SF output
    from sf.torch import _core
    w_q = torch.from_numpy(_core.dequantize_weight(sf.sf16._fmt, lin.weight.detach().double().numpy(),
                                                   "per_tensor")).float()
    y = sl(x).detach()
    g = 2 * y
    assert torch.allclose(x.grad, g @ w_q, atol=1e-6)
    assert torch.allclose(sl.weight.grad, g.T @ x.detach(), atol=1e-5)
    assert torch.allclose(sl.bias.grad, g.sum(0), atol=1e-6)
    # 3-D input, no bias
    sl2 = sft.SFLinear(6, 2, bias=False, name="g2")
    x3 = torch.randn(2, 5, 6, requires_grad=True)
    sl2(x3).sum().backward()
    assert x3.grad.shape == x3.shape and sl2.weight.grad.shape == (2, 6) and sl2.bias is None


# --- autocast and functional matmul -----------------------------------------------------------

def test_autocast_matches_convert():
    torch.manual_seed(4)
    model = nn.Sequential(nn.Linear(10, 20), nn.GELU(), nn.Linear(20, 5))
    x = torch.randn(8, 10)
    conv = sft.convert(model)
    with sft.autocast(sf.sf16):
        y_auto = model(x)
    assert torch.equal(y_auto, conv(x))
    assert any(k.startswith("autocast.linear") for k in sft.last_status())

    @sft.autocast("sf12", prefix="dec.")
    def run(inp):
        return model(inp)

    assert torch.equal(run(x), sft.convert(model, dtype="sf12")(x))
    assert any(k.startswith("dec.linear") for k in sft.last_status())
    with pytest.raises(TypeError):
        @sft.autocast()
        async def nope():
            pass
    # outside the block, plain float again
    assert torch.equal(model(x), F.linear(F.gelu(F.linear(x, model[0].weight, model[0].bias)),
                                          model[2].weight, model[2].bias))


def test_matmul_activations():
    torch.manual_seed(5)
    a = torch.randn(3, 4, 6) * 5
    b = torch.randn(6, 7)
    y = sft.matmul(a, b, name="mm")
    st = sft.last_status()["mm"]
    assert y.shape == (3, 4, 7) and st.op == "matmul" and st.saturated == 0
    # Stage 1 recomputation with the chosen scales
    m = sf.info(sf.sf16).max
    s_b = float(b.abs().max()) / m
    with sf.errstate(all="ignore"):
        A = sf.asarray(a.double().numpy() / st.act_scale, dtype=sf.sf16)
        B = sf.asarray(b.double().numpy() / s_b, dtype=sf.sf16)
        ref = (A @ B).numpy() * st.act_scale * s_b
    assert np.array_equal(y.detach().double().numpy(), ref.astype(np.float32).astype(np.float64))
    assert float((y - a @ b).abs().max()) < 1e-2 * float((a @ b).abs().max())
    v = torch.randn(6)
    assert sft.matmul(v, b).shape == (7,) and sft.matmul(a[0], v).shape == (4,)
    assert sft.matmul(v, v).shape == ()
    with sft.autocast():
        assert torch.equal(a @ b, y) and torch.equal(torch.matmul(a, b), y)
        assert torch.equal(torch.bmm(a, b.expand(3, 6, 7)), y)
    a.requires_grad_(True)
    b.requires_grad_(True)
    sft.matmul(a, b).sum().backward()
    assert a.grad.shape == a.shape and b.grad.shape == b.shape
    assert torch.allclose(b.grad, (a.detach().transpose(-1, -2) @ torch.ones(3, 4, 7)).sum(0), atol=1e-4)
    with pytest.raises(Exception):
        sft.matmul(torch.randn(2, 3), torch.randn(4, 5))


# --- the accuracy gate on a real task ---------------------------------------------------------------

@pytest.fixture(scope="module")
def digits():
    datasets = pytest.importorskip("sklearn.datasets")
    d = datasets.load_digits()
    X = torch.tensor(d.data / 16.0, dtype=torch.float32)
    Y = torch.tensor(d.target)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(0))
    return X, Y, perm[:1400], perm[1400:]


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 8, 3, padding=1)
        self.c2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 1, 8, 8)
        x = F.max_pool2d(torch.relu(self.c1(x)), 2)
        x = F.max_pool2d(torch.relu(self.c2(x)), 2)
        return self.fc(x.flatten(1))


def _train(model, X, Y, idx, epochs, lr=1e-2):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(model(X[idx]), Y[idx]).backward()
        opt.step()
    return model


def _acc(model, X, Y, idx):
    with torch.no_grad():
        return (model(X[idx]).argmax(1) == Y[idx]).float().mean().item()


@pytest.fixture(scope="module")
def trained(digits):
    X, Y, tr, te = digits
    torch.manual_seed(0)
    mlp = _train(nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 10)), X, Y, tr, 60)
    torch.manual_seed(0)
    cnn = _train(CNN(), X, Y, tr, 80)
    return {"mlp": mlp, "cnn": cnn}


@pytest.mark.parametrize("arch", ["mlp", "cnn"])
def test_accuracy_gate(digits, trained, arch, record_property):
    X, Y, tr, te = digits
    model = trained[arch]
    base = _acc(model, X, Y, te)
    assert base > 0.93
    results = {}
    for dtype in ("sf16", "sf12"):
        conv = sft.convert(model, dtype=dtype)
        results[dtype] = _acc(conv, X, Y, te)
        st = sft.last_status(conv)
        assert all(s.saturated == 0 for s in st.values())          # the bound policy, end to end
        assert abs(results[dtype] - base) <= 0.01, (dtype, results[dtype], base)
    none = _acc(sft.convert(model, headroom="none"), X, Y, te)
    record_property("accuracy", {"float": base, **results, "sf16_no_headroom": none})
    assert none < base - 0.3        # without headroom the accumulation saturates: why bound is the default


def test_quantization_aware_finetuning_recovers_sf8(digits, trained):
    """sf8 loses accuracy on the CNN; a few epochs of fine-tuning through the
    straight-through gradient recover most of it."""
    X, Y, tr, te = digits
    conv = sft.convert(trained["cnn"], dtype="sf8")
    before = _acc(conv, X, Y, te)
    torch.manual_seed(0)
    _train(conv, X, Y, tr, 25, lr=3e-3)
    after = _acc(conv, X, Y, te)
    assert after > before + 0.1, (before, after)
