"""Stage 2.0: emulation.  ``sft.quantize``: fake quantization on torch
tensors (values snapped to SF codes, float storage), straight-through
gradient.  Gate: same bits as Stage 1 on identical inputs."""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sf                       # noqa: E402
import sf.torch as sft          # noqa: E402
from conftest import EVENT_ERROR, dtype_of  # noqa: E402

WIDTHS = [4, 8, 12, 16]


def codes_of(t: torch.Tensor, bits: int, scale: float = 1.0) -> list:
    return (t.double() / scale * 2 ** (bits - 1)).round().long().tolist()


@pytest.mark.parametrize("bits", WIDTHS)
def test_same_codes_as_stage_1(bits):
    dt = dtype_of(bits)
    rng = np.random.default_rng(bits)
    lsb = 2.0 ** -(bits - 1)
    m = sf.info(dt).max_code
    values = np.concatenate([
        rng.uniform(-0.999, 0.999, 3000),
        (np.arange(-m, m) + 0.5) * lsb if bits <= 12 else (rng.integers(-m, m, 3000) + 0.5) * lsb,  # ties
        [0.0, -0.0, lsb / 2, -lsb / 2, lsb / 4, sf.info(dt).max, -sf.info(dt).max],
    ])
    t = torch.tensor(values, dtype=torch.float64)
    with sf.errstate(all="ignore"):
        q = sft.quantize(t, dtype=dt)
        ref = sf.asarray(values, dtype=dt)
    assert codes_of(q, bits) == ref.codes.tolist()
    with sf.errstate(all="ignore"):
        assert sft.to_sf(t, dt).codes.tolist() == ref.codes.tolist()
        q32 = sft.quantize(t.float(), dtype=dt)          # float32 output holds every value exactly
    assert q32.dtype == torch.float32
    assert np.array_equal(q32.double().numpy() * 2 ** (bits - 1),
                          np.round(q32.double().numpy() * 2 ** (bits - 1)))


@pytest.mark.parametrize("bits", WIDTHS)
@pytest.mark.parametrize("policy", ["raise", "warn", "ignore"])
def test_same_events_as_stage_1(bits, policy):
    """Under every policy, quantize raises / warns exactly when Stage 1 does."""
    import warnings
    dt = dtype_of(bits)
    cases = [[0.5], [1.5], [-1.0], [float("nan")], [float("inf")], [1e-12], [0.25, 2.0], [0.1, 1e-12]]

    def outcome(fn):
        with sf.errstate(all=policy), warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                fn()
                raised = None
            except sf.SFError as e:
                raised = type(e)
        return raised, sorted({x.category.__name__ for x in w})

    for values in cases:
        a = outcome(lambda: sf.asarray(np.array(values), dtype=dt))
        b = outcome(lambda: sft.quantize(torch.tensor(values, dtype=torch.float64), dtype=dt))
        assert a == b, (values, a, b)


def test_scaled_modes():
    x = torch.tensor([3.0, -1.5, 0.75, 0.0])
    q = sft.quantize(x, scale="dynamic")
    s = 3.0 / sf.info(sf.sf16).max
    assert codes_of(q, 16, s) == [32767, -16384, 8192, 0]
    assert torch.allclose(q, x, atol=s / 32768 * 0.51)
    q2 = sft.quantize(x, scale=4.0)
    assert codes_of(q2, 16, 4.0) == [24576, -12288, 6144, 0]
    with pytest.raises(sf.SFOverflowError):
        sft.quantize(x, scale=2.0)                       # 3 / 2 = 1.5 does not fit
    with pytest.raises(ValueError):
        sft.quantize(x, scale=-1.0)
    with pytest.raises(TypeError):
        sft.quantize([0.5])
    assert sft.quantize(torch.zeros(3), scale="dynamic").tolist() == [0.0, 0.0, 0.0]
    assert sft.quantize(torch.tensor([0, 0]), dtype="sf8").dtype == torch.float64   # ints -> float64


def test_straight_through_gradient():
    x = torch.tensor([0.5, -0.25, 1.5, -2.0, 0.999], requires_grad=True)
    with sf.errstate(all="ignore"):
        sft.quantize(x).sum().backward()
    assert x.grad.tolist() == [1.0, 1.0, 0.0, 0.0, 1.0]     # clipped where it saturated
    y = torch.randn(10, requires_grad=True)
    with sf.errstate(under="ignore"):
        (sft.quantize(y, scale="dynamic") * 3).sum().backward()
    assert y.grad.tolist() == [3.0] * 10                     # dynamic scale never clips
    z = torch.tensor([1.0, 3.0], requires_grad=True)
    with sf.errstate(all="ignore"):
        sft.quantize(z, scale=2.0).sum().backward()
    assert z.grad.tolist() == [1.0, 0.0]
    # gradient check of the STE against the identity inside the range
    w = torch.rand(6, dtype=torch.float64) * 0.9
    w.requires_grad_(True)
    out = sft.quantize(w)
    g = torch.autograd.grad(out.sum(), w)[0]
    assert torch.equal(g, torch.ones_like(w))


def test_dtypes():
    x = torch.tensor([0.5, 0.25], dtype=torch.bfloat16)
    q = sft.quantize(x)
    assert q.dtype == torch.float32                          # bfloat16 cannot hold SF16 values
    assert sft.quantize(torch.tensor([0.5], dtype=torch.float64)).dtype == torch.float64
    # 32767/32768 is not representable in bfloat16 or float16: it rounds to 1.0
    v = 32767 / 32768
    assert float(torch.tensor(v, dtype=torch.bfloat16)) == 1.0
    assert float(torch.tensor(v, dtype=torch.float16)) == 1.0
    assert float(sft.quantize(torch.tensor([v]))[0]) == v   # float32 holds it
    a = sf.array([0.5, -0.25, v], dtype=sf.sf16)
    t = sft.from_sf(a)
    assert t.dtype == torch.float32 and t.tolist() == [0.5, -0.25, v]
    assert sft.from_sf(sf.sf16(0.5)).item() == 0.5
    with pytest.raises(sf.SFTypeError):
        sft.from_sf(a, torch.bfloat16)
    assert sft.to_sf(t, "sf16", device="atreides").device == "atreides"


def test_quantize_under_compile():
    def f(x):
        return sft.quantize(x, scale="dynamic") * 2

    x = torch.randn(16)
    c = torch.compile(f, backend="superfloat")
    with sf.errstate(under="ignore"):
        assert torch.equal(c(x), f(x))
