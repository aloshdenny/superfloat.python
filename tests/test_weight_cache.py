"""Weights prepared once for the host's wide path (sf.torch._ops.weight_cache).

The cache must never change a result: every output and status field equals
the unprepared path, a weight changed in place is prepared again, and a new
tensor allocated where a freed one was is never mistaken for it."""
from __future__ import annotations

import gc

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import sf.torch as sft  # noqa: E402
from sf import _fastmx  # noqa: E402
from sf.torch import _core, _ops  # noqa: E402

pytestmark = pytest.mark.skipif(not _fastmx.available(), reason="the cache needs the compiled MX loop")
FIELDS = ("saturated", "underflow", "flushed", "acc_clipped", "redone", "outputs", "exact", "shift_min",
          "shift_max", "h", "guard_raised", "weight_underflow", "input_underflow", "bias_saturated")


@pytest.fixture(autouse=True)
def fresh_cache():
    c = _ops.weight_cache
    old = c.budget
    c.clear()
    yield c
    c.budget = old
    c.clear()


@pytest.mark.parametrize("bits", [16, 12, 8, 4])
@pytest.mark.parametrize("mx", [None, "v4.8.1"])
@pytest.mark.parametrize("shape,block", [((3, 64, 10), 1 << 23), ((5, 96, 301), 96 * 23), ((1, 512, 40), 1 << 23)])
def test_prepared_weights_give_the_same_bits_and_status(bits, mx, shape, block, monkeypatch):
    monkeypatch.setattr(_core, "WIDE_BLOCK", block)
    M, K, N = shape
    rng = np.random.default_rng(bits * 31 + N)
    x = rng.normal(size=(M, K))
    x[0, :] = 0.0
    x[0, 3] = 4.0                                              # outputs near max: redos
    w = rng.normal(size=(N, K)) * 0.1
    w[N // 2] = 0.9
    b = rng.normal(size=N) * 0.3
    one = _core.wide_linear(x, w, b, bits, mx=mx)
    two = _core.wide_linear(x, None, b, bits, mx=mx, prepared=_core.prepare_weight(w, bits))
    assert np.array_equal(one.y, two.y)
    assert {f: getattr(one, f) for f in FIELDS} == {f: getattr(two, f) for f in FIELDS}


def _model():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(32, 48), torch.nn.GELU(), torch.nn.Linear(48, 8)).eval()


def test_a_model_gives_the_same_outputs_with_and_without_the_cache(fresh_cache):
    c = sft.convert(_model(), accumulate="wide")
    x = torch.randn(4, 32)
    fresh_cache.budget = 0
    off = c(x)
    fresh_cache.budget = 1 << 30
    on1, on2 = c(x), c(x)
    assert torch.equal(off, on1) and torch.equal(off, on2)
    assert len(fresh_cache.items) == 2 and fresh_cache.hits == 2 and fresh_cache.misses == 2


def test_a_weight_changed_in_place_is_prepared_again(fresh_cache):
    c = sft.convert(_model(), accumulate="wide")
    x = torch.randn(4, 32)
    before = c(x)
    with torch.no_grad():
        c[0].weight.mul_(0.5)                                  # bumps the version counter
    after = c(x)
    fresh_cache.budget = 0
    assert torch.equal(after, c(x)) and not torch.equal(before, after)


def test_a_new_tensor_at_a_freed_address_is_not_mistaken(fresh_cache):
    """Build and drop models of the same shape until one reuses a freed
    address; every one must match its own uncached result."""
    x = torch.randn(3, 32)
    for seed in range(6):
        torch.manual_seed(seed)
        c = sft.convert(torch.nn.Sequential(torch.nn.Linear(32, 16)).eval(), accumulate="wide")
        fresh_cache.budget = 1 << 30
        on = c(x)
        fresh_cache.budget = 0
        assert torch.equal(on, c(x)), seed
        del c
        gc.collect()


def test_slices_and_transposes_share_their_parameter(fresh_cache):
    """MultiheadAttention passes slices of in_proj_weight and Conv1D a
    transpose: each view is cached separately and stays right."""
    torch.manual_seed(1)
    m = torch.nn.MultiheadAttention(16, 4).eval()
    c = sft.convert(torch.nn.Sequential(m), accumulate="wide")
    x = torch.randn(5, 2, 16)
    with torch.no_grad():
        on = c[0](x, x, x)[0]
        fresh_cache.budget = 0
        off = c[0](x, x, x)[0]
    assert torch.equal(on, off)


def test_the_budget_evicts_least_recently_used(fresh_cache):
    c = sft.convert(_model(), accumulate="wide")
    x = torch.randn(2, 32)
    one = _core.prepare_weight(c[0].weight.detach().double().numpy(), 16).nbytes
    fresh_cache.budget = one + 10                              # room for the first layer only
    c(x)
    assert len(fresh_cache.items) == 1 and fresh_cache.bytes <= fresh_cache.budget


def test_non_finite_weights_are_not_cached(fresh_cache):
    w = np.ones((4, 8))
    w[1, 2] = np.nan
    assert _core.prepare_weight(w, 16) is None


def test_prepared_for_another_width_is_refused():
    pw = _core.prepare_weight(np.ones((4, 8)), 16)
    with pytest.raises(ValueError, match="prepared for sf16"):
        _core.wide_linear(np.ones((2, 8)), None, None, 8, prepared=pw)
