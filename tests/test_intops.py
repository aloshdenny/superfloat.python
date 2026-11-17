"""sf's integer exp, softmax, GELU, SiLU, reciprocal and rsqrt (Alosh, 2026-11-18; spec 1.9.0).

Alosh's ``sf_intops_ref.py`` is the definition: it is kept unchanged as ``tests/data/sf_intops_ref_v0.py`` and every
ported op must give its bits, on every input code where that is cheap.  ``softmax_row`` is the fixed sequence
(three changes, pending his sign-off); ``softmax_row_v0`` must still be his.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sf import _intops as I

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("sf_intops_ref_v0", ROOT / "tests" / "data" / "sf_intops_ref_v0.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)
ALL = np.arange(-32767, 32768, dtype=np.int64)
NONPOS = np.arange(-32767, 1, dtype=np.int64)
ROW = json.loads((ROOT / "tests" / "data" / "softmax_causal_row_example.json").read_text())


# --- the port is Alosh's definition, bit for bit ------------------------------------------------------------------

@pytest.mark.parametrize("k", range(7))
def test_exp_on_every_code(k):
    assert np.array_equal(I.exp_core(NONPOS, k), R.exp_core(NONPOS, k))


@pytest.mark.parametrize("ex", range(7))
def test_gelu_and_silu_on_every_code(ex):
    assert np.array_equal(I.gelu(ALL, ex), R.gelu(ALL, ex))
    assert np.array_equal(I.silu(ALL, ex), R.silu(ALL, ex))


def test_recip_and_rsqrt_on_their_whole_domains():
    q = np.arange(I.HALF, I.MAXC + 1)
    assert np.array_equal(I.recip_norm(q), R.recip_norm(q))
    q = np.arange(I.QUARTER, I.MAXC + 1)
    assert np.array_equal(I.rsqrt_norm(q), R.rsqrt_norm(q))
    assert np.abs(I.recip_norm(np.arange(I.HALF, I.MAXC + 1)) - 2 ** 29 / np.arange(I.HALF, I.MAXC + 1)).max() < 4


def test_tables_are_alosh_s():
    assert I.EXP_TABLE == R.EXP_TABLE and I.SIG_C == R.SIG_C and I.GELU_K == R.GELU_K and I.SILU_K == R.SILU_K
    assert (I.R_A, I.R_B, I.S_A, I.S_B) == (R.R_A, R.R_B, R.S_A, R.S_B)


def test_softmax_v0_and_norms_are_alosh_s():
    rng = np.random.default_rng(1)
    for n in (1, 2, 7, 64, 942, 3000):
        s = np.clip(np.round(rng.normal(0, 1.5, n) / 16 * 32768), -32767, 32767).astype(np.int64)
        assert np.array_equal(I.softmax_row_v0(s, 4), R.softmax_row(s, 4))
    x = np.clip(np.round(rng.normal(0, 0.2, 96) * 32768), -32767, 32767).astype(np.int64)
    g = np.full(96, 20000)
    b = np.full(96, 300)
    assert np.array_equal(I.norm_row(x, g, b, 0, 0, 0, 0), R.norm_row(x, g, b, 0, 0, 0, 0))
    assert np.array_equal(I.norm_row(x, g, b, 0, 0, 0, 0, rms=True), R.norm_row(x, g, b, 0, 0, 0, 0, rms=True))


def test_gelu_and_silu_track_float():
    import math
    for ex in range(7):
        x = ALL * 2.0 ** (ex - 15)
        g = 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))
        assert np.abs(I.gelu(ALL, ex) * 2.0 ** (ex - 15) - g).max() < 2 ** (ex - 15) * 8
        assert np.abs(I.silu(ALL, ex) * 2.0 ** (ex - 15) - x / (1 + np.exp(-x))).max() < 2 ** (ex - 15) * 8


# --- the fixed softmax ------------------------------------------------------------------------------------------

def _codes(x, k=4):
    return np.clip(np.round(np.asarray(x) / 2 ** k * 32768), -32767, 32767).astype(np.int64)


def test_the_causal_row_alosh_sent():
    f = np.array(ROW["float_P"])
    s = _codes(ROW["scores"])
    v0 = I.softmax_row_v0(s, 4) / 32768
    assert abs(v0.sum() - 0.986297607421875) < 1e-12 and np.abs(v0 - f).max() > 3.6e-3      # his numbers
    t, ep = I.softmax_row(s, 4)
    p = t / 32768 * 2.0 ** ep
    assert np.abs(p - f).max() < 3e-4                                                       # 12x better
    rng = np.random.default_rng(2)
    V = rng.normal(0, 1, (len(f), 64))
    err = lambda q: np.linalg.norm(q @ V - f @ V) / np.linalg.norm(f @ V)                   # noqa: E731
    assert err(p) < err(v0) / 4
    # the tail below one ulp of P_max is still 0 (not representable at 15 bits), but it no longer inflates L;
    # what remains of sum P's shortfall is mul's truncation, under one ulp per nonzero P (v0 has the same)
    assert 0 <= f[p > 0].sum() - p.sum() < np.count_nonzero(p) * 2.0 ** (ep - 15)


def test_flat_long_rows_keep_their_bits():
    rng = np.random.default_rng(3)
    x = rng.normal(0, 0.5, 4096)
    f = np.exp(x - x.max())
    f /= f.sum()
    s = _codes(x)
    t, ep = I.softmax_row(s, 4)
    p = t / 32768 * 2.0 ** ep
    v0 = I.softmax_row_v0(s, 4) / 32768
    assert (p == 0).sum() == 0
    rel = lambda q: np.abs(q - f).max() / f.max()                                           # noqa: E731
    assert rel(p) < rel(v0) / 5                                         # v0 shifted log2(n) bits out of every P


def test_q_below_2_14_is_renormalised():
    # a dominant key and a tail whose coarse POP rounds up to a power of two: v0 leaves q at 2^14 - 3
    rng = np.random.default_rng(9)
    for _ in range(3000):
        ei = np.zeros(33, dtype=np.int64)
        ei[0] = 32767
        ei[1:] = (rng.random(32) ** rng.uniform(1, 40) * 32767).astype(np.int64)
        q, _ = R.mx_normalize(ei, 1 - 6, 0, ei - R.relu(ei - 1), 0)
        if 0 < q < R.HALF:
            break
    else:
        pytest.skip("no edge row found")
    # the same row through the fixed softmax: rebuild scores that give these e_i is not needed; the fixed
    # normalisation is checked directly
    S, Si = int(ei.sum()), int((ei - R.relu(ei - 1)).sum())
    pq = lambda sh: R.mx_sum_pop(ei, sh, 0, 0) + R.mx_sum_pop(ei - R.relu(ei - 1), sh - 1, 0, 0)  # noqa: E731
    c = pq(-5)
    _, e = R.normalize(min(c, R.MAXC))
    q0 = pq(-5 + e)
    assert q0 < R.HALF
    q1 = pq(-5 + e + 1)
    assert R.HALF <= q1 <= R.MAXC and S and Si


def test_batched_equals_row_by_row_and_masks():
    rng = np.random.default_rng(4)
    for n in (1, 3, 64, 942):
        s = np.clip(np.round(rng.normal(0, 1, (6, n)) / 16 * 32768), -32767, 32767).astype(np.int64)
        vis = rng.random((6, n)) < 0.7
        vis[:, 0] = True
        t, ep = I.softmax(s, 4, vis)
        for r in range(6):
            c, e = I.softmax_row(s[r], 4, vis[r])
            assert np.array_equal(t[r], c) and ep[r] == e
            assert (t[r][~vis[r]] == 0).all()


def test_masked_keys_change_nothing():
    """A causal row computed with masked future keys appended gives the same P on its visible keys."""
    rng = np.random.default_rng(5)
    s = _codes(rng.normal(0, 1.5, 300))
    t, ep = I.softmax_row(s[:120], 4)
    t2, ep2 = I.softmax_row(s, 4, np.arange(300) < 120)
    assert ep == ep2 and np.array_equal(t, t2[:120]) and (t2[120:] == 0).all()


def test_softmax_rows_on_floats():
    rng = np.random.default_rng(6)
    x = rng.normal(0, 2, (4, 50))
    x[1, 30:] = -np.inf
    x[2] *= 8                                                           # max |score| above 16: redone at k = 5 or 6
    p = I.softmax_rows(x)
    f = np.exp(x - x.max(-1, keepdims=True))
    f /= f.sum(-1, keepdims=True)
    assert (p[1, 30:] == 0).all() and np.abs(p - f).max() < 2e-3
    # masks written as the dtype's lowest value (Hugging Face eager attention) are masks too
    x2 = x.copy()
    x2[1, 30:] = np.finfo(np.float32).min
    x2[3, 10:] = -1e9 + x2[3, 10:]
    p2 = I.softmax_rows(x2)
    x3 = x.copy()
    x3[3, 10:] = -np.inf
    assert np.array_equal(p2, I.softmax_rows(x3))
    # raw scores (Alosh's v0 convention): beyond 2^6 = 64 they saturate and collapse onto one value; with the
    # row max taken first (sf's convention) only the gap to the max matters
    y = np.array([[1728.0, 70.0, 66.0, 0.0], [70.0, 66.0, 0.0, -1e9]])
    raw = I.softmax_rows_raw(y)
    assert raw[1, 0] == raw[1, 1] and abs(raw[0, 0] - 1) > 0.3                     # sink no longer dominates
    f = np.exp(y - y.max(-1, keepdims=True))
    f /= f.sum(-1, keepdims=True)
    assert np.abs(I.softmax_rows(y) - f).max() < 1e-3


# --- golden vectors ------------------------------------------------------------------------------------------------

def _gold(name):
    return json.loads((ROOT / "spec" / "golden" / "intops" / name).read_text())


def test_golden_vectors_replay():
    for c in _gold("exp.json")["cases"]:
        assert I.exp_core(np.array(c["d"]), c["k"]).tolist() == c["out"]
    for c in _gold("gelu.json")["cases"]:
        assert I.gelu(np.array(c["x"]), c["ex"]).tolist() == c["out"]
    for c in _gold("silu.json")["cases"]:
        assert I.silu(np.array(c["x"]), c["ex"]).tolist() == c["out"]
    c = _gold("recip.json")["cases"][0]
    assert I.recip_norm(np.array(c["q"])).tolist() == c["out"]
    c = _gold("rsqrt.json")["cases"][0]
    assert I.rsqrt_norm(np.array(c["q"])).tolist() == c["out"]
    for c in _gold("softmax.json")["cases"]:
        t, ep = I.softmax_row(np.array(c["s"]), c["k"], c.get("visible"))
        assert t.tolist() == c["p"] and ep == c["ep"], c["what"]
        if "v0" in c:
            assert R.softmax_row(np.array(c["s"]), c["k"]).tolist() == c["v0"], c["what"]


def test_the_generator_writes_exactly_these_files():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "gen_golden_intops.py"), "--check"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


# --- torch wrappers --------------------------------------------------------------------------------------------------

def test_torch_wrappers():
    torch = pytest.importorskip("torch")
    import sf.torch as sft
    torch.manual_seed(0)
    x = torch.randn(4, 33) * 3
    assert (sft.int_gelu(x) - torch.nn.functional.gelu(x, approximate="tanh")).abs().max() < 2e-3
    assert (sft.int_silu(x) - torch.nn.functional.silu(x)).abs().max() < 2e-3
    s = torch.randn(2, 3, 9, 9)
    s = s.masked_fill(torch.ones(9, 9, dtype=torch.bool).triu(1), float("-inf"))
    p = sft.int_softmax(s)
    assert p.shape == s.shape and (p - torch.softmax(s, -1)).abs().max() < 1e-3
    assert (sft.int_softmax(s.transpose(-1, -2), dim=-2) - p.transpose(-1, -2)).abs().max() == 0


# --- convert(nonlinear="integer") ------------------------------------------------------------------------------------

def _mlp_attn_model(act):
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv = torch.nn.Linear(32, 96)
            self.up = torch.nn.Linear(32, 64)
            self.act = act
            self.down = torch.nn.Linear(64, 32)

        def forward(self, x):
            q, k, v = self.qkv(x).view(x.shape[0], x.shape[1], 3, 2, 16).permute(2, 0, 3, 1, 4)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(x.shape)
            return x + a + self.down(self.act(self.up(x)))
    torch.manual_seed(0)
    return Block()


@pytest.mark.parametrize("act_name", ["gelu_tanh", "silu"])
def test_convert_runs_nonlinears_on_the_integer_definitions(act_name, monkeypatch):
    torch = pytest.importorskip("torch")
    import sf.torch as sft
    from sf.torch import _intops as TI
    act = torch.nn.GELU(approximate="tanh") if act_name == "gelu_tanh" else torch.nn.SiLU()
    m = _mlp_attn_model(act)
    x = torch.randn(2, 12, 32)
    fl = sft.convert(m, "sf16", accumulate="wide")
    it = sft.convert(m, "sf16", accumulate="wide", nonlinear="integer")
    assert isinstance(it.act, TI.IntGELU if act_name == "gelu_tanh" else TI.IntSiLU)
    rep = sft.conversion_report(it)
    assert ("act", "GELU (tanh)" if act_name == "gelu_tanh" else "SiLU") in rep.nonlinear
    assert ("attention", "softmax (integer)") in rep.nonlinear and "integer" in repr(rep)
    calls = []
    real = TI.int_softmax
    monkeypatch.setattr(TI, "int_softmax", lambda *a, **k: calls.append(1) or real(*a, **k))
    with torch.no_grad():
        yf, yi = fl(x), it(x)
    assert calls and (yf - yi).abs().max() < 2e-2 * yf.abs().max()


def test_convert_nonlinear_edges():
    torch = pytest.importorskip("torch")
    import sf.torch as sft
    m = _mlp_attn_model(torch.nn.GELU())                               # erf GELU: no integer form
    it = sft.convert(m, "sf16", nonlinear="integer")                    # chain mode: attention stays float
    assert isinstance(it.act, torch.nn.GELU)
    rep = sft.conversion_report(it)
    assert any(n == "act" and "erf" in r for n, _, r in rep.skipped)
    assert any("stays float" in w for _, w in rep.nonlinear)
    with pytest.raises(ValueError, match="nonlinear"):
        sft.convert(m, nonlinear="int")


def test_eager_attention_softmax_goes_integer(monkeypatch):
    torch = pytest.importorskip("torch")
    import sf.torch as sft
    from sf.torch import _intops as TI

    class SelfAttention(torch.nn.Module):                               # attention written with matmul
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(16, 16)
            self.k_proj = torch.nn.Linear(16, 16)
            self.v_proj = torch.nn.Linear(16, 16)

        def forward(self, x):
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            s = torch.matmul(q, k.transpose(-1, -2)) / 4
            s = s.masked_fill(torch.ones(x.shape[1], x.shape[1], dtype=torch.bool).triu(1), float("-inf"))
            return torch.matmul(torch.nn.functional.softmax(s, dim=-1, dtype=torch.float32), v)

    torch.manual_seed(1)
    m = torch.nn.Sequential(SelfAttention())
    calls = []
    real = TI.int_softmax
    monkeypatch.setattr(TI, "int_softmax", lambda *a, **k: calls.append(1) or real(*a, **k))
    it = sft.convert(m, "sf16", accumulate="wide", nonlinear="integer")
    fl = sft.convert(m, "sf16", accumulate="wide")
    x = torch.randn(1, 10, 16)
    with torch.no_grad():
        yi, yf = it(x), fl(x)
    assert sft.conversion_report(it).eager_attention == ["0"]
    assert calls and (yi - yf).abs().max() < 1e-2 * yf.abs().max()
