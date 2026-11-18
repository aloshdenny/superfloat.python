"""Every matmul of a model on sf, or the reason it is not (closing Stage 2).

Alosh's questions, 2026-11-17: does ``sft.convert`` find every layer, give
it the right width, read its bias, and cope with MoE and block-scaled
models?  Before this, three kinds of matmul ran in float without a word in
the report: MoE experts kept in one 3-D parameter (and their router),
attention written with ``torch.matmul`` instead of SDPA, and weights in a
quantized format, which were read as if they were float (FP8 with block
scales came out 100 times too large).

The classes below copy the layouts of Hugging Face's modules, so these tests
need no ``transformers``; the last tests run the real models when it is
installed.
"""
from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

import sf  # noqa: E402
import sf.kernels  # noqa: E402,F401
import sf.torch as sft  # noqa: E402
from sf.torch import _scopes  # noqa: E402

aten = torch.ops.aten
FLOAT_MATMULS = {aten.mm.default, aten.addmm.default, aten.bmm.default, aten.baddbmm.default,
                 aten.matmul.default, aten.linear.default}
for _n in ("_grouped_mm",):
    if hasattr(aten, _n):
        FLOAT_MATMULS.add(getattr(aten, _n).default)


class FloatMatmuls(TorchDispatchMode):
    """Counts matmuls that still run in torch (float) rather than on sf."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in FLOAT_MATMULS and _inner_dim(func, args) != 1:
            self.calls.append(str(func))
        return func(*args, **(kwargs or {}))


def _inner_dim(func, args):
    """K of a matmul call.  K = 1 is an outer product, no accumulation: transformers < 5.20 builds RoPE's angles
    as inv_freq (d/2, 1) @ positions (1, T) with bmm, host-side position bookkeeping like the norms."""
    a = args[1] if func in (aten.addmm.default, aten.baddbmm.default) else args[0]
    return a.shape[-1] if isinstance(a, torch.Tensor) and a.dim() else None


def float_matmuls(model, *a):
    with torch.no_grad(), FloatMatmuls() as c:
        out = model(*a)
    return c.calls, out


# --- copies of the Hugging Face MoE layout ---------------------------------------------------------

class Experts(nn.Module):
    """Like MixtralExperts after ``use_experts_implementation``: every expert in
    one 3-D parameter, layout flags, the model's own loop over experts."""

    def __init__(self, E=4, H=16, I=24, transposed=False, bias=False):
        super().__init__()
        self.num_experts, self.has_gate, self.has_bias, self.is_transposed = E, True, bias, transposed
        self.has_post_expert_norm = False
        shape_gu, shape_d = ((E, H, 2 * I), (E, I, H)) if transposed else ((E, 2 * I, H), (E, H, I))
        self.gate_up_proj = nn.Parameter(torch.randn(shape_gu) * 0.2)
        self.down_proj = nn.Parameter(torch.randn(shape_d) * 0.2)
        if bias:
            self.gate_up_proj_bias = nn.Parameter(torch.randn(E, 2 * I) * 0.1)
            self.down_proj_bias = nn.Parameter(torch.randn(E, H) * 0.1)
        self.act_fn = nn.SiLU()

    def _apply_gate(self, gu):
        g, u = gu.chunk(2, dim=-1)
        return self.act_fn(g) * u

    def forward(self, hidden, top_k_index, top_k_weights):
        out = torch.zeros_like(hidden)
        mask = F.one_hot(top_k_index, self.num_experts).permute(2, 1, 0)
        for e in range(self.num_experts):
            pos, tok = torch.where(mask[e])
            if tok.numel() == 0:
                continue
            w_gu = self.gate_up_proj[e].t() if self.is_transposed else self.gate_up_proj[e]
            w_d = self.down_proj[e].t() if self.is_transposed else self.down_proj[e]
            h = F.linear(hidden[tok], w_gu, self.gate_up_proj_bias[e] if self.has_bias else None)
            y = F.linear(self._apply_gate(h), w_d, self.down_proj_bias[e] if self.has_bias else None)
            out.index_add_(0, tok, y * top_k_weights[tok, pos, None])
        return out


class TopKRouter(nn.Module):
    def __init__(self, E=4, H=16, k=2):
        super().__init__()
        self.top_k = k
        self.weight = nn.Parameter(torch.randn(E, H) * 0.3)

    def forward(self, h):
        logits = F.linear(h, self.weight)
        p = torch.softmax(logits.float(), dim=-1)
        v, i = torch.topk(p, self.top_k, dim=-1)
        return logits, v / v.sum(-1, keepdim=True), i


class SparseMoeBlock(nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.gate = TopKRouter()
        self.experts = Experts(**kw)

    def forward(self, x):
        b, s, h = x.shape
        flat = x.reshape(-1, h)
        _, w, i = self.gate(flat)
        return self.experts(flat, i, w).reshape(b, s, h)


class MoeLM(nn.Module):
    def __init__(self, vocab=32, **kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, 16)
        self.attn_proj = nn.Linear(16, 16)
        self.moe = SparseMoeBlock(**kw)
        self.head = nn.Linear(16, vocab, bias=False)

    def get_output_embeddings(self):
        return self.head

    def forward(self, ids):
        x = self.emb(ids)
        return self.head(self.moe(torch.tanh(self.attn_proj(x))))


IDS = torch.tensor([[1, 5, 9, 3, 7, 2, 11, 30]])


@pytest.mark.parametrize("kw", [{}, {"transposed": True, "bias": True}], ids=["mixtral", "gpt-oss-layout"])
def test_moe_experts_and_router_run_on_sf_and_match_an_explicit_loop(kw):
    torch.manual_seed(0)
    m = MoeLM(**kw).eval()
    c = sft.convert(m, dtype="sf16", accumulate="wide")
    left, y = float_matmuls(c, IDS)
    assert left == []                                                     # nothing in float
    kinds = dict(sft.conversion_report(c).converted)
    assert kinds["moe.gate"] == "MoE router" and kinds["moe.experts"].startswith("MoE experts (4 experts")

    # the same maths written out with sft.linear on each expert's tokens
    ex = c.moe.experts
    with torch.no_grad():
        x = torch.tanh(sft.linear(c.emb(IDS), c.attn_proj.weight, c.attn_proj.bias, accumulate="wide"))
        flat = x.reshape(-1, 16)
        logits = sft.linear(flat, c.moe.gate.weight, accumulate="wide")
        p = torch.softmax(logits.float(), -1)
        v, i = torch.topk(p, 2, -1)
        w = v / v.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for e in range(4):
            pos, tok = torch.where((i == e).t())
            if tok.numel() == 0:
                continue
            wg = ex.gate_up_proj[e].t() if ex.is_transposed else ex.gate_up_proj[e]
            wd = ex.down_proj[e].t() if ex.is_transposed else ex.down_proj[e]
            h = sft.linear(flat[tok], wg, ex.gate_up_proj_bias[e] if ex.has_bias else None, accumulate="wide")
            yy = sft.linear(ex._apply_gate(h), wd, ex.down_proj_bias[e] if ex.has_bias else None, accumulate="wide")
            out.index_add_(0, tok, yy * w[tok, pos, None])
        want = sft.linear(out.reshape(1, -1, 16), c.head.weight, accumulate="wide")
    assert torch.equal(y, want)
    # and close to float
    with torch.no_grad():
        f = m(IDS)
    assert ((y - f).norm() / f.norm()) < 1e-3


def test_moe_statuses_name_each_expert_and_the_router_is_kept_at_sf16():
    torch.manual_seed(1)
    c = sft.convert(MoeLM(), dtype="sf8", accumulate="wide")
    rep = sft.conversion_report(c)
    assert dict(rep.kept_sf16) == {"moe.gate": "MoE router (picks the experts)",
                                   "head": "LM head (the model's get_output_embeddings())"}
    assert rep.widths == {"attn_proj": 8, "moe.gate": 16, "moe.experts": 8, "head": 16}
    assert "widths: sf8 x 2, sf16 x 2" in repr(rep)
    with sft.capture_status() as st, torch.no_grad():
        c(IDS)
    names = set(st)
    assert {"moe.gate", "attn_proj", "head"} <= names
    assert any(n.startswith("moe.experts.gate_up_proj[") for n in names)
    assert st["moe.gate"].dtype == "sf16" and all(st[n].dtype == "sf8" for n in names if "experts" in n)


def test_moe_compiled_on_the_target_equals_the_host_bit_for_bit():
    torch.manual_seed(2)
    c = sft.convert(MoeLM(), dtype="sf8", accumulate="wide").eval()
    with torch.no_grad():
        want = c(IDS)
        torch._dynamo.reset()
        n0 = len(sft.compile_reports())
        with sf.config(default_device="atreides", preflight="ignore"):
            got = torch.compile(c, backend="superfloat")(IDS)
    assert torch.equal(got, want)
    # the model compiles to several graphs (5 on torch 2.9 / 2.11: graph breaks at the host round trips), and
    # compile_report() is only the last one (the LM head): read every graph this call compiled
    on = {layer for r in sft.compile_reports()[n0:] for *_, layer, _ in r.on_device}
    assert any(n.startswith("moe.experts.down_proj[") for n in on) and "moe.gate" in on


def test_the_dense_expert_path_gives_the_sparse_paths_bits(monkeypatch):
    """Under torch.compile every expert runs on every token, weighted by 0 where
    the token was not routed there; wide mode computes rows independently."""
    torch.manual_seed(3)
    c = sft.convert(MoeLM(), dtype="sf16", accumulate="wide")
    with torch.no_grad():
        sparse = c(IDS)
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
        dense = c(IDS)
    assert torch.equal(sparse, dense)


def test_converted_moe_keeps_its_state_dict_and_survives_deepcopy():
    torch.manual_seed(4)
    m = MoeLM()
    c = sft.convert(m, dtype="sf16", accumulate="wide")
    assert list(c.state_dict()) == list(m.state_dict())
    d = copy.deepcopy(c)
    with torch.no_grad():
        assert torch.equal(d(IDS), c(IDS))
        d.moe.experts.down_proj.mul_(2)                       # the copy runs its own weights
        assert not torch.equal(d(IDS), c(IDS))


def test_an_experts_module_without_the_flags_is_taken_through_its_own_calls():
    class PlainExperts(Experts):
        def __init__(self):
            super().__init__()
            for a in ("has_gate", "has_bias", "is_transposed"):
                delattr(self, a)

        def forward(self, hidden, top_k_index, top_k_weights):
            out = torch.zeros_like(hidden)
            for e in range(self.num_experts):
                pos, tok = torch.where((top_k_index == e).t())
                h = F.linear(hidden[tok], self.gate_up_proj[e])
                out.index_add_(0, tok, F.linear(self._apply_gate(h), self.down_proj[e]) * top_k_weights[tok, pos, None])
            return out

    torch.manual_seed(5)
    m = MoeLM()
    m.moe.experts = PlainExperts()
    c = sft.convert(m, dtype="sf16", accumulate="wide")
    left, _ = float_matmuls(c, IDS)
    assert left == []
    assert "F.linear calls taken" in dict(sft.conversion_report(c).converted)["moe.experts"]


# --- attention written with torch.matmul ------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    def forward(self, pos):
        inv = torch.arange(1, 5, dtype=torch.float32)[None, :, None]
        return torch.matmul(inv, pos[:, None, :].float())             # angles: must stay float


class EagerSelfAttention(nn.Module):
    def __init__(self, d=16, h=2):
        super().__init__()
        self.h, self.dh = h, d // h
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.rotary = RotaryEmbedding()

    def forward(self, x):
        b, s, d = x.shape
        ang = self.rotary(torch.arange(s)[None])
        q, k, v = (t(x).view(b, s, self.h, self.dh).transpose(1, 2) for t in (self.q, self.k, self.v))
        q = q * torch.cos(ang).mean()
        scores = torch.matmul(q, k.transpose(2, 3)) / self.dh ** 0.5
        scores = scores.masked_fill(torch.ones(s, s, dtype=torch.bool).triu(1), float("-inf"))
        p = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        return self.o(torch.matmul(p, v).transpose(1, 2).reshape(b, s, d))


def test_eager_attention_runs_both_matmuls_on_mx_and_the_rotary_stays_float():
    torch.manual_seed(6)
    m = nn.Sequential(EagerSelfAttention()).eval()
    x = torch.randn(1, 6, 16)
    c = sft.convert(m, dtype="sf16", accumulate="wide")
    assert sft.conversion_report(c).eager_attention == ["0"]
    with sft.capture_status() as st:
        left, y = float_matmuls(c, x)
    assert left == ["aten.bmm.default"]                                  # the rotary angles, nothing else
    assert {"0.scores", "0.pv"} <= set(st) and st["0.pv"].dtype == "sf16"

    # the same as sft.attention's two ops written out
    a = c[0]
    with torch.no_grad():
        q, k, v = (t(x).view(1, 6, 2, 8).transpose(1, 2) for t in (a.q, a.k, a.v))
        q = q * torch.cos(a.rotary(torch.arange(6)[None])).mean()
        s = torch.ops.sf.matmul_wide(q, k.transpose(2, 3), 16, None, "s") / 8 ** 0.5
        s = s.masked_fill(torch.ones(6, 6, dtype=torch.bool).triu(1), float("-inf"))
        p = F.softmax(s, dim=-1, dtype=torch.float32)
        want = a.o(torch.ops.sf.matmul_wide_pv(p, v, 16, "pv").transpose(1, 2).reshape(1, 6, 16))
    assert torch.equal(y, want)


def test_eager_attention_compiles_to_the_target_with_the_same_bits():
    torch.manual_seed(7)
    c = sft.convert(nn.Sequential(EagerSelfAttention()).eval(), dtype="sf16", accumulate="wide")
    x = torch.randn(1, 5, 16)
    with torch.no_grad():
        want = c(x)
        torch._dynamo.reset()
        with sf.config(default_device="atreides", preflight="ignore"):
            got = torch.compile(c, backend="superfloat")(x)
    assert torch.equal(got, want)
    assert {"0.scores", "0.pv"} <= {layer for *_, layer, _ in sft.compile_report().on_device}


def test_eager_attention_is_left_alone_in_chain_mode_or_with_attention_off():
    m = nn.Sequential(EagerSelfAttention())
    for kw in ({"accumulate": "chain"}, {"accumulate": "wide", "attention": False}):
        c = sft.convert(m, **kw)
        assert sft.conversion_report(c).eager_attention == []
        left, _ = float_matmuls(c, torch.randn(1, 4, 16))
        assert left.count("aten.bmm.default") == 3                         # rotary + scores + pv


# --- weights in other formats ---------------------------------------------------------------------

FP8 = getattr(torch, "float8_e4m3fn", None)


class FP8BlockLinear(nn.Linear):
    """Like transformers' FP8Linear: float8 weight, one float scale per block."""

    def __init__(self, out, inn, block=(8, 8)):
        super().__init__(inn, out, bias=True)
        self.block_size = block
        self.weight = nn.Parameter(torch.empty(out, inn, dtype=FP8), requires_grad=False)
        self.weight_scale_inv = nn.Parameter(torch.rand(-(-out // block[0]), -(-inn // block[1])) * 0.01 + 1e-3,
                                             requires_grad=False)

    def forward(self, x):                                                  # the real one calls a kernel
        raise RuntimeError("FP8 kernel not available on CPU")


@pytest.mark.skipif(FP8 is None, reason="no float8 in this torch")
@pytest.mark.parametrize("block", [(8, 8), (16, 8), None])
def test_fp8_block_scaled_weights_are_read_with_their_scales(block):
    torch.manual_seed(8)
    lin = FP8BlockLinear(20, 24, block or (8, 8))                         # 20 rows: a partial last block
    with torch.no_grad():
        lin.weight.copy_((torch.randn(20, 24) * 60).clamp(-440, 440).to(FP8))
        if block is None:
            lin.block_size = None
            lin.weight_scale_inv = nn.Parameter(torch.tensor(0.004), requires_grad=False)
    s = lin.weight_scale_inv
    full = s if s.numel() == 1 else s.repeat_interleave(lin.block_size[0], 0)[:20].repeat_interleave(
        lin.block_size[1], 1)[:, :24]
    dq = lin.weight.float() * full
    c = sft.convert(nn.Sequential(lin), dtype="sf16", accumulate="wide")
    kind = dict(sft.conversion_report(c).converted)["0"]
    assert kind.startswith("Linear (FP8 weights with scales in ") and "weight_scale_inv" in kind
    x = torch.randn(3, 24)
    with torch.no_grad():
        assert torch.equal(c(x), sft.linear(x, dq, lin.bias, accumulate="wide"))
        assert ((c(x) - F.linear(x, dq, lin.bias)).norm() / F.linear(x, dq, lin.bias).norm()) < 1e-3


def test_fp8_without_scales_and_badly_tiled_scales_are_refused():
    if FP8 is None:
        pytest.skip("no float8")
    lin = FP8BlockLinear(16, 16)
    del lin.weight_scale_inv
    rep = sft.conversion_report(sft.convert(nn.Sequential(lin), accumulate="wide"))
    assert rep.converted == [] and "no scale" in rep.skipped[0][2]
    lin = FP8BlockLinear(16, 16)
    lin.block_size = (5, 5)
    rep = sft.conversion_report(sft.convert(nn.Sequential(lin), accumulate="wide"))
    assert rep.converted == [] and "do not tile" in rep.skipped[0][2]


class Int8Linear(nn.Linear):
    """An nn.Linear subclass with int8 codes in .weight (bitsandbytes-style)."""

    def __init__(self, src):
        super().__init__(src.in_features, src.out_features)
        s = src.weight.abs().amax(1, keepdim=True) / 127
        self.weight = nn.Parameter((src.weight / s).round().to(torch.int8), requires_grad=False)
        self.register_buffer("scale", s.detach())
        self.bias = src.bias

    def forward(self, x):
        return F.linear(x, self.weight.float() * self.scale, self.bias)


class GPTQStyle(nn.Module):
    def __init__(self, src):
        super().__init__()
        self.register_buffer("qweight", (src.weight * 50).round().to(torch.int32))
        self.bias = src.bias

    def forward(self, x):
        return F.linear(x, self.qweight.float() / 50, self.bias)


class LoggedLinear(nn.Linear):
    def forward(self, x):
        return super().forward(x) * 2                                     # not what the weight alone says


@pytest.mark.parametrize("make, why", [(Int8Linear, "weights stored as torch.int8"),
                                       (GPTQStyle, "holds quantized weights (qweight: torch.int32)"),
                                       (LoggedLinear, "subclass of nn.Linear with its own forward")])
def test_layers_sf_cannot_read_are_skipped_with_the_reason_and_run_as_they_were(make, why):
    torch.manual_seed(9)
    src = nn.Linear(12, 6)
    layer = make(src) if make is not LoggedLinear else LoggedLinear(12, 6)
    m = nn.Sequential(layer, nn.ReLU())
    c = sft.convert(m, dtype="sf16", accumulate="wide")
    rep = sft.conversion_report(c)
    assert rep.converted == [] and why in rep.skipped[0][2] and rep.skipped[0][0] == "0"
    x = torch.randn(2, 12)
    with torch.no_grad():
        assert torch.equal(c(x), m(x))                                    # unchanged, not wrong


def test_the_report_lists_weights_no_rule_converted_and_not_embeddings():
    class Mixer(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Parameter(torch.randn(8, 8))
            self.position_embeddings = nn.Parameter(torch.randn(1, 4, 8))
            self.register_buffer("position_ids", torch.arange(4)[None])

        def forward(self, x):
            return x @ self.proj + self.position_embeddings

    m = nn.Sequential(nn.Embedding(10, 8), Mixer(), nn.Linear(8, 8))
    rep = sft.conversion_report(sft.convert(m, accumulate="wide"))
    assert rep.skipped == [("1", "Mixer", "holds weights sf does not convert (proj); its own maths runs in float")]
    assert [n for n, _ in rep.converted] == ["2"]


def test_read_weight_and_dequantize_helpers():
    w = torch.arange(12.0).reshape(3, 4)
    s = torch.tensor([[1.0, 10.0], [100.0, 1000.0]])
    got = _scopes.dequantize_block_scaled(w, s, (2, 2))
    assert got[0].tolist() == [0, 1, 20, 30] and got[2].tolist() == [800, 900, 10000, 11000]
    assert torch.equal(_scopes.dequantize_block_scaled(w, s), got)                # blocks found from shapes
    assert _scopes.read_weight(nn.Linear(2, 2)) == (None, None, None)


# --- the real Hugging Face models, when transformers is installed ---------------------------------

SMALL = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=2, max_position_embeddings=32)


def _hf_models():
    T = pytest.importorskip("transformers")
    return {
        "mixtral": lambda: T.MixtralForCausalLM(T.MixtralConfig(**SMALL, num_local_experts=4, num_experts_per_tok=2)),
        "qwen3-moe": lambda: T.Qwen3MoeForCausalLM(T.Qwen3MoeConfig(**SMALL, num_experts=4, num_experts_per_tok=2,
                                                                    moe_intermediate_size=32, head_dim=8)),
        "llama-eager": lambda: T.LlamaForCausalLM(T.LlamaConfig(**SMALL, attn_implementation="eager")),
        "gemma2": lambda: T.Gemma2ForCausalLM(T.Gemma2Config(**SMALL, head_dim=8)),
        "qwen2-moe": lambda: T.Qwen2MoeForCausalLM(T.Qwen2MoeConfig(**SMALL, num_experts=4, num_experts_per_tok=2,
                                                                    moe_intermediate_size=32,
                                                                    shared_expert_intermediate_size=32)),
        # multi-head latent attention: one key-value head per query head (2 would be an invalid config, which
        # transformers 5.20 refuses in float too)
        "deepseek-v3": lambda: T.DeepseekV3ForCausalLM(T.DeepseekV3Config(
            **dict(SMALL, num_key_value_heads=4), n_routed_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            n_shared_experts=1, first_k_dense_replace=1, kv_lora_rank=16, q_lora_rank=16, qk_rope_head_dim=8,
            qk_nope_head_dim=8, v_head_dim=8, n_group=1, topk_group=1)),
        "gpt-oss": lambda: T.GptOssForCausalLM(T.GptOssConfig(**SMALL, num_local_experts=4, num_experts_per_tok=2,
                                                              head_dim=8)),
    }


@pytest.mark.parametrize("name", ["mixtral", "qwen3-moe", "llama-eager", "gemma2", "qwen2-moe", "deepseek-v3",
                                  "gpt-oss"])
def test_real_hf_models_have_no_float_matmul_left(name):
    torch.manual_seed(10)
    try:
        m = _hf_models()[name]().eval()
    except AttributeError as e:                       # a model class this transformers does not have
        pytest.skip(str(e))
    c = sft.convert(m, dtype="sf16", accumulate="wide")
    ids = torch.randint(0, 128, (1, 7))
    left, y = float_matmuls(c, ids)
    assert left == []
    with torch.no_grad():
        f = m(ids).logits
    assert ((y.logits - f).norm() / f.norm()) < 1e-3


@pytest.mark.skipif(FP8 is None, reason="no float8 in this torch")
def test_real_fp8linear_from_transformers():
    fp8 = pytest.importorskip("transformers.integrations.finegrained_fp8")
    lin = fp8.FP8Linear(256, 128, block_size=(128, 128))
    with torch.no_grad():
        lin.weight.copy_((torch.randn(128, 256) * 60).clamp(-440, 440).to(FP8))
        lin.weight_scale_inv.copy_(torch.tensor([[0.001, 0.004]]))
    dq = lin.weight.float() * lin.weight_scale_inv.repeat_interleave(128, 1)
    c = sft.convert(nn.Sequential(lin), accumulate="wide")
    x = torch.randn(2, 256)
    with torch.no_grad():
        assert torch.equal(c(x), sft.linear(x, dq, accumulate="wide"))
