"""MXFP4 weights (Alosh, 2026-11-18): the real gpt-oss checkpoints ship in it.

Same pattern as FP8: read the stored values exactly, then quantize to sf.
The decoder must give transformers' values bit for bit, a checkpoint loaded
by ``sft.load_mxfp4`` must equal one loaded by transformers, and an MXFP4
linear layer must convert like a float one.
"""
from __future__ import annotations

import os

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import sf.torch as sft  # noqa: E402
from sf.torch import _mxfp4  # noqa: E402

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]


def test_decoder_by_hand():
    # one row, one block: bytes 0x10 (lo 0 -> 0, hi 1 -> 0.5), 0x7F (lo 15 -> -6, hi 7 -> 6), then zeros
    blocks = torch.zeros(1, 1, 16, dtype=torch.uint8)
    blocks[0, 0, 0], blocks[0, 0, 1] = 0x10, 0x7F
    v = sft.dequantize_mxfp4(blocks, torch.tensor([[127]], dtype=torch.uint8))
    assert v.shape == (1, 32) and v[0, :4].tolist() == [0.0, 0.5, -6.0, 6.0] and v[0, 4:].abs().sum() == 0
    v = sft.dequantize_mxfp4(blocks, torch.tensor([[130]], dtype=torch.uint8))          # 2^3
    assert v[0, :4].tolist() == [0.0, 4.0, -48.0, 48.0]
    v = sft.dequantize_mxfp4(blocks, torch.tensor([[255]], dtype=torch.uint8))          # E8M0 NaN
    assert torch.isnan(v).all()
    every = torch.arange(16, dtype=torch.uint8).repeat(2).reshape(1, 2, 16)              # lo nibbles 0..15
    lo = sft.dequantize_mxfp4(every & 0x0F, torch.full((1, 2), 127, dtype=torch.uint8))[0, 0::2][:16]
    assert lo.tolist() == E2M1 + [-x for x in E2M1]
    assert torch.signbit(lo[8])                                                         # code 8 is -0


def test_scale_as_e8m0_dtype():
    e8 = getattr(torch, "float8_e8m0fnu", None)
    if e8 is None:
        pytest.skip("no float8_e8m0fnu")
    b = torch.randint(0, 256, (2, 3, 16), dtype=torch.uint8)
    s = torch.randint(110, 140, (2, 3), dtype=torch.uint8)
    assert torch.equal(sft.dequantize_mxfp4(b, s), sft.dequantize_mxfp4(b, s.view(e8)))


def test_decoder_errors():
    with pytest.raises(TypeError, match="uint8"):
        sft.dequantize_mxfp4(torch.zeros(1, 1, 16), torch.zeros(1, 1, dtype=torch.uint8))
    with pytest.raises(ValueError, match="do not match"):
        sft.dequantize_mxfp4(torch.zeros(1, 2, 16, dtype=torch.uint8), torch.zeros(1, 3, dtype=torch.uint8))
    with pytest.raises(TypeError, match="E8M0"):
        sft.dequantize_mxfp4(torch.zeros(1, 1, 16, dtype=torch.uint8), torch.zeros(1, 1))


def test_encoder_rounds_to_nearest_even_and_saturates():
    x = torch.zeros(1, 32)
    x[0, :8] = torch.tensor([6.0, 0.25, 2.5, 5.0, 7.0, -1.25, 0.75, -3.0])   # max 7 -> exponent 0
    b, s = sft.quantize_mxfp4(x)
    assert s.tolist() == [[127]]
    got = sft.dequantize_mxfp4(b, s)[0, :8].tolist()
    # 0.25 ties 0 / 0.5 -> 0 (even); 2.5 ties 2 / 3 -> 2; 5 ties 4 / 6 -> 4; 7 saturates to 6;
    # -1.25 ties -1 / -1.5 -> -1; 0.75 ties 0.5 / 1 -> 1 (code 2 is even)
    assert got == [6.0, 0.0, 2.0, 4.0, 6.0, -1.0, 1.0, -3.0]


def test_encoder_round_trip_is_exact_on_mxfp4_values():
    torch.manual_seed(0)
    w = torch.randn(5, 96) * 3
    b, s = sft.quantize_mxfp4(w)
    dq = sft.dequantize_mxfp4(b, s)
    b2, s2 = sft.quantize_mxfp4(dq)
    assert torch.equal(b, b2) and torch.equal(s, s2)
    assert ((dq - w).norm() / w.norm()) < 0.2                                            # 4-bit elements
    with pytest.raises(ValueError, match="blocks of 32"):
        sft.quantize_mxfp4(torch.randn(2, 40))


class MXFP4Linear(nn.Linear):
    """An nn.Linear with packed MXFP4 weights (out, in / 2) and E8M0 scales (out, in / 32)."""

    def __init__(self, src: nn.Linear):
        super().__init__(src.in_features, src.out_features)
        b, s = sft.quantize_mxfp4(src.weight.detach())
        self.weight = nn.Parameter(b.reshape(src.out_features, -1), requires_grad=False)
        self.register_buffer("weight_scale", s)
        self.bias = src.bias

    def forward(self, x):
        w = sft.dequantize_mxfp4(self.weight.reshape(self.out_features, -1, 16), self.weight_scale)
        return nn.functional.linear(x, w, self.bias)


def test_an_mxfp4_linear_converts_like_float():
    torch.manual_seed(1)
    lin = MXFP4Linear(nn.Linear(64, 24))
    m = nn.Sequential(lin, nn.ReLU())
    c = sft.convert(m, dtype="sf16", accumulate="wide")
    kind = dict(sft.conversion_report(c).converted)["0"]
    assert kind == "Linear (MXFP4 weights, read with their E8M0 scale per block of 32)"
    dq = sft.dequantize_mxfp4(lin.weight.reshape(24, -1, 16), lin.weight_scale)
    x = torch.randn(3, 64)
    with torch.no_grad():
        assert torch.equal(c[0](x), sft.linear(x, dq, lin.bias, accumulate="wide"))
        f = m(x)
        assert ((c(x) - f).norm() / f.norm()) < 1e-3


def test_load_mxfp4_into_a_float_model():
    torch.manual_seed(2)

    class Experts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(3, 32, 64))   # (E, in, out), like gpt-oss
            self.down_proj = nn.Parameter(torch.zeros(3, 32, 32))      # square: transposed all the same

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = Experts()
            self.proj = nn.Linear(64, 8, bias=False)

    m = M()
    src = {"experts.gate_up_proj": torch.randn(3, 64, 32), "experts.down_proj": torch.randn(3, 32, 32),
           "proj.weight": torch.randn(8, 64)}                        # stored (E, out, in) / (out, in)
    ck = {}
    for k, v in src.items():
        b, s = sft.quantize_mxfp4(v)
        ck[k + "_blocks"], ck[k + "_scales"] = b, s
    assert sft.load_mxfp4(m, ck) == ["experts.down_proj", "experts.gate_up_proj", "proj.weight"]
    for k, p in (("experts.gate_up_proj", m.experts.gate_up_proj), ("experts.down_proj", m.experts.down_proj)):
        want = sft.dequantize_mxfp4(ck[k + "_blocks"], ck[k + "_scales"]).transpose(-1, -2)
        assert torch.equal(p.detach(), want)
    assert torch.equal(m.proj.weight.detach(), sft.dequantize_mxfp4(ck["proj.weight_blocks"], ck["proj.weight_scales"]))

    with pytest.raises(KeyError, match="no parameter nope"):
        sft.load_mxfp4(m, {"nope_blocks": ck["proj.weight_blocks"], "nope_scales": ck["proj.weight_scales"]})
    assert sft.load_mxfp4(m, {"nope_blocks": ck["proj.weight_blocks"], "nope_scales": ck["proj.weight_scales"]},
                          strict=False) == []
    with pytest.raises(ValueError, match="does not fit"):
        sft.load_mxfp4(m, {"proj.weight_blocks": ck["experts.down_proj_blocks"][0],
                           "proj.weight_scales": ck["experts.down_proj_scales"][0]})
    with pytest.raises(ValueError, match="has no x_scales"):
        sft.load_mxfp4(m, {"x_blocks": ck["proj.weight_blocks"]})


def test_gpu_packed_mxfp4_experts_are_refused_with_the_way_out():
    class Mxfp4GptOssExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(2, 16, 8, dtype=torch.uint8), requires_grad=False)

    rep = sft.conversion_report(sft.convert(nn.Sequential(Mxfp4GptOssExperts()), accumulate="wide"))
    assert rep.skipped[0][0] == "0" and "sft.load_mxfp4" in rep.skipped[0][2]


def test_matches_transformers_decoder():
    mx = pytest.importorskip("transformers.integrations.mxfp4")
    g = torch.Generator().manual_seed(3)
    blocks = torch.randint(0, 256, (3, 5, 4, 16), dtype=torch.uint8, generator=g)
    scales = torch.randint(100, 150, (3, 5, 4), dtype=torch.uint8, generator=g)
    # transformers' decoder is private and has been renamed: _convert_moe_packed_tensors (5.x),
    # convert_moe_packed_tensors (4.57)
    conv = getattr(mx, "_convert_moe_packed_tensors", None) or getattr(mx, "convert_moe_packed_tensors", None)
    if conv is None:
        pytest.skip("this transformers has no MXFP4 decoder to compare with")
    theirs = conv(blocks, scales, dtype=torch.float32)
    assert torch.equal(sft.dequantize_mxfp4(blocks, scales).transpose(1, 2), theirs)


def test_a_gpt_oss_checkpoint_loads_like_transformers_and_converts(tmp_path):
    T = pytest.importorskip("transformers")
    pytest.importorskip("accelerate")                                 # transformers' MXFP4 loader needs it
    from safetensors.torch import save_file
    import json
    torch.manual_seed(4)
    cfg = T.GptOssConfig(vocab_size=128, hidden_size=64, intermediate_size=64, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16, num_local_experts=4,
                         num_experts_per_tok=2, layer_types=["sliding_attention", "full_attention"],
                         sliding_window=8, max_position_embeddings=64)
    src = T.GptOssForCausalLM(cfg).eval()
    ck = {}
    for k, v in src.state_dict().items():
        if k.endswith(("experts.gate_up_proj", "experts.down_proj")):
            ck[k + "_blocks"], ck[k + "_scales"] = sft.quantize_mxfp4(v.transpose(-1, -2).contiguous())
        else:
            ck[k] = v.contiguous()
    save_file(ck, os.path.join(tmp_path, "model.safetensors"))
    c = cfg.to_dict()
    c["quantization_config"] = {"quant_method": "mxfp4", "modules_to_not_convert": [
        "model.layers.*.self_attn", "model.layers.*.mlp.router", "model.embed_tokens", "lm_head"]}
    json.dump(c, open(os.path.join(tmp_path, "config.json"), "w"))

    ours = T.GptOssForCausalLM(cfg).eval()
    ours.load_state_dict({k: v for k, v in ck.items() if not k.endswith(("_blocks", "_scales"))}, strict=False)
    names = sft.load_mxfp4(ours, tmp_path)
    assert len(names) == 4
    hf = T.AutoModelForCausalLM.from_pretrained(tmp_path, dtype=torch.float32).eval()
    hp = dict(hf.named_parameters())
    assert all(torch.equal(hp[n].float(), dict(ours.named_parameters())[n]) for n in names)

    ids = torch.randint(0, 128, (1, 9))
    with torch.no_grad():
        f = ours(ids).logits
        conv = sft.convert(ours, dtype="sf16", accumulate="wide")
        y = conv(ids).logits
    kinds = dict(sft.conversion_report(conv).converted)
    assert kinds["model.layers.0.mlp.experts"].startswith("MoE experts (4 experts")
    assert ((y - f).norm() / f.norm()) < 1e-3 and torch.equal(y.argmax(-1), f.argmax(-1))
