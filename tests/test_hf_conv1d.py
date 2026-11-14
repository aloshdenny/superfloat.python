"""sft.convert recognises Hugging Face's Conv1D (GPT-2's layers).

Conv1D is a linear layer with the weight stored (in, out). Before this,
convert walked past it without converting it or listing it as skipped, so a
GPT-2 model kept 48 of its 49 matmul layers in float.

The class below copies the Hugging Face definition, so these tests need no
``transformers``; one extra test runs against the real class when installed.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402

import sf.torch as sft  # noqa: E402


class Conv1D(nn.Module):
    """Same as transformers.pytorch_utils.Conv1D."""

    def __init__(self, nf, nx):
        super().__init__()
        self.nf, self.nx = nf, nx
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        return torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight).view(size_out)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = Conv1D(16, 8)
        self.c_proj = Conv1D(8, 16)

    def forward(self, x):
        return self.c_proj(torch.relu(self.c_fc(x)))


def _model():
    torch.manual_seed(0)
    m = Block()
    with torch.no_grad():
        m.c_fc.bias.uniform_(-0.1, 0.1)
    return m


def test_conv1d_is_converted_and_reported():
    m = sft.convert(_model())
    assert isinstance(m.c_fc, sft.SFHFConv1D) and isinstance(m.c_proj, sft.SFHFConv1D)
    r = sft.conversion_report(m)
    assert [k for _, k in r.converted] == ["Conv1D (Hugging Face)"] * 2 and not r.skipped


def test_conv1d_gives_the_same_bits_as_the_equal_linear():
    """A Conv1D and an nn.Linear holding the transposed weight are the same
    layer, so their SF versions must agree exactly."""
    src = _model()
    x = torch.randn(2, 5, 8) * 0.5
    lin = nn.Linear(8, 16)
    with torch.no_grad():
        lin.weight.copy_(src.c_fc.weight.t())
        lin.bias.copy_(src.c_fc.bias)
    a = sft.convert(src).c_fc(x)
    b = sft.convert(lin)(x)
    assert a.shape == (2, 5, 16) and torch.equal(a, b)


def test_conv1d_keeps_the_parameter_objects_and_gradients():
    src = _model()
    w = src.c_fc.weight
    m = sft.convert(src, inplace=True)
    assert m.c_fc.weight is w                                   # tied weights and optimizers carry over
    x = torch.randn(3, 8) * 0.5
    m(x).sum().backward()
    assert w.grad is not None and w.grad.shape == (8, 16)


def test_a_similar_looking_module_is_left_alone():
    class Conv1D(nn.Module):                                    # same name, wrong layout
        def __init__(self):
            super().__init__()
            self.nf, self.nx = 4, 3
            self.weight = nn.Parameter(torch.zeros(4, 3))

    m = nn.Sequential(Conv1D())
    assert not isinstance(sft.convert(m)[0], sft.SFHFConv1D)


def test_real_gpt2_layers_are_converted():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.GPT2Config(n_layer=2, n_embd=32, n_head=2, vocab_size=64, n_positions=16)
    m = transformers.GPT2LMHeadModel(cfg).eval()
    r = sft.conversion_report(sft.convert(m))
    kinds = [k for _, k in r.converted]
    assert kinds.count("Conv1D (Hugging Face)") == 8 and kinds.count("Linear") == 1
