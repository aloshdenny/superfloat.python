"""torch.compile with the integer nonlinears (convert(nonlinear="integer")).

The integer definitions run in NumPy.  Dynamo cannot trace into NumPy, so before they were custom ops
(sf::int_softmax, sf::int_xsig) torch.compile(backend="superfloat") failed on a model converted with
nonlinear="integer" (Dynamo: "'ndarray' object has no attribute '__round__'", in sf._intops.lo_offset).  Now each
is one node in the graph, run on the host as in eager, with the eager bits.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
import sf.torch as sft  # noqa: E402


@pytest.fixture
def gpt2():
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(n_layer=2, n_embd=64, n_head=2, vocab_size=96, n_positions=32)
    return sft.convert(transformers.GPT2LMHeadModel(cfg).eval(), dtype="sf16", accumulate="wide",
                       nonlinear="integer")


def test_compile_with_integer_softmax_and_gelu_gives_the_eager_bits(gpt2):
    ids = torch.randint(0, 96, (1, 12))
    torch._dynamo.reset()
    with torch.no_grad():
        want = gpt2(ids).logits
        got = torch.compile(gpt2, backend="superfloat")(ids).logits
    assert torch.equal(got, want)


def test_the_integer_ops_are_custom_ops():
    x = torch.randn(2, 3, 8)
    assert torch.equal(sft.int_softmax(x, dim=1), torch.ops.sf.int_softmax(x, 1))
    assert torch.equal(sft.int_gelu(x), torch.ops.sf.int_xsig(x, True))
    assert torch.equal(sft.int_silu(x), torch.ops.sf.int_xsig(x, False))
    assert torch.equal(sft.int_softmax(x, dim=-2), sft.int_softmax(x, dim=1))   # negative dims are the same op
