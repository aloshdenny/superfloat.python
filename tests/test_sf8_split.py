"""The sf8 / sf16 split (Alosh's build order, items 3 and 7).

Item 7: Atreides targets hold sf4 to sf16 and run every width on the MX
engine (wide matmuls); every other op stays sf16.  A host op the default
device cannot run at its width is not offloaded.

Item 3: ``sft.convert(model, dtype="sf8", ...)`` keeps the LM head at sf16
by itself (``keep_sf16="auto"``), found from the model, not from names.
"""
from __future__ import annotations

import numpy as np
import pytest

import sf
import sf.kernels  # noqa: F401  (registers atreides-sim)

torch = pytest.importorskip("torch")
nn = torch.nn
import sf.torch as sft  # noqa: E402

DEV = "atreides"


# --- item 7: device widths ---------------------------------------------------------------------

def test_widths_per_op():
    dev = sf.get_device(DEV)
    assert dev.widths_for("matmul_wide") == tuple(range(4, 17))
    assert dev.widths_for("add") == (16,) and dev.widths_for("matmul") == (16,)
    assert dev.widths_for(None) == tuple(range(4, 17))                 # where an array may live
    assert dev.can_run("matmul_wide", sf.sf8._fmt) and not dev.can_run("mul", sf.sf8._fmt)
    assert not dev.can_run("div", sf.sf16._fmt)                         # no divide instruction


@pytest.mark.parametrize("dtype", [sf.sf4, sf.sf6, sf.sf8, sf.sf9, sf.sf12, sf.sf16])
@pytest.mark.parametrize("device", ["atreides", "atreides-sim"])
def test_wide_matmul_runs_every_width_on_the_target_with_the_host_bits(dtype, device):
    rng = np.random.default_rng(dtype._fmt.bits)
    s = np.arange(15).reshape(3, 5) % 4 - 4
    with sf.errstate(all="ignore"), sf.config(preflight="ignore"):
        a = sf.array(rng.uniform(-0.9, 0.9, (3, 24)), dtype=dtype)       # some round to 0 at sf4
        b = sf.array(rng.uniform(-0.9, 0.9, (24, 5)), dtype=dtype)
        want = sf.matmul(a, b, accumulate="wide", shift=s)
        got = sf.matmul(a.to(device), b.to(device), accumulate="wide", shift=s)
    assert got.device == device and got.dtype == dtype and np.array_equal(got.codes, want.codes)


def test_other_ops_at_narrow_widths_are_refused_with_the_way_out():
    a = sf.array([0.5, 0.25], dtype=sf.sf8, device=DEV)
    n0 = sf.get_device(DEV).launches
    for fn in (lambda: a * a, lambda: a + a, lambda: a.sum(), lambda: sf.dot(a, a),
               lambda: sf.matmul(a[None, :], a[:, None])):
        with pytest.raises(sf.SFUnsupportedError, match=r"sf8 \w+ is not supported on atreides: it runs "
                                                        r"sf8 only in wide matmuls .*accumulate=\"wide\""):
            fn()
    assert sf.get_device(DEV).launches == n0                         # refused before launch


def test_a_target_without_mx_still_refuses_narrow_arrays():
    class NoMX(sf.MockAtreides):
        name = "no-mx-widths"
        mx = False

    sf.register_device(NoMX())
    try:
        with pytest.raises(sf.SFUnsupportedError, match=r"sf8 is not supported on no-mx-widths \(supported: sf16\)"):
            sf.array([0.5], dtype=sf.sf8, device="no-mx-widths")
    finally:
        sf._device._registry.pop("no-mx-widths", None)


def test_host_ops_at_a_width_the_device_cannot_run_stay_on_the_host():
    a = sf.array([0.5, 0.25, 0.125], dtype=sf.sf8)
    dev = sf.get_device(DEV)
    n0 = dev.launches
    with sf.config(offload_threshold=0), sf.errstate(all="ignore"):
        assert (a * a).device == "host" and (a + a).device == "host"
        assert sf.matmul(a[None, :], a[:, None]).device == "host"         # chain matmul: sf16 only
        assert sf.div(a, a).device == "host"
        y = sf.matmul(a[None, :], a[:, None], accumulate="wide")           # MX takes sf8: offloads
    assert y.device == DEV and dev.launches == n0 + 1


# --- item 3: the split in sft.convert ----------------------------------------------------------

class TiedLM(nn.Module):
    def __init__(self, vocab=40, d=16):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.body = nn.Linear(d, d)
        self.out = nn.Linear(d, vocab, bias=False)
        self.out.weight = self.emb.weight

    def forward(self, i):
        return self.out(torch.relu(self.body(self.emb(i))))


class UntiedLM(TiedLM):
    def __init__(self):
        super().__init__()
        self.out = nn.Linear(16, 40)


class HFStyleLM(UntiedLM):
    def __init__(self):
        super().__init__()
        self.extra = nn.Linear(16, 40)                     # same shape, registered later: not the head

    def get_output_embeddings(self):
        return self.out


def test_the_lm_head_is_kept_at_sf16_by_each_rule():
    for model, why in ((TiedLM(), "tied to the embeddings"),
                       (UntiedLM(), "vocabulary's output size"),
                       (HFStyleLM(), "get_output_embeddings")):
        c = sft.convert(model, dtype="sf8", accumulate="wide")
        kept = sft.conversion_report(c).kept_sf16
        assert [n for n, _ in kept] == ["out"] and why in kept[0][1]
        assert c.out.sf.bits == 16 and c.body.sf.bits == 8
        assert "kept at sf16" in repr(sft.conversion_report(c))


def test_runs_and_reports_widths_per_layer():
    torch.manual_seed(0)
    c = sft.convert(TiedLM(), dtype="sf8", accumulate="wide")
    with sft.capture_status() as st:
        c(torch.tensor([[1, 2, 3]]))
    assert {n: s.dtype for n, s in st.items()} == {"body": "sf8", "out": "sf16"}


def test_keep_sf16_off_names_and_sf16():
    assert sft.convert(TiedLM(), dtype="sf8", keep_sf16=False).out.sf.bits == 8
    c = sft.convert(TiedLM(), dtype="sf8", keep_sf16=["body"])
    assert c.body.sf.bits == 16 and c.out.sf.bits == 8                   # a list replaces the rule
    assert sft.conversion_report(sft.convert(TiedLM(), dtype="sf16")).kept_sf16 == []
    mlp = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 3))                # no embedding: no LM head
    assert sft.conversion_report(sft.convert(mlp, dtype="sf8")).kept_sf16 == []
    with pytest.raises(ValueError, match="no module named 'nope'"):
        sft.convert(mlp, dtype="sf8", keep_sf16=["nope"])
    with pytest.raises(ValueError, match="keep_sf16 must be"):
        sft.convert(mlp, dtype="sf8", keep_sf16="all")


def test_sf8_wide_runs_on_the_target_through_the_compile_backend():
    torch.manual_seed(1)
    c = sft.convert(TiedLM(), dtype="sf8", accumulate="wide").eval()
    x = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        want = c(x)
        torch._dynamo.reset()
        with sf.config(default_device=DEV, preflight="ignore"):
            got = torch.compile(c, backend="superfloat")(x)
    assert torch.equal(got, want)
    on = {layer for _, _, layer, _ in sft.compile_report().on_device}
    assert on == {"body", "out"}                                          # sf8 and sf16 both on the chip
