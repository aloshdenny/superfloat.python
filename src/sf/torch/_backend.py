"""``torch.compile(model, backend="superfloat")`` (spec 2.2).

TorchDynamo hands the backend FX graphs.  Every ``sf::linear`` /
``sf::matmul`` node whose width the target supports is retargeted to a
device implementation that runs the same chip-exact core through the
Stage 1.1 heavy-op path (pre-flight gate, launch, ``result.status``).
Everything else stays on the CPU, unchanged.  A coverage report is kept
per compiled graph.

Bit-identity with eager emulation holds by construction: both call
``_core.linear`` / ``_core.matmul`` with the same inputs and differ only in
the runner, and the runner's kernels are checked against the host reference
(Stage 1.1 gate).

The backend is inference-only: the device path has no autograd.  Calling a
compiled model with gradients enabled on parameters that require them raises
``SFUnsupportedError``; train with the eager emulation (``sft.convert``).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field

import torch
from torch._dynamo import register_backend

from . import _core
from ._ops import SF_OPS, linear_impl, matmul_impl

# float ops that could have been SF if the model had been converted
_UNCONVERTED = {
    "linear", "matmul", "mm", "bmm", "conv2d", "conv1d", "conv3d", "addmm", "baddbmm",
    "einsum", "__matmul__", "scaled_dot_product_attention",
}


@dataclass
class GraphReport:
    on_device: list = field(default_factory=list)        # (node, op, layer name, device)
    host_emulated: list = field(default_factory=list)    # (node, op, layer name, reason)
    cpu: list = field(default_factory=list)              # (node, target)
    unconverted: list = field(default_factory=list)      # (node, target): float linear algebra

    @property
    def sf_ops(self) -> int:
        return len(self.on_device) + len(self.host_emulated)

    @property
    def coverage(self) -> float:
        """Fraction of the graph's linear algebra that runs on the device."""
        total = self.sf_ops + len(self.unconverted)
        return len(self.on_device) / total if total else 1.0

    def __repr__(self) -> str:
        lines = [f"GraphReport: {len(self.on_device)} SF op(s) on device, "
                 f"{len(self.host_emulated)} host-emulated, {len(self.unconverted)} float linear "
                 f"algebra op(s) not converted, {len(self.cpu)} other op(s) on CPU; "
                 f"linear-algebra coverage {self.coverage:.0%}"]
        lines += [f"  device  {n}: sf.{op} {name!r} on {d}" for n, op, name, d in self.on_device]
        lines += [f"  host    {n}: sf.{op} {name!r} ({why})" for n, op, name, why in self.host_emulated]
        lines += [f"  float   {n}: {t} (not converted; use sft.convert or sft.autocast)"
                  for n, t in self.unconverted]
        lines += [f"  cpu     {n}: {t}" for n, t in self.cpu]
        return "\n".join(lines)


_lock = threading.Lock()
_reports: list[GraphReport] = []


def compile_reports() -> list[GraphReport]:
    with _lock:
        return list(_reports)


def compile_report() -> GraphReport:
    with _lock:
        if not _reports:
            raise ValueError("no graph has been compiled with backend='superfloat' yet")
        return _reports[-1]


def _target_name(node) -> str:
    t = node.target
    if isinstance(t, str):
        return t
    return getattr(t, "__name__", None) or str(t)


def _require_no_grad(*tensors):
    if torch.is_grad_enabled() and any(isinstance(t, torch.Tensor) and t.requires_grad for t in tensors):
        import sf
        raise sf.SFUnsupportedError(
            "the superfloat compile backend is inference-only (the device path has no autograd); "
            "wrap the call in torch.no_grad(), or train with the eager emulation from sft.convert")


def _device_linear(device):
    runner = _core.DeviceRunner(device)

    def run(x, weight, bias, bits, wscale, headroom, name):
        _require_no_grad(x, weight, bias)
        return linear_impl(x, weight, bias, bits, wscale, headroom, name, runner)
    run.__name__ = f"sf_linear_on_{device}"
    return run


def _device_matmul(device):
    runner = _core.DeviceRunner(device)

    def run(a, b, bits, headroom, name):
        _require_no_grad(a, b)
        return matmul_impl(a, b, bits, headroom, name, runner)
    run.__name__ = f"sf_matmul_on_{device}"
    return run


def _arg(node, index, key):
    if len(node.args) > index:
        return node.args[index]
    return node.kwargs.get(key)


@register_backend
def superfloat(gm: torch.fx.GraphModule, example_inputs, **kwargs):
    import sf
    from .. import _config, _device
    device_name = _config.get("default_device")
    dev = _device.get_device(device_name)
    report = GraphReport()
    impl = {"linear": _device_linear(dev.name), "matmul": _device_matmul(dev.name)}
    for node in gm.graph.nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        name = _target_name(node)
        if node.op == "call_function" and str(node.target) in SF_OPS:
            op = SF_OPS[str(node.target)]
            bits = _arg(node, 3 if op == "linear" else 2, "bits")
            layer = _arg(node, 6 if op == "linear" else 4, "name")
            if bits in dev.supported_bits:
                node.target = impl[op]
                report.on_device.append((node.name, op, layer, dev.name))
            else:
                report.host_emulated.append((node.name, op, layer,
                                             f"sf{bits} is not supported on {dev.name}"))
            continue
        short = name.split(".")[-1]
        if node.op == "call_module":
            sub = gm.get_submodule(node.target)
            short = type(sub).__name__.lower()
            name = f"{node.target} ({type(sub).__name__})"
        if short in _UNCONVERTED or short in ("linear", "conv1d", "conv2d", "conv3d"):
            report.unconverted.append((node.name, name))
        else:
            report.cpu.append((node.name, name))
    gm.recompile()
    with _lock:
        _reports.append(report)
    return gm.forward
