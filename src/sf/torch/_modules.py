"""SF layers and model conversion (spec 2.1)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import _core
from ._ops import call_linear
from . import _scopes


def _bits_of(dtype) -> int:
    import sf
    return sf.info(sf.dtype(dtype)).bits


@dataclass
class SFConfig:
    """How an SF layer computes.

    ``accumulate="chain"`` (default): the FMA chain with the float scale
    policy (``scale``, ``headroom``).  ``accumulate="wide"``: the MX engine
    with the wide-mode rule; weights get one power-of-two exponent per
    channel and activations one per token, so ``scale`` and ``headroom`` are
    not used.  ``shift`` (an int) overrides the rule's shifts; ``calibration``
    names the rule, and ``"heuristic"`` is the only one."""

    bits: int = 16
    scale: str = "per_tensor"         # chain: weight scale "per_tensor" | "per_channel"
    headroom: str = "bound"            # chain: "bound" (no saturation, provable) | "none"
    accumulate: str = "chain"          # "chain" (FMA) | "wide" (MX)
    shift: int | None = None           # wide: one shift for every output, overriding the rule
    calibration: str = "heuristic"     # wide: the shift rule

    def __post_init__(self):
        _core._check(self.bits, self.scale, self.headroom, self.accumulate)
        from .. import _heuristic, _linalg
        if self.calibration not in _heuristic.CALIBRATIONS:
            raise ValueError(f"calibration must be one of {_heuristic.CALIBRATIONS}, got {self.calibration!r}")
        if self.shift is not None:
            if self.accumulate != "wide":
                raise ValueError("shift= only applies to accumulate=\"wide\"")
            self.shift = _linalg.check_shift(self.shift, allow_array=False)

    @property
    def dtype_name(self) -> str:
        return f"sf{self.bits}"

    def describe(self) -> str:
        if self.accumulate == "wide":
            return f"dtype={self.dtype_name}, accumulate=wide" + ("" if self.shift is None else f", shift={self.shift}")
        return f"dtype={self.dtype_name}, scale={self.scale}, headroom={self.headroom}"


class SFLinear(nn.Module):
    """``nn.Linear`` with chip-exact SF arithmetic.

    Parameters stay float (so training works through the straight-through
    gradient); every forward quantizes the weight with the configured scale
    policy and runs the SF accumulation chain.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, *,
                 dtype="sf16", scale: str = "per_tensor", headroom: str = "bound",
                 accumulate: str = "chain", shift: int | None = None,
                 name: str = "", device=None, param_dtype=None) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.sf = SFConfig(_bits_of(dtype), scale, headroom, accumulate, shift)
        self.sf_name = name or f"linear_{id(self):x}"
        kw = {"device": device, "dtype": param_dtype}
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **kw))
        self.bias = nn.Parameter(torch.empty(out_features, **kw)) if bias else None
        ref = nn.Linear(in_features, out_features, bias, **kw)
        with torch.no_grad():
            self.weight.copy_(ref.weight)
            if bias:
                self.bias.copy_(ref.bias)

    @classmethod
    def from_float(cls, m: nn.Linear, name: str, cfg: SFConfig) -> "SFLinear":
        """Wrap ``m``'s own parameter objects (identity preserved, so an
        optimizer built before conversion keeps working and ``requires_grad``
        is whatever it was), and copy its training / eval state."""
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.in_features, new.out_features = m.in_features, m.out_features
        new.sf = cfg
        new.sf_name = name or f"linear_{id(new):x}"
        new.weight = m.weight
        new.bias = m.bias
        new.train(m.training)
        return new

    @classmethod
    def from_weight(cls, m: nn.Module, weight: torch.Tensor, name: str, cfg: SFConfig) -> "SFLinear":
        """An SF linear for ``m`` with ``weight`` read from another format (FP8
        with block scales, dequantized to float).  The weight is a new frozen
        parameter; the bias stays ``m``'s own."""
        new = cls.__new__(cls)
        nn.Module.__init__(new)
        new.in_features, new.out_features = weight.shape[1], weight.shape[0]
        new.sf = cfg
        new.sf_name = name or f"linear_{id(new):x}"
        new.weight = nn.Parameter(weight, requires_grad=False)
        b = getattr(m, "bias", None)
        new.bias = b if isinstance(b, torch.Tensor) and b.is_floating_point() else None
        new.train(m.training)
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return call_linear(x, self.weight, self.bias, self.sf, self.sf_name)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, {self.sf.describe()}, name={self.sf_name!r}")


def _float_weight(m: nn.Module, full: str, report) -> bool:
    """False (and the reason in the report) when ``m``'s weight is not float."""
    _, _, problem = _scopes.read_weight(m)
    if problem is None and getattr(m, "weight", None) is not None and not m.weight.is_floating_point():
        problem = f"weights stored as {m.weight.dtype}"
    if problem is None and getattr(m, "weight", None) is not None and m.weight.dtype in _scopes._FP8:
        problem = f"{m.weight.dtype} weights are read only for linear layers; runs as it was"
    if problem is not None:
        report.skipped.append((full, type(m).__name__, problem))
        return False
    return True


def _pair(v):
    return tuple(v) if isinstance(v, (tuple, list)) else (v, v)


def is_hf_conv1d(m: nn.Module) -> bool:
    """True for Hugging Face's ``Conv1D`` (GPT-2 and relatives): a linear layer
    whose weight is stored transposed, ``(in, out)``.  Recognised by its shape
    and attributes, so ``transformers`` is never imported."""
    w = getattr(m, "weight", None)
    return (type(m).__name__ == "Conv1D" and isinstance(getattr(m, "nf", None), int)
            and isinstance(getattr(m, "nx", None), int) and isinstance(w, torch.Tensor)
            and tuple(w.shape) == (m.nx, m.nf))


class SFHFConv1D(nn.Module):
    """Hugging Face ``Conv1D`` with chip-exact SF arithmetic.

    The same maths as ``SFLinear``; the weight keeps its ``(in, out)`` layout
    and is the source layer's own parameter object, so tied weights, an
    existing optimizer and ``requires_grad`` all carry over."""

    def __init__(self, conv: nn.Module, name: str, cfg: SFConfig) -> None:
        super().__init__()
        if not is_hf_conv1d(conv):
            raise ValueError("not a Hugging Face Conv1D")
        self.nf, self.nx = conv.nf, conv.nx
        self.sf = cfg
        self.sf_name = name
        self.weight = conv.weight
        self.bias = conv.bias
        self.train(conv.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return call_linear(x, self.weight.t(), self.bias, self.sf, self.sf_name)

    def extra_repr(self) -> str:
        return f"nf={self.nf}, nx={self.nx}, {self.sf.describe()}, name={self.sf_name!r}"


class SFConv2d(nn.Module):
    """``nn.Conv2d`` (groups=1, zero padding) as unfold + SF linear: the patch
    extraction is data movement on the host, the reduction is the SF chain."""

    def __init__(self, conv: nn.Conv2d, name: str, cfg: SFConfig) -> None:
        super().__init__()
        if not convertible_conv(conv)[0]:
            raise ValueError(convertible_conv(conv)[1])
        self.in_channels, self.out_channels = conv.in_channels, conv.out_channels
        self.kernel_size = _pair(conv.kernel_size)
        self.stride, self.padding, self.dilation = _pair(conv.stride), _pair(conv.padding), _pair(conv.dilation)
        self.sf = cfg
        self.sf_name = name
        self.weight = conv.weight                     # same parameter objects as the source conv
        self.bias = conv.bias
        self.train(conv.training)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, h, w = x.shape
        kh, kw = self.kernel_size
        cols = F.unfold(x, (kh, kw), dilation=self.dilation, padding=self.padding, stride=self.stride)
        oh = (h + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        ow = (w + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
        y = call_linear(cols.transpose(1, 2), self.weight.reshape(self.out_channels, -1),
                        self.bias, self.sf, self.sf_name)
        return y.transpose(1, 2).reshape(n, self.out_channels, oh, ow)

    def extra_repr(self) -> str:
        return (f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
                f"stride={self.stride}, padding={self.padding}, {self.sf.describe()}, "
                f"name={self.sf_name!r}")


def convertible_conv(c: nn.Conv2d) -> tuple[bool, str]:
    if c.groups != 1:
        return False, f"groups={c.groups} (only groups=1 is converted)"
    if c.padding_mode != "zeros":
        return False, f"padding_mode={c.padding_mode!r} (only 'zeros' is converted)"
    if isinstance(c.padding, str):
        return False, f"padding={c.padding!r} (only explicit integer padding is converted)"
    return True, ""


@dataclass
class ConversionReport:
    converted: list = field(default_factory=list)     # (name, kind)
    skipped: list = field(default_factory=list)       # (name, kind, reason)
    shared: list = field(default_factory=list)        # (alias name, converted name): same module
    kept_sf16: list = field(default_factory=list)     # (name, why): converted at sf16, not the narrower dtype
    attention: str = "float (not converted)"          # where F.scaled_dot_product_attention runs
    widths: dict = field(default_factory=dict)        # name -> bits, for every converted module
    nonlinear: list = field(default_factory=list)     # (name, what): activations on sf's integer definitions
    eager_attention: list = field(default_factory=list)  # attention modules whose torch.matmul calls run on MX

    def __repr__(self) -> str:
        lines = [f"ConversionReport: {len(self.converted)} converted, {len(self.skipped)} skipped"
                 + (f", {len(self.shared)} shared alias(es)" if self.shared else "")
                 + (f", {len(self.kept_sf16)} kept at sf16" if self.kept_sf16 else "")]
        if self.widths:
            per = {}
            for b in self.widths.values():
                per[b] = per.get(b, 0) + 1
            lines += ["  widths: " + ", ".join(f"sf{b} x {k}" for b, k in sorted(per.items()))]
        lines += [f"  + {n} ({k})" for n, k in self.converted]
        lines += [f"  ^ {n} kept at sf16: {w}" for n, w in self.kept_sf16]
        lines += [f"  attention: {self.attention}"]
        lines += [f"  ~ {n}: {w}" for n, w in self.nonlinear]
        if self.eager_attention:
            lines += [f"  attention written with torch.matmul (no SDPA): caught on the MX engine at sf16 in "
                      f"{len(self.eager_attention)} attention module(s), when they use it"]
        lines += [f"  = {a} is {n} (shared module, converted once)" for a, n in self.shared]
        lines += [f"  - {n} ({k}): {r}" for n, k, r in self.skipped]
        return "\n".join(lines)


def _int_act(m: nn.Module):
    from ._intops import integer_activation
    return integer_activation(m)


def _linear_like(m: nn.Module) -> bool:
    return isinstance(m, (nn.Linear, SFLinear, SFHFConv1D)) or is_hf_conv1d(m)


def _out_features(m: nn.Module):
    return getattr(m, "out_features", None) or getattr(m, "nf", None)


def find_lm_heads(model: nn.Module) -> dict[int, str]:
    """The LM head(s) of ``model``, as {id(module): why}.  Rules, in order:
    the model's own ``get_output_embeddings()`` (Hugging Face models); else a
    linear layer whose weight is tied to an ``nn.Embedding``'s; else the last
    linear layer whose output size is an embedding's vocabulary.  A model with
    no embedding has no LM head (a classifier's last layer is not one)."""
    get = getattr(model, "get_output_embeddings", None)
    if callable(get):
        try:
            head = get()
        except Exception:                                   # noqa: BLE001  (a model's own hook may raise)
            head = None
        if isinstance(head, nn.Module) and _linear_like(head):
            return {id(head): "LM head (the model's get_output_embeddings())"}
    embs = [m for m in model.modules() if isinstance(m, nn.Embedding)]
    if not embs:
        return {}
    lins = [m for m in model.modules() if _linear_like(m)]
    tied = {}
    for m in lins:
        w = getattr(m, "weight", None)
        for e in embs:
            if w is e.weight or (w is not None and w.data_ptr() == e.weight.data_ptr()
                                 and w.numel() == e.weight.numel()):
                tied[id(m)] = "LM head (weight tied to the embeddings)"
    if tied:
        return tied
    vocab = {e.num_embeddings for e in embs}
    last = [m for m in lins if _out_features(m) in vocab]
    return {id(last[-1]): "LM head (last layer with the vocabulary's output size)"} if last else {}


def convert(model: nn.Module, dtype="sf16", scale: str = "per_tensor", headroom: str = "bound",
            inplace: bool = False, *, accumulate: str = "chain", shift: int | None = None,
            calibration: str = "heuristic", keep_sf16="auto", attention="auto",
            nonlinear: str = "float", widths: dict | None = None) -> nn.Module:
    """Swap ``nn.Linear``, ``nn.Conv2d`` and Hugging Face's ``Conv1D`` (GPT-2's
    layers) for SF versions with a scale policy.

    ``accumulate="wide"`` runs every converted layer on the MX engine and
    chooses every shift with the wide-mode rule (no calibration data);
    ``sft.last_status()`` reports the shifts and redos per layer.

    ``keep_sf16``: with a dtype narrower than sf16, which layers still run
    at sf16.  ``"auto"`` (default) keeps the LM head (``find_lm_heads``),
    which costs accuracy at sf8 and little time; ``False`` keeps none; a list
    of module names keeps exactly those.  The report lists them.

    ``attention``: ``"auto"`` (default) runs the model's
    ``F.scaled_dot_product_attention`` calls on the MX engine at sf16 when
    ``accumulate="wide"`` (``sft.attention``), named ``attention.0``,
    ``attention.1``, ... per forward; ``False`` leaves them in float.

    MoE models: the experts Hugging Face keeps in one 3-D parameter run on
    the MX engine (each expert's ``F.linear`` is taken while the experts
    module runs), at ``dtype``; the router at sf16.  Attention written with
    ``torch.matmul`` instead of SDPA (``attn_implementation="eager"``) is
    caught the same way in wide mode.  See ``sf.torch._scopes``.

    Weights in another format: FP8 with block scales (``weight_scale_inv``)
    is dequantized, then converted; any other non-float weight (int8,
    packed 4-bit, GPTQ / AWQ) is skipped with the reason, and so is a
    subclass of ``nn.Linear`` with its own ``forward``, so those layers run
    as they were instead of giving a wrong answer.

    ``widths``: {module name: dtype} for layers that run at another width
    than ``dtype`` (a width per layer); it wins over ``keep_sf16``.
    ``sft.choose_widths`` makes one from measured sensitivity and the chip's
    cost table (``plan.convert_kwargs()``).

    ``nonlinear``: ``"float"`` (default) keeps softmax and activations in
    float on the host.  ``"integer"`` runs them on sf's integer definitions
    (``sf.intops``, Alosh's chip sequences): attention's softmax (wide mode
    only, where attention runs on sf), GPT-2's tanh GELU and SiLU modules
    (MoE experts' ``act_fn`` included).  GELU with erf has no integer form
    and stays float.  Norms always stay on the host.  The report lists each.

    Other modules (normalisation, pooling, ...) stay as
    they are and run in float on the host; ``sft.conversion_report(model)``
    lists what was converted and what was skipped, and why, including any
    module that holds weights no rule converted.
    """
    cfg = SFConfig(_bits_of(dtype), scale, headroom, accumulate, shift, calibration)
    if attention not in ("auto", False, None):
        raise ValueError(f"attention must be \"auto\" or False, got {attention!r}")
    if nonlinear not in ("float", "integer"):
        raise ValueError(f"nonlinear must be \"float\" or \"integer\", got {nonlinear!r}")
    if not inplace:
        model = copy.deepcopy(model)
    report = ConversionReport()
    cfg16 = SFConfig(16, scale, headroom, accumulate, shift, calibration)
    keep_ids: dict[int, str] = {}
    keep_names: set[str] = set()
    if keep_sf16 == "auto":
        if cfg.bits < 16:
            keep_ids = find_lm_heads(model)
    elif keep_sf16 in (False, None):
        pass
    elif isinstance(keep_sf16, str) or not hasattr(keep_sf16, "__iter__"):
        raise ValueError(f"keep_sf16 must be \"auto\", False or a list of module names, got {keep_sf16!r}")
    else:
        keep_names = set(keep_sf16)
        if cfg.bits < 16:
            known = {n for n, _ in model.named_modules(remove_duplicate=False)}
            missing = sorted(keep_names - known)
            if missing:
                raise ValueError(f"keep_sf16: no module named {', '.join(map(repr, missing))}")

    per_layer: dict[str, SFConfig] = {}
    if widths:
        known = {n for n, _ in model.named_modules(remove_duplicate=False)}
        missing = sorted(set(widths) - known)
        if missing:
            raise ValueError(f"widths: no module named {', '.join(map(repr, missing))}")
        per_layer = {n: SFConfig(_bits_of(d), scale, headroom, accumulate, shift, calibration)
                     for n, d in widths.items()}

    def cfg_for(child, full):
        c = _cfg_for(child, full)
        report.widths[full] = c.bits
        return c

    def _cfg_for(child, full):
        if full in per_layer:
            return per_layer[full]
        if cfg.bits >= 16:
            return cfg
        why = keep_ids.get(id(child)) or ("named in keep_sf16" if full in keep_names else None)
        if why is None:
            return cfg
        report.kept_sf16.append((full, why))
        return cfg16

    wide_attention = attention == "auto" and cfg.accumulate == "wide"
    done: dict[int, tuple[str, nn.Module | None]] = {}     # id(source) -> (name, replacement)
    seen: set[int] = set()                                  # containers already visited

    def replacement(child: nn.Module, full: str):
        """The SF module for ``child``, or None if it stays; converts each
        distinct module once and returns the same object for every alias."""
        key = id(child)
        if key in done:
            name, rep = done[key]
            if rep is not None:
                report.shared.append((full, name))
            return rep, True
        rep = None
        if isinstance(child, (SFLinear, SFConv2d, SFHFConv1D)):
            report.converted.append((full, type(child).__name__ + " (already SF)"))
        elif isinstance(child, nn.Linear):
            dq, note, problem = _scopes.read_weight(child)
            if problem is None and dq is None and _scopes.custom_forward(child, nn.Linear):
                problem = (f"a subclass of nn.Linear with its own forward ({type(child).__name__}); sf cannot "
                           f"tell what it computes from the weight, so it runs as it was")
            if problem is not None:
                report.skipped.append((full, type(child).__name__, problem))
            elif dq is not None:
                rep = SFLinear.from_weight(child, dq, full, cfg_for(child, full))
                report.converted.append((full, f"Linear ({note})"))
            else:
                rep = SFLinear.from_float(child, full, cfg_for(child, full))
                report.converted.append((full, "Linear"))
        elif is_hf_conv1d(child) and _float_weight(child, full, report):
            rep = SFHFConv1D(child, full, cfg_for(child, full))
            report.converted.append((full, "Conv1D (Hugging Face)"))
        elif is_hf_conv1d(child):
            pass
        elif isinstance(child, nn.Conv2d) and not _float_weight(child, full, report):
            pass
        elif isinstance(child, nn.Conv2d):
            ok, why = convertible_conv(child)
            if ok:
                rep = SFConv2d(child, full, cfg_for(child, full))
                report.converted.append((full, "Conv2d"))
            else:
                report.skipped.append((full, "Conv2d", why))
        elif isinstance(child, nn.MultiheadAttention):
            from ._attention import SFMultiheadAttention, mha_unsupported
            why = mha_unsupported(child)
            if why:
                report.skipped.append((full, "MultiheadAttention", why))
            else:
                rep = SFMultiheadAttention(child, full, cfg_for(child, full))
                report.converted.append((full, "MultiheadAttention"))
        elif nonlinear == "integer" and _int_act(child)[1] is not None:
            rep, what = _int_act(child)
            if rep is None:
                report.skipped.append((full, type(child).__name__, what))
            else:
                report.nonlinear.append((full, what))
        elif isinstance(child, (nn.Conv1d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d,
                                nn.ConvTranspose3d, nn.Bilinear,
                                nn.LSTM, nn.GRU, nn.RNN)):
            report.skipped.append((full, type(child).__name__, "no SF version yet; runs in float"))
        else:
            done[key] = (full, None)
            scoped(child, full)
            return None, False                              # a container: recurse into it
        done[key] = (full, rep)
        return rep, True

    def scoped(child: nn.Module, full: str):
        """Modules that are not swapped but whose matmuls still run on sf."""
        q = _scopes.quantized_storage(child)
        if q is not None:
            report.skipped.append((full, type(child).__name__, q))
            return
        ex = _scopes.experts_params(child)
        if ex:
            c = cfg_for(child, full)
            n = getattr(child, "num_experts", None) or child._parameters[ex[0]].shape[0]
            if _scopes.hf_experts(child):
                _scopes.install_experts(child, full, c)
                how = "each expert's projections as SF linears"
            else:                                           # unknown layout: take its own F.linear calls
                _scopes.add_scope(child, "experts", full, c)
                _scopes.eager_experts(child)
                how = "its F.linear calls taken while it runs"
            report.converted.append((full, f"MoE experts ({n} experts: {', '.join(ex)}; {how})"))
            return
        if _scopes.is_router(child):
            c = cfg_for(child, full) if cfg.bits >= 16 else cfg16
            if cfg.bits < 16:
                report.kept_sf16.append((full, "MoE router (picks the experts)"))
                report.widths[full] = 16
            _scopes.add_scope(child, "router", full, c)
            report.converted.append((full, "MoE router"))
            return
        if wide_attention and _scopes.is_attention_scope(child):
            _scopes.add_scope(child, "attention", full, cfg16)
            child._sf_softmax = nonlinear
            report.eager_attention.append(full)
        elif _scopes.is_rotary(child):
            _scopes.add_suspend(child)
        left = _scopes.unconverted_weights(child)
        if left:
            report.skipped.append((full, type(child).__name__,
                                   f"holds weights sf does not convert ({', '.join(left)}); its own maths "
                                   f"runs in float"))

    def visit(parent: nn.Module, prefix: str):
        if id(parent) in seen:
            return
        seen.add(id(parent))
        # parent._modules, not named_children(): the latter yields a module
        # registered under two names only once, and the second alias would
        # keep its float version
        for child_name, child in list(parent._modules.items()):
            if child is None:
                continue
            full = f"{prefix}.{child_name}" if prefix else child_name
            if cfg.bits < 16 and _scopes.is_linear_router(parent, child_name, child) and keep_sf16 == "auto":
                keep_ids.setdefault(id(child), "MoE router (picks the experts)")
            rep, leaf = replacement(child, full)
            if rep is not None:
                setattr(parent, child_name, rep)
            elif not leaf:
                visit(child, full)

    rep, leaf = replacement(model, "")
    if rep is not None:
        model = rep
    elif not leaf:
        visit(model, "")
    if wide_attention:
        from ._attention import install
        install(model, softmax="integer" if nonlinear == "integer" else "float")
        report.attention = "F.scaled_dot_product_attention on the MX engine at sf16 (attention.0, attention.1, ...)"
        if nonlinear == "integer":
            report.attention += "; softmax on sf's integer definition"
            report.nonlinear.append(("attention", "softmax (integer)"))
    elif nonlinear == "integer":
        report.nonlinear.append(("attention", "softmax stays float: attention runs on sf only with accumulate='wide'"))
    model._sf_conversion_report = report
    return model


def conversion_report(model: nn.Module) -> ConversionReport:
    r = getattr(model, "_sf_conversion_report", None)
    if r is None:
        raise ValueError("model was not produced by sft.convert")
    return r
