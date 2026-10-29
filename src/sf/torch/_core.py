"""Chip-exact SF linear and matmul with a scale policy (spec section 8, 2.1).

Torch-free: works on float64 NumPy arrays and SF code arrays, so the eager
emulation, the autocast path and the compile backend all run exactly the same
function and differ only in the *runner* that executes the SF kernels
(host reference or a device).

Why a scale policy: real activations and weights exceed ±1, SF values do not.
Every operand is held as ``value = scale * code / 2^(N-1)`` with a float scale
kept on the host (spec section 12, "with the float scale kept on the host").

Linear ``y = x W^T + b`` (x: (R, K), W: (N, K), b: (N,)):

1. Weight scale ``s_w``: per tensor (one scale) or per channel (one scale per
   output row), chosen so the largest weight maps to the largest code.
   ``W_code = cast(W / s_w)``.
2. Activation scale ``a``: dynamic per call, ``a0 = max|x| / max``.
   With ``headroom="bound"`` (default) ``a`` is then increased until the L1
   bound proves that no partial sum of the accumulation chain, and no bias
   add, can saturate (``_linalg.matmul_bound``).  That is the pre-flight
   rule from NOTES.md applied at the source: the device is never handed an
   accumulation that can saturate, and its pre-flight check proves the same
   bound, so ``status.saturated`` is an exact 0.  The cost is precision:
   ``headroom_bits = log2(a / a0)`` bits of the activation are given up, and
   it is reported per layer.  ``headroom="none"`` uses ``a0`` and lets the
   chain saturate (counted in the status).
3. ``x_code = cast(x / a)``; ``b_code = cast(b / (a s_w))`` (in the
   accumulator scale).
4. On the runner: ``acc = x_code @ W_code^T`` (sequential fma chain with
   per-product truncation and per-step saturation, Stage 1.1), then one
   saturating add of ``b_code`` (``addmm`` semantics).
5. ``y = acc / 2^(N-1) * a * s_w``, in float64 on the host.

Matmul ``A @ B`` of two activations is the same with a dynamic per-tensor
scale on both operands and the headroom applied to ``A``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .. import _linalg, _vectorized as vec
from .._format import FORMATS, SFFormat

WSCALES = ("per_tensor", "per_channel")
HEADROOMS = ("bound", "none")
_MAX_ITER = 32


# --- runners -------------------------------------------------------------------

class HostRunner:
    """Executes SF kernels on the bit-exact host reference (exact events)."""

    placement = "host"

    def matmul(self, fmt: SFFormat, ka: np.ndarray, kb: np.ndarray):
        codes, ev = _linalg.matmul_codes(fmt, ka, kb)
        return codes, {"saturated": ev["overflow"], "underflow": ev["underflow"], "exact": True}

    def add(self, fmt: SFFormat, ka: np.ndarray, kb: np.ndarray):
        codes, ev = vec.add(fmt, ka, kb)
        return codes, {"saturated": ev["overflow"], "exact": True}


class DeviceRunner:
    """Executes SF kernels on a target through the public Stage 1.1 heavy-op
    path, so the pre-flight gate and ``result.status`` apply."""

    def __init__(self, device: str) -> None:
        self.placement = device

    def _arrays(self, fmt, *codes):
        import sf
        dt = sf.dtypes[fmt.bits - 4]
        return [sf.from_codes(np.asarray(c, dtype=np.int64), dt, device=self.placement) for c in codes]

    def matmul(self, fmt, ka, kb):
        import sf
        A, B = self._arrays(fmt, ka, kb)
        with sf.errstate(all="ignore"):
            C = A @ B
        return np.asarray(C.codes, dtype=np.int64), self._events(C, under=True)

    def add(self, fmt, ka, kb):
        import sf
        A, B = self._arrays(fmt, ka, kb)
        with sf.errstate(all="ignore"):
            C = A + B
        return np.asarray(C.codes, dtype=np.int64), self._events(C, under=False)

    @staticmethod
    def _events(C, under: bool) -> dict:
        """Per-output masks when the target reported them (flags or replay),
        so a layer can take the union of its steps' events like the host does;
        otherwise the status counts."""
        st = C.status
        masks = getattr(C, "_events", None)
        if masks is not None:
            return {"saturated": np.asarray(masks.get("overflow", np.zeros(np.shape(C.codes), bool))),
                    "underflow": np.asarray(masks["underflow"]) if "underflow" in masks else st.underflow,
                    "exact": True}
        return {"saturated": st.saturated, "underflow": st.underflow if under else 0, "exact": st.exact}


# --- results -----------------------------------------------------------------------

@dataclass
class CoreResult:
    y: np.ndarray                   # float64 result
    saturated: int | None           # outputs whose chain or bias add saturated (None: unknown)
    exact: bool
    underflow: int | None           # outputs where a nonzero product truncated to 0 (None: unknown)
    input_underflow: int            # nonzero activations quantized to 0
    weight_underflow: int           # nonzero weights quantized to 0
    bias_saturated: int             # bias values that did not fit the accumulator scale
    headroom_bits: float
    outputs: int
    act_scale: float = 1.0          # the activation scale ``a`` that was chosen


def _count(mask) -> int | None:
    if mask is None:
        return None
    if isinstance(mask, (int, np.integer)):
        return int(mask)
    return int(np.count_nonzero(mask))


def _cast(fmt: SFFormat, values: np.ndarray):
    codes, ev = vec.cast_array(fmt, values)
    return codes, ev


def _check(fmt_bits: int, wscale: str | None = None, headroom: str | None = None) -> SFFormat:
    if fmt_bits not in FORMATS:
        raise ValueError(f"unsupported SF width {fmt_bits}")
    if wscale is not None and wscale not in WSCALES:
        raise ValueError(f"scale must be one of {WSCALES}, got {wscale!r}")
    if headroom is not None and headroom not in HEADROOMS:
        raise ValueError(f"headroom must be one of {HEADROOMS}, got {headroom!r}")
    return FORMATS[fmt_bits]


# --- non-finite inputs -----------------------------------------------------------------------

def sanitize(x: np.ndarray, fmt: SFFormat, what: str) -> np.ndarray:
    """Route NaN and ±inf through the SF error policy before any scale is
    chosen, exactly as a Stage 1 cast would: NaN is an ``invalid`` event
    (device value 0), ±inf an ``overflow`` event.  Under a non-raising
    policy the returned array is finite: NaN becomes 0 and ±inf becomes the
    largest finite magnitude of the tensor with its sign (so it quantizes to
    ±max in the scale the finite values choose), or ±max itself when nothing
    is finite."""
    from .. import _errstate
    x = np.asarray(x, dtype=np.float64)
    nan = np.isnan(x)
    inf = np.isinf(x)
    if nan.any():
        _errstate.handle("invalid", f"{fmt.name} {what}: invalid (NaN) in "
                                    f"{int(nan.sum())} of {x.size} element(s)")
    if inf.any():
        _errstate.handle("overflow", f"{fmt.name} {what}: ±inf in {int(inf.sum())} of "
                                     f"{x.size} element(s) (saturates)")
    if nan.any():
        x = np.where(nan, 0.0, x)
    if inf.any():
        big = _finite_absmax(x)
        if big == 0.0:
            big = float(fmt.max)
        x = np.where(inf, np.sign(x) * big, x)
    return x


def _finite_absmax(x: np.ndarray) -> float:
    finite = np.isfinite(x)
    return float(np.abs(x[finite]).max()) if finite.any() else 0.0


# --- weights ----------------------------------------------------------------------------

def weight_scales(fmt: SFFormat, w: np.ndarray, wscale: str) -> np.ndarray:
    """One positive scale per output row (per_tensor: all equal), from the
    finite magnitudes (an inf weight saturates to ±max at that scale)."""
    m = fmt.max_code / fmt.scale
    finite = np.where(np.isfinite(w), np.abs(w), 0.0)
    if wscale == "per_tensor":
        amax = float(finite.max()) if w.size else 0.0
        s = np.full(w.shape[0], amax / m if amax > 0 else 1.0)
    else:
        amax = finite.max(axis=1) if w.size else np.zeros(w.shape[0])
        s = np.where(amax > 0, amax / m, 1.0)
    return s.astype(np.float64)


def quantize_weight(fmt: SFFormat, w: np.ndarray, wscale: str):
    """-> (codes (N, K), scales (N,), weight_underflow).  Never overflows."""
    s = weight_scales(fmt, w, wscale)
    codes, ev = _cast(fmt, w / s[:, None])
    return codes, s, _count(ev["underflow"])


def dequantize_weight(fmt: SFFormat, w: np.ndarray, wscale: str) -> np.ndarray:
    codes, s, _ = quantize_weight(fmt, w, wscale)
    return codes / fmt.scale * s[:, None]


# --- activation scale with headroom ---------------------------------------------------------

def _activation(fmt: SFFormat, x: np.ndarray, kb_for_bound, bias_over_scale, headroom: str):
    """Pick the activation scale.  ``kb_for_bound`` is the right operand in
    codes (K, N) or batched; ``bias_over_scale(a)`` returns max |b| / (a s_w)
    in accumulator codes (0 without bias).  Returns (codes, a, a0)."""
    m = fmt.max_code / fmt.scale
    amax = _finite_absmax(x) if x.size else 0.0
    a0 = amax / m if amax > 0 else 1.0
    a = a0
    kx, ev = _cast(fmt, x / a)
    if headroom == "none":
        return kx, ev, a, a0
    # the bias alone needs a >= a_bias for its codes to fit the accumulator;
    # start there so the ratio below is always finite (<= max_code)
    a_bias = bias_over_scale(None)
    if a_bias is not None and math.isfinite(a_bias) and a_bias > a:
        a = a_bias
        kx, ev = _cast(fmt, x / a)
    for _ in range(_MAX_ITER):
        ratio = bias_over_scale(a)
        bias_codes = int(math.ceil(ratio)) if ratio is not None and math.isfinite(ratio) else 0
        bound = _linalg.matmul_bound(fmt, kx, kb_for_bound) + bias_codes
        if bound <= fmt.max_code:
            return kx, ev, a, a0
        a *= max(bound / fmt.max_code, 1.0 + 2.0 ** -20)
        kx, ev = _cast(fmt, x / a)
    raise RuntimeError("sf: headroom search did not converge")   # pragma: no cover


# --- linear ---------------------------------------------------------------------------

def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray | None, bits: int,
           wscale: str = "per_tensor", headroom: str = "bound", runner=None) -> CoreResult:
    """x: (R, K) float64, w: (N, K) float64, b: (N,) float64 or None."""
    fmt = _check(bits, wscale, headroom)
    runner = runner or HostRunner()
    S = fmt.scale
    x = sanitize(x, fmt, "linear input")
    w = sanitize(w, fmt, "linear weight")
    if b is not None:
        b = sanitize(b, fmt, "linear bias")
    kw, s_w, w_under = quantize_weight(fmt, w, wscale)
    kwT = np.ascontiguousarray(kw.T)

    if b is not None and b.size:
        # |b_j| / (a s_wj) must be <= max_code / S for the bias code to fit:
        # the smallest such a is a_bias.  Computed with overflow guarded so a
        # finite bias near float64 max does not crash the search; a bias that
        # cannot be encoded by any finite a is simply counted as saturated.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            need = np.abs(b) / s_w                            # may be inf
            a_bias = float(need[np.isfinite(need)].max()) / (fmt.max_code / S) \
                if np.isfinite(need).any() else 0.0

        def bias_over_scale(a):
            if a is None:
                return a_bias
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                r = float((need / a).max() * S)
            return r
    else:
        def bias_over_scale(a):
            return None if a is None else 0.0

    kx, ev_x, a, a0 = _activation(fmt, x, kwT, bias_over_scale, headroom)
    acc, ev = runner.matmul(fmt, kx, kwT)
    sat = ev["saturated"]
    exact = ev["exact"]
    bias_sat = 0
    if b is not None:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            kb, ev_b = _cast(fmt, b / (a * s_w))
        bias_sat = _count(ev_b["overflow"])
        acc, ev2 = runner.add(fmt, acc, np.broadcast_to(kb, acc.shape))
        sat, merged_exact = _merge(sat, ev2["saturated"])
        exact = exact and ev2["exact"] and merged_exact
    with np.errstate(over="ignore"):                  # a true float64 overflow is inf, as in float
        y = acc / S * a * s_w[None, :]
    return CoreResult(
        y=y, saturated=_count(sat), exact=exact, underflow=_count(ev["underflow"]),
        input_underflow=_count(ev_x["underflow"]), weight_underflow=w_under,
        bias_saturated=bias_sat, headroom_bits=math.log2(a / a0) if a0 > 0 else 0.0,
        outputs=int(acc.size), act_scale=a)


def _merge(a, b):
    """Combine the matmul's and the bias add's per-output saturation reports
    (masks, counts or None).  Returns ``(report, exact)``.

    The layer's count is the number of outputs that saturated in either step,
    the union.  Masks give it exactly.  Two nonzero counts cannot: an output
    can saturate in both steps, so their sum is only an upper bound and the
    result is marked inexact.  A zero count adds nothing, so it stays exact."""
    if a is None or b is None:
        return None, False
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a | b, True
    ca, cb = _count(a), _count(b)
    if ca == 0:
        return b, True
    if cb == 0:
        return a, True
    return ca + cb, False


# --- matmul of two activations --------------------------------------------------------------

def matmul(a_val: np.ndarray, b_val: np.ndarray, bits: int, headroom: str = "bound",
           runner=None) -> CoreResult:
    """(..., M, K) @ (..., K, N), both operands dynamic per-tensor scale."""
    fmt = _check(bits, None, headroom)
    runner = runner or HostRunner()
    S = fmt.scale
    m = fmt.max_code / S
    a_val = sanitize(a_val, fmt, "matmul left operand")
    b_val = sanitize(b_val, fmt, "matmul right operand")
    bmax = _finite_absmax(b_val) if b_val.size else 0.0
    s_b = bmax / m if bmax > 0 else 1.0
    kb, ev_b = _cast(fmt, b_val / s_b)
    ka, ev_a, a, a0 = _activation(fmt, a_val, kb, lambda _a: None if _a is None else 0.0, headroom)
    acc, ev = runner.matmul(fmt, ka, kb)
    with np.errstate(over="ignore"):
        y = acc / S * a * s_b
    return CoreResult(
        y=y, saturated=_count(ev["saturated"]), exact=ev["exact"],
        underflow=_count(ev["underflow"]),
        input_underflow=_count(ev_a["underflow"]) + _count(ev_b["underflow"]),
        weight_underflow=0, bias_saturated=0,
        headroom_bits=math.log2(a / a0) if a0 > 0 else 0.0, outputs=int(acc.size), act_scale=a)


# --- fake quantization ---------------------------------------------------------------------

def fake_quantize(x: np.ndarray, bits: int, mode: str, scale: float):
    """Values snapped to the SF grid (spec 2.0).  ``mode``: "none" (plain
    cast, values must be in range), "dynamic" (scale = max|x| / max) or
    "fixed" (``scale`` given).  Returns (values float64, codes, scale, events)."""
    fmt = _check(bits)
    if mode == "none":
        s = 1.0
    elif mode == "dynamic":
        amax = float(np.abs(x[np.isfinite(x)]).max()) if np.isfinite(x).any() else 0.0
        s = amax / (fmt.max_code / fmt.scale) if amax > 0 else 1.0
    elif mode == "fixed":
        if not (scale > 0 and math.isfinite(scale)):
            raise ValueError("scale must be a positive finite float")
        s = float(scale)
    else:
        raise ValueError(f"unknown quantize mode {mode!r}")
    codes, ev = _cast(fmt, x / s)
    return codes / fmt.scale * s, codes, s, ev
