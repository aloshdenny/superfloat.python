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

from .. import _heuristic, _linalg, _vectorized as vec
from .._format import FORMATS, SFFormat

WSCALES = ("per_tensor", "per_channel")
HEADROOMS = ("bound", "none")
ACCUMULATES = ("chain", "wide")
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

    def matmul_wide(self, fmt, ka, kb, shift, shift_redo, mx, bias_at=None, b_prep=None):
        """POP at ``shift``; where an output's code reached +-max with a
        nonzero total, POP again at ``shift_redo`` (None: no redo).  The
        accumulation runs once, which gives the same bits as the chip
        relaunching the tile.  ``b_prep``: ``kb`` already prepared for the
        compiled loop (a cached weight); ``kb`` is then not read.

        ``bias_at(used)``: (codes, clamped) of the bias aligned to each
        output's exponent for the shifts ``used``; the codes are added to the POP codes with the
        saturating sfN add (ACT on the chip), and an output whose POP or add
        reached +-max is redone too."""
        if b_prep is not None:
            acc, clipped, F = _linalg.matmul_wide_acc_prepared(fmt, ka, b_prep, mx)
        else:
            acc, clipped, F = _linalg.matmul_wide_acc(fmt, ka, kb, mx)
        q, over, under = _linalg.mx_pop(fmt, acc, shift, F)
        redo = np.zeros(q.shape, bool)
        used = np.broadcast_to(shift, q.shape)
        y, add_sat, nz = _post(self, fmt, q, acc != 0, used, bias_at)
        if shift_redo is not None:
            # |q| or |q + b| reaches max, or b itself was clamped at this exponent
            redo = ((np.abs(q) >= fmt.max_code) | (np.abs(y) >= fmt.max_code) | add_sat) & nz
            if redo.any():
                q2, over2, under2 = _linalg.mx_pop(fmt, acc, shift_redo, F)
                used2 = np.broadcast_to(shift_redo, q.shape)
                y2, add_sat2, _ = _post(self, fmt, q2, acc != 0, used2, bias_at)
                y, add_sat = np.where(redo, y2, y), np.where(redo, add_sat2, add_sat)
                over, under = np.where(redo, over2, over), np.where(redo, under2, under)
                used = np.where(redo, shift_redo, used)
        return y, {"used": used, "redone": int(redo.sum()), "saturated": over | add_sat, "flushed": under,
                   "acc_clipped": clipped, "exact": True}


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

    def matmul_wide(self, fmt, ka, kb, shift, shift_redo, mx, bias_at=None):
        """One launch at ``shift``, then one relaunch at ``shift_redo`` if any
        output came back at +-max (the codes show it).  The chip cannot
        report flushed totals; accumulator clips were ruled out on the host.
        ``bias_at``: as on the host; the add runs on the target."""
        import sf
        A, B = self._arrays(fmt, ka, kb)
        with sf.errstate(all="ignore"):
            q = np.asarray(sf.matmul(A, B, accumulate="wide", shift=shift).codes, dtype=np.int64)
        used = np.broadcast_to(shift, q.shape)
        y, sat, _ = _post(self, fmt, q, None, used, bias_at)
        top = (np.abs(q) >= fmt.max_code) | (np.abs(y) >= fmt.max_code) | sat
        redo = np.zeros(q.shape, bool)
        if shift_redo is not None and top.any():
            with sf.errstate(all="ignore"):
                q2 = np.asarray(sf.matmul(A, B, accumulate="wide", shift=shift_redo).codes, dtype=np.int64)
            y2, sat2, _ = _post(self, fmt, q2, None, np.broadcast_to(shift_redo, q.shape), bias_at)
            redo = top
            q, y = np.where(redo, q2, q), np.where(redo, y2, y)
            used = np.where(redo, shift_redo, used)
            top = (np.abs(q) >= fmt.max_code) | (np.abs(y) >= fmt.max_code) | np.where(redo, sat2, sat)
        # +-max after the redo is an upper bound: it may be a true value
        return y, {"used": used, "redone": int(redo.sum()), "saturated": int(top.sum()), "flushed": None,
                   "acc_clipped": 0, "exact": not top.any()}

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


def _post(runner, fmt, q, nonzero, used, bias_at):
    """The step after POP: q + bias with the saturating sfN add, on the
    runner (ACT f=00 on the chip; below sf16 the masked form, compiler
    doc s5).  Returns (codes, saturated mask, nonzero mask); "saturated"
    includes a bias code clamped at this exponent, which the sum alone may
    not show (q of the other sign pulls it back below max)."""
    if bias_at is None:
        return q, np.zeros(q.shape, bool), (np.ones(q.shape, bool) if nonzero is None else nonzero)
    bq, clamped = (np.broadcast_to(a, q.shape) for a in bias_at(used))
    y, ev = runner.add(fmt, q, bq)
    sat = ev["saturated"]
    sat = np.broadcast_to(np.asarray(sat, dtype=bool), q.shape) if np.ndim(sat) else (np.abs(y) >= fmt.max_code)
    nz = (np.ones(q.shape, bool) if nonzero is None else nonzero) | (bq != 0)
    return np.asarray(y, dtype=np.int64), sat | clamped, nz


def _bias_codes(fmt, b, fe):
    """The bias at each output's exponent ``fe``: round half to even of
    b * 2^(P - fe), clamped to +-max.  Returns (codes, clamped mask)."""
    P, M = fmt.bits - 1, fmt.max_code
    with np.errstate(over="ignore"):
        raw = np.rint(np.ldexp(np.asarray(b, dtype=np.float64), (P - fe).astype(np.int64)))
    return np.clip(raw, -M, M).astype(np.int64), np.abs(raw) > M


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
    accumulate: str = "chain"
    shift_min: int | None = None    # wide: range of the shifts used
    shift_max: int | None = None
    h: int | None = None            # wide: the table value for this K
    redone: int = 0                 # wide: outputs popped again at the l2 exponent
    flushed: int | None = None      # wide: nonzero totals that rounded to 0 (None: unknown)
    acc_clipped: int | None = None  # wide: totals clipped at the accumulator's limit
    guard_raised: int = 0           # wide: tokens whose exponent the accumulator guard raised


def _count(mask) -> int | None:
    if mask is None:
        return None
    if isinstance(mask, (int, np.integer)):
        return int(mask)
    return int(np.count_nonzero(mask))


def _cast(fmt: SFFormat, values: np.ndarray):
    codes, ev = vec.cast_array(fmt, values)
    return codes, ev


def _check(fmt_bits: int, wscale: str | None = None, headroom: str | None = None,
           accumulate: str | None = None) -> SFFormat:
    if accumulate is not None and accumulate not in ACCUMULATES:
        raise ValueError(f"accumulate must be one of {ACCUMULATES}, got {accumulate!r}")
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


# --- wide mode (the MX engine) ---------------------------------------------------------------

WIDE_BLOCK = 1 << 23            # weight elements quantized at once by wide_linear (memory, not bits)


def _wide(fmt, x_rows, qa, ea, qb, eb, in_under, w_under, shift, runner, mx, eo=None, bias=None):
    """The common wide step for (..., M, K) codes @ (..., K, N) codes.

    ``bias`` (N,) float64: added after POP on the chip, in sfN at each
    output's exponent (``_bias_codes``), with the saturating add."""
    qa, ea, raised = _heuristic.acc_guard(fmt, x_rows, qa, ea, qb, mx)
    return _wide_cols(fmt, qa, ea, qb, eb, in_under, w_under, shift, runner, mx, eo, raised, bias)


def _wide_cols(fmt, qa, ea, qb, eb, in_under, w_under, shift, runner, mx, eo=None, raised=0, bias=None,
               prep=None):
    """The wide step after the guard: plan, launch, bias, redo, dequantize.
    ``prep``: a ``PreparedWeight`` block standing in for ``qb`` (host only)."""
    base = ea[..., :, None] + eb[..., None, :]
    if shift is None:
        sh, sh_redo, h = _heuristic.plan(fmt, qa, ea, qb, eb, eo, None if bias is None else np.abs(bias),
                                         nb=None if prep is None else prep.nb)
    else:                                          # an explicit shift overrides the rule
        sh, sh_redo, h = _linalg.check_shift(shift, allow_array=False), None, None
    bias_at = None if bias is None else (lambda used: _bias_codes(fmt, bias, base - used))
    if prep is None:
        q, ev = runner.matmul_wide(fmt, qa, qb, sh, sh_redo, mx, bias_at)
    else:
        q, ev = runner.matmul_wide(fmt, qa, None, sh, sh_redo, mx, bias_at, b_prep=(prep.m16, prep.s8))
    used = np.broadcast_to(ev["used"], q.shape)
    fe = base - used                                         # each output's exponent
    with np.errstate(over="ignore"):
        y = np.ldexp(q.astype(np.float64), (fe - (fmt.bits - 1)).astype(np.int64))
    bias_sat = 0 if bias is None else int(np.count_nonzero(_bias_codes(fmt, bias, fe)[1]))
    return y, dict(
        saturated=_count(ev["saturated"]), exact=ev["exact"], underflow=_count(ev["flushed"]),
        input_underflow=in_under, weight_underflow=w_under, bias_saturated=bias_sat, headroom_bits=0.0,
        outputs=int(q.size), act_scale=float(np.ldexp(1.0, int(ea.max()))) if ea.size else 1.0,
        accumulate="wide", shift_min=int(used.min()) if used.size else None,
        shift_max=int(used.max()) if used.size else None, h=h, redone=ev["redone"],
        flushed=_count(ev["flushed"]), acc_clipped=_count(ev["acc_clipped"]), guard_raised=raised)


def wide_linear(x: np.ndarray, w: np.ndarray | None, b: np.ndarray | None, bits: int,
                shift: int | None = None, runner=None, mx=None, prepared=None) -> CoreResult:
    """``y = x W^T + b`` on the MX engine with the wide-mode rule.

    x: (R, K), w: (N, K), b: (N,) or None, all float64.  Activations get one
    power-of-two exponent per token, weights one per output channel (the
    chip only shifts), every shift comes from ``_heuristic``, and outputs
    that saturate are redone at the l2 exponent.  ``shift`` (an int)
    overrides the rule.  The bias is added on the chip after POP (decided
    by Alosh, 2026-11-17): quantized to each output's exponent and added
    with the saturating sfN add (one ACT per output), so the whole layer is
    SF arithmetic; an output whose add reaches +-max is redone at an
    exponent that bounds |x||w| + |b|.

    ``prepared``: the weight prepared once by ``prepare_weight`` (then ``w``
    is not read): what a model's forward reuses, call after call.
    """
    fmt = _check(bits)
    runner = runner or HostRunner()
    if prepared is not None:
        return _wide_linear_prepared(fmt, x, prepared, b, shift, runner, mx)
    x = sanitize(x, fmt, "linear input")
    w = sanitize(w, fmt, "linear weight")
    qa, ea, in_under = _heuristic.quantize_pow2(fmt, x, -1)
    bias = None if b is None else sanitize(b, fmt, "linear bias")
    N, K = w.shape
    rows = max(1, WIDE_BLOCK // max(1, K))
    if N <= rows:
        qw, ew, w_under = _heuristic.quantize_pow2(fmt, w, -1)
        y, info = _wide(fmt, x, qa, ea, np.ascontiguousarray(qw.T), ew, in_under, w_under, shift, runner, mx,
                        bias=bias)
    else:
        y, info = _wide_linear_blocks(fmt, x, qa, ea, in_under, w, rows, shift, runner, mx, bias)
    return CoreResult(y=y, **info)


def wide_matmul(a_val: np.ndarray, b_val: np.ndarray, bits: int, shift: int | None = None,
                runner=None, mx=None, out_rule: str = "heuristic") -> CoreResult:
    """(..., M, K) @ (..., K, N) of two activations on the MX engine: one
    exponent per row of ``a`` and per column of ``b``, shifts from the rule.

    ``out_rule="max_b"`` (attention's P @ V, rows of ``a`` nonnegative): see
    ``_wide_pv``.  Every exponent of row t comes from the keys row t can see
    (``a[t, k] != 0``), so keys appended after it never change its result,
    and no output can saturate, so P @ V is never redone."""
    fmt = _check(bits)
    runner = runner or HostRunner()
    a_val = sanitize(a_val, fmt, "matmul left operand")
    b_val = sanitize(b_val, fmt, "matmul right operand")
    if out_rule not in ("heuristic", "max_b"):
        raise ValueError(f"out_rule must be 'heuristic' or 'max_b', got {out_rule!r}")
    if out_rule == "max_b":
        return _wide_pv(fmt, a_val, b_val, shift, runner, mx)
    qa, ea, ua = _heuristic.quantize_pow2(fmt, a_val, -1)
    qb, eb, ub = _heuristic.quantize_pow2(fmt, b_val, -2)
    y, info = _wide(fmt, a_val, qa, ea, qb, eb, ua + ub, 0, shift, runner, mx)
    return CoreResult(y=y, **info)


PV_PER_TOKEN = True     # V quantized per key (Alosh, 2026-11-18); False: per channel over every key (DSL <= 0.15.0)


def _wide_pv(fmt, p, v, shift, runner, mx):
    """Attention's P @ V (``p`` (..., M, S) nonnegative, 0 on masked keys; ``v`` (..., S, N)).

    Prefix invariance (Alosh, 2026-11-18): row t's result depends only on the keys it can see, so appending
    keys (a KV cache growing, more of a sequence in the batch) never changes it.
      * V is quantized per key, one exponent e_k per row of V: written once, like a KV cache entry.
      * Each row folds its keys' exponents into P, exactly (a power of two): P'[t, k] = P[t, k] 2^(e_k - E_t),
        with E_t = max e_k over the keys row t sees.  The MX engine then runs one P' @ V matmul at one scale
        per row, and E_t moves into the row's exponent.
      * The output exponent bounds the product exactly from the quantized operands,
        (sum_k |P'[t, k]|) * max |V[k, c]| over the visible k, with the margin that keeps POP's rounding
        below the largest code: no P @ V output can saturate, none is redone.
      * The accumulator guard bounds each row against the column norms of its visible keys only.
    ``PV_PER_TOKEN = False`` keeps the per-channel V of DSL 0.15.0 (exponents over every key, not invariant)."""
    P = fmt.bits - 1
    vis = p != 0
    lead = np.broadcast_shapes(p.shape[:-2], v.shape[:-2])
    visb = np.broadcast_to(vis, lead + p.shape[-2:])
    if PV_PER_TOKEN:
        qv, ek, ub = _heuristic.quantize_pow2(fmt, v, -1)                       # (..., S, N), e per key (..., S)
        ekb = np.broadcast_to(ek[..., None, :], lead + p.shape[-2:])           # (..., M, S)
        E = np.where(visb, ekb, np.iinfo(np.int64).min).max(axis=-1) if visb.shape[-1] else \
            np.zeros(visb.shape[:-1], np.int64)
        E = np.where(visb.any(axis=-1), E, 0)                                   # (..., M)
        rows = np.where(visb, np.ldexp(np.broadcast_to(p, visb.shape), np.where(visb, ekb - E[..., None], 0)), 0.0)
        eb = np.zeros(v.shape[:-2] + v.shape[-1:], np.int64)
    else:
        qv, eb, ub = _heuristic.quantize_pow2(fmt, v, -2)
        rows, E = p, 0
    qa, ea, ua = _heuristic.quantize_pow2(fmt, rows, -1)
    bmag = _linalg._mx_mags(fmt, qv, _linalg.mx_block(fmt)).astype(np.float64)
    bmax = np.sqrt(np.matmul(visb.astype(np.float64), bmag * bmag)).max(axis=-1) if bmag.size else \
        np.zeros(visb.shape[:-1])                                               # (..., M): visible keys only
    qa, ea, raised = _heuristic.acc_guard(fmt, rows, qa, ea, None, mx, bmax=bmax)
    ea = ea + E
    pd = np.abs(qa) * np.ldexp(1.0, (ea - P).astype(np.int64))[..., None]       # P' as the engine sees it
    vd = np.abs(qv) * np.ldexp(1.0, (eb - P).astype(np.int64))[..., None, :]
    eo = visible_max_exp(visb, vd, pd.sum(axis=-1) * (2.0 ** P / (2.0 ** P - 1)))
    y, info = _wide_cols(fmt, qa, ea, qv, eb, ua + ub, 0, shift, runner, mx, eo, raised)
    return CoreResult(y=y, **info)


def visible_max_exp(p: np.ndarray, v: np.ndarray, scale=None, chunk: int = 1 << 24) -> np.ndarray:
    """pow2_exp of max |v[k, c]| over the k with p[t, k] != 0, for every (t, c):
    (..., M, S) and (..., S, N) -> (..., M, N); times ``scale[..., t]`` first
    when given.  Rows are done in chunks so
    the (..., rows, S, N) intermediate stays below ``chunk`` elements."""
    vis = np.asarray(p) != 0
    av = np.abs(np.asarray(v, dtype=np.float64))
    lead = np.broadcast_shapes(vis.shape[:-2], av.shape[:-2])
    vis = np.broadcast_to(vis, lead + vis.shape[-2:])
    av = np.broadcast_to(av, lead + av.shape[-2:])
    M, S, N = vis.shape[-2], vis.shape[-1], av.shape[-1]
    out = np.zeros(lead + (M, N))
    step = max(1, chunk // max(1, int(np.prod(lead, dtype=np.int64)) * S * N))
    for i in range(0, M, step):
        sl = slice(i, min(M, i + step))
        out[..., sl, :] = np.where(vis[..., sl, :, None], av[..., None, :, :], 0.0).max(axis=-2) if S else 0.0
    if scale is not None:
        out = out * np.broadcast_to(np.asarray(scale, dtype=np.float64), lead + (M,))[..., None]
    return _heuristic.pow2_exp(out)


def _wide_linear_blocks(fmt, x, qa, ea, in_under, w, rows, shift, runner, mx, bias=None):
    """``wide_linear`` for a weight too big to quantize at once (an LM head of
    150,000 rows): the same bits, ``rows`` output channels at a time.  Each
    channel's exponent depends only on its own row, and the guard needs only
    the largest column norm, so pass 1 finds that and pass 2 computes."""
    N = w.shape[0]
    blocks = [slice(i, min(N, i + rows)) for i in range(0, N, rows)]
    bmax, w_under = 0.0, 0
    for s in blocks:
        qw, _, wu = _heuristic.quantize_pow2(fmt, w[s], -1)
        bmax = max(bmax, float(_heuristic.col_norm_max(fmt, qw.T)[0]))
        w_under += wu
    qa, ea, raised = _heuristic.acc_guard(fmt, x, qa, ea, None, mx, bmax=np.array([bmax]))
    ys, infos = [], []
    for s in blocks:
        qw, ew, _ = _heuristic.quantize_pow2(fmt, w[s], -1)
        yb, ib = _wide_cols(fmt, qa, ea, np.ascontiguousarray(qw.T), ew, in_under, w_under, shift, runner, mx,
                            None, raised, None if bias is None else bias[s])
        ys.append(yb)
        infos.append(ib)

    def total(key):
        vals = [i[key] for i in infos]
        return None if any(v is None for v in vals) else sum(vals)

    info = dict(infos[0])
    info.update(saturated=total("saturated"), underflow=total("underflow"), flushed=total("flushed"),
                acc_clipped=total("acc_clipped"), redone=sum(i["redone"] for i in infos),
                bias_saturated=sum(i["bias_saturated"] for i in infos),
                outputs=sum(i["outputs"] for i in infos), exact=all(i["exact"] for i in infos),
                shift_min=min(i["shift_min"] for i in infos), shift_max=max(i["shift_max"] for i in infos))
    return np.concatenate(ys, axis=1), info


# --- weights prepared once ----------------------------------------------------------------------

class PreparedWeight:
    """A linear layer's weight, prepared once for the host's wide path.

    Holds what every call needs and nothing it can recompute cheaply: each
    output channel's exponent ``ew``, the dequantized column norms ``nb``
    (for the rule's l2), the largest column norm ``bmax`` (for the guard),
    the weight-underflow count, and the codes as the compiled loop reads
    them (``m16`` uint16 magnitudes, ``s8`` int8 signs, (N, K)): 3 bytes per
    weight.  Built a block of rows at a time, like ``_wide_linear_blocks``,
    so the bits are those of the unprepared path."""

    def __init__(self, ew, nb, bmax, w_under, m16, s8, bits):
        self.ew, self.nb, self.bmax, self.w_under = ew, nb, bmax, w_under
        self.m16, self.s8, self.bits = m16, s8, bits
        self.N, self.K = m16.shape

    @property
    def nbytes(self) -> int:
        return self.m16.nbytes + self.s8.nbytes + self.ew.nbytes + self.nb.nbytes

    def rows(self, s):
        return PreparedWeight(self.ew[s], self.nb[s], self.bmax, self.w_under, self.m16[s], self.s8[s], self.bits)


def prepare_weight(w: np.ndarray, bits: int):
    """``PreparedWeight`` for ``w`` (N, K), or None when the host cannot use
    one (no compiled loop) or the weight holds NaN or inf (its events must
    fire on every call, so it takes the unprepared path)."""
    from .. import _fastmx
    fmt = _check(bits)
    if not (_linalg.FAST_MX and _fastmx.available()):
        return None
    w = np.asarray(w, dtype=np.float64)
    if not np.isfinite(w).all():
        return None
    N, K = w.shape
    rows = max(1, WIDE_BLOCK // max(1, K))
    blk = _linalg.mx_block(fmt)
    P = fmt.bits - 1
    m16 = np.empty((N, K), dtype=np.uint16)
    s8 = np.empty((N, K), dtype=np.int8)
    ew = np.empty(N, dtype=np.int64)
    nb = np.empty(N, dtype=np.float64)
    bmax, w_under = 0.0, 0
    for i in range(0, N, rows):
        s = slice(i, min(N, i + rows))
        qw, e, wu = _heuristic.quantize_pow2(fmt, w[s], -1)
        qb = qw.T
        bm, bs = _fastmx.prep_b(_linalg._mx_mags(fmt, qb, blk), qb < 0)
        m16[s], s8[s], ew[s] = bm, bs, e
        nb[s] = _heuristic._norms(fmt, qb, -2) * np.ldexp(1.0, (e - P).astype(np.int64))
        bmax = max(bmax, float(_heuristic.col_norm_max(fmt, qb)[0]))
        w_under += wu
    return PreparedWeight(ew, nb, bmax, w_under, m16, s8, bits)


def _wide_linear_prepared(fmt, x, pw, b, shift, runner, mx):
    """``wide_linear`` with a ``PreparedWeight``: the same steps and bits, in
    the same blocks of rows, without touching the float weight."""
    if pw.bits != fmt.bits:
        raise ValueError(f"prepared for sf{pw.bits}, called at {fmt.name}")
    x = sanitize(x, fmt, "linear input")
    bias = None if b is None else sanitize(b, fmt, "linear bias")
    qa, ea, in_under = _heuristic.quantize_pow2(fmt, x, -1)
    qa, ea, raised = _heuristic.acc_guard(fmt, x, qa, ea, None, mx, bmax=np.array([pw.bmax]))
    rows = max(1, WIDE_BLOCK // max(1, pw.K))
    ys, infos = [], []
    for i in range(0, pw.N, rows):
        s = slice(i, min(pw.N, i + rows))
        part = pw.rows(s)
        yb, ib = _wide_cols(fmt, qa, ea, None, part.ew, in_under, pw.w_under, shift, runner, mx, None, raised,
                            None if bias is None else bias[s], prep=part)
        ys.append(yb)
        infos.append(ib)
    if len(infos) == 1:
        return CoreResult(y=ys[0], **infos[0])
    return CoreResult(y=np.concatenate(ys, axis=1), **_merge_infos(infos))


def _merge_infos(infos):
    def total(key):
        vals = [i[key] for i in infos]
        return None if any(v is None for v in vals) else sum(vals)

    info = dict(infos[0])
    info.update(saturated=total("saturated"), underflow=total("underflow"), flushed=total("flushed"),
                acc_clipped=total("acc_clipped"), redone=sum(i["redone"] for i in infos),
                bias_saturated=sum(i["bias_saturated"] for i in infos),
                outputs=sum(i["outputs"] for i in infos), exact=all(i["exact"] for i in infos),
                shift_min=min(i["shift_min"] for i in infos), shift_max=max(i["shift_max"] for i in infos))
    return info
