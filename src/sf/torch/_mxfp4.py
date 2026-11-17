"""MXFP4 weights (OCP Microscaling v1.0), the format the gpt-oss checkpoints ship in.

Same pattern as FP8 with block scales: sf reads the stored values exactly,
then quantizes them to sf like any float weight.  Nothing about MXFP4 reaches
the chip.

The format.  Every block of 32 values along a row shares one scale:

* elements: FP4 E2M1, 4 bits, sign-magnitude:
  0, 0.5, 1, 1.5, 2, 3, 4, 6 and their negatives (code 8 is -0);
* scale: E8M0, one byte, the power of two 2^(s - 127); s = 255 is NaN;
* packing: two elements per byte, the low nibble first (element 2i in bits
  0..3, element 2i + 1 in bits 4..7), as gpt-oss and ``transformers`` store it.

``dequantize_mxfp4(blocks, scales)``: ``blocks`` uint8 (..., G, 16), ``scales``
uint8 (or float8_e8m0fnu) (..., G) -> float (..., G * 32).  Every value is
an E2M1 value times a power of two, so it is exact in float32 (and bf16).

``load_mxfp4(model, source)``: a checkpoint's ``<name>_blocks`` /
``<name>_scales`` pairs, dequantized into the model's float parameter
``<name>``.  gpt-oss stores its experts as (E, out, in) blocks while the model
keeps (E, in, out), so every 3-D pair is transposed, as ``transformers``
does (``experts_transposed=False`` keeps them as stored).  2-D pairs (linear
weights, (out, in)) are loaded as stored.

``quantize_mxfp4(w)``: the encoder, for tests and for writing a checkpoint:
the OCP rule, shared exponent floor(log2(max|block|)) - 2, elements rounded
to nearest E2M1 (ties to even) and saturated at +-6.
"""
from __future__ import annotations

import os

import numpy as np
import torch

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
BLOCK = 32
SCALE_BIAS = 127
SCALE_NAN = 255
_E8M0 = getattr(torch, "float8_e8m0fnu", None)


def _scale_bytes(scales: torch.Tensor) -> torch.Tensor:
    if _E8M0 is not None and scales.dtype == _E8M0:
        return scales.view(torch.uint8)
    if scales.dtype not in (torch.uint8, torch.int32, torch.int64, torch.int16):
        raise TypeError(f"MXFP4 scales must be uint8 (E8M0 bytes), got {scales.dtype}")
    if scales.dtype != torch.uint8 and (scales.min() < 0 or scales.max() > 255):
        raise ValueError("MXFP4 scales must be bytes 0..255")
    return scales.to(torch.uint8)


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """(..., G, B) packed bytes and (..., G) E8M0 scales -> (..., G * 2B) values."""
    if blocks.dtype != torch.uint8:
        raise TypeError(f"MXFP4 blocks must be uint8 (two E2M1 values per byte), got {blocks.dtype}")
    s = _scale_bytes(scales)
    if tuple(blocks.shape[:-1]) != tuple(s.shape):
        raise ValueError(f"MXFP4 blocks {tuple(blocks.shape)} do not match scales {tuple(s.shape)}: "
                         f"expected scales of shape {tuple(blocks.shape[:-1])}")
    lut = torch.tensor(E2M1, dtype=torch.float32)
    lo = lut[(blocks & 0x0F).long()]
    hi = lut[(blocks >> 4).long()]
    vals = torch.stack((lo, hi), dim=-1).reshape(*blocks.shape[:-1], blocks.shape[-1] * 2)
    e = s.to(torch.int32) - SCALE_BIAS
    out = torch.ldexp(vals, e.unsqueeze(-1).to(torch.float32))
    out = torch.where((s == SCALE_NAN).unsqueeze(-1), torch.full_like(out, float("nan")), out)
    return out.reshape(*blocks.shape[:-2], blocks.shape[-2] * blocks.shape[-1] * 2).to(dtype)


def _round_e2m1(x: np.ndarray) -> np.ndarray:
    """Nearest E2M1 code (0..15) for each value, ties to even code, saturating at +-6."""
    mags = np.array(E2M1[:8])
    a = np.minimum(np.abs(x), 6.0)
    d = np.abs(a[..., None] - mags)                          # (..., 8)
    best = d.min(-1, keepdims=True)
    ties = d == best
    # among tied magnitudes pick the even code (E2M1's mantissa bit 0): codes 0..7, even = mantissa 0
    idx = np.where(ties & (np.arange(8) % 2 == 0), np.arange(8), 99).min(-1)
    idx = np.where(idx == 99, d.argmin(-1), idx)
    return (idx + 8 * ((x < 0) & (idx > 0))).astype(np.uint8)


def quantize_mxfp4(w: torch.Tensor):
    """(..., K) float, K a multiple of 32 -> (blocks uint8 (..., K/32, 16), scales uint8 (..., K/32))."""
    x = w.detach().to(torch.float64).cpu().numpy()
    K = x.shape[-1]
    if K % BLOCK:
        raise ValueError(f"MXFP4 needs the last axis in blocks of {BLOCK}, got {K}")
    xb = x.reshape(*x.shape[:-1], K // BLOCK, BLOCK)
    amax = np.abs(xb).max(-1)
    with np.errstate(divide="ignore"):
        e = np.where(amax > 0, np.floor(np.log2(np.where(amax > 0, amax, 1.0))) - 2, -SCALE_BIAS)
    e = np.clip(e, -SCALE_BIAS, SCALE_BIAS).astype(np.int64)
    codes = _round_e2m1(np.ldexp(xb, -e[..., None]))
    packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).astype(np.uint8)
    return torch.from_numpy(packed), torch.from_numpy((e + SCALE_BIAS).astype(np.uint8))


def _pairs(source):
    """{name: (blocks, scales)} from a dict of tensors, a .safetensors file or a folder of them."""
    if isinstance(source, (str, os.PathLike)):
        from safetensors.torch import load_file
        p = os.fspath(source)
        files = [p] if os.path.isfile(p) else sorted(os.path.join(p, f) for f in os.listdir(p)
                                                      if f.endswith(".safetensors"))
        if not files:
            raise FileNotFoundError(f"no .safetensors files in {p}")
        tensors = {}
        for f in files:
            sd = load_file(f)
            tensors.update({k: v for k, v in sd.items() if k.endswith(("_blocks", "_scales"))})
    else:
        tensors = dict(source)
    pairs = {}
    for k, v in tensors.items():
        if k.endswith("_blocks"):
            name = k[: -len("_blocks")]
            if name + "_scales" not in tensors:
                raise ValueError(f"{k} has no {name}_scales next to it")
            pairs[name] = (v, tensors[name + "_scales"])
    return pairs


@torch.no_grad()
def load_mxfp4(model: torch.nn.Module, source, strict: bool = True, experts_transposed: bool = True) -> list:
    """Dequantize every MXFP4 pair in ``source`` into ``model``'s float
    parameter of the same name.  Returns the names loaded.  ``strict``: a pair
    with no parameter of that name raises."""
    params = dict(model.named_parameters())
    loaded = []
    for name, (blocks, scales) in sorted(_pairs(source).items()):
        p = params.get(name)
        if p is None:
            if strict:
                raise KeyError(f"MXFP4 tensor {name}_blocks has no parameter {name} in the model")
            continue
        if not p.is_floating_point():
            raise TypeError(f"{name} is {p.dtype}; load MXFP4 into a float model (it holds the dequantized values)")
        v = dequantize_mxfp4(blocks, scales)
        if v.dim() == 3 and experts_transposed:
            v = v.transpose(-1, -2)                           # gpt-oss experts: stored (E, out, in), used (E, in, out)
        if tuple(v.shape) != tuple(p.shape):
            raise ValueError(f"MXFP4 {name}: dequantized shape {tuple(v.shape)} does not fit the parameter "
                             f"{tuple(p.shape)}" + (" (3-D tensors are transposed as gpt-oss stores them; "
                                                    "pass experts_transposed=False for another layout)"
                                                    if v.dim() == 3 else ""))
        p.copy_(v.to(p.dtype))
        loaded.append(name)
    return loaded


def is_mxfp4_linear(m) -> bool:
    """A linear-like module with packed MXFP4 weights next to its E8M0 scales:
    weight uint8 (out, in / 2) and a scale of shape (out, in / 32)."""
    w = getattr(m, "weight", None)
    if w is None:
        w = getattr(m, "weight_packed", None)
    s = None
    for n in ("weight_scale", "weight_scales", "scales", "weight_scale_inv"):
        t = getattr(m, n, None)
        if isinstance(t, torch.Tensor):
            s = t
            break
    if not isinstance(w, torch.Tensor) or s is None or w.dtype != torch.uint8 or w.dim() != 2 or s.dim() != 2:
        return False
    if not (s.dtype == torch.uint8 or (_E8M0 is not None and s.dtype == _E8M0)):
        return False
    return s.shape[0] == w.shape[0] and (w.shape[1] * 2) == s.shape[1] * BLOCK


def read_mxfp4_linear(m) -> torch.Tensor:
    w = getattr(m, "weight", None)
    if w is None or w.dtype != torch.uint8:
        w = m.weight_packed
    s = next(getattr(m, n) for n in ("weight_scale", "weight_scales", "scales", "weight_scale_inv")
             if isinstance(getattr(m, n, None), torch.Tensor))
    out, half = w.shape
    blocks = w.detach().reshape(out, half * 2 // BLOCK, BLOCK // 2)
    return dequantize_mxfp4(blocks, s.detach())
