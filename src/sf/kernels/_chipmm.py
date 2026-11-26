"""Wide matmul on the chip as a library call: Alosh's hand-written gemm / gemv programs, no compiler.

    from sf.kernels._chipmm import ChipMatmul
    mm = ChipMatmul()                            # superfloat.sim, programs from a superfloat.gpu checkout
    codes = mm(a, w, shift, bits=16)             # (M, K) @ (K, N) codes, one POP shift per output
    mm.report                                    # launches, cycles, bus reads and writes

This is the "matmuls as library calls first, full SF IR later" path (the
compiler meeting): a linear layer runs as a call to a hand-tuned chip program
(superfloat.gpu ``test/gpt2/kernels_v4.py``) instead of a compiled kernel.
The host side is ours: it splits the matmul into blocks of 8 output columns
(gemv: one token; gemm: 8 tokens), writes each launch's data window in the
program's layout, runs the launches and reads the outputs back.  The answer
must equal ``sf._linalg.matmul_wide_codes`` (the MX model) bit for bit.

The window of one launch (compiler_v4.9.2.md 4.3, kernels_v4.py):
  words 0 .. 127   the block table: block b's x, weight, output and POP operand bases at 4b .. 4b + 3
  then             per output-column tile, its weights in WL order (chunk by chunk, rows K = 7 .. 0)
                   per token tile, its activations (gemv: x[k]; gemm: chunk by chunk, token by token)
                   per block, its 8 (gemv) or 64 (gemm) POP operands, slot << 8 | (shift & 63)
                   per block, its 8 or 64 outputs
Codes are sign-magnitude, left-aligned in 16 bits (the SF8 unit takes the top
byte).  At most 31 blocks a launch; a launch's x region never crosses 0x8000
(the program's loop compare is signed).

The programs are Superfloat's IP: they are imported from a superfloat.gpu
checkout (``SUPERFLOAT_GPU``, or ``gpu=`` here), never copied into sf.
"""
from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from .._errors import SFUnsupportedError

N = 8                       # columns per block (one per thread), K values per chunk, tokens per gemm block
MAX_BLOCKS = 31
TABLE_WORDS = 4 * MAX_BLOCKS + 4
TOP = 0xFFC0                # 0xFFC0 .. 0xFFFF is the scratchpad: the host's data there is never seen


def _kernels(gpu: str | None):
    """kernels_v4 from a superfloat.gpu checkout (it puts its own helpers on sys.path)."""
    gpu = gpu or os.environ.get("SUPERFLOAT_GPU")
    if not gpu:
        raise SFUnsupportedError("the chip's matmul programs come from a superfloat.gpu checkout: "
                                 "set SUPERFLOAT_GPU or pass gpu=")
    path = os.path.join(gpu, "test", "gpt2")
    if not os.path.isfile(os.path.join(path, "kernels_v4.py")):
        raise SFUnsupportedError(f"{path} has no kernels_v4.py (a superfloat.gpu checkout at v4.9.2?)")
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module("kernels_v4")


def to_word(code: int, bits: int) -> int:
    """A signed sfN code as the MX engine's operand word: sign-magnitude, left-aligned in 16 bits."""
    return (0x8000 if code < 0 else 0) | (abs(int(code)) << (16 - bits))


def from_word(word: int, bits: int) -> int:
    mag = (word & 0x7FFF) >> (16 - bits)
    return -mag if word & 0x8000 else mag


def unit_for(bits: int) -> int:
    """The MX block that takes sfN: SF8 for sf4 to sf8, SF16 for sf9 to sf16 (spec 1.5.0)."""
    if not 4 <= bits <= 16:
        raise SFUnsupportedError(f"sf{bits}: the MX engine takes sf4 to sf16")
    return 8 if bits <= 8 else 16


@dataclass
class Report:
    kernel: str = ""
    launches: list = field(default_factory=list)   # superfloat.sim's report per launch

    @property
    def cycles(self) -> int:
        return sum(r["cycles"] for r in self.launches)

    @property
    def data_reads(self) -> int:
        return sum(r["data_reads"] for r in self.launches)


def weight_words(W: np.ndarray) -> np.ndarray:
    """A (K, 8) tile of operand words in WL order: chunk by chunk, rows K = 7 .. 0, each row's 8 columns
    (kernels_v4.weight_words, unpacked; tests/test_chipmm.py checks they agree)."""
    return W.reshape(-1, N, N)[:, ::-1, :].reshape(-1)


def act_words(A: np.ndarray) -> np.ndarray:
    """An (8, K) token tile in gemm order: chunk by chunk, tokens 0 .. 7, lanes 0 .. 7 (kernels_v4.act_words)."""
    return A.reshape(N, -1, N).transpose(1, 0, 2).reshape(-1)


def encode(codes: np.ndarray, bits: int) -> np.ndarray:
    """``to_word`` for an array."""
    c = np.asarray(codes, dtype=np.int64)
    return np.where(c < 0, 0x8000, 0) | (np.abs(c) << (16 - bits))


def decode(words: np.ndarray, bits: int) -> np.ndarray:
    w = np.asarray(words, dtype=np.int64)
    mag = (w & 0x7FFF) >> (16 - bits)
    return np.where(w & 0x8000, -mag, mag)


class _Window:
    """One launch's data window, as a dense word array (written to the chip in one go)."""

    def __init__(self):
        self.words = np.zeros(TOP, dtype=np.int64)
        self.top = TABLE_WORDS
        self.blocks = 0

    @staticmethod
    def _base(top: int, words: int, no_cross: bool) -> int:
        """Where a region of ``words`` goes when the window is filled up to ``top``."""
        return 0x8000 if no_cross and top < 0x8000 < top + words else top   # x loops compare signed

    def fits(self, regions) -> bool:
        """Whether every (words, no_cross) region still fits, placed in order."""
        top = self.top
        for words, no_cross in regions:
            top = self._base(top, words, no_cross) + words
        return top <= TOP

    def alloc(self, words: int, no_cross: bool = False) -> int:
        base = self._base(self.top, words, no_cross)
        self.top = base + words
        assert self.top <= TOP, "window overflow"
        return base

    def put(self, base: int, words) -> None:
        words = np.asarray(words)
        self.words[base:base + words.size] = words


class ChipMatmul:
    """``(a @ w)`` with one POP shift per output, on superfloat.sim, through the chip's own programs.

    ``a`` (M, K), ``w`` (K, N) and ``shift`` (broadcastable to (M, N)) are
    signed sfN codes and ints in -32..31.  M = 1 runs the gemv program (one
    token a block), any other M the gemm program (8 tokens a block); M, K and N
    are padded with zeros to whole tiles of 8.
    """

    def __init__(self, gpu: str | None = None, sim=None, version: str = "v4.9.2") -> None:
        self.K = _kernels(gpu)
        if sim is None:
            from ._chipsim import _chip_class
            sim = _chip_class()(version)
        self.chip = sim
        self.report = Report()
        self._programs: dict = {}          # (gemv, Kp, unit, bits) -> program

    def __call__(self, a, w, shift, bits: int = 16) -> np.ndarray:
        a, w = np.asarray(a, dtype=np.int64), np.asarray(w, dtype=np.int64)
        (M, K), (K2, Nc) = a.shape, w.shape
        if K != K2:
            raise ValueError(f"a is {a.shape} and w is {w.shape}")
        s = np.broadcast_to(np.asarray(shift, dtype=np.int64), (M, Nc))
        if s.size and (s.min() < -32 or s.max() > 31):
            raise ValueError("POP shifts are -32..31")
        unit, gemv = unit_for(bits), M == 1
        lim = (1 << (bits - 1)) - 1
        for name, x in (("a", a), ("w", w)):
            if x.size and np.abs(x).max() > lim:
                raise ValueError(f"{name} holds a code outside sf{bits}'s +-{lim}")
        tm = 1 if gemv else N
        Mp, Kp, Np = -(-M // tm) * tm, max(N, -(-K // N) * N), -(-Nc // N) * N
        A = np.zeros((Mp, Kp), np.int64); A[:M, :K] = a
        W = np.zeros((Kp, Np), np.int64); W[:K, :Nc] = w
        S = np.zeros((Mp, Np), np.int64); S[:M, :Nc] = s
        Aw, Ww = encode(A, bits), encode(W, bits)
        Kv = self.K
        key = (gemv, Kp, unit, bits)
        if key not in self._programs:
            self._programs[key] = (Kv.gemv_program if gemv else Kv.gemm_program)(Kp, unit, bits)[0]
        prog = self._programs[key]
        self.report = Report("gemv" if gemv else "gemm")
        x_words = Kp * tm
        out = np.zeros((Mp, Np), np.int64)
        slots = (np.arange(tm)[:, None] << 8)                      # POP operand: slot t << 8 | (shift & 63)
        blocks = [(m0, n0) for m0 in range(0, Mp, tm) for n0 in range(0, Np, N)]
        i = 0
        while i < len(blocks):
            win, wbase, xbase, placed = _Window(), {}, {}, []
            while i < len(blocks) and win.blocks < MAX_BLOCKS:
                m0, n0 = blocks[i]
                need = ([(x_words, True)] if m0 not in xbase else []) + \
                       ([(Kp * N, False)] if n0 not in wbase else []) + [(tm * N, False)] * 2   # POP operands, outputs
                if not win.fits(need):
                    break
                if m0 not in xbase:                       # x first: low addresses, never across 0x8000
                    xbase[m0] = xb = win.alloc(x_words, no_cross=True)
                    win.put(xb, Aw[m0] if gemv else act_words(Aw[m0:m0 + N]))
                if n0 not in wbase:
                    wbase[n0] = wb = win.alloc(Kp * N)
                    win.put(wb, weight_words(Ww[:, n0:n0 + N]))
                sb, yb = win.alloc(tm * N), win.alloc(tm * N)
                win.put(sb, ((S[m0:m0 + tm, n0:n0 + N] & 63) | slots).reshape(-1))
                win.put(4 * win.blocks, [xbase[m0], wbase[n0], yb, sb])
                win.blocks += 1
                placed.append((m0, n0, yb))
                i += 1
            if not placed:
                raise SFUnsupportedError(f"one block of a {Kp}-deep matmul does not fit in the 64K-word window")
            self._launch(prog, win)
            mem = self.chip.memory()
            for m0, n0, yb in placed:
                out[m0:m0 + tm, n0:n0 + N] = decode(mem[yb:yb + tm * N], bits).reshape(tm, N)
        return out[:M, :Nc]

    def _launch(self, prog, win) -> None:
        chip = self.chip
        chip.chip_reset()
        chip.load_program(prog)
        chip.load_data(win.words[:win.top].astype(np.uint16).tolist(), base=0)
        self.report.launches.append(chip.launch(N * win.blocks))


def library(mm: ChipMatmul | None = None, check: bool = False) -> dict:
    """The ops ``HardwareAtreides.attach_library`` takes: ``matmul_wide`` on ``mm``.

        HardwareAtreides.attach_library(library(check=True))
        sf.matmul(a, b, accumulate="wide", shift=s)       # a, b on device="atreides-hw": runs on the chip

    Batched operands run one 2-D matmul per batch element (``mm.report`` then
    holds all their launches).  ``check=True``
    also computes every call on the host MX model and raises
    ``SFDeviceError`` if a single output differs (a bring-up check; it
    doubles the work).  The chip has no event flags, so none are returned:
    the runtime counts outputs at +-max, or recomputes the events with
    ``sf.config(status="replay")``.
    """
    mm = mm or ChipMatmul()

    def matmul_wide(fmt, a, b, shift=0, mx=None):
        from .. import _linalg
        from .._errors import SFDeviceError
        if mx is not None and _linalg.mx_version(mx) != "v4.9.2":
            raise SFUnsupportedError(f"the chip model is v4.9.2; this launch asks for MX {mx}")
        a, b = np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64)
        batch = np.broadcast_shapes(a.shape[:-2], b.shape[:-2])
        shape = batch + (a.shape[-2], b.shape[-1])
        s = np.broadcast_to(np.asarray(shift, dtype=np.int64), shape)
        A, B = np.broadcast_to(a, batch + a.shape[-2:]), np.broadcast_to(b, batch + b.shape[-2:])
        out = np.zeros(shape, np.int64)
        total = Report()
        for idx in np.ndindex(*batch):
            out[idx] = mm(A[idx], B[idx], s[idx], fmt.bits)
            total.kernel = mm.report.kernel
            total.launches += mm.report.launches
        mm.report = total                              # every batch element's launches
        if check:
            ref, _ = _linalg.matmul_wide_codes(fmt, a, b, s, "v4.9.2")
            if not np.array_equal(out, ref):
                raise SFDeviceError(f"chip matmul_wide differs from the host MX model in "
                                    f"{int((out != ref).sum())} of {out.size} outputs")
        return out

    return {"matmul_wide": matmul_wide}
