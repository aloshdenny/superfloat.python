"""Write the SF IR example pack for the compiler: examples/sfir/<kernel>/.

Each kernel folder holds three files:

* ``kernel.sfir.json``: the traced kernel (``KernelIR.to_json``).
* ``inputs.json``: the launch: grid, constexpr values and every argument.
  SF data are integer codes (signed: code k means k / 2^(N-1)), never floats.
  The driver turns them into sign-magnitude words for the chip.
* ``expected.json``: the output buffers as codes, from the simulator, checked
  here against the host reference.  ``target`` names the chip whose MX engine
  the wide kernels model (v4.9.2).

The kernels: axpy, an elementwise add, a dot chain (``sf.dot``, chain
accumulation), a wide gemv, a wide gemm (``sf.dot`` on the MX engine, one
POP shift per output from an int table) and a k-tiled matmul.  The wide shifts are the tightest
that do not saturate, so each output uses its full range.

Run from the repository root:  python tools/export_sfir_examples.py
``tests/test_sfir_examples.py`` rebuilds the pack in memory and checks the
files match, so they can never go stale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sf  # noqa: E402
from sf import _linalg  # noqa: E402
from sf._format import FORMATS  # noqa: E402
from sf.kernels._runtime import Kernel  # noqa: E402

DEV = "atreides-sim"
TARGET = "v4.9.2"
F16 = FORMATS[16]
OUT_DIR = ROOT / "examples" / "sfir"


# --- the kernels --------------------------------------------------------------------------------

@sf.jit
def axpy(a, x_ptr, y_ptr, out_ptr, n, BLOCK: sf.constexpr):
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    mask = offs < n
    x = sf.load(x_ptr + offs, mask=mask)
    y = sf.load(y_ptr + offs, mask=mask)
    sf.store(out_ptr + offs, sf.fma(a, x, y), mask=mask)


@sf.jit
def add(x_ptr, y_ptr, out_ptr, n, BLOCK: sf.constexpr):
    offs = sf.program_id(0) * BLOCK + sf.arange(0, BLOCK)
    mask = offs < n
    sf.store(out_ptr + offs, sf.load(x_ptr + offs, mask=mask) + sf.load(y_ptr + offs, mask=mask), mask=mask)


@sf.jit
def dot_chain(a_ptr, b_ptr, c_ptr, M: sf.constexpr, K: sf.constexpr, N: sf.constexpr):
    i = sf.arange(0, M)[:, None]
    k = sf.arange(0, K)
    j = sf.arange(0, N)[None, :]
    x = sf.load(a_ptr + i * K + k[None, :])
    w = sf.load(b_ptr + k[:, None] * N + j)
    sf.store(c_ptr + i * N + j, sf.dot(x, w, accumulate="chain"))


@sf.jit
def matmul_wide(a_ptr, b_ptr, s_ptr, c_ptr, M: sf.constexpr, K: sf.constexpr, N: sf.constexpr):
    i = sf.arange(0, M)[:, None]
    k = sf.arange(0, K)
    j = sf.arange(0, N)[None, :]
    x = sf.load(a_ptr + i * K + k[None, :])
    w = sf.load(b_ptr + k[:, None] * N + j)
    s = sf.load(s_ptr + i * N + j)                     # one POP shift per output
    sf.store(c_ptr + i * N + j, sf.dot(x, w, accumulate="wide", shift=s))


# dot_k_tiles: its k loop is one ``for`` op (``sf.fori_loop``, SF IR format_version 2), so the
# file holds the loop once.  Last in the file and in the pack, with its own random numbers, so
# the older examples keep their exact bytes.
@sf.jit
def dot_k_tiles(a_ptr, b_ptr, c_ptr, M: sf.constexpr, K: sf.constexpr, N: sf.constexpr, BK: sf.constexpr):
    i = sf.arange(0, M)[:, None]
    kk = sf.arange(0, BK)
    j = sf.arange(0, N)[None, :]

    def k_tile(t, acc):                                # one trip per BK columns of a: the loop is in the file once
        k = t * BK + kk
        x = sf.load(a_ptr + i * K + k[None, :])
        w = sf.load(b_ptr + k[:, None] * N + j)
        return acc + sf.dot(x, w, accumulate="chain")

    acc = sf.fori_loop(0, K // BK, k_tile, sf.zeros((M, N)))
    sf.store(c_ptr + i * N + j, acc)


# --- inputs and reference -----------------------------------------------------------------------

def _codes(rng, shape, lo=-0.9, hi=0.9):
    with sf.errstate(all="ignore"):
        return sf.array(rng.uniform(lo, hi, shape), dtype=sf.sf16).codes.astype(np.int64)


def _tight_shifts(a, b):
    """Per output, the largest shift in -32..31 whose POP does not saturate."""
    acc, _, F = _linalg.matmul_wide_acc(F16, a, b)
    best = np.zeros(acc.shape, dtype=np.int64)
    for s in range(-32, 32):
        _, over, _ = _linalg._mx_pop(F16, acc, s, F)
        best = np.where(~over & (acc != 0), s, best)
    return best


def _cases():
    rng = np.random.default_rng(20261121)
    n = 20                                              # 3 blocks of 8, the last one partial
    x, y = _codes(rng, n), _codes(rng, n)
    a = int(_codes(rng, 1)[0])
    with sf.errstate(all="ignore"):
        fx, fy = sf.from_codes(x, sf.sf16), sf.from_codes(y, sf.sf16)
        ref_axpy = sf.fma(sf.from_codes(np.array([a]), sf.sf16)[0], fx, fy).codes
        ref_add = (fx + fy).codes
    yield dict(kernel=axpy, name="axpy", grid=(3,), constexpr={"BLOCK": 8},
               args=[("a", "sf", a), ("x_ptr", "ptr", x), ("y_ptr", "ptr", y),
                     ("out_ptr", "ptr", np.zeros(n, np.int64)), ("n", "int", n)],
               outputs=["out_ptr"], reference={"out_ptr": ref_axpy})
    yield dict(kernel=add, name="add", grid=(3,), constexpr={"BLOCK": 8},
               args=[("x_ptr", "ptr", x), ("y_ptr", "ptr", y),
                     ("out_ptr", "ptr", np.zeros(n, np.int64)), ("n", "int", n)],
               outputs=["out_ptr"], reference={"out_ptr": ref_add})

    M, K, N = 8, 16, 8
    A, B = _codes(rng, (M, K), -0.3, 0.3), _codes(rng, (K, N), -0.3, 0.3)
    with sf.errstate(all="ignore"):
        ref = (sf.from_codes(A, sf.sf16) @ sf.from_codes(B, sf.sf16)).codes
    yield dict(kernel=dot_chain, name="dot_chain", grid=(1,), constexpr={"M": M, "K": K, "N": N},
               args=[("a_ptr", "ptr", A.reshape(-1)), ("b_ptr", "ptr", B.reshape(-1)),
                     ("c_ptr", "ptr", np.zeros(M * N, np.int64))],
               outputs=["c_ptr"], reference={"c_ptr": ref.reshape(-1)})

    for name, M in (("gemv_wide", 1), ("gemm_wide", 8)):
        K, N = 64, 8
        A, B = _codes(rng, (M, K)), _codes(rng, (K, N))
        S = _tight_shifts(A, B)
        ref, _ = _linalg.matmul_wide_codes(F16, A, B, S, mx=TARGET)
        yield dict(kernel=matmul_wide, name=name, grid=(1,), constexpr={"M": M, "K": K, "N": N},
                   args=[("a_ptr", "ptr", A.reshape(-1)), ("b_ptr", "ptr", B.reshape(-1)),
                         ("s_ptr", "table", S.reshape(-1)), ("c_ptr", "ptr", np.zeros(M * N, np.int64))],
                   outputs=["c_ptr"], reference={"c_ptr": ref.reshape(-1)})

    M, K, N, BK = 8, 32, 8, 8
    rng = np.random.default_rng(20261123)
    A, B = _codes(rng, (M, K), -0.3, 0.3), _codes(rng, (K, N), -0.3, 0.3)
    with sf.errstate(all="ignore"):
        ref = sf.zeros((M, N), "sf16")
        for t in range(K // BK):                       # the host reference: tile by tile, in order
            ref = ref + (sf.from_codes(A[:, t * BK:(t + 1) * BK], sf.sf16) @ sf.from_codes(B[t * BK:(t + 1) * BK], sf.sf16))
    yield dict(kernel=dot_k_tiles, name="dot_k_tiles", grid=(1,), constexpr={"M": M, "K": K, "N": N, "BK": BK},
               args=[("a_ptr", "ptr", A.reshape(-1)), ("b_ptr", "ptr", B.reshape(-1)),
                     ("c_ptr", "ptr", np.zeros(M * N, np.int64))],
               outputs=["c_ptr"], reference={"c_ptr": ref.codes.reshape(-1)})

# --- build -------------------------------------------------------------------------------------

def _run(case):
    """Launch on the simulator; return (ir, {output name: codes})."""
    k: Kernel = case["kernel"]
    k.cache.clear()
    live, bufs = [], {}
    for name, kind, v in case["args"]:
        if kind == "ptr":
            bufs[name] = sf.from_codes(np.asarray(v), sf.sf16, device=DEV)
            live.append(bufs[name])
        elif kind == "table":
            live.append(np.asarray(v, dtype=np.int64).copy())
        elif kind == "sf":
            live.append(sf.from_codes(np.array([v]), sf.sf16)[0])
        else:
            live.append(int(v))
    with sf.config(kernel_backend="interp"), sf.errstate(all="ignore"):
        k[case["grid"]](*live, **case["constexpr"])
    (ir,) = k.cache.values()
    return ir, {o: bufs[o].codes.astype(np.int64) for o in case["outputs"]}


def _inputs_json(case) -> dict:
    args = []
    for name, kind, v in case["args"]:
        if kind == "ptr":
            args.append({"name": name, "kind": "ptr", "type": "ptr<sf16>",
                         "codes": [int(c) for c in np.asarray(v).reshape(-1)]})
        elif kind == "table":
            args.append({"name": name, "kind": "ptr", "type": "ptr<int>",
                         "values": [int(c) for c in np.asarray(v).reshape(-1)]})
        elif kind == "sf":
            args.append({"name": name, "kind": "sf", "type": "sf16", "code": int(v)})
        else:
            args.append({"name": name, "kind": "int", "type": "int", "value": int(v)})
    return {"kernel": case["name"], "grid": list(case["grid"]), "constexpr": case["constexpr"],
            "args": args, "target": TARGET}


def build() -> dict[str, dict[str, str]]:
    """{kernel folder: {file name: text}} for the whole pack, checked against the host."""
    pack = {}
    for case in _cases():
        ir, outs = _run(case)
        for o in case["outputs"]:
            if not np.array_equal(outs[o], case["reference"][o]):
                raise AssertionError(f"{case['name']}: simulator and host reference disagree on {o}")
        expected = {"kernel": case["name"], "target": TARGET,
                    "outputs": {o: {"type": "sf16", "codes": [int(c) for c in outs[o]]}
                                for o in case["outputs"]}}
        dump = lambda d: json.dumps(d, sort_keys=True, indent=1) + "\n"   # noqa: E731
        pack[case["name"]] = {"kernel.sfir.json": ir.to_json(),
                              "inputs.json": dump(_inputs_json(case)),
                              "expected.json": dump(expected)}
    return pack


def main() -> None:
    for name, files in build().items():
        d = OUT_DIR / name
        d.mkdir(parents=True, exist_ok=True)
        for fname, text in files.items():
            (d / fname).write_text(text)
        print(f"wrote {d.relative_to(ROOT)}/ ({', '.join(files)})")


if __name__ == "__main__":
    main()
