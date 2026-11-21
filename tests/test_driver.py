"""The Atreides driver (sf.kernels._driver): kernel files, memory layout, packing,
launch splitting, the compiler call and its cache.

The end-to-end tests run Kartik's hand-written axpy program (tests/data/
atreides_kernel_axpy_draft.json) on a toy model of the 7 instructions it uses
(CMP, CONST, LDR, ADD, FMA, STR, RET, as compiler_v4.9.2.md sections 2 and 3
define them), on the memory the driver lays out, and compare with the host.
The toy model is for these tests only; it is not the chip.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import sf
from sf.kernels import _driver as D

from atreides_toy import run_toy  # noqa: E402

HERE = Path(__file__).resolve().parent
SAMPLE = json.loads((HERE / "data" / "atreides_kernel_axpy_draft.json").read_text())
FAKE = [sys.executable, str(HERE / "fake_superfloat_compile.py")]
AXPY_IR = (HERE.parent / "examples" / "sfir" / "axpy" / "kernel.sfir.json").read_text()


def _axpy_args(n, seed=0):
    rng = np.random.default_rng(seed)
    with sf.errstate(all="ignore"):
        x = sf.array(rng.uniform(-0.9, 0.9, n), dtype=sf.sf16)
        y = sf.array(rng.uniform(-0.9, 0.9, n), dtype=sf.sf16)
        a = sf.sf16(float(rng.uniform(-0.9, 0.9)))
    return {"a": a, "x_ptr": x, "y_ptr": y, "out_ptr": sf.zeros(n, "sf16"), "n": n}


# --- reading kernel files -----------------------------------------------------------------------

def test_reads_the_sample():
    k = D.read_kernel(copy.deepcopy(SAMPLE))
    assert len(k["program"]["words"]) == 25 and k["kernel"] == "axpy"


@pytest.mark.parametrize("change,msg", [
    (lambda k: k.update(format="sfir"), "format"),
    (lambda k: k.update(format_version="2.0"), "format_version"),
    (lambda k: k["program"].update(words=[0] * 513), "at most 512"),
    (lambda k: k["program"].update(words=[0x10000]), "0xFFFF"),
    (lambda k: k["launch"].update(max_blocks_per_launch=32), "255 threads"),
    (lambda k: k["params"]["slots"].append({"offset": 1, "name": "z", "kind": "int"}), "share an offset"),
])
def test_rejects_bad_kernel_files(change, msg):
    k = copy.deepcopy(SAMPLE)
    change(k)
    with pytest.raises(sf.SFCompileError, match=msg):
        D.read_kernel(k)


# --- layout, packing, launches ------------------------------------------------------------------

def test_layout_of_axpy():
    args = _axpy_args(20)
    p = D.plan_launch(SAMPLE, args)
    assert p.program == SAMPLE["program"]["words"]
    x, y, out = p.buffers["x_ptr"], p.buffers["y_ptr"], p.buffers["out_ptr"]
    assert (x.base, y.base, out.base) == (5, 25, 45)               # right after the 5 reserved words
    assert (p.memory[1], p.memory[2], p.memory[3]) == (5, 25, 45)
    assert p.memory[4] == sf.sf16(args["a"]).bits                  # a, sign-magnitude
    assert [p.memory[5 + i] for i in range(20)] == [sf.from_codes(np.array([c]), sf.sf16)[0].bits
                                                     for c in args["x_ptr"].codes]
    assert not any(45 <= addr < 65 for addr in p.memory)          # the output is not uploaded
    assert [(L.threads, L.block_offset, L.writes) for L in p.launches] == [(20, 0, {0: 0})]


def test_launches_split_at_31_blocks():
    p = D.plan_launch(SAMPLE, _axpy_args(600))
    assert [(L.threads, L.block_offset) for L in p.launches] == [(248, 0), (248, 31), (104, 62)]
    assert [L.writes for L in p.launches] == [{0: 0}, {0: 31}, {0: 62}]


@pytest.mark.parametrize("n", [1, 8, 20, 248, 249, 600])
def test_axpy_end_to_end_on_the_toy_model(n):
    args = _axpy_args(n, seed=n)
    plan = D.plan_launch(SAMPLE, args)
    got = D.read_back(plan, run_toy(plan, plan.memory))["out_ptr"]
    with sf.errstate(all="ignore"):
        want = sf.fma(args["a"], args["x_ptr"], args["y_ptr"]).codes
    assert np.array_equal(got, want)


@pytest.mark.parametrize("bits", [8, 12, 16])
def test_sf_words_round_trip_left_aligned(bits):
    M = sf._format.FORMATS[bits].max_code
    for code in range(-M, M + 1, max(1, M // 300)):
        w = D._sf_word(code, bits)
        assert 0 <= w < 1 << 16 and w & ((1 << (16 - bits)) - 1) == 0   # low bits empty
        assert D._sf_code(w, bits) == code
    assert D._sf_word(-1, bits) >> 15 == 1                         # the sign is bit 15


def test_int_tables_and_ints_are_16_bit_twos_complement():
    k = copy.deepcopy(SAMPLE)
    k["params"]["slots"] += [{"offset": 5, "name": "s_ptr", "kind": "ptr", "access": "read"},
                             {"offset": 6, "name": "m", "kind": "int"}]
    k["memory"]["reserved_words"] = [0, 6]
    args = _axpy_args(8) | {"s_ptr": np.array([-32, -1, 0, 31]), "m": -5}
    p = D.plan_launch(k, args)
    s = p.buffers["s_ptr"]
    assert [p.memory[s.base + i] for i in range(4)] == [0xFFE0, 0xFFFF, 0, 31]
    assert p.memory[6] == 0xFFFB
    with pytest.raises(sf.SFLaunchError, match="16-bit"):
        D.plan_launch(k, args | {"m": 40000})


def test_memory_limits():
    with pytest.raises(sf.SFLaunchError, match="does not fit below 0xffc0"):
        D.plan_launch(SAMPLE, _axpy_args(22000))                    # 3 x 22000 words > 0xFFC0
    k = copy.deepcopy(SAMPLE)
    k["memory"]["pointers_compared_with_cmp"] = True
    p = D.plan_launch(k, _axpy_args(12000))
    for b in p.buffers.values():                                    # no buffer straddles 0x8000
        assert not b.base < 0x8000 <= b.base + b.length - 1
    assert p.buffers["out_ptr"].base == 0x8000


def test_missing_and_wrong_arguments():
    args = _axpy_args(8)
    del args["y_ptr"]
    with pytest.raises(sf.SFLaunchError, match="missing arguments"):
        D.plan_launch(SAMPLE, args)
    with pytest.raises(sf.SFLaunchError, match="SFScalar"):
        D.plan_launch(SAMPLE, _axpy_args(8) | {"a": 3})
    with pytest.raises(sf.SFLaunchError, match="at least 1"):
        D.plan_launch(SAMPLE, _axpy_args(8) | {"n": 0})


# --- the compiler call and its cache ------------------------------------------------------------

@pytest.fixture
def cache(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "_CACHE_DIR", str(tmp_path / "cache"))
    count = tmp_path / "count"
    monkeypatch.setenv("FAKE_COMPILE_COUNT", str(count))
    return lambda: len(count.read_text().splitlines()) if count.exists() else 0


def test_compile_runs_once_then_hits_the_cache(cache):
    k1 = D.compile_kernel(AXPY_IR, compiler=FAKE)
    k2 = D.compile_kernel(AXPY_IR, compiler=FAKE)
    assert k1 == k2 and k1["target"]["isa"] == D.DEFAULT_TARGET
    assert cache() == 1
    D.compile_kernel(AXPY_IR, target="atreides-v4.8.1", compiler=FAKE)   # another target: another key
    assert cache() == 2
    D.compile_kernel(AXPY_IR.replace('"line": 4', '"line": 5', 1), compiler=FAKE)   # the IR changed
    assert cache() == 3


def test_cache_key_covers_all_four_parts():
    base = D.cache_key("ir", "v1", "t", ())
    assert len({base, D.cache_key("ir2", "v1", "t", ()), D.cache_key("ir", "v2", "t", ()),
                D.cache_key("ir", "v1", "t2", ()), D.cache_key("ir", "v1", "t", ("-O1",))}) == 5


def test_compiler_failure_is_a_clean_error(cache, monkeypatch):
    monkeypatch.setenv("FAKE_COMPILE_FAIL", "1")
    with pytest.raises(sf.SFCompileError, match="assertion failed"):
        D.compile_kernel(AXPY_IR, compiler=FAKE)
    with pytest.raises(sf.SFCompileError, match="not SF IR|failed"):
        D.compile_kernel('{"format": "other"}', compiler=FAKE)


def test_missing_compiler():
    with pytest.raises(sf.SFUnsupportedError, match="not installed"):
        D.compile_kernel(AXPY_IR, compiler="no-such-superfloat-compile")
