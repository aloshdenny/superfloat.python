"""The Atreides driver (sf.kernels._driver): kernel files, memory layout, packing,
launch splitting, the compiler call and its cache.

The end-to-end tests run Kartik's hand-written axpy program (tests/data/
atreides_kernel_axpy_v0.3.json, written by his axpy_sample.py; BLOCK 16) on a toy model of the 7 instructions it uses
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
SAMPLE = json.loads((HERE / "data" / "atreides_kernel_axpy_v0.3.json").read_text())
SAMPLE_02 = json.loads((HERE / "data" / "atreides_kernel_axpy_v0.2.json").read_text())
SAMPLE_01 = json.loads((HERE / "data" / "atreides_kernel_axpy_draft.json").read_text())
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
    assert k["format_version"] == "0.3-draft" and k["target"] == {"isa": "atreides-v4.9.2", "mx": "v4.9.2"}
    assert "cache_key" not in k and all("file" in e for e in k["source_map"])
    assert (k["launch"]["lanes_per_program"], k["launch"]["threads_per_program"]) == (16, 16)


def test_still_reads_the_earlier_drafts():
    for old, v in ((SAMPLE_01, "0.1-draft"), (SAMPLE_02, "0.2-draft")):
        k = D.read_kernel(copy.deepcopy(old))
        assert k["format_version"] == v and k["program"]["words"] == SAMPLE["program"]["words"]
    p = D.plan_launch(SAMPLE_02, _axpy_args(20))                   # no lanes fields: 8 per program
    assert p.grid == (3,) and p.lanes == 8


@pytest.mark.parametrize("change,msg", [
    (lambda k: k.update(format="sfir"), "format"),
    (lambda k: k.update(format_version="2.0"), "format_version"),
    (lambda k: k.update(format_version="0.10"), "format_version"),
    (lambda k: k["program"].update(words=[0] * 513), "at most 512"),
    (lambda k: k["program"].update(words=[0x10000]), "0xFFFF"),
    (lambda k: k["launch"].update(max_blocks_per_launch=32), "255 threads"),
    (lambda k: k["params"]["slots"].append({"offset": 1, "name": "z", "kind": "int"}), "share an offset"),
    (lambda k: k["params"]["slots"].append({"offset": 7, "name": "z", "kind": "int"}), "does not cover"),
    (lambda k: k["launch"].update(total_threads="N"), "not one of the kernel's arguments"),
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


def test_buffers_start_after_the_last_slot_even_with_gaps():
    k = copy.deepcopy(SAMPLE)
    k["params"]["slots"].append({"offset": 7, "name": "m", "kind": "int"})   # offsets 5 and 6 unused
    del k["memory"]["reserved_words"]
    p = D.plan_launch(k, _axpy_args(8) | {"m": 3})
    assert p.memory[7] == 3 and min(b.base for b in p.buffers.values()) == 8


def _v03(lanes=16, tpp=16):
    k = copy.deepcopy(SAMPLE)
    k["format_version"] = "0.3-draft"
    k["launch"].update(lanes_per_program=lanes, threads_per_program=tpp)
    return k


def test_reads_format_0_3_and_refuses_unknown_versions():
    assert D.read_kernel(_v03())["format_version"] == "0.3-draft"
    with pytest.raises(sf.SFCompileError, match="format_version"):
        D.read_kernel(_v03() | {"format_version": "0.4-draft"})


def test_lanes_and_threads_per_program():
    p = D.plan_launch(_v03(16, 16), _axpy_args(20))               # BLOCK 16 on 16 threads
    assert p.grid == (2,) and p.lanes == 16 and sum(L.threads for L in p.launches) == 20
    p = D.plan_launch(_v03(16, 8), _axpy_args(20))                # 2 lanes per thread: 10 threads
    assert p.grid == (2,) and sum(L.threads for L in p.launches) == 10
    p = D.plan_launch(_v03(16, 16), _axpy_args(600))              # launches hold whole programs
    assert [L.threads for L in p.launches] == [240, 240, 120]
    assert [L.block_offset for L in p.launches] == [0, 30, 60]
    with pytest.raises(sf.SFCompileError, match="whole number"):
        D.read_kernel(_v03(16, 6))


@pytest.mark.parametrize("lanes,tpp", [(6, 6), (12, 12), (4, 4), (16, 8)])
def test_launches_end_on_block_boundaries(lanes, tpp):
    """The chip numbers threads from 0 in every block, so a launch that ended mid-block would
    start the next one on the wrong element.  Runs the real axpy on the toy model to check."""
    n = 600
    args = _axpy_args(n, seed=lanes)
    plan = D.plan_launch(_v03(lanes, tpp), args)
    assert all(L.threads % 8 == 0 for L in plan.launches[:-1])
    if lanes == tpp:                                   # Kartik's axpy maps one lane to one thread
        got = D.read_back(plan, run_toy(plan, plan.memory))["out_ptr"]
        with sf.errstate(all="ignore"):
            assert np.array_equal(got, sf.fma(args["a"], args["x_ptr"], args["y_ptr"]).codes)


def test_launch_rule_is_checked_once_and_clearly():
    k = copy.deepcopy(SAMPLE)
    del k["launch"]["lanes_per_program"]
    with pytest.raises(sf.SFCompileError, match="without lanes_per_program"):
        D.read_kernel(k)
    k = _gemm_2d()
    k["launch"]["grid_axes"] = "2"                     # a string still means 2-D
    p = D.plan_launch(k, {"a_ptr": sf.zeros(8, "sf16"), "c_ptr": sf.zeros(8, "sf16")},
                      {"M": 16, "BM": 8, "N": 8, "BN": 8})
    assert p.grid == (2, 1)
    for cx, msg in [({"M": 16, "BM": 0, "N": 8, "BN": 8}, "divides by zero"),
                    ({"M": (16,), "BM": 8, "N": 8, "BN": 8}, "not an int")]:
        with pytest.raises(sf.SFLaunchError, match=msg):
            D.plan_launch(_gemm_2d(), {"a_ptr": sf.zeros(8, "sf16"), "c_ptr": sf.zeros(8, "sf16")}, cx)


def test_2d_block_offset_is_the_chip_block():
    k = _gemm_2d()
    k["launch"].update(lanes_per_program=16, threads_per_program=16)
    k["params"]["slots"][0] = {"offset": 0, "name": "__pid0", "kind": "axis_index"}
    k["params"]["slots"].append({"offset": 4, "name": "__block_offset", "kind": "block_offset"})
    k["memory"]["reserved_words"] = [0, 4]
    p = D.plan_launch(k, {"a_ptr": sf.zeros(8, "sf16"), "c_ptr": sf.zeros(8, "sf16")},
                      {"M": 40 * 16, "BM": 16, "N": 8, "BN": 8})
    second = p.launches[1]                             # 15 programs of 16 in the first launch
    assert second.block_offset == 30 and second.writes == {4: 30, 0: 15, 3: 0}


def test_grid_must_match_the_programs():
    p = D.plan_launch(_v03(16, 16), _axpy_args(20))
    assert D.check_grid(p, (2,)) == 2 and D.check_grid(p, (2, 1, 1)) == 2
    for g in [(1,), (3,), (2, 2)]:
        with pytest.raises(sf.SFLaunchError, match="does not match"):
            D.check_grid(p, g)


def _gemm_2d():
    k = copy.deepcopy(SAMPLE)
    k["format_version"] = "0.3-draft"
    k["params"]["slots"] = [{"offset": 0, "name": "__pid0", "kind": "block_offset"},
                            {"offset": 1, "name": "a_ptr", "kind": "ptr", "dtype": "sf16", "access": "read"},
                            {"offset": 2, "name": "c_ptr", "kind": "ptr", "dtype": "sf16", "access": "write"},
                            {"offset": 3, "name": "__pid1", "kind": "axis_index"}]
    k["params"]["not_in_block"] = []
    k["memory"]["reserved_words"] = [0, 3]
    k["launch"] = {"threads_per_block": 8, "max_blocks_per_launch": 31, "grid_axes": 2,
                   "lanes_per_program": 8, "threads_per_program": 8,
                   "axes": [{"size": "cdiv(M, BM)", "param_slot": 0}, {"size": "cdiv(N, BN)", "param_slot": 3}]}
    return k


def test_2d_grid_runs_as_row_launches():
    a = sf.zeros(8, "sf16")
    args = {"a_ptr": a, "c_ptr": sf.zeros(8, "sf16")}
    p = D.plan_launch(_gemm_2d(), args, {"M": 40 * 8, "BM": 8, "N": 24, "BN": 8})
    assert p.grid == (40, 3)                                      # 40 programs on axis 0, 3 on axis 1
    assert [(L.threads, L.writes) for L in p.launches[:3]] == [
        (31 * 8, {0: 0, 3: 0}), (9 * 8, {0: 31, 3: 0}), (31 * 8, {0: 0, 3: 1})]
    assert len(p.launches) == 6
    with pytest.raises(sf.SFLaunchError, match="cdiv"):
        D.plan_launch(_gemm_2d(), args, {"M": 8})                 # BM, N, BN unknown


def test_2d_launch_needs_real_slots():
    k = _gemm_2d()
    k["launch"]["axes"][1]["param_slot"] = 9
    with pytest.raises(sf.SFCompileError, match="not a parameter slot"):
        D.read_kernel(k)


def test_launches_split_at_31_blocks():
    p = D.plan_launch(SAMPLE_02, _axpy_args(600))                   # 8 lanes per program: 31 blocks
    assert [(L.threads, L.block_offset) for L in p.launches] == [(248, 0), (248, 31), (104, 62)]
    assert [L.writes for L in p.launches] == [{0: 0}, {0: 31}, {0: 62}]
    p = D.plan_launch(SAMPLE, _axpy_args(600))                      # 16 lanes: 15 whole programs
    assert [(L.threads, L.block_offset) for L in p.launches] == [(240, 0), (240, 30), (120, 60)]


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
    k["params"]["slots"] += [{"offset": 5, "name": "s_ptr", "kind": "ptr", "dtype": "int16",
                              "encoding": "twos_complement", "access": "read"},
                             {"offset": 6, "name": "m", "kind": "int"}]
    k["memory"]["reserved_words"] = [0, 6]
    args = _axpy_args(8) | {"s_ptr": np.array([-32, -1, 0, 31]), "m": -5}
    p = D.plan_launch(k, args)
    s = p.buffers["s_ptr"]
    assert [p.memory[s.base + i] for i in range(4)] == [0xFFE0, 0xFFFF, 0, 31]
    assert p.memory[6] == 0xFFFB
    assert s.dtype == "int16"
    with pytest.raises(sf.SFLaunchError, match="16-bit"):
        D.plan_launch(k, args | {"m": 40000})
    k["params"]["slots"][5]["dtype"] = "int"                         # Kartik: the file names the real width
    with pytest.raises(sf.SFLaunchError, match="wants int, got int16"):
        D.plan_launch(k, args)
    k["params"]["slots"][5].update(dtype="int16", encoding="sign_magnitude")
    with pytest.raises(sf.SFLaunchError, match="twos_complement"):
        D.plan_launch(k, args)


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


def test_cache_key_covers_the_ir_the_version_and_the_exact_arguments():
    base = D.cache_key("ir", "v1", ["--target=t"])
    assert len({base, D.cache_key("ir2", "v1", ["--target=t"]), D.cache_key("ir", "v2", ["--target=t"]),
                D.cache_key("ir", "v1", ["--target=t2"]), D.cache_key("ir", "v1", ["--target=t", "-O1"])}) == 5


def test_compiler_gets_target_equals_and_the_options(cache, tmp_path):
    D.compile_kernel(AXPY_IR, options=["-O1"], compiler=FAKE)
    assert (tmp_path / "count").read_text().split() == ["axpy", "--target=atreides-v4.9.2", "-O1"]


def test_dirty_compiler_builds_are_not_cached(cache, monkeypatch):
    monkeypatch.setattr(D, "_versions", {})
    monkeypatch.setenv("FAKE_COMPILE_VERSION", "superfloat-compile 0.1.0 (3f2a9c1-dirty)")
    D.compile_kernel(AXPY_IR, compiler=FAKE)
    D.compile_kernel(AXPY_IR, compiler=FAKE)
    assert cache() == 2                                          # compiled both times
    assert D.is_dirty("superfloat-compile 0.1.0 (3f2a9c1-dirty)")
    assert not D.is_dirty("superfloat-compile 0.1.0 (3f2a9c1)")


def test_debug_options_are_refused(cache):
    with pytest.raises(sf.SFUnsupportedError, match="debug options"):
        D.compile_kernel(AXPY_IR, options=["--emit=mlir"], compiler=FAKE)


def test_cache_key_keeps_the_order_of_options():
    assert (D.cache_key("ir", "v1", ["--unroll", "4", "--regs", "8"])
            != D.cache_key("ir", "v1", ["--unroll", "8", "--regs", "4"]))


def test_compiler_failure_is_a_clean_error(cache, monkeypatch):
    monkeypatch.setenv("FAKE_COMPILE_FAIL", "1")
    with pytest.raises(sf.SFCompileError, match="assertion failed"):
        D.compile_kernel(AXPY_IR, compiler=FAKE)
    with pytest.raises(sf.SFCompileError, match="not SF IR|failed"):
        D.compile_kernel('{"format": "other"}', compiler=FAKE)


def test_kernel_for_another_target_is_refused(cache, monkeypatch):
    monkeypatch.setenv("FAKE_COMPILE_WRONG_TARGET", "atreides-v4.8.1")
    with pytest.raises(sf.SFCompileError, match="asked for target 'atreides-v4.9.2'"):
        D.compile_kernel(AXPY_IR, compiler=FAKE)


def test_missing_compiler():
    with pytest.raises(sf.SFUnsupportedError, match="not installed"):
        D.compile_kernel(AXPY_IR, compiler="no-such-superfloat-compile")
