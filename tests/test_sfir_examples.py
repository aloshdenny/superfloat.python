"""The SF IR example pack in examples/sfir/ matches what the code builds today.

If this fails after a change to the DSL or the IR, regenerate the pack with
``python tools/export_sfir_examples.py`` and review the diff.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import export_sfir_examples as E  # noqa: E402

from sf.kernels._ir import KernelIR  # noqa: E402

PACK = E.build()


@pytest.mark.parametrize("name", sorted(PACK))
def test_pack_files_are_current(name):
    for fname, text in PACK[name].items():
        assert (E.OUT_DIR / name / fname).read_text() == text, f"{name}/{fname} is stale"


@pytest.mark.parametrize("name", sorted(PACK))
def test_pack_ir_reads_back(name):
    text = (E.OUT_DIR / name / "kernel.sfir.json").read_text()
    assert KernelIR.from_json(text).to_json() == text


def test_pack_has_the_five_kernels_and_a_shift_table():
    assert sorted(PACK) == ["add", "axpy", "dot_chain", "gemm_wide", "gemv_wide"]
    ins = json.loads(PACK["gemm_wide"]["inputs.json"])
    table = next(a for a in ins["args"] if a["name"] == "s_ptr")
    assert table["type"] == "ptr<int>" and len(table["values"]) == 64
    ir = json.loads(PACK["gemm_wide"]["kernel.sfir.json"])
    dw = next(o for o in ir["ops"] if o["op"] == "dot_wide")
    assert len(dw["args"]) == 3 and dw["attrs"]["shift"] is None
