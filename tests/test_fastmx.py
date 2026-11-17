"""The compiled MX accumulator (sf._fastmx) gives the reference's exact
integers: every width, both chips, random, edge and clipping codes, batches,
and the golden files.  The reference is _linalg.matmul_wide_acc with
FAST_MX = False (the bit-by-bit partial-product walk)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sf import _fastmx, _linalg
from sf._format import FORMATS

pytestmark = pytest.mark.skipif(not _fastmx.available(), reason="no C compiler: sf uses the reference")
ROOT = Path(__file__).resolve().parent.parent


def _both(fmt, a, b, mx):
    old = _linalg.FAST_MX
    try:
        _linalg.FAST_MX = True
        fast = _linalg.matmul_wide_acc(fmt, a, b, mx)
        _linalg.FAST_MX = False
        ref = _linalg.matmul_wide_acc(fmt, a, b, mx)
    finally:
        _linalg.FAST_MX = old
    return fast, ref


def test_the_package_versions_agree():
    import re
    import sf
    text = (ROOT / "pyproject.toml").read_text()
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == sf.__version__


def test_the_correction_table_is_the_dropped_columns():
    """C[x][y] equals what dropping the columns below tk removes from x*y."""
    for tk in (6, 9):
        C = _fastmx.correction_table(tk)
        rng = np.random.default_rng(tk)
        x = rng.integers(0, 1 << 15, 2000)
        y = rng.integers(0, 1 << 15, 2000)
        keep = ~((1 << tk) - 1)
        dropped = sum(((x >> i) & 1) * ((y << i) & keep) for i in range(15))
        assert np.array_equal(x * y - dropped, C[x & ((1 << tk) - 1), y & ((1 << tk) - 1)])


@pytest.mark.parametrize("mx", ["v4.9.2", "v4.8.1"])
@pytest.mark.parametrize("bits", [4, 5, 7, 8, 9, 11, 13, 16])
def test_random_codes_every_width(mx, bits):
    fmt = FORMATS[bits]
    M = fmt.max_code
    rng = np.random.default_rng(bits * 7 + len(mx))
    for (m, k, n) in ((1, 1, 1), (3, 7, 5), (4, 8, 4), (5, 9, 3), (6, 130, 7)):
        a = rng.integers(-M, M + 1, (m, k))
        b = rng.integers(-M, M + 1, (k, n))
        (fa, fc, ff), (ra, rc, rf) = _both(fmt, a, b, mx)
        assert ff == rf and np.array_equal(fa, ra) and np.array_equal(fc, rc)


@pytest.mark.parametrize("mx", ["v4.9.2", "v4.8.1"])
def test_clipping_and_edge_codes(mx):
    """All max codes over long K reach the limit; the clip mask and the
    saturated totals match, also when the sum comes back after a clip."""
    for bits in (8, 16):
        fmt = FORMATS[bits]
        M = fmt.max_code
        a = np.full((3, 2048), M)
        a[1, ::2] = -M
        a[2, :1024] = M
        a[2, 1024:] = -M                                  # clips, then comes back down
        b = np.full((2048, 2), M)
        b[:, 1] = -M
        (fa, fc, _), (ra, rc, _) = _both(fmt, a, b, mx)
        assert np.array_equal(fa, ra) and np.array_equal(fc, rc)
        assert fc.any() == (bits == 16 or mx == "v4.8.1")     # v4.9.2's SF8 block (+-131072) never clips here
        z = np.zeros((2, 5), dtype=np.int64)                  # zeros and +-1
        o = np.array([[1, -1, 0, 1, -1]] * 5)
        (fa, fc, _), (ra, rc, _) = _both(fmt, np.vstack([z, o[:2]]), o, mx)
        assert np.array_equal(fa, ra) and np.array_equal(fc, rc)


def test_batches_broadcast():
    fmt = FORMATS[16]
    rng = np.random.default_rng(3)
    a = rng.integers(-32767, 32768, (2, 3, 4, 20))
    b = rng.integers(-32767, 32768, (3, 20, 6))
    (fa, fc, _), (ra, rc, _) = _both(fmt, a, b, None)
    assert fa.shape == (2, 3, 4, 6) and np.array_equal(fa, ra) and np.array_equal(fc, rc)


def test_the_golden_dot_wide_cases_on_the_fast_path():
    """The emulator's 240 cases, on the compiled path (the default)."""
    mx = json.loads((ROOT / "tests" / "data" / "mx_golden.json").read_text())
    assert _linalg.FAST_MX
    for c in mx["cases"]:
        a, b = np.array(c["a"]), np.array(c["b"])
        q, _ = _linalg.matmul_wide_codes(FORMATS[c["bits"]], a, b, c["shift"], mx["target"])
        assert q.tolist() == c["out"]
