"""Shared fixtures and helpers.

* ``EXHAUSTIVE_WIDTHS`` (4..10) are tested over every operand pair; sf9 and
  sf10 are marked ``slow`` so they can be skipped while iterating with
  ``-m "not slow"``.  They still run on every commit.
* ``SAMPLED_WIDTHS`` (11..16) get boundary vectors, hypothesis property
  tests and random sampling.
"""
from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))      # make ``oracle`` importable

import sf                                           # noqa: E402
from oracle import Fmt                              # noqa: E402

EXHAUSTIVE_WIDTHS = list(range(4, 11))
SAMPLED_WIDTHS = list(range(11, 17))
ALL_WIDTHS = EXHAUSTIVE_WIDTHS + SAMPLED_WIDTHS

SEED = int(os.environ.get("SF_TEST_SEED", "20261017"))
N_RANDOM = int(os.environ.get("SF_TEST_RANDOM_PAIRS", "20000"))


EVENT_ERROR = {
    None: None,
    "overflow": sf.SFOverflowError,
    "underflow": sf.SFUnderflowError,
    "divide": sf.SFZeroDivisionError,
    "invalid": sf.SFInvalidError,
}

_RAISE_ALL = sf.errstate(all="raise")
_IGNORE_ALL = sf.errstate(all="ignore")


def run(fn, *args):
    """Call ``fn(*args)`` with every event set to raise.

    Returns ``(hardware_value, error_type_or_None)``: the type of the SF error
    raised, if any, and the value the op produces with events ignored (the
    hardware value).  Cheaper than pytest.raises in exhaustive loops.
    """
    with _RAISE_ALL:
        try:
            return fn(*args), None
        except sf.SFError as e:
            raised = type(e)
    with _IGNORE_ALL:
        return fn(*args), raised


def width_params(widths):
    return [pytest.param(w, marks=pytest.mark.slow) if w >= 9 else w for w in widths]


def dtype_of(bits: int):
    return sf.dtypes[bits - 4]


def pairs(bits: int):
    """Every (code_a, code_b) pair for exhaustive widths, a random sample
    of ``N_RANDOM`` pairs (plus the boundary corners) otherwise."""
    m = sf.info(dtype_of(bits)).max_code
    if bits in EXHAUSTIVE_WIDTHS:
        rng = range(-m, m + 1)
        return ((a, b) for a in rng for b in rng)
    rnd = random.Random(SEED * 1000 + bits)
    corners = [-m, -m + 1, -1, 0, 1, m - 1, m]
    out = [(a, b) for a in corners for b in corners]
    out += [(rnd.randint(-m, m), rnd.randint(-m, m)) for _ in range(N_RANDOM)]
    # small magnitudes exercise underflow in mul and overflow in div
    out += [(rnd.randint(-300, 300), rnd.randint(-300, 300)) for _ in range(N_RANDOM // 4)]
    return out


def singles(bits: int):
    m = sf.info(dtype_of(bits)).max_code
    if bits in EXHAUSTIVE_WIDTHS or bits <= 13:
        return range(-m, m + 1)
    rnd = random.Random(SEED * 1000 + bits)
    return [-m, -m + 1, -1, 0, 1, m - 1, m] + [rnd.randint(-m, m) for _ in range(N_RANDOM)]


@pytest.fixture(params=width_params(ALL_WIDTHS), ids=lambda b: f"sf{b}")
def width(request):
    return request.param


@pytest.fixture
def dt(width):
    return dtype_of(width)


@pytest.fixture
def oracle(width):
    return Fmt(width)


@pytest.fixture(autouse=True)
def _default_errstate():
    """Every test starts from the default policy and leaves it unchanged."""
    before = sf.geterr()
    yield
    sf.seterr(**before)
