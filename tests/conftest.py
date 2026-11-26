"""Shared fixtures and helpers.

* ``EXHAUSTIVE_WIDTHS`` (4..10) are tested over every operand pair; sf9 and
  sf10 are marked ``slow`` so they can be skipped while iterating with
  ``-m "not slow"``.  They still run on every commit.
* ``SAMPLED_WIDTHS`` (11..16) get boundary vectors, hypothesis property
  tests and random sampling for operand *pairs*; single-value tests are
  exhaustive on every width.
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

if os.environ.get("SF_IR_ROUNDTRIP") == "1":
    # Send every traced kernel through SF IR JSON and back before it runs, so every
    # bit-exact kernel test in the suite also checks that the JSON loses nothing.
    from sf.kernels._ir import KernelIR             # noqa: E402
    from sf.kernels._runtime import Kernel          # noqa: E402

    import json                                     # noqa: E402
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"))
    from check_sfir import check, load_spec         # noqa: E402

    _trace = Kernel.trace
    _spec = load_spec()

    def _round_trip_trace(self, *a, **k):
        ir = _trace(self, *a, **k)
        text = ir.to_json()
        errs = check(json.loads(text), _spec)       # and it must pass the spec's per-op schema
        assert not errs, f"{ir.name}: SF IR JSON does not match spec/ops.json:\n" + "\n".join(errs)
        back = KernelIR.from_json(text)
        assert back.to_json() == text, f"{ir.name}: SF IR JSON round trip changed the file"
        return back

    Kernel.trace = _round_trip_trace

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


class AllOpsMock(sf.MockAtreides):
    """A test-only target that has every heavy op, divide and square root included.

    The Atreides targets refuse those two, because v4.8.1 has no instruction
    for them.  The device-path rules for them (div0 and invalid are read off
    the inputs, exact, before launch) still apply to any target that does have
    them, so they stay under test here."""

    name = "mock-all-ops"
    supported_ops = None


ALL_OPS = AllOpsMock.name
sf.register_device(AllOpsMock())


from sf.kernels._runtime import SimulatorAtreides  # noqa: E402


class AllOpsSimulator(SimulatorAtreides):
    """The kernel simulator with divide and square root, for tests of their
    kernel semantics (events, C builtins).  ``atreides-sim`` refuses both."""

    name = "sim-all-ops"
    supported_ops = None


SIM_ALL_OPS = AllOpsSimulator.name
sf.register_device(AllOpsSimulator())


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
    """Every code of the width.  Single-value tests are exhaustive on all
    widths: sf16 has only 65,535 values, which is cheap."""
    m = sf.info(dtype_of(bits)).max_code
    return range(-m, m + 1)


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
