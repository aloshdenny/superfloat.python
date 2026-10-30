"""Device interface and targets.

A *target* executes heavy ops.  Everything above it (placement, promotion,
pre-flight, launch, readback, status) is the runtime and is target
independent.  A target only needs to run an op on codes and hand back
codes, like the hardware hands back ``done`` and data.

Targets available now:

* ``host``: not a target; host arrays run the vectorised reference directly
  (light ops).
* ``atreides``: ``MockAtreides``, which executes the same vectorised
  reference in host memory and reports nothing but the result, exactly as
  Atreides v1 does (no saturation flag or status register).  It is the
  stand-in until the simulator and hardware are wired in: the bit-identical
  test is "target == host reference", and it does not change when the mock
  is swapped out.

Target definition (spec sections 3.1 and 12): accumulation order and
accumulator width are attributes of the target, and the host reference
reads them from here.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from . import _vectorized as vec
from ._errors import SFLaunchError, SFUnsupportedError
from ._format import SFFormat

HOST = "host"


class Device:
    """Base class of every target."""

    name: str = ""
    supported_bits: tuple[int, ...] = ()
    accumulation_order: str = "sequential"   # left to right over the reduced axis
    accumulator_bits: int | None = None      # None: same width as the data (v1)
    sticky_flags: bool = False               # per-tile SAT / DIV0 / INVALID flags
    max_elements: int | None = None          # launch limit, None = unbounded
    supported_ops: frozenset[str] | None = None  # heavy ops the target has instructions for; None = all

    def check(self, fmt: SFFormat, n_elements: int) -> None:
        if fmt.bits not in self.supported_bits:
            raise SFUnsupportedError(
                f"{fmt.name} is not supported on {self.name} "
                f"(supported: {', '.join(f'sf{b}' for b in self.supported_bits)})")
        if self.max_elements is not None and n_elements > self.max_elements:
            raise SFLaunchError(f"{n_elements} elements exceed the {self.name} limit "
                                f"of {self.max_elements}")

    def run(self, op: str, fmt: SFFormat, *operands: np.ndarray, **kw):
        """Execute ``op`` on code arrays.

        Returns ``(codes, flags)``.  ``flags`` is ``None`` for a target with
        no event reporting (Atreides v1), or a dict of boolean masks over the
        output (``"overflow"`` at least, ``"underflow"`` where the op can
        underflow) for a target with sticky flags.  The runtime trusts
        ``flags`` only when ``sticky_flags`` is True.

        An op not in ``supported_ops`` (when that is set) is refused on the
        host with ``SFUnsupportedError`` before ``run`` is called: Atreides
        v1, for example, has no divide or square-root instruction.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"<sf device {self.name}>"


class MockAtreides(Device):
    """Atreides v1 stand-in: runs the vectorised reference and returns codes
    only, exactly like hardware with no status register."""

    name = "atreides"
    supported_bits = (16,)
    accumulation_order = "sequential"
    accumulator_bits = None
    sticky_flags = False
    max_elements = 1 << 28

    def __init__(self) -> None:
        self.launches = 0                    # for tests: proves pre-flight skipped a launch

    def run(self, op: str, fmt: SFFormat, *operands: np.ndarray, **kw):
        self.launches += 1
        fn: Callable = _KERNELS[op]
        codes, _events = fn(fmt, *operands, **kw)
        return np.asarray(codes, dtype=np.int64), None


class MockAtreidesFlags(MockAtreides):
    """A hypothetical Atreides with per-tile sticky flags (spec section 12):
    the same kernels, but the event masks come back with the result.  Used
    to exercise the exact-status path until such hardware exists."""

    name = "atreides-flags"
    sticky_flags = True

    def run(self, op: str, fmt: SFFormat, *operands: np.ndarray, **kw):
        self.launches += 1
        codes, events = _KERNELS[op](fmt, *operands, **kw)
        return np.asarray(codes, dtype=np.int64), {k: np.asarray(v, dtype=bool)
                                                   for k, v in events.items() if k != "first"}


def _matmul_kernel(fmt: SFFormat, a: np.ndarray, b: np.ndarray):
    from ._linalg import matmul_codes
    return matmul_codes(fmt, a, b, order="sequential")


def _sum_kernel(fmt: SFFormat, a: np.ndarray, axis: int):
    from ._linalg import sum_codes
    return sum_codes(fmt, a, axis, order="sequential")


_KERNELS: dict[str, Callable] = {
    "add": vec.add, "sub": vec.sub, "mul": vec.mul, "div": vec.div, "fma": vec.fma,
    "neg": vec.neg, "abs": vec.abs_, "sqrt": vec.sqrt, "pow": vec.pow_int,
    "matmul": _matmul_kernel, "sum": _sum_kernel,
}

_registry: dict[str, Device] = {"atreides": MockAtreides(), "atreides-flags": MockAtreidesFlags()}


def get_device(name) -> Device:
    if isinstance(name, Device):
        return name
    if name not in _registry:
        raise SFUnsupportedError(f"unknown device {name!r}; known: host, "
                                 + ", ".join(_registry))
    return _registry[name]


def register_device(device: Device) -> None:
    """Add or replace a target (the simulator or hardware, when they arrive)."""
    _registry[device.name] = device


def devices() -> tuple[str, ...]:
    return (HOST,) + tuple(_registry)
