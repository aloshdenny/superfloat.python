"""A driver runner on superfloat.sim, Alosh's cycle-accurate, bit-exact model of the Atreides chip.

    from sf.kernels._chipsim import SuperfloatSimRunner
    from sf.kernels._driver import AtreidesDriver
    HardwareAtreides.attach(AtreidesDriver(runner=SuperfloatSimRunner()))

superfloat.sim checks every launch against the v4.9.2 RTL in lockstep (cycle
count, pin-bus reads and writes, every memory word), so a plan that gives the
right bits here gives them on the RTL.  It plays the host as well: it answers
the chip's pin bus from its own copy of data memory, which is the job a runner
on the real chip has to do.

superfloat.sim is Superfloat's private SDK (its own private GitHub repository),
not a dependency of sf: build it from its checkout and put its ``python/``
folder on ``PYTHONPATH`` (or set ``SUPERFLOAT_SIM`` to the checkout for the
tests).  Nothing of it is copied here.
"""
from __future__ import annotations

from .._errors import SFUnsupportedError


def _chip_class():
    try:
        from superfloat_sim import Chip
    except ImportError:
        raise SFUnsupportedError(
            "superfloat.sim is not importable: build it (cmake -S . -B build/cmake && cmake --build build/cmake, "
            "in its checkout, with the Python that runs sf) and put its python/ folder on PYTHONPATH") from None
    return Chip


class SuperfloatSimRunner:
    """``runner(plan) -> memory`` for ``AtreidesDriver``: every launch of the plan on superfloat.sim.

    The chip is reset, the program and the data memory are loaded, and each
    launch writes its parameter-block words and runs START to DONE.  After a
    call, ``launches`` holds superfloat.sim's report for each launch (cycles,
    pin-bus data reads and writes) and ``cycles`` their total.

    ``host_delay`` and ``host_seed`` set how fast the simulated host answers the
    bus (see superfloat.sim's ``Chip.set_host``); they change cycle counts, never bits.
    """

    def __init__(self, version: str = "v4.9.2", host_delay: int = 0, host_seed: int | None = None) -> None:
        self.chip = _chip_class()(version, host_delay, host_seed)
        self.version = version
        self.launches: list[dict] = []

    @property
    def cycles(self) -> int:
        return sum(r["cycles"] for r in self.launches)

    def __call__(self, plan) -> dict:
        chip = self.chip
        chip.chip_reset()
        chip.load_program(plan.program)
        chip.load_data(plan.memory)
        self.launches = []
        for launch in plan.launches:
            chip.load_data(launch.writes, clear=False)
            self.launches.append(chip.launch(launch.threads))
        mem = chip.memory()
        return {int(a): int(mem[a]) for a in mem.nonzero()[0]}       # unwritten words read 0
