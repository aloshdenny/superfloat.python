"""sf: the Python DSL for Superfloat on the Atreides GPU.

Stage 1.0 (scalar arithmetic) and Stage 1.1 (arrays and linear algebra):

>>> import sf
>>> a, b = sf.sf16(0.5), sf.sf16(0.25)
>>> a + b
sf16(0.75)
>>> a < 1          # comparisons are exact
True
>>> x = sf.array([0.10, -0.25, 0.50], dtype=sf.sf16)      # host array: light
>>> h = sf.asarray(np_inputs, dtype=sf.sf16, device="atreides")
>>> y = h @ w                                              # heavy: built-in kernel
>>> y.status                                               # SFStatus(saturated<=12, ...)
>>> y.numpy()                                              # read back to host
"""
from __future__ import annotations

from ._array import SFArray, addmm, dot, matmul, sum_ as sum  # noqa: A001
from ._config import config, getconfig
from ._device import Device, MockAtreides, devices, get_device, register_device
from ._errors import *            # noqa: F401,F403
from ._errors import __all__ as _errors_all
from ._errstate import errstate, geterr, seterr
from ._format import FORMATS, MAX_BITS, MIN_BITS, SFFormat
from ._functions import *         # noqa: F401,F403
from ._functions import __all__ as _functions_all
from ._scalar import DTYPES, SFScalar
from ._status import SFStatus
from .kernels import (HardwareAtreides, Kernel, LaunchResult, Limits, SimulatorAtreides,  # noqa: F401
                      arange, autotune, barrier, block_dim, block_idx, cdiv, constexpr, jit, kernel,
                      load, predicate, program_id, shared, store, synchronize, thread_idx)

__version__ = "0.4.0"           # Stage 3

# sf.sf4 .. sf.sf16
globals().update({cls.__name__: cls for cls in DTYPES.values()})
dtypes = tuple(DTYPES[b] for b in sorted(DTYPES))


def device(name="atreides-sim"):
    """A target by name (``sf.device("atreides-sim")``); see ``sf.devices()``."""
    return get_device(name)


def max(x, axis=None):          # noqa: A001
    """Largest element (exact selection, no events)."""
    from .kernels import _lang, is_traced
    if is_traced(x):
        return _lang.max(x, axis)
    from ._array import _minmax
    import numpy as _np
    return _minmax(x, axis, _np.max)


def min(x, axis=None):          # noqa: A001
    """Smallest element (exact selection, no events)."""
    from .kernels import _lang, is_traced
    if is_traced(x):
        return _lang.min(x, axis)
    from ._array import _minmax
    import numpy as _np
    return _minmax(x, axis, _np.min)


def argmax(x, axis=None):
    from ._array import _as_sfarray
    return _as_sfarray(x, "argmax").argmax(axis)


def argmin(x, axis=None):
    from ._array import _as_sfarray
    return _as_sfarray(x, "argmin").argmin(axis)


__all__ = (
    ["SFScalar", "SFArray", "SFFormat", "SFStatus", "FORMATS", "MIN_BITS", "MAX_BITS", "dtypes",
     "errstate", "geterr", "seterr", "config", "getconfig",
     "Device", "MockAtreides", "devices", "get_device", "register_device", "device",
     "sum", "max", "min", "argmax", "argmin", "dot", "matmul", "addmm", "__version__",
     "kernel", "jit", "autotune", "constexpr", "program_id", "block_idx", "thread_idx", "block_dim",
     "arange", "load", "store", "shared", "barrier", "predicate", "cdiv", "synchronize",
     "Kernel", "LaunchResult", "Limits", "SimulatorAtreides", "HardwareAtreides"]
    + [cls.__name__ for cls in dtypes]
    + list(_errors_all) + list(_functions_all)
)
