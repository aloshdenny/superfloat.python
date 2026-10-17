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

__version__ = "0.2.0"           # Stage 1.1

# sf.sf4 .. sf.sf16
globals().update({cls.__name__: cls for cls in DTYPES.values()})
dtypes = tuple(DTYPES[b] for b in sorted(DTYPES))


def max(x, axis=None):          # noqa: A001
    """Largest element (exact selection, no events)."""
    from ._array import _minmax
    import numpy as _np
    return _minmax(x, axis, _np.max)


def min(x, axis=None):          # noqa: A001
    """Smallest element (exact selection, no events)."""
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
     "Device", "MockAtreides", "devices", "get_device", "register_device",
     "sum", "max", "min", "argmax", "argmin", "dot", "matmul", "addmm", "__version__"]
    + [cls.__name__ for cls in dtypes]
    + list(_errors_all) + list(_functions_all)
)
