"""sf: the Python DSL for Superfloat on the Atreides GPU.

Stage 1.0 (scalar arithmetic) of the language roadmap:

>>> import sf
>>> a, b = sf.sf16(0.5), sf.sf16(0.25)
>>> a + b
sf16(0.75)
>>> a * b
sf16(0.125)
>>> b / a
sf16(0.5)
>>> a < 1          # comparisons are exact
True
>>> with sf.errstate(over="ignore"):
...     sf.sf16(0.75) + 0.5          # silent saturation, same bits as the GPU
sf16(0.999969482421875)
"""
from __future__ import annotations

from ._errors import *            # noqa: F401,F403
from ._errors import __all__ as _errors_all
from ._errstate import errstate, geterr, seterr
from ._format import FORMATS, MAX_BITS, MIN_BITS, SFFormat
from ._functions import *         # noqa: F401,F403
from ._functions import __all__ as _functions_all
from ._scalar import DTYPES, SFScalar

__version__ = "0.1.0"           # Stage 1.0

# sf.sf4 .. sf.sf16
globals().update({cls.__name__: cls for cls in DTYPES.values()})
dtypes = tuple(DTYPES[b] for b in sorted(DTYPES))

__all__ = (
    ["SFScalar", "SFFormat", "FORMATS", "MIN_BITS", "MAX_BITS", "dtypes",
     "errstate", "geterr", "seterr", "__version__"]
    + [cls.__name__ for cls in dtypes]
    + list(_errors_all) + list(_functions_all)
)
