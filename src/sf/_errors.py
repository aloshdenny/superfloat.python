"""Exception and warning hierarchy (spec section 3.2).

Every SF exception also inherits from ``SFError`` and every SF warning from
``SFWarning``, so ``except sf.SFError`` / ``warnings.filterwarnings(...,
category=sf.SFWarning)`` catch everything the package can raise or warn.

The spec names one Python type per event.  Because the error policy can
turn any event into a raise *or* a warn (``sf.seterr``), each numeric event
has both an error class and a warning class; the ones the spec does not
name are additions and are marked as such below.
"""
from __future__ import annotations


class SFError(Exception):
    """Mixin base for every exception raised by the sf package."""


class SFWarning(RuntimeWarning):
    """Base for every warning issued by the sf package."""


# --- static errors: caught on the host before anything runs ----------------

class SFTypeError(SFError, TypeError):
    """Bad dtype, or a narrowing mix that needs an explicit sf.cast."""


class SFShapeError(SFError, ValueError):
    """Shape mismatch, caught on the host before launch (Stage 1.1)."""


class SFUnsupportedError(SFError, NotImplementedError):
    """Operation or width not supported, on the host or on the target."""


class SFLaunchError(SFError, RuntimeError):
    """Grid or data larger than the target allows (Stage 3)."""


class SFCompileError(SFError, RuntimeError):
    """Backend (Clang / MLIR) failure, with IR and backend log (Stage 3)."""


class SFDeviceError(SFError, RuntimeError):
    """Device fault: done never asserted, timeout, bus error (Stage 3)."""


# --- numeric events: subject to the error policy ---------------------------

class SFOverflowError(SFError, OverflowError):
    """Exact result outside ±max.  The hardware value is saturation to ±max."""


class SFOverflowWarning(SFWarning):
    """Addition: the warn form of an overflow event (seterr(over='warn'))."""


class SFUnderflowError(SFError, ArithmeticError):
    """Addition: the raise form of an underflow event (seterr(under='raise'))."""


class SFUnderflowWarning(SFWarning):
    """A nonzero exact result became 0."""


class SFZeroDivisionError(SFError, ZeroDivisionError):
    """Division by an SF zero."""


class SFZeroDivisionWarning(SFWarning):
    """Addition: the warn form of a divide-by-zero event."""


class SFInvalidError(SFError, ValueError):
    """Invalid input (NaN) or domain error (sqrt of a negative)."""


class SFInvalidWarning(SFWarning):
    """Addition: the warn form of an invalid event."""


class SFDeviceOverflowWarning(SFWarning):
    """Saturation detected in a heavy op from result.status (Stage 1.1)."""


__all__ = [
    "SFError", "SFWarning",
    "SFTypeError", "SFShapeError", "SFUnsupportedError",
    "SFLaunchError", "SFCompileError", "SFDeviceError",
    "SFOverflowError", "SFOverflowWarning",
    "SFUnderflowError", "SFUnderflowWarning",
    "SFZeroDivisionError", "SFZeroDivisionWarning",
    "SFInvalidError", "SFInvalidWarning",
    "SFDeviceOverflowWarning",
]
