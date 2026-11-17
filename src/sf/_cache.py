"""A private folder for the shared libraries sf compiles.

Two paths compile C once and load the result with ctypes: the MX
accumulator (``_fastmx``) and the kernel C backend (``kernels._cgen``).
Loading a library runs its code, so it must come from a folder that only
this user can write.  A fixed name in the shared temp directory is not
one: another user on the machine can create that folder first and put a
library there under the predictable name.

So the cache lives under the user's own cache folder
(``$XDG_CACHE_HOME/sf`` or ``~/.cache/sf``), or under ``$SF_CACHE_DIR``
when that is set.  The folder is created with mode 0700, and on POSIX both
the folder and every library are checked before use: owned by this user,
not a symbolic link, and not writable by group or others.  A check that
fails raises ``OSError``; the callers then use their reference path
(``_fastmx``) or report the C backend as unavailable (``_cgen``).

With no home folder at all, each process gets its own ``mkdtemp`` folder
(private by construction), so nothing is shared and nothing is reused.
"""
from __future__ import annotations

import os
import stat
import tempfile

_POSIX = hasattr(os, "getuid")
_fallback: dict[str, str] = {}


def private_dir(name: str) -> str:
    """The private cache folder for ``name``, created if missing and checked."""
    base = os.environ.get("SF_CACHE_DIR")
    if not base:
        root = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
        if not os.path.isabs(root):                    # no usable home folder
            d = _fallback.get(name)
            if d is None:
                d = _fallback[name] = tempfile.mkdtemp(prefix=f"sf-{name}-")
            return d
        base = os.path.join(root, "sf")
    return ensure_private(os.path.join(base, name))


def ensure_private(path: str) -> str:
    """Create ``path`` (mode 0700) if missing, then check it is ours alone."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    _check(path, want_dir=True)
    return path


def check_file(path: str) -> None:
    """Raise ``OSError`` unless ``path`` is a regular file only we can write."""
    _check(path, want_dir=False)


def _check(path: str, want_dir: bool) -> None:
    if not _POSIX:                                     # Windows: the user profile is private already
        return
    st = os.lstat(path)
    kind = stat.S_ISDIR(st.st_mode) if want_dir else stat.S_ISREG(st.st_mode)
    if not kind:
        raise OSError(f"{path} is not a plain {'folder' if want_dir else 'file'} (a link or another type)")
    if st.st_uid != os.getuid():
        raise OSError(f"{path} belongs to another user")
    if st.st_mode & 0o022:
        raise OSError(f"{path} is writable by group or others")
