"""The compiled-library cache is private (sf/_cache.py).

Loading a .so runs its code, so sf must never load one from a folder another
user can write.  These check the folder and file rules, and that both C paths
refuse a cache that fails them instead of loading from it."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from sf import _cache

posix = pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX permission checks")


def test_private_dir_is_created_0700_under_sf_cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SF_CACHE_DIR", str(tmp_path / "c"))
    d = _cache.private_dir("fastmx")
    assert d == str(tmp_path / "c" / "fastmx")
    if hasattr(os, "getuid"):
        assert os.stat(d).st_mode & 0o777 == 0o700


def test_default_is_the_user_cache_folder_not_tmp(tmp_path, monkeypatch):
    monkeypatch.delenv("SF_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert _cache.private_dir("kernels-c") == str(tmp_path / "sf" / "kernels-c")


@posix
def test_a_folder_others_can_write_is_refused(tmp_path):
    d = tmp_path / "open"
    d.mkdir()
    os.chmod(d, 0o777)
    with pytest.raises(OSError, match="writable by group or others"):
        _cache.ensure_private(str(d))


@posix
def test_a_link_is_refused(tmp_path):
    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    (tmp_path / "link").symlink_to(real)
    with pytest.raises(OSError, match="not a plain folder"):
        _cache.ensure_private(str(tmp_path / "link"))
    f = real / "x.so"
    f.write_bytes(b"")
    (real / "y.so").symlink_to(f)
    with pytest.raises(OSError, match="not a plain file"):
        _cache.check_file(str(real / "y.so"))


@posix
def test_a_library_others_can_write_is_refused(tmp_path):
    f = tmp_path / "x.so"
    f.write_bytes(b"")
    os.chmod(f, 0o666)
    with pytest.raises(OSError, match="writable by group or others"):
        _cache.check_file(str(f))


def _run(code, env):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                          env={**os.environ, **env})


@posix
def test_fastmx_uses_the_reference_when_the_cache_is_not_private(tmp_path):
    bad = tmp_path / "shared"
    (bad / "fastmx").mkdir(parents=True)
    os.chmod(bad / "fastmx", 0o777)
    r = _run("""
        from sf import _fastmx
        print(_fastmx.available())
    """, {"SF_CACHE_DIR": str(bad), "SF_FAST_MX": "1"})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "False"
    assert not [n for n in os.listdir(bad / "fastmx")]             # nothing was built or loaded there


@posix
def test_the_kernel_backend_refuses_a_cache_that_is_not_private(tmp_path):
    from sf.kernels import _cgen
    if not _cgen.compiler():
        pytest.skip("no C compiler")
    bad = tmp_path / "shared"
    (bad / "kernels-c").mkdir(parents=True)
    os.chmod(bad / "kernels-c", 0o777)
    r = _run("""
        import numpy as np, sf
        @sf.jit
        def k(x, out, BLOCK: sf.constexpr):
            o = sf.arange(0, BLOCK)
            sf.store(out + o, sf.load(x + o) * 0.5)
        dev = sf.device("atreides-sim")
        x = dev.upload(np.full(8, 0.25)); out = dev.alloc(8)
        try:
            with sf.config(kernel_backend="c"):
                k[(1,)](x, out, BLOCK=8)
        except sf.SFUnsupportedError as e:
            print("refused:", e)
    """, {"SF_CACHE_DIR": str(bad)})
    assert r.returncode == 0, r.stderr
    assert "refused:" in r.stdout and "writable by group or others" in r.stdout
    assert not os.listdir(bad / "kernels-c")
