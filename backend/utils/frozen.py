"""Detect a frozen (PyInstaller) server binary that was replaced on disk.

PyInstaller keeps Python modules compressed inside the executable and reopens
that file *by path* on every deferred import. If the binary is rebuilt or
re-bundled while the server is running, later imports read the new file with
the old offsets and fail with ``zlib.error: Error -3 while decompressing data``
(or a bare ``ImportError``). The traceback points at ``pyimod01_archive.py``
and says nothing about the real cause, so we record the executable's identity
at startup and turn that failure into an actionable message.
"""

import os
import struct
import sys
import zlib

_START_STAT: tuple[int, int, int] | None = None

# Exceptions the PyInstaller archive reader raises when it reads a file whose
# layout no longer matches the table of contents held in memory.
_LOADER_ERRORS = (zlib.error, ImportError, struct.error, EOFError)


def is_frozen() -> bool:
    return bool(getattr(sys, "frozen", False))


def _stat_executable() -> tuple[int, int, int] | None:
    try:
        st = os.stat(sys.executable)
    except OSError:
        return None
    return (st.st_ino, st.st_size, st.st_mtime_ns)


def snapshot() -> tuple[int, int, int] | None:
    """Record the executable's identity once. Runs at import, i.e. at startup."""
    global _START_STAT
    if _START_STAT is None and is_frozen():
        _START_STAT = _stat_executable()
    return _START_STAT


def executable_replaced() -> bool:
    """True when the frozen binary on disk is no longer the one that started."""
    if not is_frozen() or _START_STAT is None:
        return False
    return _stat_executable() != _START_STAT


def _chain(exc: BaseException):
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def looks_like_loader_failure(exc: BaseException) -> bool:
    return any(isinstance(e, _LOADER_ERRORS) for e in _chain(exc))


def describe_failure(exc: BaseException) -> str:
    """``str(exc)``, with a hint appended when the frozen binary was swapped."""
    message = str(exc) or exc.__class__.__name__
    if looks_like_loader_failure(exc) and executable_replaced():
        return (
            f"{message}. The Voicebox server binary ({sys.executable}) was replaced on "
            "disk after it started (rebuild while running), so it can no longer load its "
            "own modules. Quit and relaunch Voicebox; rebuilding again will not help while "
            "this process is alive."
        )
    return message


snapshot()
