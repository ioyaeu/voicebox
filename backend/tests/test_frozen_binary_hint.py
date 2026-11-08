"""``backend.utils.frozen`` turns a swapped PyInstaller binary into a clear error.

Regression for: rebuilding / re-bundling Voicebox while the packaged server is
running makes the next deferred import fail with
``zlib.error: Error -3 while decompressing data: incorrect header check``,
which surfaced verbatim in the generation history and speech sessions.

Usage:
    python -m pytest backend/tests/test_frozen_binary_hint.py -v
"""

import os
import sys
import zlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.utils import frozen  # noqa: E402

ZLIB_MESSAGE = "Error -3 while decompressing data: incorrect header check"


@pytest.fixture
def frozen_binary(tmp_path, monkeypatch):
    """Pretend we run as a PyInstaller onefile binary at ``tmp_path/voicebox-server``."""
    binary = tmp_path / "voicebox-server"
    binary.write_bytes(b"build-1" * 100)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(binary))
    monkeypatch.setattr(frozen, "_START_STAT", None)
    assert frozen.snapshot() is not None
    return binary


def _replace(binary: Path) -> None:
    # Same path, new inode and a different size: what `cp` / `tauri build` do.
    fresh = binary.with_name("voicebox-server.new")
    fresh.write_bytes(b"build-2" * 101)
    os.replace(fresh, binary)


def test_plain_message_when_not_frozen(monkeypatch):
    monkeypatch.setattr(sys, "frozen", False, raising=False)
    monkeypatch.setattr(frozen, "_START_STAT", None)
    assert frozen.snapshot() is None
    assert frozen.executable_replaced() is False
    assert frozen.describe_failure(zlib.error(ZLIB_MESSAGE)) == ZLIB_MESSAGE


def test_plain_message_while_binary_is_untouched(frozen_binary):
    assert frozen.executable_replaced() is False
    assert frozen.describe_failure(zlib.error(ZLIB_MESSAGE)) == ZLIB_MESSAGE


def test_hint_when_binary_replaced_and_loader_failed(frozen_binary):
    _replace(frozen_binary)
    assert frozen.executable_replaced() is True

    message = frozen.describe_failure(zlib.error(ZLIB_MESSAGE))

    assert message.startswith(ZLIB_MESSAGE)
    assert str(frozen_binary) in message
    assert "relaunch Voicebox" in message


def test_hint_follows_the_exception_chain(frozen_binary):
    _replace(frozen_binary)
    try:
        try:
            raise zlib.error(ZLIB_MESSAGE)
        except zlib.error as inner:
            raise RuntimeError(str(inner)) from inner
    except RuntimeError as wrapped:
        message = frozen.describe_failure(wrapped)

    assert "relaunch Voicebox" in message


def test_unrelated_error_is_not_blamed_on_the_binary(frozen_binary):
    _replace(frozen_binary)
    assert frozen.describe_failure(ValueError("bad seed")) == "bad seed"


def test_missing_binary_counts_as_replaced(frozen_binary):
    frozen_binary.unlink()
    assert frozen.executable_replaced() is True
