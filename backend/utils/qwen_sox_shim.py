"""Fallbacks for qwen_tts' optional SoX command-line dependency.

The qwen_tts speech tokenizer imports the Python ``sox`` package and uses
``sox.Transformer().norm(-6)`` for x-vector reference normalization. The Python
package shells out to a system ``sox`` binary, which is not guaranteed to exist
on user machines or inside the frozen Voicebox sidecar.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def sox_binary_available() -> bool:
    return shutil.which("sox") is not None


def suppress_sox_import_warning_if_missing() -> None:
    """Silence the Python sox package's import-time warning when no CLI exists."""
    if not sox_binary_available():
        logging.getLogger("sox").disabled = True


@contextlib.contextmanager
def sox_import_probe_stub_if_missing():
    """Provide a temporary ``sox -h`` stub while importing qwen_tts.

    The Python ``sox`` package checks the external executable at import time via
    ``os.popen("sox -h")``. This context makes that probe succeed without
    leaving a fake SoX executable on PATH for real runtime work.
    """
    suppress_sox_import_warning_if_missing()
    if sox_binary_available():
        yield
        return

    old_path = os.environ.get("PATH", "")
    with tempfile.TemporaryDirectory(prefix="voicebox-qwen-sox-") as tmp:
        stub = Path(tmp) / ("sox.bat" if os.name == "nt" else "sox")
        if os.name == "nt":
            stub.write_text(
                "@echo off\necho AUDIO FILE FORMATS: wav mp3 flac ogg m4a webm\n",
                encoding="utf-8",
            )
        else:
            stub.write_text(
                "#!/bin/sh\nprintf '%s\\n' 'AUDIO FILE FORMATS: wav mp3 flac ogg m4a webm'\n",
                encoding="utf-8",
            )
            stub.chmod(0o755)
        os.environ["PATH"] = tmp + os.pathsep + old_path
        try:
            yield
        finally:
            os.environ["PATH"] = old_path


def install_qwen_sox_norm_shim() -> bool:
    """Patch qwen_tts to normalize reference audio without the SoX binary.

    Returns ``True`` when the shim is installed. If a real SoX executable is on
    PATH, leaves qwen_tts untouched.
    """
    if sox_binary_available():
        return False

    try:
        from qwen_tts.core.tokenizer_25hz.vq import speech_vq
    except Exception as exc:
        logger.debug("Could not install qwen_tts SoX shim: %r", exc)
        return False

    extractor = getattr(speech_vq, "XVectorExtractor", None)
    if extractor is None:
        return False
    if getattr(extractor, "_voicebox_sox_norm_shim", False):
        return True

    def _numpy_norm(self, audio):
        arr = np.asarray(audio)
        if arr.size == 0:
            return arr.astype(np.float32, copy=True)

        out = arr.astype(np.float32, copy=True)
        peak = float(np.max(np.abs(out)))
        if not np.isfinite(peak) or peak <= 1e-8:
            return out

        # SoX "norm -6" normalizes peak level to -6 dBFS.
        target_peak = 10 ** (-6.0 / 20.0)
        out *= target_peak / peak
        return np.clip(out, -1.0, 1.0).astype(np.float32, copy=False)

    extractor.sox_norm = _numpy_norm
    extractor._voicebox_sox_norm_shim = True
    logger.info("Installed qwen_tts SoX normalization shim")
    return True
