"""Offline RVC engine smoke test (step 02, deliverable 5).

Drives a full file-in -> array-out conversion through the real
:class:`~backend.backends.rvc.pipeline.RVCPipeline` and asserts the output is
well-formed. It is gated on a local community checkpoint fixture: with no
fixture present -- the default, and the state the suite runs in on CI -- the
single test SKIPS, so ``pytest`` stays green without a GPU, network access, or a
multi-hundred-megabyte model download.

Placing a fixture
-----------------
Drop a community RVC ``.pth`` checkpoint (any 32k/40k/48k, v1 or v2, f0 or
non-f0) into ``backend/tests/fixtures/rvc/``. It must be an *extracted*
inference checkpoint -- the kind RVC WebUI's ckpt export produces -- i.e. one
that passes :func:`validate_rvc_checkpoint`: it carries the
``weight``/``config``/``f0``/``version``/``sr`` keys. Raw training generators
such as ``f0G40k.pth`` (whose top-level keys are ``model``/``iteration``/...)
do *not* qualify and are rejected before conversion. A small 40k v2 model
(~55 MB) is ideal, for example::

    from huggingface_hub import hf_hub_download
    hf_hub_download(
        "trojblue/rvc-kanade-voice",
        "_weights_unsorted/keruanv2.pth",
        local_dir="backend/tests/fixtures/rvc",
    )

The first ``*.pth`` in the directory (sorted) is used; an optional matching
``*.index`` FAISS file in the same directory is picked up and exercises the
retrieval blend, otherwise the conversion runs index-free.

The conversion also needs the shared ContentVec and RMVPE backbones. When they
are absent from the HuggingFace cache and there is no network to fetch them, the
test SKIPS instead of failing (the same offline contract as the fixture gate);
when they *are* cached, a download-shaped error is a real failure and is raised.
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend.backends.base import is_model_cached
from backend.backends.rvc import get_rvc_engine
from backend.backends.rvc.checkpoint import (
    load_rvc_checkpoint,
    validate_rvc_checkpoint,
)

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rvc"

# Mirror the shared-backbone repos declared in ``features.py`` / ``pitch.py``.
# Kept as literals (not imported) so this module -- and its skip path -- never
# depends on those modules' private constants staying importable.
_CONTENTVEC_REPO = "lengyue233/content-vec-best"
_RMVPE_REPO = "lj1995/VoiceConversionWebUI"
_RMVPE_FILENAME = "rmvpe.pt"


def _first_glob(pattern: str):
    if not _FIXTURE_DIR.is_dir():
        return None
    matches = sorted(_FIXTURE_DIR.glob(pattern))
    return str(matches[0]) if matches else None


_CHECKPOINT = _first_glob("*.pth")

# Whether both backbones are already cached decides how an offline-shaped
# failure is handled below: raised (real bug) when present, skipped when not.
_BACKBONES_CACHED = is_model_cached(_CONTENTVEC_REPO) and is_model_cached(
    _RMVPE_REPO, required_files=[_RMVPE_FILENAME]
)

# requests' ConnectionError and huggingface_hub's LocalEntryNotFoundError /
# HTTP errors all derive from OSError, so a single base covers "could not fetch
# a model" without swallowing unrelated failures (assertions, ValueError, ...).
_DOWNLOAD_ERROR = OSError


def _write_voiced_wav(path: Path, seconds: float = 3.0, sr: int = 16000) -> float:
    """Write a synthetic voiced tone and return its duration in seconds.

    RMVPE only emits a non-zero f0 on voiced frames, so silence or noise would
    exercise nothing; a fundamental plus a few harmonics with light vibrato and
    a fade envelope gives both the pitch estimator and the synthesizer real
    content to convert.
    """
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    f0 = 150.0 * (2.0 ** (0.05 * np.sin(2 * np.pi * 5.0 * t)))  # ~150 Hz + vibrato
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros_like(t)
    for harmonic, amp in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125)):
        sig += amp * np.sin(harmonic * phase)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / seconds, 0.0, 1.0)))  # fade in/out
    sig = (sig * env).astype(np.float32)
    sig /= np.abs(sig).max() + 1e-9
    sf.write(str(path), sig * 0.8, sr)
    return seconds


@pytest.mark.skipif(
    _CHECKPOINT is None,
    reason=(
        "no RVC checkpoint fixture under backend/tests/fixtures/rvc/ "
        "(see this module's docstring to place one)"
    ),
)
def test_offline_conversion_produces_wellformed_audio(tmp_path):
    """A 3 s clip converts to non-silent, NaN-free audio at the checkpoint sr."""
    info = validate_rvc_checkpoint(load_rvc_checkpoint(_CHECKPOINT))

    src = tmp_path / "src.wav"
    in_seconds = _write_voiced_wav(src, seconds=3.0)

    engine = get_rvc_engine()
    try:
        engine.load(_CHECKPOINT, _first_glob("*.index"))
        audio, sr = engine.convert_file(
            str(src),
            f0_up_key=0,
            f0_method="rmvpe",
            index_rate=0.75,
            rms_mix_rate=0.25,
            protect=0.33,
        )
    except _DOWNLOAD_ERROR as exc:
        if _BACKBONES_CACHED:
            raise  # backbones are local, so a fetch-shaped error is a real bug
        pytest.skip(f"ContentVec/RMVPE unavailable offline: {exc}")
    finally:
        engine.unload()

    assert isinstance(audio, np.ndarray)
    assert audio.ndim == 1
    assert audio.size > 0

    # Output is at the checkpoint's native sample rate (callers resample).
    assert sr == info.sample_rate

    assert not np.isnan(audio).any(), "conversion produced NaNs"
    assert not np.isinf(audio).any(), "conversion produced non-finite samples"

    out_seconds = audio.size / sr
    drift = abs(out_seconds - in_seconds) / in_seconds
    assert drift < 0.05, (
        f"output duration {out_seconds:.3f}s drifted {drift:.1%} from "
        f"input {in_seconds:.3f}s (>5%)"
    )

    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    assert rms > 1e-4, f"output is effectively silent (rms={rms:.2e})"
