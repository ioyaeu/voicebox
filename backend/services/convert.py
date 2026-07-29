"""Offline RVC voice-conversion orchestration.

Mirrors the TTS generation flow (``services/generation.py``): ``start_conversion``
validates the request, persists a ``generations`` row, and enqueues the heavy
work on the shared serial queue; ``run_conversion`` is the background coroutine
that loads the checkpoint, runs the pipeline off the event loop, and writes the
result WAV into the same storage layout as any other generation. Status and
result are therefore served by the existing ``GET /generate/{id}/status`` and
``GET /audio/{id}`` endpoints — no conversion-specific job system.
"""

from __future__ import annotations

import asyncio
import logging
import os
import traceback
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from .. import config
from ..database import VoiceProfile as DBVoiceProfile, get_db
from ..models import ConvertRequest, ConvertResponse
from ..utils.audio import save_audio, validate_and_load_reference_audio
from ..utils.tasks import get_task_manager
from . import history
from .task_queue import enqueue_generation

logger = logging.getLogger(__name__)

# Conversion is designed for full-length clips (the pipeline chunks long input),
# so the reference-audio loader is reused only for its "is this decodable?"
# guarantee — the 2-30 s bounds it defaults to for TTS references are widened.
CONVERT_MIN_DURATION = 0.3
CONVERT_MAX_DURATION = 3600.0


def ensure_crepe_available(f0_method: str) -> None:
    """Raise ``ValueError`` if crepe pitch is requested but its model is absent.

    The Crepe backbone is an on-demand download the frozen build does not bundle,
    so requesting ``f0_method="crepe"`` without it must fail with a clear,
    user-facing message (mapped to 400 by the routes) instead of crashing
    mid-inference. Single source of truth shared by offline conversion, the
    realtime stream handshake, and the TTS->RVC chain.
    """
    if f0_method != "crepe":
        return
    from ..backends.base import get_crepe_model_path

    crepe_path = get_crepe_model_path()
    if crepe_path is None or not crepe_path.exists():
        raise ValueError(
            'The Crepe pitch model is not downloaded. Download '
            '"Crepe (pitch, full)" in the Models tab to use f0_method="crepe".'
        )


def resolve_conversion_target(profile, f0_method: str) -> tuple[Path, str | None]:
    """Validate an RVC profile for offline conversion and resolve its files.

    Returns ``(model_path, index_path_str)``. Raises ``ValueError`` with a
    user-facing message (mapped to 400) when the profile is not an rvc profile,
    has no uploaded model, its model file is missing on disk, or crepe pitch is
    requested without its model. Shared by :func:`start_conversion` and the
    ``POST /convert`` route so the same checks can run *before* the request body
    is streamed to disk.
    """
    voice_type = getattr(profile, "voice_type", None) or "cloned"
    if voice_type != "rvc":
        raise ValueError("Voice conversion is only supported for RVC voice profiles")

    model_stored = getattr(profile, "rvc_model_path", None)
    if not model_stored:
        raise ValueError(
            "This RVC profile has no uploaded model. Upload a .pth checkpoint first."
        )
    model_path = config.resolve_storage_path(model_stored)
    if model_path is None or not model_path.exists():
        raise ValueError("The profile's RVC model file is missing on disk.")

    ensure_crepe_available(f0_method)

    index_path = config.resolve_storage_path(getattr(profile, "rvc_index_path", None))
    index_path_str = str(index_path) if index_path is not None and index_path.exists() else None
    return model_path, index_path_str


def sweep_orphan_files() -> int:
    """Remove stranded RVC upload/convert temp files left by crashes or failures.

    **Startup-only.** Safe precisely because no upload or conversion is in flight
    when the server boots, so every matching temp file is an orphan. Sweeps:

      * ``tmp*.pth`` / ``tmp*.index`` in each profile directory — partial RVC
        model/index uploads that never reached the atomic commit in
        ``services.profiles._validate_and_store_rvc_files``.
      * ``*_source*`` in the generations directory — offline-conversion source
        clips whose worker never ran (enqueue failure, or a crash before
        ``_cleanup_source``).

    Committed artifacts (``model.pth`` / ``model.index``) and finished
    generations never match these globs. Returns the number of files removed.

    Not wired here: startup runs in ``backend/app.py`` (outside this agent's
    ownership). The coordinator should call this from ``_run_startup``.
    """
    removed = 0

    profiles_dir = config.get_profiles_dir()
    if profiles_dir.is_dir():
        for profile_dir in profiles_dir.iterdir():
            if not profile_dir.is_dir():
                continue
            for pattern in ("tmp*.pth", "tmp*.index"):
                for stray in profile_dir.glob(pattern):
                    try:
                        stray.unlink()
                        removed += 1
                    except OSError as e:
                        logger.warning("orphan sweep: could not remove %s: %s", stray, e)

    generations_dir = config.get_generations_dir()
    if generations_dir.is_dir():
        for stray in generations_dir.glob("*_source*"):
            try:
                stray.unlink()
                removed += 1
            except OSError as e:
                logger.warning("orphan sweep: could not remove %s: %s", stray, e)

    if removed:
        logger.info("orphan sweep: removed %d stranded RVC temp file(s)", removed)
    return removed


async def start_conversion(
    *,
    params: ConvertRequest,
    source_tmp_path: str,
    source_ext: str,
    source_filename: str,
    db: Session,
) -> ConvertResponse:
    """Validate the request, persist a generation row, and enqueue the job.

    Raises:
        ValueError: If the profile is not an RVC profile, has no uploaded model,
            or the source audio cannot be decoded. The temp source is left in
            place for the caller to clean up; nothing is persisted on failure.
    """
    profile = db.query(DBVoiceProfile).filter_by(id=params.profile_id).first()
    if not profile:
        raise ValueError("Profile not found")

    # Profile-type / model / crepe validation, shared with the POST /convert route
    # (which runs it *before* streaming the body) so the checks and their
    # user-facing messages live in exactly one place.
    model_path, index_path_str = resolve_conversion_target(profile, params.f0_method)

    # Same loader add_profile_sample uses, off the event loop; widened bounds.
    is_valid, error_msg, _audio, _sr = await asyncio.to_thread(
        validate_and_load_reference_audio,
        source_tmp_path,
        CONVERT_MIN_DURATION,
        CONVERT_MAX_DURATION,
    )
    if not is_valid:
        raise ValueError(f"Invalid source audio: {error_msg}")

    generation_id = str(uuid.uuid4())

    # The temp source lives in the generations dir (same filesystem), so it can
    # be committed to a stable name with an atomic rename for the worker to read.
    source_dest = config.get_generations_dir() / f"{generation_id}_source{source_ext}"
    os.replace(source_tmp_path, source_dest)

    display_text = Path(source_filename).stem or "Voice conversion"

    generation = await history.create_generation(
        profile_id=params.profile_id,
        text=display_text,
        language=profile.language or "en",
        audio_path="",
        duration=0,
        seed=None,
        db=db,
        generation_id=generation_id,
        status="generating",
        engine="rvc",
        model_size=None,
        source="convert",
    )

    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=params.profile_id,
        text=display_text,
    )

    enqueue_generation(
        generation_id,
        run_conversion(
            generation_id=generation_id,
            source_path=str(source_dest),
            model_path=str(model_path),
            index_path=index_path_str,
            f0_up_key=params.f0_up_key,
            f0_method=params.f0_method,
            index_rate=params.index_rate,
            rms_mix_rate=params.rms_mix_rate,
            protect=params.protect,
        ),
    )

    return ConvertResponse(
        task_id=generation.id,
        profile_id=params.profile_id,
        status=generation.status or "generating",
        created_at=generation.created_at,
    )


async def run_conversion(
    *,
    generation_id: str,
    source_path: str,
    model_path: str,
    index_path: str | None,
    f0_up_key: int,
    f0_method: str,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
) -> None:
    """Background worker: load the checkpoint, convert, persist the WAV.

    Enqueued through ``task_queue.enqueue_generation`` so it shares the serial
    queue with TTS jobs (the RVC pipeline holds one model at a time).
    """
    from ..backends.rvc import acquire as rvc_acquire, release as rvc_release

    task_manager = get_task_manager()
    bg_db = next(get_db())

    # Fail fast if a live Voice Changer stream (or another job) owns the engine:
    # a shared single-model pipeline cannot serve both without corrupting one.
    lease_owner = f"convert:{generation_id}"
    lease_held = rvc_acquire(lease_owner)

    try:
        if not lease_held:
            raise RuntimeError(
                "Voice Changer live session is active — stop it to run conversions"
            )

        await history.update_generation_status(generation_id, "loading_model", bg_db)

        audio, sample_rate = await asyncio.to_thread(
            _convert_sync,
            source_path,
            model_path,
            index_path,
            f0_up_key,
            f0_method,
            index_rate,
            rms_mix_rate,
            protect,
        )

        duration = len(audio) / sample_rate if sample_rate else 0.0
        out_path = config.get_generations_dir() / f"{generation_id}.wav"
        await asyncio.to_thread(save_audio, audio, str(out_path), sample_rate)

        await history.update_generation_status(
            generation_id=generation_id,
            status="completed",
            db=bg_db,
            audio_path=config.to_storage_path(out_path),
            duration=duration,
        )
    except asyncio.CancelledError:
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error="Conversion cancelled",
        )
    except Exception as e:
        traceback.print_exc()
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error=str(e),
        )
    finally:
        if lease_held:
            rvc_release(lease_owner)
        task_manager.complete_generation(generation_id)
        _cleanup_source(source_path)
        bg_db.close()


def _convert_sync(
    source_path: str,
    model_path: str,
    index_path: str | None,
    f0_up_key: int,
    f0_method: str,
    index_rate: float,
    rms_mix_rate: float,
    protect: float,
):
    """Blocking load + convert on the shared RVC engine (runs in a thread)."""
    from ..backends.rvc import get_rvc_engine

    engine = get_rvc_engine()
    engine.load(model_path, index_path)
    return engine.convert_file(
        source_path,
        f0_up_key=f0_up_key,
        f0_method=f0_method,
        index_rate=index_rate,
        rms_mix_rate=rms_mix_rate,
        protect=protect,
    )


def _cleanup_source(source_path: str) -> None:
    """Best-effort removal of the persisted source clip once conversion ends."""
    try:
        Path(source_path).unlink(missing_ok=True)
    except OSError:
        pass
