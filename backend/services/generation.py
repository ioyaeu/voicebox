"""
Unified TTS generation orchestration.

Replaces the three near-identical closures (_run_generation, _run_retry,
_run_regenerate) that lived in main.py with a single ``run_generation()``
function parameterized by *mode*.

Mode differences:
  - "generate"   : full pipeline -- save clean version, optionally apply
                    effects and create a processed version.
  - "retry"      : re-runs a failed generation with the same seed.
                    No effects, no version creation.
  - "regenerate" : re-runs with seed=None for variation.  Creates a new
                    version with an auto-incremented "take-N" label.
"""

from __future__ import annotations

import asyncio
import traceback
from typing import Literal, Optional

from .. import config
from . import history, profiles
from ..database import get_db
from ..utils.tasks import get_task_manager


def _raise_if_realtime_stream_active() -> None:
    try:
        from ..backends.rvc import realtime_stream_active

        active = realtime_stream_active()
    except Exception:
        active = False
    if active:
        raise RuntimeError(
            "Voice Changer real-time streaming is active. Stop the live stream "
            "before starting a TTS generation."
        )


async def run_generation(
    *,
    generation_id: str,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: Optional[int],
    normalize: bool = False,
    effects_chain: Optional[list] = None,
    instruct: Optional[str] = None,
    mode: Literal["generate", "retry", "regenerate"],
    max_chunk_chars: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
    version_id: Optional[str] = None,
) -> None:
    """Execute TTS inference and persist the result.

    This is the single entry point for all background generation work.
    It is designed to be enqueued via ``services.task_queue.enqueue_generation``.
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..database import VoiceProfile as DBVoiceProfile
    from ..utils.audio import has_tts_runaway, normalize_audio, save_audio, trim_tts_output
    from ..utils.chunked_tts import generate_chunked

    task_manager = get_task_manager()
    bg_db = next(get_db())

    try:
        profile_row = bg_db.query(DBVoiceProfile).filter_by(id=profile_id).first()
        voice_type = getattr(profile_row, "voice_type", None) or "cloned"

        if voice_type == "rvc":
            # TTS->RVC chain: render the text with a cloning-free base preset
            # voice, then convert into the profile's RVC voice. Everything
            # downstream (normalize, effects, storage, versions) is shared with
            # the non-rvc path below.
            audio, sample_rate = await _generate_rvc_chained(
                profile=profile_row,
                generation_id=generation_id,
                bg_db=bg_db,
                text=text,
                language=language,
                seed=seed if mode != "regenerate" else None,
                instruct=instruct,
                max_chunk_chars=max_chunk_chars,
                crossfade_ms=crossfade_ms,
            )
        else:
            tts_model = get_tts_backend_for_engine(engine)
            _raise_if_realtime_stream_active()

            if not tts_model.is_loaded():
                await history.update_generation_status(generation_id, "loading_model", bg_db)

            await load_engine_model(engine, model_size)
            _raise_if_realtime_stream_active()

            voice_prompt = await profiles.create_voice_prompt_for_profile(
                profile_id,
                bg_db,
                use_cache=True,
                engine=engine,
                language=language,
            )

            await history.update_generation_status(generation_id, "generating", bg_db)
            trim_fn = trim_tts_output if engine_needs_trim(engine) else None
            runaway_detector = has_tts_runaway if engine_retries_runaway(engine) else None

            gen_kwargs: dict = dict(
                language=language,
                seed=seed if mode != "regenerate" else None,
                instruct=instruct,
                trim_fn=trim_fn,
                runaway_detector=runaway_detector,
            )
            if max_chunk_chars is not None:
                gen_kwargs["max_chunk_chars"] = max_chunk_chars
            if crossfade_ms is not None:
                gen_kwargs["crossfade_ms"] = crossfade_ms

            audio, sample_rate = await generate_chunked(tts_model, text, voice_prompt, **gen_kwargs)

        # --- Normalize (generate and regenerate always; retry skips) -----
        if normalize or mode == "regenerate":
            audio = normalize_audio(audio)

        duration = len(audio) / sample_rate

        # --- Persist audio and update status -----------------------------
        if mode == "generate":
            final_path = _save_generate(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                effects_chain=effects_chain,
                save_audio=save_audio,
                db=bg_db,
            )
        elif mode == "retry":
            final_path = _save_retry(
                generation_id=generation_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
            )
        elif mode == "regenerate":
            final_path = _save_regenerate(
                generation_id=generation_id,
                version_id=version_id,
                audio=audio,
                sample_rate=sample_rate,
                save_audio=save_audio,
                db=bg_db,
            )

        await history.update_generation_status(
            generation_id=generation_id,
            status="completed",
            db=bg_db,
            audio_path=final_path,
            duration=duration,
        )

    except asyncio.CancelledError:
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error="Generation cancelled",
        )
        _notify_speak_end(generation_id, status="cancelled")
    except Exception as e:
        traceback.print_exc()
        await history.update_generation_status(
            generation_id=generation_id,
            status="failed",
            db=bg_db,
            error=str(e),
        )
        _notify_speak_end(generation_id, status="failed")
    else:
        _notify_speak_end(generation_id, status="completed")
    finally:
        task_manager.complete_generation(generation_id)
        bg_db.close()


class _ChainStageError(Exception):
    """Wraps a failure in one TTS->RVC chain stage so the history error can name
    the stage the user cares about.

    ``str()`` is ``"<stage> stage: <original>"`` (e.g. ``"TTS stage: ..."`` /
    ``"Conversion stage: ..."``), so ``run_generation``'s generic failure handler
    persists an attributed message without any chain-specific branching.
    """

    def __init__(self, stage: str, original: BaseException):
        self.stage = stage
        self.original = original
        super().__init__(f"{stage} stage: {original}")


def _resolve_base_model_size(base_engine: str, language: str | None = None) -> str:
    """Pick the model size to load for a chained base-TTS engine.

    Single-model engines (kokoro, chatterbox, …) use ``"default"``. For
    multi-size engines (Qwen, Qwen CustomVoice, TADA) the old code
    hardcoded ``"1.7B"``, which forced a multi-GB download even when the user
    already had the smaller variant installed. Prefer an already-cached size;
    fall back to the first configured size so a fresh install still resolves to a
    real, downloadable model (the frontend then shows the normal download dialog).
    """
    from ..backends import (
        engine_has_model_sizes,
        get_tts_backend_for_engine,
        get_tts_model_configs,
        resolve_model_size_for_engine,
    )

    if not engine_has_model_sizes(base_engine):
        return "default"

    configs = [c for c in get_tts_model_configs() if c.engine == base_engine]
    if not configs:
        return "default"

    backend = get_tts_backend_for_engine(base_engine)
    for cfg in configs:
        if language and language not in cfg.languages:
            continue
        if backend._is_model_cached(cfg.model_size):
            return cfg.model_size
    # None installed yet: use the first configured size; the download flow (and
    # the frontend dialog) handle fetching it.
    return resolve_model_size_for_engine(base_engine, language=language) or configs[0].model_size


def _resolve_base_profile_engine(base_profile) -> str:
    """Engine a non-rvc base profile would use if it were generated directly.

    Mirrors the non-rvc branch of
    ``routes.generations._resolve_generation_engine`` (there is no request engine
    here): the profile's default engine, then its preset engine, then the qwen
    fallback. Preset profiles are locked to their ``preset_engine``. Keeping this
    identical to the direct-generation resolution is what makes a profile base
    "the same code the base profile would use if targeted directly".
    """
    voice_type = getattr(base_profile, "voice_type", None) or "cloned"
    if voice_type == "preset":
        return getattr(base_profile, "preset_engine", None) or "qwen"
    return (
        getattr(base_profile, "default_engine", None)
        or getattr(base_profile, "preset_engine", None)
        or "qwen"
    )


async def _generate_rvc_chained(
    *,
    profile,
    generation_id: str,
    bg_db,
    text: str,
    language: str,
    seed: Optional[int],
    instruct: Optional[str],
    max_chunk_chars: Optional[int],
    crossfade_ms: Optional[int],
):
    """Render *text* with the profile's base TTS voice, then RVC-convert it.

    The base voice is a cloning-free preset (``rvc_base_voice``); the conversion
    uses the profile's uploaded checkpoint/index and ``rvc_params``. The base TTS
    engine is unloaded before the RVC model loads so the two never need to fit in
    VRAM together. Returns ``(audio, sample_rate)`` for the shared post-processing
    (normalize/effects/storage) in :func:`run_generation`.
    """
    import asyncio

    from ..backends import (
        engine_needs_trim,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..backends.rvc import (
        acquire as rvc_acquire,
        get_rvc_engine,
        release as rvc_release,
    )
    from ..database import VoiceProfile as DBVoiceProfile
    from ..utils.audio import trim_tts_output
    from ..utils.chunked_tts import generate_chunked
    from .profiles import (
        RVC_PROFILE_BASE_PREFIX,
        load_rvc_params,
        parse_rvc_base_voice,
        rvc_base_is_profile_ref,
    )

    params = load_rvc_params(getattr(profile, "rvc_params", None))

    # Resolve the base voice: either a cloning-free preset ("engine:voice_id") or
    # an existing non-rvc profile ("profile:{id}"). For a profile base the audio
    # is rendered through the *same* generation path the base profile uses when
    # targeted directly (create_voice_prompt_for_profile + its resolved engine);
    # the resulting audio is then RVC-converted exactly like the preset case.
    raw_base_voice = getattr(profile, "rvc_base_voice", None)
    base_is_profile = rvc_base_is_profile_ref(raw_base_voice)
    base_voice_id = None
    base_profile = None
    if base_is_profile:
        base_profile_id = raw_base_voice[len(RVC_PROFILE_BASE_PREFIX):].strip()
        base_profile = bg_db.query(DBVoiceProfile).filter_by(id=base_profile_id).first()
        if base_profile is None:
            raise ValueError(
                f"This RVC profile's base voice '{base_profile_id}' no longer exists."
            )
        # Runtime recursion guard: validation and execution are separated by time
        # and DB edits, so re-assert the base is non-rvc here (ValueError, never a
        # silent skip) — an rvc-on-rvc chain must not run.
        base_type = getattr(base_profile, "voice_type", None) or "cloned"
        if base_type == "rvc":
            raise ValueError(
                "An RVC profile cannot be used as a base voice for another RVC "
                "profile (no chain-of-chains)."
            )
        base_engine = _resolve_base_profile_engine(base_profile)
    else:
        base_engine, base_voice_id = parse_rvc_base_voice(raw_base_voice)

    model_stored = getattr(profile, "rvc_model_path", None)
    if not model_stored:
        raise ValueError(
            "This RVC profile has no uploaded model. Upload a .pth checkpoint first."
        )
    model_path = config.resolve_storage_path(model_stored)
    if model_path is None or not model_path.exists():
        raise ValueError("The profile's RVC model file is missing on disk.")

    index_resolved = config.resolve_storage_path(getattr(profile, "rvc_index_path", None))
    index_path = str(index_resolved) if index_resolved is not None and index_resolved.exists() else None

    from .convert import ensure_crepe_available

    ensure_crepe_available(params["f0_method"])

    # Fail fast if a live Voice Changer stream (or another job) owns the engine;
    # done before base TTS so contention does not waste a whole render. Held for
    # both stages and released in the finally below.
    lease_owner = f"chain:{generation_id}"
    if not rvc_acquire(lease_owner):
        raise RuntimeError(
            "Voice Changer live session is active — stop it to run conversions"
        )

    try:
        # --- Stage 1: base TTS (preset voice) ---------------------------------
        try:
            base_model = get_tts_backend_for_engine(base_engine)
            if not base_model.is_loaded():
                await history.update_generation_status(generation_id, "loading_model", bg_db)

            base_model_size = _resolve_base_model_size(base_engine, language)
            await load_engine_model(base_engine, base_model_size)

            if base_is_profile:
                # Single choke point: build the base profile's voice prompt through
                # the exact path a direct generation would (cloned samples /
                # designed prompt / preset reference), so there is no second
                # voice-prompt implementation to drift.
                voice_prompt = await profiles.create_voice_prompt_for_profile(
                    base_profile.id,
                    bg_db,
                    use_cache=True,
                    engine=base_engine,
                    language=language,
                )
            else:
                voice_prompt = {
                    "voice_type": "preset",
                    "preset_engine": base_engine,
                    "preset_voice_id": base_voice_id,
                }

            await history.update_generation_status(generation_id, "generating", bg_db)
            trim_fn = trim_tts_output if engine_needs_trim(base_engine) else None

            gen_kwargs: dict = dict(
                language=language,
                seed=seed,
                instruct=instruct,
                trim_fn=trim_fn,
            )
            if max_chunk_chars is not None:
                gen_kwargs["max_chunk_chars"] = max_chunk_chars
            if crossfade_ms is not None:
                gen_kwargs["crossfade_ms"] = crossfade_ms

            audio, sample_rate = await generate_chunked(base_model, text, voice_prompt, **gen_kwargs)
        except Exception as e:
            # Attribute the failure to the base-TTS stage so the history error
            # tells the user which half of the chain broke.
            raise _ChainStageError("TTS", e) from e

        # --- Stage 2: unload base engine, then RVC-convert --------------------
        # The base engine and RVC synthesizer may not fit in VRAM together; free
        # the base engine (services/tts lifecycle) before the RVC model loads.
        # unload_model touches torch, so run it off the event loop.
        try:
            await asyncio.to_thread(base_model.unload_model)

            await history.update_generation_status(generation_id, "converting", bg_db)

            rvc_engine = get_rvc_engine()
            audio, sample_rate = await asyncio.to_thread(
                _convert_chained_sync,
                rvc_engine,
                str(model_path),
                index_path,
                audio,
                sample_rate,
                params,
            )
        except Exception as e:
            raise _ChainStageError("Conversion", e) from e
        return audio, sample_rate
    finally:
        rvc_release(lease_owner)


# Convert long base audio through RVC in bounded segments. RVC's f0 (RMVPE) —
# and any other whole-audio step — runs once per convert_audio call; a single
# RMVPE pass over many minutes drifts (its BiGRU degrades over ~10^5 frames), so
# the pitch track worsens as the clip grows and the conversion turns
# progressively metallic. Splitting into short clips keeps every f0 pass fresh.
# Only long audio is segmented; a normal (short) generation still converts in
# one call, so its output is byte-identical to before.
_RVC_SEGMENT_SECONDS = 30
_RVC_SEGMENT_MAX_SECONDS = 45  # convert whole when the base audio is <= this


def _silence_aware_bounds(audio, sr, seg_len, search):
    """Yield ``(start, end)`` sample bounds ~``seg_len`` apart, each cut snapped
    to the quietest 20 ms window within ``±search`` of the target so segment
    joins land in pauses rather than mid-word."""
    import numpy as np

    n = len(audio)
    win = max(1, int(sr * 0.02))
    bounds = []
    start = 0
    while start < n:
        target = start + seg_len
        if target >= n - search:
            bounds.append((start, n))
            break
        lo = max(target - search, start + win + 1)
        hi = min(target + search, n - win)
        if hi <= lo:
            cut = target
        else:
            region = np.abs(audio[lo : hi + win].astype(np.float64))
            csum = np.concatenate(([0.0], np.cumsum(region)))
            m = hi - lo
            energy = csum[win : m + win] - csum[0:m]
            cut = lo + int(np.argmin(energy)) + win // 2
        bounds.append((start, cut))
        start = cut
    return bounds


def _convert_chained_sync(rvc_engine, model_path, index_path, audio, sr, params):
    """Blocking load + in-memory convert on the shared RVC engine (thread).

    Long base audio is converted in silence-snapped segments (see
    ``_silence_aware_bounds``) so RVC's per-call f0 extraction never drifts; the
    converted segments are concatenated (cuts are in pauses, so a hard join has
    no click). Unloads the RVC stack in a finally (success or failure) so the
    next plain TTS generation does not coexist with a resident synthesizer +
    ContentVec + RMVPE + Crepe. Runs in a worker thread, so unload() stays off
    the event loop.
    """
    import numpy as np

    def _convert(segment):
        return rvc_engine.convert_audio(
            segment,
            sr,
            f0_up_key=params["f0_up_key"],
            f0_method=params["f0_method"],
            index_rate=params["index_rate"],
            rms_mix_rate=params["rms_mix_rate"],
            protect=params["protect"],
        )

    try:
        rvc_engine.load(model_path, index_path)

        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) <= int(sr * _RVC_SEGMENT_MAX_SECONDS):
            return _convert(audio)

        seg_len = int(sr * _RVC_SEGMENT_SECONDS)
        search = int(sr * 2.0)
        parts = []
        out_sr = int(sr)
        for s, e in _silence_aware_bounds(audio, sr, seg_len, search):
            converted, out_sr = _convert(audio[s:e])
            parts.append(np.asarray(converted, dtype=np.float32))
        return np.concatenate(parts), out_sr
    finally:
        rvc_engine.unload()


def _notify_speak_end(generation_id: str, *, status: str) -> None:
    """Publish a speak-end event; the frontend ignores unknown ids."""
    try:
        from ..mcp_server import events as mcp_events

        mcp_events.publish(
            "speak-end",
            {"generation_id": generation_id, "status": status},
        )
    except Exception:
        # Never let event pub/sub break generation completion.
        pass


def _save_generate(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    effects_chain: Optional[list],
    save_audio,
    db,
) -> str:
    """Save clean version and optionally an effects-processed version.

    Returns the final audio path (processed if effects were applied,
    otherwise clean).
    """
    from . import versions as versions_mod

    clean_audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(clean_audio_path), sample_rate)

    has_effects = effects_chain and any(e.get("enabled", True) for e in effects_chain)

    versions_mod.create_version(
        generation_id=generation_id,
        label="original",
        audio_path=config.to_storage_path(clean_audio_path),
        db=db,
        effects_chain=None,
        is_default=not has_effects,
    )

    final_audio_path = str(clean_audio_path)

    if has_effects:
        from ..utils.effects import apply_effects, validate_effects_chain

        assert effects_chain is not None

        error_msg = validate_effects_chain(effects_chain)
        if error_msg:
            import logging
            logging.getLogger(__name__).warning("invalid effects chain, skipping: %s", error_msg)
            versions_mod.set_default_version(
                versions_mod.list_versions(generation_id, db)[0].id, db
            )
        else:
            processed_audio = apply_effects(audio, sample_rate, effects_chain)
            processed_path = config.get_generations_dir() / f"{generation_id}_processed.wav"
            save_audio(processed_audio, str(processed_path), sample_rate)
            final_audio_path = str(processed_path)
            versions_mod.create_version(
                generation_id=generation_id,
                label="version-2",
                audio_path=config.to_storage_path(processed_path),
                db=db,
                effects_chain=effects_chain,
                is_default=True,
            )

    return config.to_storage_path(final_audio_path)


def _save_retry(
    *,
    generation_id: str,
    audio,
    sample_rate: int,
    save_audio,
) -> str:
    """Save retry output -- single file, no versions.

    Returns the audio path.
    """
    audio_path = config.get_generations_dir() / f"{generation_id}.wav"
    save_audio(audio, str(audio_path), sample_rate)
    return config.to_storage_path(audio_path)


async def generate_audio_sync(
    *,
    profile_id: str,
    text: str,
    language: str,
    engine: str,
    model_size: str,
    seed: Optional[int] = None,
    instruct: Optional[str] = None,
    normalize: bool = True,
    max_chunk_chars: Optional[int] = None,
    crossfade_ms: Optional[int] = None,
) -> bytes:
    """Run a TTS generation synchronously and return the resulting wav bytes.

    Unlike :func:`run_generation`, this path does not touch the
    ``generations`` table, enqueue work, or write anything to the
    generations directory. It's used by ``POST /profiles/{id}/speak``
    when the caller passes ``persist=false`` — they just want the audio
    back in the HTTP response without polluting their history.

    Loads the engine model on demand, runs ``generate_chunked``, optional
    normalize, then encodes in-memory via :func:`tts.audio_to_wav_bytes`
    (same helper ``/generate/stream`` uses).
    """
    from ..backends import (
        engine_needs_trim,
        engine_retries_runaway,
        get_tts_backend_for_engine,
        load_engine_model,
    )
    from ..utils.chunked_tts import generate_chunked
    from ..utils.audio import has_tts_runaway, normalize_audio, trim_tts_output
    from . import tts

    bg_db = next(get_db())
    try:
        tts_model = get_tts_backend_for_engine(engine)
        await load_engine_model(engine, model_size)

        voice_prompt = await profiles.create_voice_prompt_for_profile(
            profile_id,
            bg_db,
            use_cache=True,
            engine=engine,
            language=language,
        )
    finally:
        bg_db.close()

    trim_fn = trim_tts_output if engine_needs_trim(engine) else None
    runaway_detector = has_tts_runaway if engine_retries_runaway(engine) else None

    gen_kwargs: dict = dict(
        language=language,
        seed=seed,
        instruct=instruct,
        trim_fn=trim_fn,
        runaway_detector=runaway_detector,
    )
    if max_chunk_chars is not None:
        gen_kwargs["max_chunk_chars"] = max_chunk_chars
    if crossfade_ms is not None:
        gen_kwargs["crossfade_ms"] = crossfade_ms

    audio, sample_rate = await generate_chunked(
        tts_model, text, voice_prompt, **gen_kwargs
    )

    if normalize:
        audio = normalize_audio(audio)

    return tts.audio_to_wav_bytes(audio, sample_rate)


def _save_regenerate(
    *,
    generation_id: str,
    version_id: Optional[str],
    audio,
    sample_rate: int,
    save_audio,
    db,
) -> str:
    """Save regeneration output as a new version with auto-label.

    Returns the audio path.
    """
    from . import versions as versions_mod

    import uuid as _uuid

    suffix = _uuid.uuid4().hex[:8]
    audio_path = config.get_generations_dir() / f"{generation_id}_{suffix}.wav"
    save_audio(audio, str(audio_path), sample_rate)

    # Count via DB query rather than list length to avoid TOCTOU race
    from ..database import GenerationVersion as DBGenerationVersion

    count = db.query(DBGenerationVersion).filter_by(generation_id=generation_id).count()
    label = f"take-{count + 1}"

    versions_mod.create_version(
        generation_id=generation_id,
        label=label,
        audio_path=config.to_storage_path(audio_path),
        db=db,
        effects_chain=None,
        is_default=True,
    )

    return config.to_storage_path(audio_path)
