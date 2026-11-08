"""
Voxtral TTS backend implementation.

Wraps Mistral's Voxtral-4B-TTS via the MLX community 4-bit conversion for
local Apple Silicon speech generation. Voxtral uses built-in preset voices
rather than Voicebox-style zero-shot voice cloning.
"""

import gc
import logging
import math
import traceback
from contextlib import contextmanager

import numpy as np

from ..services.voxtral_memory import VoxtralMemoryGuard
from ..utils.platform_detect import get_backend_type
from .base import (
    combine_voice_prompts as _combine_voice_prompts,
    is_model_cached,
    model_load_progress,
)
from .mlx_backend import (
    _run_on_mlx_thread,
    _run_on_mlx_thread_blocking,
    ensure_realtime_stream_not_active,
)

logger = logging.getLogger(__name__)

VOXTRAL_HF_REPO = "mlx-community/Voxtral-4B-TTS-2603-mlx-4bit"
VOXTRAL_MODEL_NAME = "voxtral-4b-tts-4bit"
VOXTRAL_SAMPLE_RATE = 24000
VOXTRAL_DEFAULT_VOICE = "casual_male"
VOXTRAL_STREAMING_INTERVAL = 2.0
VOXTRAL_FRAME_SECONDS = 0.08


def _max_audio_tokens(text: str) -> int:
    # Allow slow speech plus startup pauses, but never minutes of runaway audio
    # for one short text segment (mlx-audio defaults to 4096 frames / 328 s).
    seconds = min(60.0, max(15.0, len(text) / 8 + 5))
    return math.ceil(seconds / VOXTRAL_FRAME_SECONDS)


# All available Voxtral voices: (voice_id, display_name, gender, lang_code)
VOXTRAL_VOICES = [
    ("casual_male", "Casual Male", "male", "en"),
    ("casual_female", "Casual Female", "female", "en"),
    ("cheerful_female", "Cheerful Female", "female", "en"),
    ("neutral_male", "Neutral Male", "male", "en"),
    ("neutral_female", "Neutral Female", "female", "en"),
    ("fr_male", "French Male", "male", "fr"),
    ("fr_female", "French Female", "female", "fr"),
    ("es_male", "Spanish Male", "male", "es"),
    ("es_female", "Spanish Female", "female", "es"),
    ("de_male", "German Male", "male", "de"),
    ("de_female", "German Female", "female", "de"),
    ("it_male", "Italian Male", "male", "it"),
    ("it_female", "Italian Female", "female", "it"),
    ("pt_male", "Portuguese Male", "male", "pt"),
    ("pt_female", "Portuguese Female", "female", "pt"),
    ("nl_male", "Dutch Male", "male", "nl"),
    ("nl_female", "Dutch Female", "female", "nl"),
    ("ar_male", "Arabic Male", "male", "ar"),
    ("hi_male", "Hindi Male", "male", "hi"),
    ("hi_female", "Hindi Female", "female", "hi"),
]

VOXTRAL_VOICE_IDS = {voice_id for voice_id, _name, _gender, _lang in VOXTRAL_VOICES}


class VoxtralTTSBackend:
    """Voxtral-4B TTS backend for Apple Silicon MLX."""

    # Long agent answers are split at sentence boundaries. Voxtral's preset
    # generations can come back with noticeably different loudness per chunk;
    # match them before concatenation so the final global normalization does
    # not push a later chunk toward clipping.
    match_chunk_loudness = True
    stabilize_chunk_loudness = True
    chunk_loudness_max_gain_db = 6.0

    preserve_seed_across_chunks = True
    max_chunk_chars = 320

    def __init__(self):
        self.model = None
        self.model_size = "default"

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return VOXTRAL_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(
            VOXTRAL_HF_REPO,
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    def _ensure_loaded_sync(self) -> None:
        if self.model is not None:
            return
        self._load_model_sync()

    @contextmanager
    def _memory_guard(self, mx, operation):
        try:
            with VoxtralMemoryGuard(mx, operation) as guard:
                yield guard
        except Exception as exc:
            # Also release a resident model when guard entry refuses the job.
            traceback.clear_frames(exc.__traceback__)
            self._unload_model_sync()
            # A failed load may allocate buffers without ever assigning model.
            gc.collect()
            mx.clear_cache()
            raise

    async def load_model(self, model_size: str = "default") -> None:
        ensure_realtime_stream_not_active("Voxtral model loading")

        def _load_guarded():
            import mlx.core as mx

            with self._memory_guard(mx, "load") as guard:
                try:
                    self._ensure_loaded_sync()
                    guard.check()
                except Exception as exc:
                    traceback.clear_frames(exc.__traceback__)
                    self._unload_model_sync()
                    raise

        await _run_on_mlx_thread(_load_guarded)

    def _load_model_sync(self) -> None:
        if get_backend_type() != "mlx":
            raise RuntimeError("Voxtral TTS requires the Apple Silicon MLX backend.")

        is_cached = self._is_model_cached()
        with model_load_progress(VOXTRAL_MODEL_NAME, is_cached):
            from mlx_audio.tts import load

            logger.info("Loading Voxtral 4B TTS from %s...", VOXTRAL_HF_REPO)
            self.model = load(VOXTRAL_HF_REPO)

        logger.info("Voxtral 4B TTS loaded successfully")

    def unload_model(self) -> None:
        _run_on_mlx_thread_blocking(self._unload_model_sync)

    def _unload_model_sync(self) -> None:
        if self.model is None:
            return

        import mlx.core as mx

        active_before = mx.get_active_memory() / 1e6
        cache_before = mx.get_cache_memory() / 1e6
        del self.model
        self.model = None
        gc.collect()
        active_after_del = mx.get_active_memory() / 1e6
        mx.clear_cache()
        logger.info(
            "Voxtral unloaded — Metal MB: active %.0f->%.0f (after clear %.0f), cache %.0f->%.0f",
            active_before,
            active_after_del,
            mx.get_active_memory() / 1e6,
            cache_before,
            mx.get_cache_memory() / 1e6,
        )

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
        language: str | None = None,
    ) -> tuple[dict, bool]:
        return {
            "voice_type": "preset",
            "preset_engine": "voxtral",
            "preset_voice_id": VOXTRAL_DEFAULT_VOICE,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(
            audio_paths,
            reference_texts,
            sample_rate=VOXTRAL_SAMPLE_RATE,
        )

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        ensure_realtime_stream_not_active("Voxtral generation")

        voice_name = voice_prompt.get("preset_voice_id") or VOXTRAL_DEFAULT_VOICE
        if voice_name not in VOXTRAL_VOICE_IDS:
            raise ValueError(f"Unknown Voxtral voice: {voice_name}")
        if len(text) > self.max_chunk_chars:
            raise ValueError(f"Voxtral requires text segments of at most {self.max_chunk_chars} characters")

        def _generate_sync() -> tuple[np.ndarray, int]:
            import mlx.core as mx

            with self._memory_guard(mx, f"generate chars={len(text)} voice={voice_name}") as guard:
                results = None
                result = chunk = None
                audio_chunks: list[np.ndarray] = []
                try:
                    self._ensure_loaded_sync()
                    guard.check()
                    if seed is not None:
                        np.random.seed(seed)
                        mx.random.seed(seed)

                    max_tokens = _max_audio_tokens(text)
                    frames = 0
                    samples = 0
                    # Non-streaming codec decoding builds quadratic attention
                    # matrices over the entire audio. Decode bounded windows;
                    # mlx-audio trims its context, so append each result once.
                    results = self.model.generate(
                        text=text,
                        voice=voice_name,
                        max_tokens=max_tokens,
                        stream=True,
                        streaming_interval=VOXTRAL_STREAMING_INTERVAL,
                    )
                    for result in results:
                        guard.check()
                        frames += getattr(result, "token_count", 0)
                        chunk = getattr(result, "audio", result)
                        if chunk is None:
                            continue
                        sample_rate = getattr(result, "sample_rate", VOXTRAL_SAMPLE_RATE)
                        if sample_rate != VOXTRAL_SAMPLE_RATE:
                            raise RuntimeError(f"Unexpected Voxtral sample rate: {sample_rate}")
                        # A copy detaches NumPy from MLX's buffer and lazy graph.
                        chunk = np.array(chunk, dtype=np.float32, copy=True).reshape(-1)
                        samples += chunk.size
                        if frames >= max_tokens or samples >= max_tokens * 1920:
                            raise RuntimeError("Voxtral audio duration limit reached; generation stopped")
                        audio_chunks.append(chunk)
                        result = chunk = None
                        guard.check()

                    if not samples:
                        raise RuntimeError("Voxtral generated no audio")
                    return np.concatenate(audio_chunks), VOXTRAL_SAMPLE_RATE
                except Exception as exc:
                    traceback.clear_frames(exc.__traceback__)
                    self.model = None
                    raise
                finally:
                    if results is not None:
                        results.close()
                    result = chunk = results = None
                    audio_chunks.clear()
                    gc.collect()
                    mx.clear_cache()

        return await _run_on_mlx_thread(_generate_sync)
