"""
Voxtral TTS backend implementation.

Wraps Mistral's Voxtral-4B-TTS via the MLX community 4-bit conversion for
local Apple Silicon speech generation. Voxtral uses built-in preset voices
rather than Voicebox-style zero-shot voice cloning.
"""

import logging

import numpy as np

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

    async def load_model(self, model_size: str = "default") -> None:
        ensure_realtime_stream_not_active("Voxtral model loading")
        await _run_on_mlx_thread(self._ensure_loaded_sync)

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

        def _generate_sync() -> tuple[np.ndarray, int]:
            self._ensure_loaded_sync()

            if seed is not None:
                import mlx.core as mx

                np.random.seed(seed)
                mx.random.seed(seed)

            audio_chunks: list[np.ndarray] = []
            sample_rate = VOXTRAL_SAMPLE_RATE

            for result in self.model.generate(text=text, voice=voice_name):
                chunk = getattr(result, "audio", result)
                if chunk is None:
                    continue
                audio_chunks.append(np.asarray(chunk, dtype=np.float32).reshape(-1))
                sample_rate = getattr(result, "sample_rate", sample_rate) or sample_rate

            if not audio_chunks:
                return np.zeros(VOXTRAL_SAMPLE_RATE, dtype=np.float32), VOXTRAL_SAMPLE_RATE

            return np.concatenate(audio_chunks).astype(np.float32), int(sample_rate)

        return await _run_on_mlx_thread(_generate_sync)
