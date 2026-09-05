"""Chatterbox multilingual TTS backend for Apple Silicon MLX."""

import logging
from pathlib import Path
from typing import ClassVar

import numpy as np

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

CHATTERBOX_MLX_HF_REPO = "mlx-community/chatterbox-multilingual-v3"

_CHATTERBOX_MLX_WEIGHT_FILES = ["model.safetensors", "config.json", "tokenizer.json"]


class ChatterboxMLXTTSBackend:
    """Chatterbox Multilingual TTS backend using mlx-audio on Metal."""

    max_chunk_chars = 500

    _DEFAULTS: ClassVar[dict] = {
        "exaggeration": 0.1,
        "cfg_weight": 0.5,
        "temperature": 0.8,
        "repetition_penalty": 1.2,
    }

    def __init__(self):
        self.model = None
        self.model_size = "default"

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "default") -> str:
        return CHATTERBOX_MLX_HF_REPO

    def _is_model_cached(self, model_size: str = "default") -> bool:
        return is_model_cached(
            CHATTERBOX_MLX_HF_REPO,
            required_files=_CHATTERBOX_MLX_WEIGHT_FILES,
        )

    def _ensure_loaded_sync(self) -> None:
        if self.model is not None:
            return
        self._load_model_sync()

    async def load_model(self, model_size: str = "default") -> None:
        ensure_realtime_stream_not_active("Chatterbox MLX model loading")
        await _run_on_mlx_thread(self._ensure_loaded_sync)

    def _load_model_sync(self) -> None:
        is_cached = self._is_model_cached()

        with model_load_progress("chatterbox-tts", is_cached):
            from huggingface_hub import snapshot_download
            from mlx_audio.tts.models.chatterbox.chatterbox import Model

            logger.info("Loading Chatterbox Multilingual TTS on MLX from %s...", CHATTERBOX_MLX_HF_REPO)
            ckpt_dir = snapshot_download(CHATTERBOX_MLX_HF_REPO)
            self.model = Model.from_pretrained(ckpt_dir)

        logger.info("Chatterbox Multilingual TTS (MLX) loaded successfully")

    def unload_model(self) -> None:
        _run_on_mlx_thread_blocking(self._unload_model_sync)

    def _unload_model_sync(self) -> None:
        if self.model is None:
            return

        try:
            import mlx.core as mx

            active_before = mx.get_active_memory() / 1e6
            cache_before = mx.get_cache_memory() / 1e6
            del self.model
            self.model = None
            active_after_del = mx.get_active_memory() / 1e6
            mx.clear_cache()
            logger.info(
                "Chatterbox MLX unloaded - Metal MB: active %.0f->%.0f (after clear %.0f), cache %.0f->%.0f",
                active_before,
                active_after_del,
                mx.get_active_memory() / 1e6,
                cache_before,
                mx.get_cache_memory() / 1e6,
            )
        except Exception:
            del self.model
            self.model = None
            logger.debug("mlx cache not cleared while unloading Chatterbox MLX", exc_info=True)
            logger.info("Chatterbox MLX unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
        language: str | None = None,
    ) -> tuple[dict, bool]:
        return {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        ensure_realtime_stream_not_active("Chatterbox MLX generation")

        ref_audio = voice_prompt.get("ref_audio")
        if ref_audio and not Path(ref_audio).exists():
            logger.warning("Reference audio not found: %s", ref_audio)
            ref_audio = None

        def _generate_sync() -> tuple[np.ndarray, int]:
            self._ensure_loaded_sync()

            if seed is not None:
                import mlx.core as mx

                np.random.seed(seed)
                mx.random.seed(seed)

            logger.info("[Chatterbox MLX] Generating: lang=%s", language)

            chunks = [
                np.asarray(result.audio, dtype=np.float32).reshape(-1)
                for result in self.model.generate(
                    text,
                    ref_audio=ref_audio,
                    lang_code=language,
                    verbose=False,
                    **self._DEFAULTS,
                )
            ]
            audio = np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)
            sample_rate = getattr(self.model, "sr", None) or 24000
            return audio, int(sample_rate)

        return await _run_on_mlx_thread(_generate_sync)
