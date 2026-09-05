"""MLX backend for HumeAI TADA on Apple Silicon."""

import importlib
import logging
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .. import config
from ..utils.cache import get_cache_key
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

MLX_TADA_1B_REPO = "HumeAI/mlx-tada-1b"
MLX_TADA_3B_REPO = "HumeAI/mlx-tada-3b"
MLX_TADA_TOKENIZER_REPO = "unsloth/Llama-3.2-1B"

_MLX_TADA_WEIGHT_FILES = [
    "model/weights.safetensors",
    "encoder/weights.safetensors",
    "decoder/weights.safetensors",
    "aligner/weights.safetensors",
]
_TOKENIZER_FILES = ["tokenizer.json", "tokenizer_config.json"]


class MLXTadaBackend:
    """HumeAI TADA backend using the official MLX weights/package."""

    max_chunk_chars = 500

    def __init__(self, model_size: str = "1B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size = None
        self._weights_dir: str | None = None

    def is_loaded(self) -> bool:
        return self.model is not None

    def _get_model_path(self, model_size: str = "1B") -> str:
        model_map = {
            "1B": MLX_TADA_1B_REPO,
            "3B": MLX_TADA_3B_REPO,
        }
        if model_size not in model_map:
            raise ValueError(f"Unknown TADA model size: {model_size}")
        return model_map[model_size]

    def _is_model_cached(self, model_size: str = "1B") -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            required_files=_MLX_TADA_WEIGHT_FILES,
        ) and is_model_cached(MLX_TADA_TOKENIZER_REPO, required_files=_TOKENIZER_FILES)

    def _ensure_loaded_sync(self, model_size: str = "1B") -> None:
        if self.model is not None and self._current_model_size == model_size:
            return
        if self.model is not None:
            self._unload_model_sync()

        self._load_model_sync(model_size)

    async def load_model(self, model_size: str = "1B") -> None:
        ensure_realtime_stream_not_active("MLX TADA model loading")
        await _run_on_mlx_thread(self._ensure_loaded_sync, model_size)

    def _load_model_sync(self, model_size: str = "1B") -> None:
        model_name = "tada-3b-ml" if model_size == "3B" else "tada-1b"
        repo = self._get_model_path(model_size)
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from huggingface_hub import snapshot_download

            logger.info("Downloading MLX TADA weights from %s...", repo)
            weights_dir = snapshot_download(
                repo_id=repo,
                token=None,
                allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
            )

            logger.info("Downloading Llama tokenizer (ungated mirror)...")
            tokenizer_path = snapshot_download(
                repo_id=MLX_TADA_TOKENIZER_REPO,
                token=None,
                allow_patterns=["tokenizer*", "special_tokens*"],
            )

            mlx_tada_model = importlib.import_module("mlx_tada.model")
            with _mlx_tada_tokenizer_redirect(mlx_tada_model, tokenizer_path):
                self.model = mlx_tada_model.TadaForCausalLM.from_weights(
                    weights_dir,
                    quantize=4,
                )

        self._weights_dir = weights_dir
        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("MLX TADA %s loaded successfully from %s", model_size, repo)

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
        self._current_model_size = None
        self._weights_dir = None
        active_after_del = mx.get_active_memory() / 1e6
        mx.clear_cache()
        logger.info(
            "MLX TADA unloaded — Metal MB: active %.0f->%.0f (after clear %.0f), cache %.0f->%.0f, peak %.0f",
            active_before,
            active_after_del,
            mx.get_active_memory() / 1e6,
            cache_before,
            mx.get_cache_memory() / 1e6,
            mx.get_peak_memory() / 1e6,
        )

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
        language: str | None = None,
    ) -> tuple[dict, bool]:
        cache_key = "mlx_tada_" + get_cache_key(
            audio_path,
            f"language={language or 'en'}\n{reference_text}",
        )
        reference_path = _reference_cache_path(cache_key)
        voice_prompt = {
            "reference_path": str(reference_path),
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }

        if use_cache and reference_path.exists():
            return voice_prompt, True

        def _encode_sync() -> None:
            self._ensure_loaded_sync(self.model_size)
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            reference = self.model.load_reference(str(audio_path), reference_text)
            reference.save(str(reference_path))

        await _run_on_mlx_thread(_encode_sync)
        return voice_prompt, False

    async def combine_voice_prompts(
        self,
        audio_paths: list[str],
        reference_texts: list[str],
    ) -> tuple[np.ndarray, str]:
        return await _combine_voice_prompts(audio_paths, reference_texts, sample_rate=24000)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        ensure_realtime_stream_not_active("MLX TADA generation")

        def _generate_sync() -> tuple[np.ndarray, int]:
            self._ensure_loaded_sync(self.model_size)

            if seed is not None:
                import mlx.core as mx

                np.random.seed(seed)
                mx.random.seed(seed)

            mlx_tada_config = importlib.import_module("mlx_tada.config")
            reference_path = voice_prompt.get("reference_path")
            ref_audio = voice_prompt.get("ref_audio")
            ref_text = voice_prompt.get("ref_text", "")

            if reference_path and Path(reference_path).exists():
                reference = mlx_tada_config.Reference.load(reference_path)
            elif ref_audio and Path(ref_audio).exists():
                reference = self.model.load_reference(ref_audio, ref_text)
            else:
                raise ValueError("MLX TADA voice prompt is missing its reference audio")

            options = mlx_tada_config.InferenceOptions(num_flow_matching_steps=10)
            logger.info("[MLX TADA] Generating (%s), text length: %s", language, len(text))
            output = self.model.generate(text, reference, inference_options=options)
            return np.asarray(output.audio, dtype=np.float32), 24000

        return await _run_on_mlx_thread(_generate_sync)


def _reference_cache_path(cache_key: str) -> Path:
    return config.get_cache_dir() / f"{cache_key}.npz"


@contextmanager
def _mlx_tada_tokenizer_redirect(mlx_tada_model, tokenizer_path: str):
    """Redirect mlx-tada's hardcoded gated Llama tokenizer to our local mirror."""
    original_loader = mlx_tada_model.AutoTokenizer

    class _LocalTokenizerLoader:
        @staticmethod
        def from_pretrained(repo_id: str, *args, **kwargs):
            if repo_id == "meta-llama/Llama-3.2-1B":
                return original_loader.from_pretrained(tokenizer_path, *args, **kwargs)
            return original_loader.from_pretrained(repo_id, *args, **kwargs)

    mlx_tada_model.AutoTokenizer = _LocalTokenizerLoader
    try:
        yield
    finally:
        mlx_tada_model.AutoTokenizer = original_loader
