"""
MLX backend implementation for TTS and STT using mlx-audio.
"""

import asyncio
import contextvars
import functools
import logging
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

# PATCH: Import and apply offline patch BEFORE any huggingface_hub usage
# This prevents mlx_audio from making network requests when models are cached
from ..utils.hf_offline_patch import ensure_original_qwen_config_cached, patch_huggingface_hub_offline

patch_huggingface_hub_offline()
ensure_original_qwen_config_cached()

from ..utils.audio import estimate_max_new_tokens  # noqa: E402
from ..utils.cache import cache_voice_prompt, get_cache_key, get_cached_voice_prompt  # noqa: E402
from . import LANGUAGE_CODE_TO_NAME, WHISPER_HF_REPOS  # noqa: E402
from .base import (  # noqa: E402
    combine_voice_prompts as _combine_voice_prompts,
    is_model_cached,
    model_load_progress,
)

logger = logging.getLogger(__name__)

# MLX streams are thread-local and mlx-audio caches one on the model at load
# time, so model load and inference must run on the same OS thread.
_mlx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
_mlx_worker_state = threading.local()


def _mark_mlx_worker[T](func: Callable[..., T], *args: object, **kwargs: object) -> T:
    _mlx_worker_state.active = True
    try:
        return func(*args, **kwargs)
    finally:
        _mlx_worker_state.active = False


async def _run_on_mlx_thread[T](func: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Run a blocking MLX call on the single dedicated MLX worker thread."""
    loop = asyncio.get_running_loop()
    ctx = contextvars.copy_context()
    return await loop.run_in_executor(
        _mlx_executor,
        functools.partial(_mark_mlx_worker, ctx.run, func, *args, **kwargs),
    )


def _run_on_mlx_thread_blocking[T](func: Callable[..., T], *args: object, **kwargs: object) -> T:
    """Synchronously run a blocking MLX call on the dedicated MLX worker."""
    if getattr(_mlx_worker_state, "active", False):
        return func(*args, **kwargs)
    ctx = contextvars.copy_context()
    return _mlx_executor.submit(_mark_mlx_worker, ctx.run, func, *args, **kwargs).result()


def ensure_realtime_stream_not_active(operation: str = "MLX inference") -> None:
    """Keep heavy MLX work from starving the realtime RVC stream."""
    try:
        from .rvc import realtime_stream_active

        active = realtime_stream_active()
    except Exception:
        active = False
    if active:
        raise RuntimeError(
            f"{operation} is blocked while Voice Changer real-time streaming is active. "
            "Stop the live stream and try again."
        )


class MLXTTSBackend:
    """MLX-based TTS backend using mlx-audio."""

    max_chunk_chars = 1000

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size = None

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        """
        Get the MLX model path.

        Args:
            model_size: Model size (1.7B or 0.6B)

        Returns:
            HuggingFace Hub model ID for MLX
        """
        mlx_model_map = {
            "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        }

        if model_size not in mlx_model_map:
            raise ValueError(f"Unknown model size: {model_size}")

        hf_model_id = mlx_model_map[model_size]
        logger.info("Will download MLX model from HuggingFace Hub: %s", hf_model_id)

        return hf_model_id

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    def _ensure_loaded_sync(self, model_size: str | None = None) -> None:
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self._current_model_size == model_size:
            return

        if self.model is not None and self._current_model_size != model_size:
            self._unload_model_sync()

        self._load_model_sync(model_size)

    async def load_model_async(self, model_size: str | None = None):
        """
        Lazy load the MLX TTS model.

        Args:
            model_size: Model size to load (1.7B or 0.6B)
        """
        ensure_realtime_stream_not_active("MLX TTS model loading")
        await _run_on_mlx_thread(self._ensure_loaded_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        model_path = self._get_model_path(model_size)
        model_name = f"qwen-tts-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            from mlx_audio.tts import load

            logger.info("Loading MLX TTS model %s...", model_size)

            self.model = load(model_path)

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("MLX TTS model %s loaded successfully", model_size)

    def unload_model(self):
        _run_on_mlx_thread_blocking(self._unload_model_sync)

    def _unload_model_sync(self):
        """Unload the model and release MLX's Metal buffer pool.

        `del self.model` drops MLX's *live* buffers, but MLX keeps a separate
        Metal *cache* pool that is not freed by that — and torch's
        `mps.empty_cache()` (called on the shared unload path) only touches
        torch's pool, never MLX's. The RVC chain unloads the base engine here to
        free VRAM before loading the torch/MPS RVC stack, so release MLX's cache
        too or the two frameworks contend for Metal memory. EXPERIMENTAL: done
        only here (unload), never per-chunk, so the resident non-RVC hot path is
        untouched. The before/after log is the probe — if `active` stays high
        after `del`, a live MLX reference survived elsewhere (clear_cache only
        frees the cache, not live buffers)."""
        if self.model is not None:
            import mlx.core as mx

            active_before = mx.get_active_memory() / 1e6
            cache_before = mx.get_cache_memory() / 1e6
            del self.model
            self.model = None
            self._current_model_size = None
            active_after_del = mx.get_active_memory() / 1e6
            mx.clear_cache()
            logger.info(
                "MLX TTS model unloaded — Metal MB: active %.0f->%.0f (after clear %.0f), cache %.0f->%.0f, peak %.0f",
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
        """
        Create voice prompt from reference audio.

        MLX backend stores voice prompt as a dict with audio path and text.
        The actual voice prompt processing happens during generation.

        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of reference audio
            use_cache: Whether to use cached prompt if available

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        await self.load_model_async(None)

        # Check cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cached_prompt = get_cached_voice_prompt(cache_key)
            if cached_prompt is not None and isinstance(cached_prompt, dict):
                # Validate that the cached audio file still exists
                cached_audio_path = cached_prompt.get("ref_audio") or cached_prompt.get("ref_audio_path")
                if cached_audio_path and Path(cached_audio_path).exists():
                    return cached_prompt, True
                # Cached file no longer exists, invalidate cache
                logger.warning("Cached audio file not found: %s, regenerating prompt", cached_audio_path)

        # MLX voice prompt format - store audio path and text
        # The model will process this during generation
        voice_prompt_items = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }

        # Cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cache_voice_prompt(cache_key, voice_prompt_items)

        return voice_prompt_items, False

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: int | None = None,
        instruct: str | None = None,
    ) -> tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary with ref_audio and ref_text
            language: Language code (en or zh) - may not be fully supported by MLX
            seed: Random seed for reproducibility
            instruct: Natural language instruction (may not be supported by MLX)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        logger.info("Generating audio for text: %s", text)
        ensure_realtime_stream_not_active("MLX TTS generation")

        def _generate_sync():
            """Run synchronous generation in thread pool."""
            # MLX generate() returns a generator yielding GenerationResult objects
            audio_chunks = []
            sample_rate = 24000
            lang = LANGUAGE_CODE_TO_NAME.get(language, "auto")

            # Set seed if provided (MLX uses numpy random)
            if seed is not None:
                import mlx.core as mx

                np.random.seed(seed)
                mx.random.seed(seed)

            # Extract voice prompt info
            ref_audio = voice_prompt.get("ref_audio") or voice_prompt.get("ref_audio_path")
            ref_text = voice_prompt.get("ref_text", "")
            max_tokens = estimate_max_new_tokens(text)

            # Validate that the audio file exists
            if ref_audio and not Path(ref_audio).exists():
                logger.warning("Audio file not found: %s", ref_audio)
                logger.warning("This may be due to a cached voice prompt referencing a deleted temp file.")
                logger.warning("Regenerating without voice prompt.")
                ref_audio = None

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state. Forcing offline here (previously used to avoid lazy
            # mlx_audio lookups hanging when the network drops mid-inference,
            # issue #462) regressed online users because libraries make
            # legitimate metadata calls during generation.
            try:
                if ref_audio:
                    # Check if generate accepts ref_audio parameter
                    import inspect

                    sig = inspect.signature(self.model.generate)
                    if "ref_audio" in sig.parameters:
                        # Generate with voice cloning
                        for result in self.model.generate(
                            text,
                            ref_audio=ref_audio,
                            ref_text=ref_text,
                            lang_code=lang,
                            max_tokens=max_tokens,
                        ):
                            audio_chunks.append(np.array(result.audio))
                            sample_rate = result.sample_rate
                    else:
                        # Fallback: generate without voice cloning
                        for result in self.model.generate(text, lang_code=lang, max_tokens=max_tokens):
                            audio_chunks.append(np.array(result.audio))
                            sample_rate = result.sample_rate
                else:
                    # No voice prompt, generate normally
                    for result in self.model.generate(text, lang_code=lang, max_tokens=max_tokens):
                        audio_chunks.append(np.array(result.audio))
                        sample_rate = result.sample_rate
            except Exception as e:
                # If voice cloning fails, try without it
                logger.warning("Voice cloning failed, generating without voice prompt: %s", e)
                for result in self.model.generate(text, lang_code=lang, max_tokens=max_tokens):
                    audio_chunks.append(np.array(result.audio))
                    sample_rate = result.sample_rate

            # Concatenate all chunks
            if audio_chunks:
                audio = np.concatenate([np.asarray(chunk, dtype=np.float32) for chunk in audio_chunks])
            else:
                # Fallback: empty audio
                audio = np.array([], dtype=np.float32)

            return audio, sample_rate

        def _load_and_generate():
            self._ensure_loaded_sync(None)
            return _generate_sync()

        audio, sample_rate = await _run_on_mlx_thread(_load_and_generate)

        return audio, sample_rate


class MLXSTTBackend:
    """MLX-based STT backend using mlx-audio Whisper."""

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        hf_repo = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
        return is_model_cached(hf_repo, weight_extensions=(".safetensors", ".bin", ".npz"))

    def _ensure_loaded_sync(self, model_size: str | None = None) -> None:
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self.model_size == model_size:
            return

        if self.model is not None:
            self._unload_model_sync()

        self._load_model_sync(model_size)

    async def load_model_async(self, model_size: str | None = None):
        """
        Lazy load the MLX Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        ensure_realtime_stream_not_active("MLX Whisper model loading")
        await _run_on_mlx_thread(self._ensure_loaded_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        progress_model_name = f"whisper-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(progress_model_name, is_cached):
            from mlx_audio.stt import load

            model_name = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
            logger.info("Loading MLX Whisper model %s...", model_size)

            self.model = load(model_name)

        self.model_size = model_size
        logger.info("MLX Whisper model %s loaded successfully", model_size)

    def unload_model(self):
        _run_on_mlx_thread_blocking(self._unload_model_sync)

    def _unload_model_sync(self):
        """Unload the model to free memory."""
        if self.model is not None:
            import mlx.core as mx

            del self.model
            self.model = None
            mx.clear_cache()
            logger.info("MLX Whisper model unloaded")

    async def transcribe(
        self,
        audio_path: str,
        language: str | None = None,
        model_size: str | None = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Optional language hint
            model_size: Optional model size override

        Returns:
            Transcribed text
        """
        ensure_realtime_stream_not_active("MLX Whisper transcription")

        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # MLX Whisper transcription using generate method
            # The generate method accepts audio path directly
            decode_options = {}
            if language:
                decode_options["language"] = language

            # Inference runs with the process's default HF_HUB_OFFLINE
            # state — see the comment in MLXTTSBackend.generate for the
            # regression this revert fixes (issue #462).
            result = self.model.generate(str(audio_path), **decode_options)

            # Extract text from result
            if isinstance(result, str):
                return result.strip()
            if isinstance(result, dict):
                return result.get("text", "").strip()
            if hasattr(result, "text"):
                return result.text.strip()
            return str(result).strip()

        def _load_and_transcribe():
            self._ensure_loaded_sync(model_size)
            return _transcribe_sync()

        return await _run_on_mlx_thread(_load_and_transcribe)
