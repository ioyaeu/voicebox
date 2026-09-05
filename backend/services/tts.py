"""
TTS inference module - delegates to backend abstraction layer.
"""

import io

import numpy as np
import soundfile as sf

from ..backends import TTSBackend, get_tts_backend
from ..utils.cache import clear_voice_prompt_memory_cache


def get_tts_model() -> TTSBackend:
    """
    Get TTS backend instance (MLX or PyTorch based on platform).

    Returns:
        TTS backend instance
    """
    return get_tts_backend()


def unload_tts_model():
    """Unload TTS model to free memory."""
    backend = get_tts_backend()
    was_loaded = backend.is_loaded()
    backend.unload_model()
    if was_loaded:
        clear_voice_prompt_memory_cache()


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Convert audio array to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="WAV")
    buffer.seek(0)
    return buffer.read()
