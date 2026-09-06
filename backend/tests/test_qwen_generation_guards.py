import numpy as np
import pytest

from backend.backends.mlx_backend import MLXTTSBackend
from backend.backends.pytorch_backend import PyTorchSTTBackend, PyTorchTTSBackend
from backend.backends.qwen_custom_voice_backend import QwenCustomVoiceBackend
from backend.utils.audio import (
    estimate_max_new_tokens,
    exceeds_plausible_speech_duration,
)


class _FakePyTorchBaseModel:
    def __init__(self):
        self.calls: list[dict] = []

    def generate_voice_clone(self, **kwargs):
        self.calls.append(kwargs)
        return [np.ones(16, dtype=np.float32)], 24_000


class _FakeQwenCustomVoiceModel:
    def __init__(self):
        self.calls: list[dict] = []

    def generate_custom_voice(self, **kwargs):
        self.calls.append(kwargs)
        return [np.ones(16, dtype=np.float32)], 24_000


class _FakeMLXResult:
    audio = np.ones(16, dtype=np.float32)
    sample_rate = 24_000


class _FakeMLXModel:
    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, **kwargs})
        yield _FakeMLXResult()


def test_estimate_max_new_tokens_uses_stripped_text_length():
    assert estimate_max_new_tokens("bonjour") == 1500
    assert estimate_max_new_tokens(("x" * 700) + (" " * 2000)) == 8192


def test_duration_detector_flags_implausibly_long_audio():
    audio = np.ones(120 * 1000, dtype=np.float32)

    assert exceeds_plausible_speech_duration(audio, 1000, "short text") is True
    assert exceeds_plausible_speech_duration(audio, 1000, "x" * 500) is False


def test_duration_detector_ignores_whitespace_padding():
    audio = np.ones(120 * 1000, dtype=np.float32)
    text = "short text" + (" " * 2000)

    assert exceeds_plausible_speech_duration(audio, 1000, text) is True


@pytest.mark.asyncio
async def test_pytorch_qwen_base_passes_text_scaled_decode_budget(monkeypatch):
    backend = PyTorchTTSBackend()
    backend.model = _FakePyTorchBaseModel()

    async def _no_load(model_size=None):
        return None

    monkeypatch.setattr(backend, "load_model_async", _no_load)

    await backend.generate("Bonjour le monde.", {}, language="fr")

    assert backend.model.calls[0]["max_new_tokens"] == estimate_max_new_tokens("Bonjour le monde.")


@pytest.mark.asyncio
async def test_qwen_custom_voice_passes_text_scaled_decode_budget(monkeypatch):
    backend = QwenCustomVoiceBackend()
    backend.model = _FakeQwenCustomVoiceModel()

    async def _no_load(model_size=None):
        return None

    monkeypatch.setattr(backend, "load_model_async", _no_load)

    await backend.generate(
        "Bonjour le monde.",
        {"preset_voice_id": "Ryan"},
        language="fr",
    )

    assert backend.model.calls[0]["max_new_tokens"] == estimate_max_new_tokens("Bonjour le monde.")


@pytest.mark.asyncio
async def test_mlx_qwen_passes_text_scaled_decode_budget(monkeypatch):
    backend = MLXTTSBackend()
    backend.model = _FakeMLXModel()

    def _no_load(model_size=None):
        return None

    monkeypatch.setattr(backend, "_ensure_loaded_sync", _no_load)

    await backend.generate("Bonjour le monde.", {}, language="fr")

    assert backend.model.calls[0]["max_tokens"] == estimate_max_new_tokens("Bonjour le monde.")


def test_qwen_backends_cap_voicebox_chunk_size():
    assert MLXTTSBackend.max_chunk_chars == 1000
    assert PyTorchTTSBackend.max_chunk_chars == 1000
    assert QwenCustomVoiceBackend.max_chunk_chars == 1000


def test_pytorch_qwen_backends_request_mps(monkeypatch):
    calls: list[dict] = []

    def _fake_get_torch_device(**kwargs):
        calls.append(kwargs)
        return "mps"

    monkeypatch.setattr("backend.backends.pytorch_backend.get_torch_device", _fake_get_torch_device)
    monkeypatch.setattr(
        "backend.backends.qwen_custom_voice_backend.get_torch_device",
        _fake_get_torch_device,
    )

    assert PyTorchTTSBackend().device == "mps"
    assert PyTorchSTTBackend().device == "mps"
    assert QwenCustomVoiceBackend().device == "mps"
    assert all(call["allow_mps"] is True for call in calls)


def test_qwen_custom_voice_unload_clears_selected_device_cache(monkeypatch):
    backend = QwenCustomVoiceBackend()
    backend.model = object()
    backend.device = "mps"
    calls: list[str] = []

    monkeypatch.setattr(
        "backend.backends.qwen_custom_voice_backend.empty_device_cache",
        lambda device: calls.append(device),
    )

    backend.unload_model()

    assert calls == ["mps"]
