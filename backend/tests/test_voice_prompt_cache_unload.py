import pytest
import torch

from backend import backends
from backend.backends import ModelConfig
from backend.services import tts
from backend.utils import cache as cache_utils


@pytest.fixture(autouse=True)
def clear_prompt_memory_cache():
    cache_utils._memory_cache.clear()
    yield
    cache_utils._memory_cache.clear()


class FakeLoadedBackend:
    def __init__(self, model_size: str = "default", loaded: bool = True):
        self._current_model_size = model_size
        self.unloaded = False
        self._loaded = loaded

    def is_loaded(self):
        return self._loaded

    def unload_model(self):
        self.unloaded = True
        self._loaded = False


def test_clear_voice_prompt_memory_cache_preserves_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_utils, "_get_cache_dir", lambda: tmp_path)
    cache_key = "voice-prompt"
    prompt = torch.tensor([1.0, 2.0, 3.0])

    cache_utils.cache_voice_prompt(cache_key, prompt)
    assert cache_utils.get_cached_voice_prompt(cache_key) is prompt

    cache_utils.clear_voice_prompt_memory_cache()

    cached_prompt = cache_utils.get_cached_voice_prompt(cache_key)
    assert torch.equal(cached_prompt, prompt)
    assert (tmp_path / f"{cache_key}.prompt").exists()


def test_unload_tts_model_clears_voice_prompt_memory_cache(monkeypatch):
    backend = FakeLoadedBackend()
    cache_utils._memory_cache["prompt"] = torch.tensor([1.0])
    monkeypatch.setattr(tts, "get_tts_backend", lambda: backend)

    tts.unload_tts_model()

    assert backend.unloaded is True
    assert cache_utils._memory_cache == {}


def test_unload_tts_model_keeps_prompt_memory_when_model_was_not_loaded(monkeypatch):
    backend = FakeLoadedBackend(loaded=False)
    prompt = torch.tensor([1.0])
    cache_utils._memory_cache["prompt"] = prompt
    monkeypatch.setattr(tts, "get_tts_backend", lambda: backend)

    tts.unload_tts_model()

    assert backend.unloaded is True
    assert cache_utils._memory_cache["prompt"] is prompt


def test_unload_model_by_config_clears_prompt_memory_for_custom_voice(monkeypatch):
    backend = FakeLoadedBackend(model_size="1.7B")
    cache_utils._memory_cache["prompt"] = torch.tensor([1.0])
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda engine: backend)

    unloaded = backends.unload_model_by_config(
        ModelConfig(
            model_name="qwen-custom-voice-1.7B",
            display_name="Qwen CustomVoice 1.7B",
            engine="qwen_custom_voice",
            model_size="1.7B",
        )
    )

    assert unloaded is True
    assert backend.unloaded is True
    assert cache_utils._memory_cache == {}


def test_unload_model_by_config_clears_prompt_memory_for_generic_tts(monkeypatch):
    backend = FakeLoadedBackend()
    cache_utils._memory_cache["prompt"] = torch.tensor([1.0])
    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda engine: backend)

    unloaded = backends.unload_model_by_config(
        ModelConfig(
            model_name="voxtral-4b-tts-4bit",
            display_name="Voxtral 4B TTS",
            engine="voxtral",
        )
    )

    assert unloaded is True
    assert backend.unloaded is True
    assert cache_utils._memory_cache == {}


def test_unload_model_by_config_does_not_clear_prompt_memory_for_llm(monkeypatch):
    backend = FakeLoadedBackend(model_size="4B")
    prompt = torch.tensor([1.0])
    cache_utils._memory_cache["prompt"] = prompt
    monkeypatch.setattr("backend.services.llm.get_llm_model", lambda: backend)

    unloaded = backends.unload_model_by_config(
        ModelConfig(
            model_name="qwen3-4b",
            display_name="Qwen3 4B",
            engine="qwen_llm",
            model_size="4B",
        )
    )

    assert unloaded is True
    assert backend.unloaded is True
    assert cache_utils._memory_cache["prompt"] is prompt
