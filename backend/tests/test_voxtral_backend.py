import logging
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from backend.backends import TTS_ENGINES, get_model_config, voxtral_backend as voxtral
from backend.backends.voxtral_backend import (
    VOXTRAL_HF_REPO,
    VOXTRAL_VOICE_IDS,
    VOXTRAL_VOICES,
    VoxtralTTSBackend,
)
from backend.models import GenerationRequest, MCPClientBindingUpsert, SpeakRequest
from backend.services import voxtral_memory as memory
from backend.services.profiles import _get_preset_voice_ids, _preset_voice_language
from backend.tests.test_voxtral_memory import FakeMLX


def test_voxtral_model_config_is_registered():
    cfg = get_model_config("voxtral-4b-tts-4bit")

    assert TTS_ENGINES["voxtral"] == "Voxtral 4B TTS"
    assert cfg is not None
    assert cfg.engine == "voxtral"
    assert cfg.hf_repo_id == VOXTRAL_HF_REPO
    assert cfg.languages == ["en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi"]


def test_voxtral_preset_voice_table_is_exposed_to_profile_validation():
    assert len(VOXTRAL_VOICES) == 20
    assert _get_preset_voice_ids("voxtral") == VOXTRAL_VOICE_IDS
    assert _preset_voice_language("voxtral", "fr_female") == "fr"
    assert _preset_voice_language("voxtral", "hi_male") == "hi"


def test_voxtral_matches_loudness_across_long_form_chunks():
    assert VoxtralTTSBackend.match_chunk_loudness is True
    assert VoxtralTTSBackend.stabilize_chunk_loudness is True
    assert VoxtralTTSBackend.preserve_seed_across_chunks is True


def test_voxtral_is_accepted_by_public_engine_validators():
    generation = GenerationRequest(
        profile_id="profile-id",
        text="Bonjour",
        language="fr",
        engine="voxtral",
    )
    speak = SpeakRequest(text="Bonjour", engine="voxtral")
    binding = MCPClientBindingUpsert(client_id="client-id", default_engine="voxtral")

    assert generation.engine == "voxtral"
    assert speak.engine == "voxtral"
    assert binding.default_engine == "voxtral"


@pytest.fixture
def guarded_backend(monkeypatch):
    mx = FakeMLX()
    monkeypatch.setitem(sys.modules, "mlx.core", mx)
    monkeypatch.setitem(sys.modules, "mlx", SimpleNamespace(core=mx))
    monkeypatch.delenv(memory.MEMORY_ENV, raising=False)
    monkeypatch.setattr(memory, "process_footprint", lambda: 3 * memory.GIB)
    monkeypatch.setattr(memory, "_diagnostic_logger", lambda: logging.getLogger("test-voxtral-memory"))

    async def inline(fn):
        return fn()

    monkeypatch.setattr(voxtral, "_run_on_mlx_thread", inline)
    return VoxtralTTSBackend(), mx


def test_voxtral_token_budget_is_proportional_and_bounded():
    assert voxtral._max_audio_tokens("Bonjour") == 188
    assert voxtral._max_audio_tokens("a" * 320) == 563
    assert voxtral._max_audio_tokens("a" * 5000) == 750


async def test_bounded_codec_windows_are_concatenated_once_and_detached(guarded_backend):
    backend, mx = guarded_backend
    source = np.arange(24, dtype=np.float32)
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        yield SimpleNamespace(audio=source[:12], sample_rate=24000, token_count=1)
        # If the previous output is a view backed by MLX this mutation leaks.
        source[:12] = -1
        yield SimpleNamespace(audio=source[12:], sample_rate=24000, token_count=1)
        yield SimpleNamespace(audio=np.zeros(0), sample_rate=24000, token_count=0, is_final_chunk=True)

    backend.model = SimpleNamespace(generate=generate)
    audio, rate = await backend.generate("Bonjour", {"preset_voice_id": "fr_female"})
    np.testing.assert_array_equal(audio, np.arange(24))
    assert rate == 24000
    assert calls[0]["stream"] is True
    assert calls[0]["streaming_interval"] == 2.0
    assert calls[0]["max_tokens"] == 188
    assert mx.clears >= 2


async def test_runaway_fails_without_returning_truncated_audio_and_releases_model(guarded_backend):
    backend, mx = guarded_backend
    closed = []

    def generate(**kwargs):
        try:
            yield SimpleNamespace(audio=np.ones(8), token_count=kwargs["max_tokens"], sample_rate=24000)
        finally:
            closed.append(True)

    backend.model = SimpleNamespace(generate=generate)
    with pytest.raises(RuntimeError, match="duration limit"):
        await backend.generate("Bonjour", {})
    assert backend.model is None
    assert closed == [True]
    assert mx.get_cache_memory() == 0


async def test_memory_spike_closes_generator_and_allows_subsequent_request(guarded_backend):
    backend, mx = guarded_backend
    closed = []

    def generate(**kwargs):
        try:
            mx.active = 9 * memory.GIB
            yield SimpleNamespace(audio=np.ones(8), token_count=1, sample_rate=24000)
        finally:
            mx.active = 2 * memory.GIB
            closed.append(True)

    backend.model = SimpleNamespace(generate=generate)
    with pytest.raises(memory.VoxtralMemoryError):
        await backend.generate("Bonjour", {})
    assert backend.model is None
    assert closed == [True]
    backend.model = SimpleNamespace(generate=lambda **kwargs: iter_results())
    audio, _ = await backend.generate("Bonjour", {})
    assert audio.size == 8


def iter_results():
    yield SimpleNamespace(audio=np.ones(8), token_count=1, sample_rate=24000)


async def test_direct_long_input_is_rejected_before_loading(guarded_backend):
    backend, _ = guarded_backend
    with pytest.raises(ValueError, match="at most 320"):
        await backend.generate("a" * 321, {})
    assert backend.model is None


async def test_pressure_before_load_releases_resident_model(guarded_backend, monkeypatch):
    backend, mx = guarded_backend
    backend.model = SimpleNamespace(generate=lambda **kwargs: iter_results())
    monkeypatch.setattr(memory, "process_footprint", lambda: 9 * memory.GIB)
    with pytest.raises(memory.VoxtralMemoryError):
        await backend.load_model()
    assert backend.model is None
    assert mx.clears >= 2
