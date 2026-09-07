"""Tests for the single-active-TTS-engine lifecycle."""

import pytest

from backend import backends


class _FakeBackend:
    def __init__(self, loaded: bool):
        self.loaded = loaded
        self.unload_calls = 0

    def is_loaded(self):
        return self.loaded

    def unload_model(self):
        self.unload_calls += 1
        self.loaded = False


@pytest.mark.asyncio
async def test_unload_other_tts_models_keeps_active_engine(monkeypatch):
    active = _FakeBackend(loaded=True)
    stale = _FakeBackend(loaded=True)
    idle = _FakeBackend(loaded=False)
    monkeypatch.setattr(
        backends,
        "_tts_backends",
        {"voxtral": active, "kokoro": stale, "qwen": idle},
    )

    unloaded = await backends.unload_other_tts_models("voxtral")

    assert unloaded == ["kokoro"]
    assert active.unload_calls == 0
    assert stale.unload_calls == 1
    assert idle.unload_calls == 0
