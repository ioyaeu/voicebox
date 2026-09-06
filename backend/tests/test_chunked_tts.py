import numpy as np
import pytest

from backend.utils.chunked_tts import generate_chunked


class _SeedRecordingBackend:
    def __init__(self, preserve_seed_across_chunks: bool = False):
        self.preserve_seed_across_chunks = preserve_seed_across_chunks
        self.seeds: list[int | None] = []

    async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
        self.seeds.append(seed)
        return np.ones(16, dtype=np.float32), 24_000


class _MaxChunkRecordingBackend(_SeedRecordingBackend):
    max_chunk_chars = 20

    def __init__(self):
        super().__init__()
        self.texts: list[str] = []

    async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
        self.texts.append(text)
        return await super().generate(text, voice_prompt, language, seed, instruct)


class _LevelChangingBackend:
    match_chunk_loudness = True

    def __init__(self):
        self.levels = [0.05, 0.2, 0.1]
        self.calls = 0

    async def generate(self, text, voice_prompt, language="en", seed=None, instruct=None):
        level = self.levels[self.calls]
        self.calls += 1
        silence = np.zeros(10, dtype=np.float32)
        speech = np.full(80, level, dtype=np.float32)
        return np.concatenate([silence, speech, silence]), 1000


@pytest.mark.asyncio
async def test_generate_chunked_offsets_seed_by_default():
    backend = _SeedRecordingBackend()

    await generate_chunked(
        backend,
        "First sentence. Second sentence. Third sentence.",
        {},
        seed=42,
        max_chunk_chars=18,
        crossfade_ms=0,
    )

    assert backend.seeds == [42, 43, 44]


@pytest.mark.asyncio
async def test_generate_chunked_can_preserve_seed_for_voice_consistency():
    backend = _SeedRecordingBackend(preserve_seed_across_chunks=True)

    await generate_chunked(
        backend,
        "First sentence. Second sentence. Third sentence.",
        {},
        seed=42,
        max_chunk_chars=18,
        crossfade_ms=0,
    )

    assert backend.seeds == [42, 42, 42]


@pytest.mark.asyncio
async def test_generate_chunked_honors_backend_max_chunk_chars():
    backend = _MaxChunkRecordingBackend()

    await generate_chunked(
        backend,
        "First sentence. Second sentence. Third sentence.",
        {},
        max_chunk_chars=100,
        crossfade_ms=0,
    )

    assert backend.texts == ["First sentence.", "Second sentence.", "Third sentence."]


@pytest.mark.asyncio
async def test_generate_chunked_matches_active_speech_loudness_when_backend_requests_it():
    backend = _LevelChangingBackend()

    audio, sample_rate = await generate_chunked(
        backend,
        "First sentence. Second sentence. Third sentence.",
        {},
        max_chunk_chars=18,
        crossfade_ms=0,
    )

    assert sample_rate == 1000
    assert backend.calls == 3

    speech_rms = [float(np.sqrt(np.mean(audio[start + 10 : start + 90] ** 2))) for start in (0, 100, 200)]
    assert speech_rms == pytest.approx([0.1, 0.1, 0.1], abs=0.003)
