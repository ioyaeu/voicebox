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
