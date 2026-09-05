"""Regression tests for MLX worker-thread affinity."""

import asyncio
import threading
import time

import numpy as np
import pytest

from backend.backends.mlx_backend import (
    MLXSTTBackend,
    MLXTTSBackend,
    _run_on_mlx_thread,
    ensure_realtime_stream_not_active,
)
from backend.backends.mlx_tada_backend import MLXTadaBackend
from backend.backends.qwen_llm_backend import MLXQwenLLMBackend, PyTorchQwenLLMBackend
from backend.backends.rvc import acquire as rvc_acquire, release as rvc_release
from backend.backends.voxtral_backend import VOXTRAL_SAMPLE_RATE, VoxtralTTSBackend


class _AudioResult:
    audio = np.ones(16, dtype=np.float32)
    sample_rate = 24000


class _StreamingAudioModel:
    def __init__(self, owner):
        self.owner = owner

    def generate(self, *args, **kwargs):
        resident = self.owner.model
        assert resident is self, "model changed before audio generation"
        time.sleep(0.02)
        assert self.owner.model is resident, "model changed during audio generation"
        yield _AudioResult()


class _STTModel:
    def __init__(self, owner, model_size: str):
        self.owner = owner
        self.model_size = model_size

    def generate(self, *args, **kwargs) -> str:
        resident = self.owner.model
        assert resident is self, "model changed before transcription"
        time.sleep(0.02)
        assert self.owner.model is resident, "model changed during transcription"
        return self.model_size


class _TadaOutput:
    audio = np.ones(16, dtype=np.float32)


class TestMLXThreadAffinity:
    """MLX model load and inference must stay pinned to one OS thread."""

    @pytest.mark.asyncio
    async def test_calls_share_single_thread_under_concurrency(self):
        seen_thread_ids: set[int] = set()

        def record_thread_id() -> int:
            thread_id = threading.get_ident()
            seen_thread_ids.add(thread_id)
            return thread_id

        results = await asyncio.gather(*(_run_on_mlx_thread(record_thread_id) for _ in range(25)))

        assert len(set(results)) == 1
        assert len(seen_thread_ids) == 1

    @pytest.mark.asyncio
    async def test_thread_local_state_survives_between_calls(self):
        local_state = threading.local()

        def load_model() -> int:
            local_state.stream_id = "stream-created-at-load"
            return threading.get_ident()

        def generate() -> tuple[int, str]:
            return threading.get_ident(), local_state.stream_id

        load_thread_id = await _run_on_mlx_thread(load_model)
        generation_results = await asyncio.gather(*(_run_on_mlx_thread(generate) for _ in range(10)))

        assert generation_results
        for thread_id, stream_id in generation_results:
            assert thread_id == load_thread_id
            assert stream_id == "stream-created-at-load"

    def test_realtime_stream_blocks_heavy_mlx_work(self):
        owner = "stream:mlx-affinity-test"
        assert rvc_acquire(owner) is True
        try:
            with pytest.raises(RuntimeError, match="real-time streaming is active"):
                ensure_realtime_stream_not_active("test MLX work")
        finally:
            rvc_release(owner)

        ensure_realtime_stream_not_active("test MLX work")

    @pytest.mark.asyncio
    async def test_mlx_qwen_llm_load_uses_mlx_worker_thread(self, monkeypatch):
        backend = MLXQwenLLMBackend()
        seen: dict[str, int | str] = {}

        def fake_load_model_sync(model_size: str) -> None:
            seen["thread_name"] = threading.current_thread().name
            seen["thread_id"] = threading.get_ident()
            backend.model = object()
            backend.tokenizer = object()
            backend._current_model_size = model_size

        monkeypatch.setattr(backend, "_load_model_sync", fake_load_model_sync)

        await backend.load_model("0.6B")
        worker_thread_id = await _run_on_mlx_thread(threading.get_ident)

        assert seen["thread_name"] == "mlx_0"
        assert seen["thread_id"] == worker_thread_id

    @pytest.mark.asyncio
    async def test_pytorch_qwen_llm_load_does_not_use_mlx_worker_thread(self, monkeypatch):
        backend = PyTorchQwenLLMBackend()
        seen: dict[str, str] = {}

        def fake_load_model_sync(model_size: str) -> None:
            seen["thread_name"] = threading.current_thread().name
            backend.model = object()
            backend.tokenizer = object()
            backend._current_model_size = model_size

        monkeypatch.setattr(backend, "_load_model_sync", fake_load_model_sync)

        await backend.load_model("0.6B")

        assert not seen["thread_name"].startswith("mlx")

    @pytest.mark.asyncio
    async def test_mlx_qwen_llm_generate_load_and_infer_are_atomic(self, monkeypatch):
        backend = MLXQwenLLMBackend()
        seen_thread_ids: set[int] = set()

        def fake_load_model_sync(model_size: str) -> None:
            seen_thread_ids.add(threading.get_ident())
            time.sleep(0.02)
            backend.model = {"size": model_size}
            backend.tokenizer = object()
            backend._current_model_size = model_size
            backend.model_size = model_size

        def fake_generate_sync(*args, **kwargs) -> str:
            seen_thread_ids.add(threading.get_ident())
            resident = backend.model
            assert resident is not None, "model was unloaded before generation"
            time.sleep(0.02)
            assert backend.model is resident, "model changed during generation"
            return resident["size"]

        monkeypatch.setattr(backend, "_load_model_sync", fake_load_model_sync)
        monkeypatch.setattr(backend, "_generate_sync", fake_generate_sync)

        small, large = await asyncio.gather(
            backend.generate("a", model_size="0.6B"),
            backend.generate("b", model_size="4B"),
        )

        assert (small, large) == ("0.6B", "4B")
        assert len(seen_thread_ids) == 1

    @pytest.mark.asyncio
    async def test_mlx_qwen_tts_unload_waits_for_generation(self, monkeypatch):
        backend = MLXTTSBackend()
        seen_thread_ids: set[int] = set()

        def fake_load_model_sync(model_size: str) -> None:
            seen_thread_ids.add(threading.get_ident())
            backend.model = _StreamingAudioModel(backend)
            backend._current_model_size = model_size
            backend.model_size = model_size

        def fake_unload_model_sync() -> None:
            seen_thread_ids.add(threading.get_ident())
            backend.model = None
            backend._current_model_size = None

        monkeypatch.setattr(backend, "_load_model_sync", fake_load_model_sync)
        monkeypatch.setattr(backend, "_unload_model_sync", fake_unload_model_sync)

        task = asyncio.create_task(backend.generate("hi", {"ref_audio": None}))
        await asyncio.sleep(0.005)
        await asyncio.to_thread(backend.unload_model)
        audio, sample_rate = await task

        assert audio.shape == (16,)
        assert sample_rate == 24000
        assert backend.model is None
        assert len(seen_thread_ids) == 1

    @pytest.mark.asyncio
    async def test_mlx_whisper_transcribe_load_and_infer_are_atomic(self, monkeypatch, tmp_path):
        audio_path = tmp_path / "audio.wav"
        audio_path.write_bytes(b"fake")
        backend = MLXSTTBackend()
        seen_thread_ids: set[int] = set()

        def fake_load_model_sync(model_size: str) -> None:
            seen_thread_ids.add(threading.get_ident())
            time.sleep(0.02)
            backend.model = _STTModel(backend, model_size)
            backend.model_size = model_size

        def fake_unload_model_sync() -> None:
            seen_thread_ids.add(threading.get_ident())
            backend.model = None

        monkeypatch.setattr(backend, "_load_model_sync", fake_load_model_sync)
        monkeypatch.setattr(backend, "_unload_model_sync", fake_unload_model_sync)

        base, small = await asyncio.gather(
            backend.transcribe(str(audio_path), model_size="base"),
            backend.transcribe(str(audio_path), model_size="small"),
        )

        assert (base, small) == ("base", "small")
        assert len(seen_thread_ids) == 1

    @pytest.mark.asyncio
    async def test_mlx_tada_unload_waits_for_generation(self, monkeypatch, tmp_path):
        ref_audio = tmp_path / "ref.wav"
        ref_audio.write_bytes(b"fake")
        backend = MLXTadaBackend()
        seen_thread_ids: set[int] = set()

        class _TadaModel:
            def generate(self, *args, **kwargs):
                resident = backend.model
                assert resident is self, "model changed before TADA generation"
                time.sleep(0.02)
                assert backend.model is resident, "model changed during TADA generation"
                return _TadaOutput()

            def load_reference(self, *args, **kwargs):
                return object()

        class _Reference:
            @staticmethod
            def load(path):
                return object()

        class _Options:
            def __init__(self, *args, **kwargs):
                pass

        class _Config:
            Reference = _Reference
            InferenceOptions = _Options

        def fake_load_model_sync(model_size: str) -> None:
            seen_thread_ids.add(threading.get_ident())
            backend.model = _TadaModel()
            backend._current_model_size = model_size
            backend.model_size = model_size

        def fake_unload_model_sync() -> None:
            seen_thread_ids.add(threading.get_ident())
            backend.model = None
            backend._current_model_size = None

        monkeypatch.setattr(backend, "_load_model_sync", fake_load_model_sync)
        monkeypatch.setattr(backend, "_unload_model_sync", fake_unload_model_sync)
        monkeypatch.setattr("backend.backends.mlx_tada_backend.importlib.import_module", lambda name: _Config)

        task = asyncio.create_task(backend.generate("bonjour", {"ref_audio": str(ref_audio)}))
        await asyncio.sleep(0.005)
        await asyncio.to_thread(backend.unload_model)
        audio, sample_rate = await task

        assert audio.shape == (16,)
        assert sample_rate == 24000
        assert backend.model is None
        assert len(seen_thread_ids) == 1

    @pytest.mark.asyncio
    async def test_voxtral_unload_waits_for_generation(self, monkeypatch):
        backend = VoxtralTTSBackend()
        seen_thread_ids: set[int] = set()

        def fake_load_model_sync() -> None:
            seen_thread_ids.add(threading.get_ident())
            backend.model = _StreamingAudioModel(backend)

        def fake_unload_model_sync() -> None:
            seen_thread_ids.add(threading.get_ident())
            backend.model = None

        monkeypatch.setattr(backend, "_load_model_sync", fake_load_model_sync)
        monkeypatch.setattr(backend, "_unload_model_sync", fake_unload_model_sync)

        task = asyncio.create_task(backend.generate("bonjour", {"preset_voice_id": "fr_male"}))
        await asyncio.sleep(0.005)
        await asyncio.to_thread(backend.unload_model)
        audio, sample_rate = await task

        assert audio.shape == (16,)
        assert sample_rate == VOXTRAL_SAMPLE_RATE
        assert backend.model is None
        assert len(seen_thread_ids) == 1
