"""Regression tests for MLX worker-thread affinity."""

import asyncio
import threading

import pytest

from backend.backends.mlx_backend import _run_on_mlx_thread, ensure_realtime_stream_not_active
from backend.backends.qwen_llm_backend import MLXQwenLLMBackend, PyTorchQwenLLMBackend
from backend.backends.rvc import acquire as rvc_acquire, release as rvc_release


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
