from backend import backends
from backend.backends import ModelConfig
from backend.backends.mlx_tada_backend import MLX_TADA_3B_REPO, MLXTadaBackend


def test_tada_config_uses_mlx_repo_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(backends, "get_backend_type", lambda: "mlx")

    cfg = backends.get_model_config("tada-3b-ml")

    assert cfg is not None
    assert cfg.engine == "tada"
    assert cfg.model_size == "3B"
    assert cfg.hf_repo_id == MLX_TADA_3B_REPO
    assert "MLX 4-bit" in cfg.display_name


def test_tada_config_keeps_pytorch_repo_off_mlx(monkeypatch):
    monkeypatch.setattr(backends, "get_backend_type", lambda: "pytorch")

    cfg = backends.get_model_config("tada-3b-ml")

    assert cfg is not None
    assert cfg.hf_repo_id == "HumeAI/tada-3b-ml"
    assert "MLX" not in cfg.display_name


def test_tada_backend_resolves_to_mlx_backend_on_apple_silicon(monkeypatch):
    monkeypatch.setattr(backends, "get_backend_type", lambda: "mlx")
    backends._tts_backends.pop("tada", None)

    try:
        backend = backends.get_tts_backend_for_engine("tada")
    finally:
        backends._tts_backends.pop("tada", None)

    assert isinstance(backend, MLXTadaBackend)


def test_tada_download_loader_uses_config_model_size(monkeypatch):
    calls = []

    class _FakeBackend:
        def load_model(self, model_size):
            calls.append(model_size)

    monkeypatch.setattr(backends, "get_tts_backend_for_engine", lambda engine: _FakeBackend())

    load_func = backends.get_model_load_func(
        ModelConfig(
            model_name="tada-3b-ml",
            display_name="TADA 3B Multilingual",
            engine="tada",
            hf_repo_id="HumeAI/mlx-tada-3b",
            model_size="3B",
        )
    )
    load_func()

    assert calls == ["3B"]
