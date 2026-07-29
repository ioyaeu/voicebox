"""Tests for crepe pitch download-on-demand (Phase A packaging correction).

torchcrepe's ``full.pth`` (~85 MB) is deliberately excluded from the frozen
binary and torchcrepe has no download fallback, so ``crepe-full`` is a
direct-download registry model and ``f0_method="crepe"`` without it must raise a
clear, user-actionable error (surfaced as an HTTP 400 by ``POST /convert``),
never a raw ``FileNotFoundError`` / crash.

These tests are network-free: they never download the 85 MB weights. The full
crepe conversion round-trip is exercised by the live verification transcript, not
in CI.
"""

from pathlib import Path

import numpy as np
import pytest

from backend.backends import get_model_config
from backend.backends.base import (
    get_crepe_model_path,
    get_direct_download_path,
    is_direct_download_cached,
)
from backend.backends.rvc import pitch

# The pinned upstream source + digest (commit-pinned GitHub raw URL). If either
# changes, the download flow's sha256 verification would reject the real file, so
# these constants are asserted against the registry to catch accidental edits.
_PINNED_URL = (
    "https://raw.githubusercontent.com/maxrmorrison/torchcrepe/"
    "19e2ec3d494c0797a5ff2a11408ec5838fba6681/torchcrepe/assets/full.pth"
)
_PINNED_SHA256 = "133225604dedd2e4005f8bbd1bd0a2ec073ba8b7a6cd31ff6d5edbbfa3539986"


# -- registry -------------------------------------------------------------


def test_crepe_full_is_registered_with_pinned_source():
    cfg = get_model_config("crepe-full")
    assert cfg is not None, "crepe-full missing from the model registry"
    assert cfg.engine == "rvc"
    assert cfg.display_name == "Crepe (pitch, full)"
    assert cfg.download_url == _PINNED_URL
    assert cfg.sha256 == _PINNED_SHA256
    assert cfg.file_name == "full.pth"
    # It is a direct-download model, not an HF repo.
    assert not cfg.hf_repo_id
    assert 80 <= cfg.size_mb <= 90


def test_crepe_sha256_constant_matches_pinned_source():
    # A guard against silently editing the digest away from the verified file.
    cfg = get_model_config("crepe-full")
    assert cfg.sha256 == _PINNED_SHA256
    assert len(cfg.sha256) == 64
    int(cfg.sha256, 16)  # must be valid hex


def test_crepe_model_path_is_single_source_of_truth():
    cfg = get_model_config("crepe-full")
    from_helper = get_crepe_model_path()
    from_generic = get_direct_download_path(cfg)
    assert from_helper is not None
    assert from_helper == from_generic
    assert from_helper.name == "full.pth"
    # Lives under the model cache root, keyed by model name.
    assert from_helper.parent.name == "crepe-full"


def test_hf_model_has_no_direct_download_path():
    # A regular HF model must not resolve to a direct-download path.
    hf_cfg = get_model_config("rmvpe")
    assert hf_cfg is not None
    assert get_direct_download_path(hf_cfg) is None
    assert is_direct_download_cached(hf_cfg) is False


# -- absent-model error ---------------------------------------------------


def test_extract_f0_crepe_raises_clear_error_when_absent(monkeypatch, tmp_path):
    """With the weights absent, the loader must raise the actionable message —
    not a raw ``FileNotFoundError`` from torchcrepe's bundled-asset load."""
    missing = tmp_path / "crepe-full" / "full.pth"
    monkeypatch.setattr(pitch, "get_crepe_model_path", lambda: missing)
    # Reset any cached preload so the loader re-checks the (patched) path.
    monkeypatch.setattr(pitch, "_crepe_key", None, raising=False)

    audio = np.zeros(1600, dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        pitch.extract_f0_crepe(audio, device="cpu")

    msg = str(excinfo.value)
    assert "not downloaded" in msg
    assert "Crepe (pitch, full)" in msg


def test_ensure_crepe_loaded_raises_clear_error_when_absent(monkeypatch, tmp_path):
    missing = tmp_path / "crepe-full" / "full.pth"
    monkeypatch.setattr(pitch, "get_crepe_model_path", lambda: missing)
    monkeypatch.setattr(pitch, "_crepe_key", None, raising=False)

    with pytest.raises(ValueError, match="not downloaded"):
        pitch._ensure_crepe_loaded("cpu", "full")


# -- real conversion round-trip is skip-gated on the model being present ---


@pytest.mark.skipif(
    get_crepe_model_path() is None or not get_crepe_model_path().exists(),
    reason="crepe full.pth not downloaded (network-free CI); run the download flow first",
)
def test_extract_f0_crepe_loads_from_registry_file():
    """When the model is present, the extractor produces a valid (f0_coarse, f0)
    pair loaded from OUR registry file (not the package assets)."""
    sr = 16000
    t = np.linspace(0.0, 1.0, sr, endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 200.0 * t)).astype(np.float32)

    f0_coarse, f0 = pitch.extract_f0_crepe(audio, device="cpu")

    assert f0_coarse.shape == f0.shape
    assert f0_coarse.dtype == np.int32
    assert f0_coarse.min() >= 0 and f0_coarse.max() <= 255
    assert np.any(f0 > 0), "no voiced frames detected on a pure tone"
