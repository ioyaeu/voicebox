"""API tests for the offline RVC conversion surface (step 03).

Drives the model-upload endpoints and ``POST /convert`` through a real FastAPI
:class:`TestClient` backed by a temporary data dir, so database rows and on-disk
artifacts are observed exactly as a live server produces them. The full
conversion round-trip is gated on a checkpoint fixture (skip-if-absent), the same
offline contract ``test_rvc_engine.py`` uses, so the suite stays green without a
GPU, network access, or a multi-hundred-megabyte download.
"""

import io
import os
import time
import uuid
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from starlette.testclient import TestClient

from backend import config
from backend.backends.base import is_model_cached
from backend.backends.rvc.checkpoint import (
    load_rvc_checkpoint,
    validate_rvc_checkpoint,
)

# -- fixture gating (mirrors test_rvc_engine.py) --------------------------

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rvc"
_CONTENTVEC_REPO = "lengyue233/content-vec-best"
_RMVPE_REPO = "lj1995/VoiceConversionWebUI"
_RMVPE_FILENAME = "rmvpe.pt"


def _first_glob(pattern: str):
    if not _FIXTURE_DIR.is_dir():
        return None
    matches = sorted(_FIXTURE_DIR.glob(pattern))
    return str(matches[0]) if matches else None


_CHECKPOINT = _first_glob("*.pth")
_BACKBONES_CACHED = is_model_cached(_CONTENTVEC_REPO) and is_model_cached(
    _RMVPE_REPO, required_files=[_RMVPE_FILENAME]
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """A TestClient over the real app, pinned to a throwaway data dir.

    Entering the context runs the app lifespan, which initialises the database
    (against the temp dir) and starts the shared generation-queue worker on the
    client's event loop — the same worker ``POST /convert`` enqueues onto.
    """
    from backend.app import app

    data_dir = tmp_path_factory.mktemp("convert_api_data")
    original = config.get_data_dir()
    config.set_data_dir(str(data_dir))
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        config.set_data_dir(str(original))


# -- helpers --------------------------------------------------------------


def _create_profile(client: TestClient, voice_type: str = "cloned") -> str:
    resp = client.post(
        "/profiles",
        json={"name": f"conv-{voice_type}-{uuid.uuid4().hex[:8]}", "voice_type": voice_type},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["id"]


def _fake_checkpoint_dict() -> dict:
    """A minimal *valid* extracted v2/40k/f0 checkpoint dict (from step 01's tests).

    Carries the ``weight``/``config``/``f0``/``version``/``sr`` keys that
    ``validate_rvc_checkpoint`` requires — including ``emb_g.weight`` (the speaker
    embedding strictness now demands) and the full 18-hyperparameter ``config`` —
    enough to pass upload validation without being a real synthesizer (real
    inference is the fixture-gated test).
    """
    return {
        "weight": {
            "emb_g.weight": torch.zeros(109, 256),
            "enc_p.emb_phone.weight": torch.zeros(2, 2),
        },
        "config": [
            1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
            [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000,
        ],
        "f0": 1,
        "version": "v2",
        "sr": 40000,
    }


def _checkpoint_bytes(ckpt: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    return buf.getvalue()


def _write_fake_checkpoint(path: Path) -> None:
    torch.save(_fake_checkpoint_dict(), path)


def _wav_bytes(seconds: float = 1.0, sr: int = 16000) -> bytes:
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    return buf.getvalue()


def _voiced_wav_bytes(seconds: float = 3.0, sr: int = 16000) -> bytes:
    """A voiced tone with harmonics + vibrato so RMVPE has real f0 to track."""
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    f0 = 150.0 * (2.0 ** (0.05 * np.sin(2 * np.pi * 5.0 * t)))
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros_like(t)
    for harmonic, amp in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125)):
        sig += amp * np.sin(harmonic * phase)
    env = 0.5 * (1 - np.cos(2 * np.pi * np.clip(t / seconds, 0.0, 1.0)))
    sig = (sig * env).astype(np.float32)
    sig /= np.abs(sig).max() + 1e-9
    buf = io.BytesIO()
    sf.write(buf, sig * 0.8, sr, format="WAV")
    return buf.getvalue()


# -- upload rejection -----------------------------------------------------


def test_upload_rejected_for_non_rvc_profile(client):
    profile_id = _create_profile(client, voice_type="cloned")

    resp = client.post(
        f"/profiles/{profile_id}/rvc-model",
        files={"model": ("model.pth", b"\x00" * 1024, "application/octet-stream")},
    )

    assert resp.status_code == 400, resp.text
    assert "rvc" in resp.json()["detail"].lower()
    # A non-RVC profile is rejected before streaming — no artifact written.
    assert not (config.get_profiles_dir() / profile_id / "model.pth").exists()


def test_upload_rejects_malicious_pickle_and_persists_nothing(client, tmp_path):
    profile_id = _create_profile(client, voice_type="rvc")
    sentinel = tmp_path / "pwned.txt"

    class Exploit:
        def __reduce__(self):
            return (os.system, (f'touch "{sentinel}"',))

    malicious = tmp_path / "malicious.pth"
    torch.save({"weight": Exploit()}, malicious)

    with malicious.open("rb") as f:
        resp = client.post(
            f"/profiles/{profile_id}/rvc-model",
            files={"model": ("model.pth", f, "application/octet-stream")},
        )

    assert resp.status_code == 400, resp.text
    assert "not a valid RVC checkpoint" in resp.json()["detail"]

    # weights_only load must never have executed the payload...
    assert not sentinel.exists(), "malicious pickle payload executed"
    # ...and no model artifact (nor the streamed temp) may survive validation.
    profile_dir = config.get_profiles_dir() / profile_id
    assert not (profile_dir / "model.pth").exists()
    assert list(profile_dir.glob("*.pth")) == []

    # No usable model left on the profile (storage paths are internal; the
    # response exposes only the rvc_has_model flag).
    prof = client.get(f"/profiles/{profile_id}").json()
    assert prof["rvc_has_model"] is False


def test_upload_rejects_oversized_model_during_streaming(client, monkeypatch):
    # Lower the checkpoint cap so a small body trips the streaming guard; the
    # route re-reads this constant per call, so patching the module suffices.
    from backend.backends.rvc import checkpoint as ckpt_mod

    monkeypatch.setattr(ckpt_mod, "MAX_CHECKPOINT_BYTES", 1 * 1024 * 1024)

    profile_id = _create_profile(client, voice_type="rvc")
    oversized = b"\x00" * (2 * 1024 * 1024)  # 2 MB > 1 MB cap

    resp = client.post(
        f"/profiles/{profile_id}/rvc-model",
        files={"model": ("model.pth", oversized, "application/octet-stream")},
    )

    assert resp.status_code == 413, resp.text
    assert "too large" in resp.json()["detail"].lower()
    # Cut off mid-stream: nothing persisted.
    profile_dir = config.get_profiles_dir() / profile_id
    assert not (profile_dir / "model.pth").exists()
    assert list(profile_dir.glob("*.pth")) == []


def test_upload_rejects_checkpoint_missing_emb_g_weight(client):
    """A checkpoint whose state dict lacks ``emb_g.weight`` is a 400 at upload,
    not a raw 500 at conversion (``build_synthesizer`` dereferences it)."""
    profile_id = _create_profile(client, voice_type="rvc")
    ckpt = _fake_checkpoint_dict()
    del ckpt["weight"]["emb_g.weight"]

    resp = client.post(
        f"/profiles/{profile_id}/rvc-model",
        files={"model": ("model.pth", _checkpoint_bytes(ckpt), "application/octet-stream")},
    )

    assert resp.status_code == 400, resp.text
    assert "emb_g.weight" in resp.json()["detail"]
    profile_dir = config.get_profiles_dir() / profile_id
    assert not (profile_dir / "model.pth").exists()
    assert list(profile_dir.glob("*.pth")) == []


def test_upload_rejects_checkpoint_with_wrong_config_arity(client):
    """A ``config`` that is not the 18 required hyperparameters is a 400 at upload
    (a shorter list would crash conversion with a raw ``IndexError``)."""
    profile_id = _create_profile(client, voice_type="rvc")
    ckpt = _fake_checkpoint_dict()
    ckpt["config"] = ckpt["config"][:11]

    resp = client.post(
        f"/profiles/{profile_id}/rvc-model",
        files={"model": ("model.pth", _checkpoint_bytes(ckpt), "application/octet-stream")},
    )

    assert resp.status_code == 400, resp.text
    assert "18" in resp.json()["detail"]
    profile_dir = config.get_profiles_dir() / profile_id
    assert not (profile_dir / "model.pth").exists()
    assert list(profile_dir.glob("*.pth")) == []


# -- upload success -------------------------------------------------------


def test_upload_success_sets_columns_and_writes_files(client, tmp_path):
    profile_id = _create_profile(client, voice_type="rvc")

    fake = tmp_path / "fake.pth"
    _write_fake_checkpoint(fake)

    with fake.open("rb") as f:
        resp = client.post(
            f"/profiles/{profile_id}/rvc-model",
            files={"model": ("keruanv2.pth", f, "application/octet-stream")},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["rvc_has_model"] is True, "rvc_has_model should be true after upload"
    assert body["rvc_version"] == "v2"
    assert body["rvc_sample_rate"] == 40000
    assert body["rvc_f0"] == 1

    profile_dir = config.get_profiles_dir() / profile_id
    assert (profile_dir / "model.pth").exists(), "model.pth not written to disk"
    assert (profile_dir / "rvc_model.json").exists(), "metadata sidecar not written"


# -- convert guardrails ---------------------------------------------------


def test_convert_without_model_returns_400(client):
    profile_id = _create_profile(client, voice_type="rvc")

    resp = client.post(
        "/convert",
        data={"profile_id": profile_id},
        files={"file": ("source.wav", _wav_bytes(), "audio/wav")},
    )

    assert resp.status_code == 400, resp.text
    assert "no uploaded model" in resp.json()["detail"].lower()


# -- concurrency: engine lease arbitration --------------------------------


def test_offline_convert_fails_fast_when_engine_leased(client, tmp_path):
    """An offline ``/convert`` enqueued while the RVC engine is leased (a live
    Voice Changer stream holds it) fails fast on the shared queue with the
    user-readable message, and never usurps the lease from its holder.

    The lease is held directly here rather than through a real WS session so the
    test is deterministic; it drives the exact ``run_conversion`` acquire/fail
    path a live stream triggers. Because the lease acquire fails first, no model
    is ever loaded, so the fake checkpoint is never actually built.
    """
    from backend.backends.rvc import (
        acquire as rvc_acquire,
        lease_holder,
        release as rvc_release,
    )

    profile_id = _create_profile(client, voice_type="rvc")
    fake = tmp_path / "fake.pth"
    _write_fake_checkpoint(fake)
    with fake.open("rb") as f:
        up = client.post(
            f"/profiles/{profile_id}/rvc-model",
            files={"model": ("keruanv2.pth", f, "application/octet-stream")},
        )
    assert up.status_code == 200, up.text

    stream_owner = "stream:test-live-session"
    assert rvc_acquire(stream_owner) is True
    try:
        resp = client.post(
            "/convert",
            data={"profile_id": profile_id},
            files={"file": ("source.wav", _wav_bytes(), "audio/wav")},
        )
        assert resp.status_code == 200, resp.text  # request is accepted + enqueued
        task_id = resp.json()["task_id"]

        deadline = time.time() + 30
        status = None
        error = None
        while time.time() < deadline:
            payload = client.get(f"/history/{task_id}").json()
            status = payload["status"]
            error = payload.get("error")
            if status in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert status == "failed", f"convert must fail fast while engine leased (last={status})"
        assert "Voice Changer live session is active" in (error or "")
        # The live stream keeps the lease — the job never swapped it out.
        assert lease_holder() == stream_owner
    finally:
        rvc_release(stream_owner)


# -- full conversion round-trip (fixture-gated) ---------------------------


@pytest.mark.skipif(
    _CHECKPOINT is None,
    reason=(
        "no RVC checkpoint fixture under backend/tests/fixtures/rvc/ "
        "(see test_rvc_engine.py's docstring to place one)"
    ),
)
def test_full_conversion_roundtrip(client):
    """Upload a real checkpoint, enqueue a convert, poll to done, parse the WAV."""
    expected_sr = validate_rvc_checkpoint(load_rvc_checkpoint(_CHECKPOINT)).sample_rate

    profile_id = _create_profile(client, voice_type="rvc")

    with open(_CHECKPOINT, "rb") as f:
        up = client.post(
            f"/profiles/{profile_id}/rvc-model",
            files={"model": (Path(_CHECKPOINT).name, f, "application/octet-stream")},
        )
    assert up.status_code == 200, up.text

    index_path = _first_glob("*.index")
    if index_path is not None:
        with open(index_path, "rb") as fi, open(_CHECKPOINT, "rb") as fm:
            up = client.post(
                f"/profiles/{profile_id}/rvc-model",
                files={
                    "model": (Path(_CHECKPOINT).name, fm, "application/octet-stream"),
                    "index": (Path(index_path).name, fi, "application/octet-stream"),
                },
            )
        assert up.status_code == 200, up.text

    resp = client.post(
        "/convert",
        data={"profile_id": profile_id, "f0_method": "rmvpe"},
        files={"file": ("source.wav", _voiced_wav_bytes(seconds=3.0), "audio/wav")},
    )
    assert resp.status_code == 200, resp.text
    task_id = resp.json()["task_id"]

    deadline = time.time() + 600
    status = "generating"
    error = None
    while time.time() < deadline:
        hist = client.get(f"/history/{task_id}")
        assert hist.status_code == 200, hist.text
        payload = hist.json()
        status = payload["status"]
        error = payload.get("error")
        if status in ("completed", "failed"):
            break
        time.sleep(2)

    if status == "failed":
        # Same offline contract as test_rvc_engine.py: a fetch-shaped failure is
        # a skip only when the backbones genuinely are not cached locally.
        if not _BACKBONES_CACHED:
            pytest.skip(f"ContentVec/RMVPE unavailable offline: {error}")
        pytest.fail(f"conversion failed: {error}")
    assert status == "completed", f"conversion did not finish in time (last={status})"

    audio_resp = client.get(f"/audio/{task_id}")
    assert audio_resp.status_code == 200, audio_resp.text

    data, sr = sf.read(io.BytesIO(audio_resp.content))
    assert sr == expected_sr, f"served WAV sr {sr} != checkpoint sr {expected_sr}"
    assert data.size > 0
    duration = data.shape[0] / sr
    assert 0.5 < duration < 30.0, f"unexpected output duration {duration:.2f}s"
