"""End-to-end tests for the TTS->RVC generation chain (step 07, Phase C).

Everything here targets the single choke point in
``backend.services.generation.run_generation`` — the same function the
generation tab, stories, ``POST /generate``, ``POST /speak`` and MCP all route
through — so there is exactly one place the chain can live and exactly one place
to test it.

Three layers, cheapest first:

1. ``test_chain_order_params_and_postprocessing`` — an rvc profile with the base
   TTS engine (``generate_chunked``) and the RVC pipeline (``get_rvc_engine``)
   monkeypatched. Asserts the *order* of stages (base TTS renders, base engine
   unloads, then RVC converts), that the profile's ``rvc_params`` reach the RVC
   convert verbatim, and that the *converted* array — at the checkpoint sample
   rate, not the base TTS rate — is what enters post-processing (the real
   ``_save_generate`` writes it to disk and creates the "original" version). No
   model or network is touched; the RVC convert is identity-patched.

2. ``test_non_rvc_generation_does_not_invoke_chain`` (MANDATORY regression) —
   cloned/preset/designed profiles must take the untouched non-rvc path: the
   chain function ``_generate_rvc_chained`` and ``get_rvc_engine`` are never
   invoked and the base engine's output flows through unchanged (same samples,
   same sample rate). This guards the hottest path in the app against an
   accidental chain.

3. ``test_generate_and_speak_through_rvc_chain`` (real fixture, skip-if-absent)
   — drives the real FastAPI app: create an rvc profile, upload a checkpoint,
   generate short text via ``POST /generate`` (engine=null, mirroring the
   generation tab) and ``POST /speak``, poll to completion, and assert the
   served audio is non-silent at the checkpoint sample rate. Requires the Kokoro
   base model, the ContentVec/RMVPE backbones, an importable ``kokoro`` package,
   and an RVC checkpoint (``fixtures/rvc/*.pth`` or the cached
   ``trojblue/rvc-kanade-voice`` keruanv2.pth) all present locally; otherwise it
   skips (same offline contract as ``test_rvc_engine.py``).
"""

import io
import json
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from backend import config
from backend.backends.base import is_model_cached


# ----------------------------------------------------------------------------
# Shared fixtures / helpers for the in-process (no-server) tests
# ----------------------------------------------------------------------------


@pytest.fixture
def data_env(tmp_path):
    """Point config + DB at a throwaway data dir and initialise the schema.

    ``run_generation`` opens its own session via ``get_db`` (bound to the global
    ``SessionLocal`` that ``init_db`` sets), so re-initialising per test gives a
    clean, isolated SQLite file with the step-07 columns migrated in.
    """
    from backend.database import session as db_session

    original = config.get_data_dir()
    config.set_data_dir(str(tmp_path))
    db_session.init_db()
    try:
        yield tmp_path
    finally:
        config.set_data_dir(str(original))


def _insert_rvc_profile(db, *, base_voice="kokoro:af_heart", params=None):
    """Insert an rvc profile whose ``rvc_model_path`` points at a real file.

    The file's *contents* are irrelevant to the in-process tests — the RVC
    engine's ``load`` is monkeypatched — but the path must exist on disk because
    ``_generate_rvc_chained`` guards against a missing checkpoint before loading.
    """
    from backend.database import VoiceProfile as DBVoiceProfile

    pid = str(uuid.uuid4())
    profile_dir = config.get_profiles_dir() / pid
    profile_dir.mkdir(parents=True, exist_ok=True)
    model_file = profile_dir / "model.pth"
    model_file.write_bytes(b"placeholder-checkpoint-bytes")

    row = DBVoiceProfile(
        id=pid,
        name=f"rvc-{pid[:8]}",
        language="en",
        voice_type="rvc",
        default_engine="rvc",
        rvc_base_voice=base_voice,
        rvc_params=json.dumps(params) if params is not None else None,
        rvc_model_path=config.to_storage_path(model_file),
        rvc_index_path=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    return pid


def _insert_profile(db, *, voice_type):
    from backend.database import VoiceProfile as DBVoiceProfile

    pid = str(uuid.uuid4())
    row = DBVoiceProfile(
        id=pid,
        name=f"{voice_type}-{pid[:8]}",
        language="en",
        voice_type=voice_type,
        default_engine=None,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    return pid


def _insert_generation(db, profile_id, *, text="Hello there.", engine="rvc"):
    from backend.database import Generation as DBGeneration

    gid = str(uuid.uuid4())
    row = DBGeneration(
        id=gid,
        profile_id=profile_id,
        text=text,
        language="en",
        audio_path="",
        duration=0,
        seed=None,
        status="generating",
        engine=engine,
        model_size=None,
        source="manual",
        created_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    return gid


def _sine(seconds=1.0, sr=24000, freq=220.0, amp=0.5):
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


# ----------------------------------------------------------------------------
# 1. Chain path — order, params, converted output enters post-processing
# ----------------------------------------------------------------------------


def test_chain_order_params_and_postprocessing(data_env, monkeypatch):
    import asyncio

    from backend import backends as backends_pkg
    from backend.backends import rvc as rvc_pkg
    from backend.services import generation as gen_mod
    from backend.services import history as history_mod
    from backend.utils import chunked_tts as chunked_mod
    from backend.database import (
        Generation as DBGeneration,
        GenerationVersion as DBGenerationVersion,
        get_db,
    )

    order: list[str] = []

    # Base TTS output: distinctive tone at the base engine rate (24 kHz).
    base_audio = _sine(seconds=1.0, sr=24000, freq=220.0, amp=0.5)
    BASE_SR = 24000
    # The checkpoint's native rate the RVC stage returns — deliberately != BASE_SR
    # so "which array reached _save_generate" is unambiguous from the saved file.
    RVC_SR = 40000

    custom_params = {
        "f0_up_key": 5,
        "f0_method": "rmvpe",
        "index_rate": 0.5,
        "rms_mix_rate": 0.4,
        "protect": 0.2,
    }

    class _FakeBaseBackend:
        def is_loaded(self):
            return False

        def unload_model(self):
            order.append("unload_base")

    fake_base = _FakeBaseBackend()

    load_calls: list[tuple] = []

    async def _fake_load_engine_model(engine, model_size="default"):
        order.append("load_base")
        load_calls.append((engine, model_size))

    def _fake_get_backend(engine):
        return fake_base

    gen_chunked_calls: list[dict] = []

    async def _fake_generate_chunked(backend, text, voice_prompt, **kwargs):
        order.append("tts")
        gen_chunked_calls.append(
            {
                "backend": backend,
                "text": text,
                "voice_prompt": voice_prompt,
                "kwargs": kwargs,
            }
        )
        return base_audio.copy(), BASE_SR

    class _FakeRVCEngine:
        def __init__(self):
            self.loaded_with = None
            self.convert_calls: list[dict] = []

        def load(self, model_path, index_path=None):
            order.append("load_rvc")
            self.loaded_with = (model_path, index_path)

        def convert_audio(
            self,
            audio,
            sr,
            *,
            f0_up_key,
            f0_method,
            index_rate,
            rms_mix_rate,
            protect,
        ):
            order.append("rvc")
            self.convert_calls.append(
                {
                    "audio": np.asarray(audio, dtype=np.float32).copy(),
                    "sr": sr,
                    "params": {
                        "f0_up_key": f0_up_key,
                        "f0_method": f0_method,
                        "index_rate": index_rate,
                        "rms_mix_rate": rms_mix_rate,
                        "protect": protect,
                    },
                }
            )
            # Identity conversion, but returned at the checkpoint sample rate.
            return np.asarray(audio, dtype=np.float32).copy(), RVC_SR

        def unload(self):
            # The chain unloads the RVC stack in a finally after conversion
            # (step 08 item 1.4) so a following plain TTS generation does not
            # coexist with a resident synthesizer + ContentVec + RMVPE.
            order.append("unload_rvc")

    fake_rvc = _FakeRVCEngine()

    # Capture the staged status updates (loading_model / generating / converting
    # / completed) while still writing them to the DB via the original.
    statuses: list[str] = []
    original_update = history_mod.update_generation_status

    async def _spy_update(generation_id, status, db, **kwargs):
        statuses.append(status)
        return await original_update(generation_id, status, db, **kwargs)

    monkeypatch.setattr(backends_pkg, "load_engine_model", _fake_load_engine_model)
    monkeypatch.setattr(backends_pkg, "get_tts_backend_for_engine", _fake_get_backend)
    monkeypatch.setattr(rvc_pkg, "get_rvc_engine", lambda: fake_rvc)
    monkeypatch.setattr(chunked_mod, "generate_chunked", _fake_generate_chunked)
    monkeypatch.setattr(history_mod, "update_generation_status", _spy_update)

    db = next(get_db())
    try:
        pid = _insert_rvc_profile(db, base_voice="kokoro:af_heart", params=custom_params)
        gid = _insert_generation(db, pid, text="Convert me.", engine="rvc")
    finally:
        db.close()

    # The path the chain resolves from the profile's stored rvc_model_path and
    # hands to the RVC engine's load().
    expected_model_path = str((config.get_profiles_dir() / pid / "model.pth").resolve())

    asyncio.run(
        gen_mod.run_generation(
            generation_id=gid,
            profile_id=pid,
            text="Convert me.",
            language="en",
            engine="rvc",
            model_size=None,
            seed=None,
            normalize=False,  # keep the identity assertion exact
            effects_chain=None,
            mode="generate",
        )
    )

    # --- Order: base TTS renders, base engine unloads, then RVC converts. ---
    assert order == ["load_base", "tts", "unload_base", "load_rvc", "rvc", "unload_rvc"], order

    # --- Base engine resolved from rvc_base_voice ("kokoro:af_heart"). ---
    assert load_calls and load_calls[0][0] == "kokoro"
    assert len(gen_chunked_calls) == 1
    call = gen_chunked_calls[0]
    assert call["backend"] is fake_base
    assert call["text"] == "Convert me."
    assert call["voice_prompt"] == {
        "voice_type": "preset",
        "preset_engine": "kokoro",
        "preset_voice_id": "af_heart",
    }

    # --- rvc_params from the profile reach the RVC convert verbatim. ---
    assert len(fake_rvc.convert_calls) == 1
    conv = fake_rvc.convert_calls[0]
    assert conv["params"] == custom_params
    # The RVC stage received the *base TTS* output (at the base rate) to convert.
    assert conv["sr"] == BASE_SR
    assert np.array_equal(conv["audio"], base_audio)
    # Checkpoint loaded from the profile's stored model path, no index.
    assert fake_rvc.loaded_with == (expected_model_path, None)

    # --- The converted array (at RVC_SR) is what reached post-processing. ---
    clean_path = config.get_generations_dir() / f"{gid}.wav"
    assert clean_path.exists(), "converted output was not written by _save_generate"
    saved, saved_sr = sf.read(str(clean_path))
    assert saved_sr == RVC_SR, (
        f"saved sr {saved_sr} != RVC stage sr {RVC_SR}; the base-TTS output "
        "bypassed conversion into post-processing"
    )
    assert saved.shape[0] == base_audio.shape[0]
    assert np.allclose(saved, base_audio, atol=1e-3)
    assert float(np.sqrt(np.mean(saved.astype(np.float64) ** 2))) > 1e-3

    # --- DB state: completed, audio_path set, an "original" version exists. ---
    db = next(get_db())
    try:
        row = db.query(DBGeneration).filter_by(id=gid).first()
        assert row.status == "completed"
        assert row.audio_path
        versions = db.query(DBGenerationVersion).filter_by(generation_id=gid).all()
        assert any(v.label == "original" for v in versions)
    finally:
        db.close()

    # --- Two-stage progress: base TTS ("generating") then RVC ("converting"). ---
    assert "generating" in statuses
    assert "converting" in statuses
    assert statuses.index("generating") < statuses.index("converting")
    assert statuses[-1] == "completed"


# ----------------------------------------------------------------------------
# 2. Regression — non-rvc profiles never touch the RVC chain (MANDATORY)
# ----------------------------------------------------------------------------


@pytest.mark.parametrize("voice_type", ["cloned", "preset", "designed"])
def test_non_rvc_generation_does_not_invoke_chain(data_env, monkeypatch, voice_type):
    import asyncio

    from backend import backends as backends_pkg
    from backend.backends import rvc as rvc_pkg
    from backend.services import generation as gen_mod
    from backend.services import profiles as profiles_mod
    from backend.utils import chunked_tts as chunked_mod
    from backend.database import Generation as DBGeneration, get_db

    base_audio = _sine(seconds=1.0, sr=24000, freq=330.0, amp=0.4)
    BASE_SR = 24000

    invoked = {"rvc_chain": False, "rvc_engine": False}

    async def _spy_rvc_chain(*args, **kwargs):
        invoked["rvc_chain"] = True
        raise AssertionError("RVC chain invoked for a non-rvc profile")

    def _spy_get_rvc_engine(*args, **kwargs):
        invoked["rvc_engine"] = True
        raise AssertionError("get_rvc_engine called for a non-rvc profile")

    class _FakeBackend:
        def is_loaded(self):
            return True  # skip the loading_model status write

        def unload_model(self):
            pass

    async def _fake_load_engine_model(engine, model_size="default"):
        pass

    async def _fake_create_voice_prompt(profile_id, db, use_cache=True, engine="qwen", language=None):
        return {"voice_type": voice_type}

    gen_chunked_calls: list[dict] = []

    async def _fake_generate_chunked(backend, text, voice_prompt, **kwargs):
        gen_chunked_calls.append({"text": text})
        return base_audio.copy(), BASE_SR

    monkeypatch.setattr(gen_mod, "_generate_rvc_chained", _spy_rvc_chain)
    monkeypatch.setattr(rvc_pkg, "get_rvc_engine", _spy_get_rvc_engine)
    monkeypatch.setattr(backends_pkg, "get_tts_backend_for_engine", lambda e: _FakeBackend())
    monkeypatch.setattr(backends_pkg, "load_engine_model", _fake_load_engine_model)
    monkeypatch.setattr(profiles_mod, "create_voice_prompt_for_profile", _fake_create_voice_prompt)
    monkeypatch.setattr(chunked_mod, "generate_chunked", _fake_generate_chunked)

    db = next(get_db())
    try:
        pid = _insert_profile(db, voice_type=voice_type)
        gid = _insert_generation(db, pid, text="No conversion here.", engine="qwen")
    finally:
        db.close()

    asyncio.run(
        gen_mod.run_generation(
            generation_id=gid,
            profile_id=pid,
            text="No conversion here.",
            language="en",
            engine="qwen",
            model_size=None,
            seed=None,
            normalize=False,
            effects_chain=None,
            mode="generate",
        )
    )

    # The RVC-specific code path is never entered for a non-rvc profile.
    assert invoked["rvc_chain"] is False
    assert invoked["rvc_engine"] is False

    # The non-rvc path ran and produced the base engine's output unchanged.
    assert len(gen_chunked_calls) == 1
    clean_path = config.get_generations_dir() / f"{gid}.wav"
    assert clean_path.exists()
    saved, saved_sr = sf.read(str(clean_path))
    assert saved_sr == BASE_SR, "non-rvc output sample rate must be untouched"
    assert saved.shape[0] == base_audio.shape[0]
    assert np.allclose(saved, base_audio, atol=1e-3)

    db = next(get_db())
    try:
        row = db.query(DBGeneration).filter_by(id=gid).first()
        assert row.status == "completed"
    finally:
        db.close()


# ----------------------------------------------------------------------------
# 3. Real fixture end-to-end through the FastAPI app (skip-if-absent)
# ----------------------------------------------------------------------------

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rvc"
_KOKORO_REPO = "hexgrad/Kokoro-82M"
_CONTENTVEC_REPO = "lengyue233/content-vec-best"
_RMVPE_REPO = "lj1995/VoiceConversionWebUI"
_RMVPE_FILENAME = "rmvpe.pt"


def _resolve_checkpoint():
    """Return a usable RVC checkpoint path, or None to skip.

    Prefers a local ``fixtures/rvc/*.pth`` (matching ``test_rvc_engine.py``),
    then falls back to the HuggingFace cache for ``trojblue/rvc-kanade-voice``
    (offline only — never triggers a download).
    """
    if _FIXTURE_DIR.is_dir():
        matches = sorted(_FIXTURE_DIR.glob("*.pth"))
        if matches:
            return str(matches[0])
    try:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(
            "trojblue/rvc-kanade-voice",
            "_weights_unsorted/keruanv2.pth",
            local_files_only=True,
        )
    except Exception:
        return None


def _kokoro_importable():
    try:
        import kokoro  # noqa: F401

        return True
    except Exception:
        return False


_CHECKPOINT = _resolve_checkpoint()
_KOKORO_CACHED = is_model_cached(_KOKORO_REPO, required_files=["config.json", "kokoro-v1_0.pth"])
_BACKBONES_CACHED = is_model_cached(_CONTENTVEC_REPO) and is_model_cached(
    _RMVPE_REPO, required_files=[_RMVPE_FILENAME]
)
_KOKORO_IMPORTABLE = _kokoro_importable()

_E2E_READY = bool(_CHECKPOINT) and _KOKORO_CACHED and _BACKBONES_CACHED and _KOKORO_IMPORTABLE
_E2E_SKIP_REASON = (
    "TTS->RVC e2e needs a checkpoint (fixtures/rvc/*.pth or cached "
    "trojblue/rvc-kanade-voice), the Kokoro base model, the ContentVec/RMVPE "
    "backbones, and an importable kokoro package — one or more is absent: "
    f"checkpoint={bool(_CHECKPOINT)} kokoro_cached={_KOKORO_CACHED} "
    f"backbones_cached={_BACKBONES_CACHED} kokoro_importable={_KOKORO_IMPORTABLE}"
)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """TestClient over the real app on a throwaway data dir.

    Entering the context runs the app lifespan (init_db against the temp dir +
    the serial generation-queue worker on the client's loop) — the same worker
    ``POST /generate`` and ``POST /speak`` enqueue onto.
    """
    from starlette.testclient import TestClient

    from backend.app import app

    data_dir = tmp_path_factory.mktemp("rvc_chain_e2e_data")
    original = config.get_data_dir()
    config.set_data_dir(str(data_dir))
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        config.set_data_dir(str(original))


def _poll_and_assert_audio(client, generation_id, expected_sr, deadline_s=600):
    deadline = time.time() + deadline_s
    status = "generating"
    error = None
    while time.time() < deadline:
        hist = client.get(f"/history/{generation_id}")
        assert hist.status_code == 200, hist.text
        payload = hist.json()
        status = payload["status"]
        error = payload.get("error")
        if status in ("completed", "failed"):
            break
        time.sleep(2)

    if status == "failed":
        # Backbones/base model are all cached (gated above), so a failure here is
        # a real bug, not an offline skip.
        pytest.fail(f"chained generation failed: {error}")
    assert status == "completed", f"generation did not finish in time (last={status})"

    audio_resp = client.get(f"/audio/{generation_id}")
    assert audio_resp.status_code == 200, audio_resp.text
    data, sr = sf.read(io.BytesIO(audio_resp.content))
    assert sr == expected_sr, f"served WAV sr {sr} != checkpoint sr {expected_sr}"
    if data.ndim > 1:
        data = data.mean(axis=1)
    rms = float(np.sqrt(np.mean(data.astype(np.float64) ** 2)))
    assert rms > 1e-4, f"converted output is effectively silent (rms={rms:.2e})"
    duration = data.shape[0] / sr
    assert 0.2 < duration < 60.0, f"unexpected output duration {duration:.2f}s"
    return duration, rms


@pytest.mark.skipif(not _E2E_READY, reason=_E2E_SKIP_REASON)
def test_generate_and_speak_through_rvc_chain(client):
    from backend.backends.rvc.checkpoint import (
        load_rvc_checkpoint,
        validate_rvc_checkpoint,
    )

    expected_sr = validate_rvc_checkpoint(load_rvc_checkpoint(_CHECKPOINT)).sample_rate

    resp = client.post(
        "/profiles",
        json={
            "name": f"rvc-chain-{uuid.uuid4().hex[:8]}",
            "voice_type": "rvc",
            "rvc_base_voice": "kokoro:af_heart",
        },
    )
    assert resp.status_code == 200, resp.text
    profile = resp.json()
    pid = profile["id"]
    assert profile["voice_type"] == "rvc"
    assert profile["rvc_base_voice"] == "kokoro:af_heart"

    with open(_CHECKPOINT, "rb") as f:
        up = client.post(
            f"/profiles/{pid}/rvc-model",
            files={"model": (Path(_CHECKPOINT).name, f, "application/octet-stream")},
        )
    assert up.status_code == 200, up.text

    # generation tab payload for an rvc profile: engine=null defers to the
    # profile, and the service resolves the base engine (kokoro) server-side.
    gen = client.post(
        "/generate",
        json={"profile_id": pid, "text": "Hello from the chain.", "engine": None},
    )
    assert gen.status_code == 200, gen.text
    _poll_and_assert_audio(client, gen.json()["id"], expected_sr)

    # /speak goes through the same generation service — no engine passed at all.
    sp = client.post("/speak", json={"profile": pid, "text": "Speaking through RVC."})
    assert sp.status_code == 200, sp.text
    _poll_and_assert_audio(client, sp.json()["id"], expected_sr)


# ----------------------------------------------------------------------------
# 4. Chain path with a *profile* base (step 09) — cloned base profile
# ----------------------------------------------------------------------------


def _insert_cloned_base_profile(db, *, language="ru", default_engine="qwen"):
    """Insert a non-rvc cloned profile to be used as an RVC base voice.

    The chain resolves this profile's own generation path
    (``create_voice_prompt_for_profile`` + its resolved engine); those are
    monkeypatched in the test, so no samples/model are needed on disk here.
    """
    from backend.database import VoiceProfile as DBVoiceProfile

    pid = str(uuid.uuid4())
    row = DBVoiceProfile(
        id=pid,
        name=f"cloned-base-{pid[:8]}",
        language=language,
        voice_type="cloned",
        default_engine=default_engine,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    return pid


def test_chain_with_cloned_profile_base(data_env, monkeypatch):
    """A ``profile:{id}`` cloned base renders through the base profile's own path.

    Asserts the same chain invariants as the preset-base test, plus the two that
    are specific to a profile base: the base voice prompt is built via
    ``create_voice_prompt_for_profile`` (single choke point — not a second
    implementation) using the base profile's *resolved engine*, and the request
    ``language`` is passed through to the base TTS stage.
    """
    import asyncio

    from backend import backends as backends_pkg
    from backend.backends import rvc as rvc_pkg
    from backend.services import generation as gen_mod
    from backend.services import profiles as profiles_mod
    from backend.utils import chunked_tts as chunked_mod
    from backend.database import Generation as DBGeneration, get_db

    order: list[str] = []
    base_audio = _sine(seconds=1.0, sr=24000, freq=200.0, amp=0.5)
    BASE_SR = 24000
    RVC_SR = 40000  # deliberately != BASE_SR so the saved-file sr is unambiguous
    custom_params = {
        "f0_up_key": 2,
        "f0_method": "rmvpe",
        "index_rate": 0.6,
        "rms_mix_rate": 0.3,
        "protect": 0.25,
    }

    class _FakeBaseBackend:
        def is_loaded(self):
            return False

        def unload_model(self):
            order.append("unload_base")

    fake_base = _FakeBaseBackend()

    load_calls: list[tuple] = []

    async def _fake_load_engine_model(engine, model_size="default"):
        order.append("load_base")
        load_calls.append((engine, model_size))

    monkeypatch.setattr(backends_pkg, "load_engine_model", _fake_load_engine_model)
    monkeypatch.setattr(backends_pkg, "get_tts_backend_for_engine", lambda e: fake_base)
    # Keep base model-size resolution hermetic (no real backend config lookups).
    monkeypatch.setattr(
        gen_mod, "_resolve_base_model_size", lambda e, language=None: "default"
    )

    cvp_calls: list[dict] = []

    async def _fake_create_voice_prompt(profile_id, db, use_cache=True, engine="qwen", language=None):
        cvp_calls.append({"profile_id": profile_id, "engine": engine})
        return {"voice_type": "cloned", "engine": engine}

    monkeypatch.setattr(
        profiles_mod, "create_voice_prompt_for_profile", _fake_create_voice_prompt
    )

    gen_calls: list[dict] = []

    async def _fake_generate_chunked(backend, text, voice_prompt, **kwargs):
        order.append("tts")
        gen_calls.append(
            {
                "backend": backend,
                "voice_prompt": voice_prompt,
                "language": kwargs.get("language"),
                "runaway_detector": kwargs.get("runaway_detector"),
            }
        )
        return base_audio.copy(), BASE_SR

    monkeypatch.setattr(chunked_mod, "generate_chunked", _fake_generate_chunked)

    class _FakeRVCEngine:
        def __init__(self):
            self.loaded_with = None
            self.convert_calls: list[dict] = []

        def load(self, model_path, index_path=None):
            order.append("load_rvc")
            self.loaded_with = (model_path, index_path)

        def convert_audio(
            self, audio, sr, *, f0_up_key, f0_method, index_rate, rms_mix_rate, protect
        ):
            order.append("rvc")
            self.convert_calls.append(
                {
                    "audio": np.asarray(audio, dtype=np.float32).copy(),
                    "sr": sr,
                    "params": {
                        "f0_up_key": f0_up_key,
                        "f0_method": f0_method,
                        "index_rate": index_rate,
                        "rms_mix_rate": rms_mix_rate,
                        "protect": protect,
                    },
                }
            )
            return np.asarray(audio, dtype=np.float32).copy(), RVC_SR

        def unload(self):
            order.append("unload_rvc")

    fake_rvc = _FakeRVCEngine()
    monkeypatch.setattr(rvc_pkg, "get_rvc_engine", lambda: fake_rvc)

    db = next(get_db())
    try:
        base_id = _insert_cloned_base_profile(db, language="ru", default_engine="qwen")
        pid = _insert_rvc_profile(db, base_voice=f"profile:{base_id}", params=custom_params)
        gid = _insert_generation(db, pid, text="Convert me.", engine="rvc")
    finally:
        db.close()

    expected_model_path = str((config.get_profiles_dir() / pid / "model.pth").resolve())

    asyncio.run(
        gen_mod.run_generation(
            generation_id=gid,
            profile_id=pid,
            text="Convert me.",
            language="ru",
            engine="rvc",
            model_size=None,
            seed=None,
            normalize=False,
            effects_chain=None,
            mode="generate",
        )
    )

    # --- Same stage order as the preset base: TTS renders, base unloads, RVC. ---
    assert order == ["load_base", "tts", "unload_base", "load_rvc", "rvc", "unload_rvc"], order

    # --- Base engine resolved from the base profile's default_engine. ---
    assert load_calls and load_calls[0][0] == "qwen"

    # --- Voice prompt built through the base profile's OWN generation path. ---
    assert cvp_calls == [{"profile_id": base_id, "engine": "qwen"}]

    # --- language passed through to the base TTS stage. ---
    assert len(gen_calls) == 1
    assert gen_calls[0]["language"] == "ru"
    assert gen_calls[0]["runaway_detector"] is not None
    assert gen_calls[0]["backend"] is fake_base
    assert gen_calls[0]["voice_prompt"] == {"voice_type": "cloned", "engine": "qwen"}

    # --- rvc_params verbatim; base audio at base sr handed to convert. ---
    assert len(fake_rvc.convert_calls) == 1
    conv = fake_rvc.convert_calls[0]
    assert conv["params"] == custom_params
    assert conv["sr"] == BASE_SR
    assert np.array_equal(conv["audio"], base_audio)
    assert fake_rvc.loaded_with == (expected_model_path, None)

    # --- Converted array (at RVC_SR) is what reached post-processing / disk. ---
    clean_path = config.get_generations_dir() / f"{gid}.wav"
    assert clean_path.exists(), "converted output was not written by _save_generate"
    saved, saved_sr = sf.read(str(clean_path))
    assert saved_sr == RVC_SR, "base-TTS output bypassed conversion into post-processing"
    assert saved.shape[0] == base_audio.shape[0]

    db = next(get_db())
    try:
        row = db.query(DBGeneration).filter_by(id=gid).first()
        assert row.status == "completed"
    finally:
        db.close()


def test_chain_rejects_rvc_base_at_runtime(data_env):
    """Runtime recursion guard: an rvc base (bypassing create-time validation via a
    direct DB write) fails the generation with a clear error, never silently.

    Validation and execution are separated by time and DB edits, so the chain
    re-asserts the base is non-rvc at the call site. The generation is marked
    ``failed`` with a chain-of-chains message rather than running an rvc-on-rvc
    chain or crashing.
    """
    import asyncio

    from backend.services import generation as gen_mod
    from backend.database import Generation as DBGeneration, get_db

    db = next(get_db())
    try:
        rvc_base_id = _insert_profile(db, voice_type="rvc")
        pid = _insert_rvc_profile(db, base_voice=f"profile:{rvc_base_id}")
        gid = _insert_generation(db, pid, text="No chain of chains.", engine="rvc")
    finally:
        db.close()

    asyncio.run(
        gen_mod.run_generation(
            generation_id=gid,
            profile_id=pid,
            text="No chain of chains.",
            language="en",
            engine="rvc",
            model_size=None,
            seed=None,
            normalize=False,
            effects_chain=None,
            mode="generate",
        )
    )

    db = next(get_db())
    try:
        row = db.query(DBGeneration).filter_by(id=gid).first()
        assert row.status == "failed"
        assert "chain-of-chains" in (row.error or ""), row.error
    finally:
        db.close()


# ----------------------------------------------------------------------------
# 5. Real fixture cloned-base end-to-end (skip-if-absent)
# ----------------------------------------------------------------------------

_SAMPLE_FIXTURE_DIR = _FIXTURE_DIR.parent / "samples"


def _resolve_sample():
    """A real speech wav to clone the base voice from, or None to skip.

    Cloning engines need actual speech (a synthetic tone would clone to noise),
    so this e2e requires a hand-provided fixture under ``fixtures/samples/``.
    """
    if _SAMPLE_FIXTURE_DIR.is_dir():
        matches = sorted(_SAMPLE_FIXTURE_DIR.glob("*.wav"))
        if matches:
            return str(matches[0])
    return None


def _qwen_base_ready():
    """True when a Qwen (default cloned-engine) model is cached locally."""
    try:
        from backend.backends import get_tts_backend_for_engine, get_tts_model_configs

        configs = [c for c in get_tts_model_configs() if c.engine == "qwen"]
        if not configs:
            return False
        backend = get_tts_backend_for_engine("qwen")
        return any(backend._is_model_cached(c.model_size) for c in configs)
    except Exception:
        return False


_SAMPLE = _resolve_sample()
_QWEN_READY = _qwen_base_ready()
_CLONED_E2E_READY = bool(_CHECKPOINT) and _BACKBONES_CACHED and _QWEN_READY and bool(_SAMPLE)
_CLONED_E2E_SKIP_REASON = (
    "cloned-base TTS->RVC e2e needs an RVC checkpoint, the ContentVec/RMVPE "
    "backbones, a cached Qwen cloning model, and a real speech sample under "
    "fixtures/samples/*.wav — one or more is absent: "
    f"checkpoint={bool(_CHECKPOINT)} backbones_cached={_BACKBONES_CACHED} "
    f"qwen_cached={_QWEN_READY} sample={bool(_SAMPLE)}"
)


@pytest.mark.skipif(not _CLONED_E2E_READY, reason=_CLONED_E2E_SKIP_REASON)
def test_cloned_base_chain_e2e(client):
    from backend.backends.rvc.checkpoint import (
        load_rvc_checkpoint,
        validate_rvc_checkpoint,
    )

    expected_sr = validate_rvc_checkpoint(load_rvc_checkpoint(_CHECKPOINT)).sample_rate

    # A cloned base profile (Russian) with a real speech sample.
    base = client.post(
        "/profiles",
        json={
            "name": f"cloned-base-{uuid.uuid4().hex[:8]}",
            "voice_type": "cloned",
            "language": "ru",
        },
    )
    assert base.status_code == 200, base.text
    base_id = base.json()["id"]

    with open(_SAMPLE, "rb") as f:
        s = client.post(
            f"/profiles/{base_id}/samples",
            files={"file": (Path(_SAMPLE).name, f, "audio/wav")},
            data={"reference_text": "Reference sample for cloning."},
        )
    assert s.status_code == 200, s.text

    # The RVC profile chained on top of the cloned base.
    rvc = client.post(
        "/profiles",
        json={
            "name": f"rvc-cloned-{uuid.uuid4().hex[:8]}",
            "voice_type": "rvc",
            "rvc_base_voice": f"profile:{base_id}",
        },
    )
    assert rvc.status_code == 200, rvc.text
    profile = rvc.json()
    pid = profile["id"]
    assert profile["rvc_base_voice"] == f"profile:{base_id}"
    # Effective language defaulted from the base voice.
    assert profile["language"] == "ru"

    with open(_CHECKPOINT, "rb") as f:
        up = client.post(
            f"/profiles/{pid}/rvc-model",
            files={"model": (Path(_CHECKPOINT).name, f, "application/octet-stream")},
        )
    assert up.status_code == 200, up.text

    gen = client.post(
        "/generate",
        json={"profile_id": pid, "text": "Привет из цепочки.", "engine": None},
    )
    assert gen.status_code == 200, gen.text
    _poll_and_assert_audio(client, gen.json()["id"], expected_sr)
