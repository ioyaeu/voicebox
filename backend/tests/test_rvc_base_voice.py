"""Step 09 — cloned/designed/preset profiles as RVC base voices.

Covers the ``profile:{id}`` base-voice grammar added in step 09 to
``services/profiles.py``:

* :func:`_validate_rvc_base_voice` — accepts a non-rvc profile reference
  (cloned-with-samples, designed-with-prompt, preset), normalizes a preset
  profile to its ``engine:voice_id`` form, and rejects the recursion/garbage
  cases (rvc base, missing profile, sample-less cloned, promptless designed,
  metadata-less preset, missing db session).
* :func:`create_profile`/:func:`update_profile` — persist a valid profile base
  and default the rvc profile's effective ``language`` from the base voice
  (user-overridable).
* :func:`find_rvc_base_dependents` + ``DELETE /profiles/{id}`` — deleting a
  profile that is in use as an RVC base is blocked with a 409 that names the
  dependent rvc profiles (name + id); deleting after unlinking succeeds.

These are the contracts B1 landed on disk (read them there — do not assume);
this file pins them. No model or network is touched.
"""

import shutil
import tempfile
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend import config
from backend.database import (
    Base,
    ProfileSample as DBProfileSample,
    VoiceProfile as DBVoiceProfile,
)
from backend.models import VoiceProfileCreate
from backend.services.profiles import (
    RVC_PROFILE_BASE_PREFIX,
    _validate_rvc_base_voice,
    create_profile,
    find_rvc_base_dependents,
    update_profile,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def test_db():
    """A throwaway SQLite session with the full schema (profiles + samples)."""
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test.db"
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_profiles_dir(monkeypatch, tmp_path):
    """Point ``config.get_profiles_dir`` at a temp dir so create_profile mkdirs there."""
    monkeypatch.setattr(config, "get_profiles_dir", lambda: tmp_path)
    return tmp_path


def _insert_base_profile(
    db,
    *,
    voice_type,
    language="en",
    design_prompt=None,
    preset_engine=None,
    preset_voice_id=None,
    default_engine=None,
    with_sample=False,
    rvc_base_voice=None,
):
    """Insert a base profile row directly (bypassing create_profile validation)."""
    pid = str(uuid.uuid4())
    row = DBVoiceProfile(
        id=pid,
        name=f"{voice_type}-{pid[:8]}",
        language=language,
        voice_type=voice_type,
        design_prompt=design_prompt,
        preset_engine=preset_engine,
        preset_voice_id=preset_voice_id,
        default_engine=default_engine,
        rvc_base_voice=rvc_base_voice,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(row)
    db.commit()
    if with_sample:
        db.add(
            DBProfileSample(
                id=str(uuid.uuid4()),
                profile_id=pid,
                audio_path=f"{pid}.wav",
                reference_text="reference sample",
            )
        )
        db.commit()
    return pid


# ---------------------------------------------------------------------------
# 1. _validate_rvc_base_voice — profile:{id} grammar (accepted cases)
# ---------------------------------------------------------------------------


def test_validate_profile_base_cloned_with_sample(test_db):
    base_id = _insert_base_profile(test_db, voice_type="cloned", language="ru", with_sample=True)
    error, normalized, language = _validate_rvc_base_voice(f"profile:{base_id}", test_db)
    assert error is None, error
    # A cloned base is stored verbatim as profile:{id} (only preset collapses).
    assert normalized == f"profile:{base_id}"
    assert language == "ru"


def test_validate_profile_base_designed_with_prompt(test_db):
    base_id = _insert_base_profile(
        test_db, voice_type="designed", language="fr", design_prompt="A calm narrator."
    )
    error, normalized, language = _validate_rvc_base_voice(f"profile:{base_id}", test_db)
    assert error is None, error
    assert normalized == f"profile:{base_id}"
    assert language == "fr"


def test_validate_profile_base_preset_collapses_to_engine_voice(test_db):
    base_id = _insert_base_profile(
        test_db,
        voice_type="preset",
        language="fr",
        preset_engine="kokoro",
        preset_voice_id="ff_siwis",
        default_engine="kokoro",
    )
    error, normalized, language = _validate_rvc_base_voice(f"profile:{base_id}", test_db)
    assert error is None, error
    # A preset profile base collapses to its engine:voice_id so the chain has one
    # resolution path and never stores a profile ref pointing at a preset.
    assert normalized == "kokoro:ff_siwis"
    assert language == "fr"


# ---------------------------------------------------------------------------
# 2. _validate_rvc_base_voice — rejected cases
# ---------------------------------------------------------------------------


def test_validate_profile_base_rvc_is_rejected(test_db):
    base_id = _insert_base_profile(test_db, voice_type="rvc")
    error, normalized, language = _validate_rvc_base_voice(f"profile:{base_id}", test_db)
    assert error is not None
    assert "chain-of-chains" in error
    assert normalized == f"profile:{base_id}"
    assert language is None


def test_validate_profile_base_missing_is_rejected(test_db):
    missing = str(uuid.uuid4())
    error, _normalized, language = _validate_rvc_base_voice(f"profile:{missing}", test_db)
    assert error is not None
    assert "does not exist" in error
    assert language is None


def test_validate_profile_base_cloned_without_samples_is_rejected(test_db):
    base_id = _insert_base_profile(test_db, voice_type="cloned", with_sample=False)
    error, _normalized, _language = _validate_rvc_base_voice(f"profile:{base_id}", test_db)
    assert error is not None
    assert "no voice samples" in error


def test_validate_profile_base_designed_without_prompt_is_rejected(test_db):
    base_id = _insert_base_profile(test_db, voice_type="designed", design_prompt=None)
    error, _normalized, _language = _validate_rvc_base_voice(f"profile:{base_id}", test_db)
    assert error is not None
    assert "design prompt" in error


def test_validate_profile_base_preset_without_metadata_is_rejected(test_db):
    base_id = _insert_base_profile(
        test_db, voice_type="preset", preset_engine=None, preset_voice_id=None
    )
    error, _normalized, _language = _validate_rvc_base_voice(f"profile:{base_id}", test_db)
    assert error is not None
    assert "engine metadata" in error


def test_validate_profile_base_empty_id_is_rejected(test_db):
    error, _normalized, _language = _validate_rvc_base_voice("profile:", test_db)
    assert error is not None
    assert "missing profile id" in error


def test_validate_profile_base_requires_db_session():
    # A profile ref cannot be resolved without a session — this is a clear error,
    # not a crash at generation time.
    error, _normalized, _language = _validate_rvc_base_voice(f"profile:{uuid.uuid4()}", None)
    assert error is not None
    assert "database session" in error


# ---------------------------------------------------------------------------
# 3. _validate_rvc_base_voice — engine:voice_id grammar still works
# ---------------------------------------------------------------------------


def test_validate_preset_engine_grammar_ok():
    error, normalized, language = _validate_rvc_base_voice("kokoro:ff_siwis", None)
    assert error is None, error
    assert normalized == "kokoro:ff_siwis"
    assert language == "fr"


def test_validate_preset_engine_grammar_bad_engine():
    error, _normalized, _language = _validate_rvc_base_voice("bogus:whatever", None)
    assert error is not None
    assert "engine" in error


def test_validate_preset_engine_grammar_bad_voice():
    error, _normalized, _language = _validate_rvc_base_voice("kokoro:not_a_real_voice", None)
    assert error is not None
    assert "not a valid" in error


# ---------------------------------------------------------------------------
# 4. create_profile / update_profile — persist base + default effective language
# ---------------------------------------------------------------------------


async def test_create_rvc_profile_with_cloned_profile_base(test_db, mock_profiles_dir):
    base_id = _insert_base_profile(test_db, voice_type="cloned", language="ru", with_sample=True)

    created = await create_profile(
        VoiceProfileCreate(
            name="rvc-on-cloned",
            voice_type="rvc",
            rvc_base_voice=f"profile:{base_id}",
        ),
        test_db,
    )
    assert created.voice_type == "rvc"
    assert created.rvc_base_voice == f"profile:{base_id}"
    # Effective language defaulted from the base (create left it at "en").
    assert created.language == "ru"


async def test_create_rvc_profile_with_preset_profile_base_is_normalized(test_db, mock_profiles_dir):
    base_id = _insert_base_profile(
        test_db,
        voice_type="preset",
        language="fr",
        preset_engine="kokoro",
        preset_voice_id="ff_siwis",
        default_engine="kokoro",
    )
    created = await create_profile(
        VoiceProfileCreate(
            name="rvc-on-preset",
            voice_type="rvc",
            rvc_base_voice=f"profile:{base_id}",
        ),
        test_db,
    )
    # A preset profile base is normalized to engine:voice_id at write time.
    assert created.rvc_base_voice == "kokoro:ff_siwis"
    assert created.language == "fr"


async def test_create_rvc_profile_language_default_from_preset_engine_base(test_db, mock_profiles_dir):
    created = await create_profile(
        VoiceProfileCreate(
            name="rvc-preset-engine-fr",
            voice_type="rvc",
            rvc_base_voice="kokoro:ff_siwis",  # French preset voice
        ),
        test_db,
    )
    assert created.rvc_base_voice == "kokoro:ff_siwis"
    assert created.language == "fr"


async def test_create_rvc_profile_language_user_override_honored(test_db, mock_profiles_dir):
    base_id = _insert_base_profile(test_db, voice_type="cloned", language="ru", with_sample=True)
    created = await create_profile(
        VoiceProfileCreate(
            name="rvc-override-de",
            voice_type="rvc",
            language="de",  # explicit, non-default → must win over the base default
            rvc_base_voice=f"profile:{base_id}",
        ),
        test_db,
    )
    assert created.language == "de"


async def test_create_rvc_profile_rejects_rvc_base(test_db, mock_profiles_dir):
    rvc_base_id = _insert_base_profile(test_db, voice_type="rvc")
    with pytest.raises(ValueError) as exc:
        await create_profile(
            VoiceProfileCreate(
                name="rvc-on-rvc",
                voice_type="rvc",
                rvc_base_voice=f"profile:{rvc_base_id}",
            ),
            test_db,
        )
    assert "chain-of-chains" in str(exc.value)


async def test_create_rvc_profile_rejects_missing_base(test_db, mock_profiles_dir):
    with pytest.raises(ValueError) as exc:
        await create_profile(
            VoiceProfileCreate(
                name="rvc-on-missing",
                voice_type="rvc",
                rvc_base_voice=f"profile:{uuid.uuid4()}",
            ),
            test_db,
        )
    assert "does not exist" in str(exc.value)


async def test_update_rvc_profile_defaults_language_from_new_base(test_db, mock_profiles_dir):
    # Start with an English preset base.
    created = await create_profile(
        VoiceProfileCreate(
            name="rvc-relanguage",
            voice_type="rvc",
            rvc_base_voice="kokoro:af_heart",
        ),
        test_db,
    )
    assert created.language == "en"

    ru_base_id = _insert_base_profile(test_db, voice_type="cloned", language="ru", with_sample=True)
    updated = await update_profile(
        created.id,
        VoiceProfileCreate(
            name="rvc-relanguage",
            voice_type="rvc",
            rvc_base_voice=f"profile:{ru_base_id}",
        ),
        test_db,
    )
    assert updated is not None
    assert updated.rvc_base_voice == f"profile:{ru_base_id}"
    # Language left at the schema default "en" on update → defaulted from the base.
    assert updated.language == "ru"


# ---------------------------------------------------------------------------
# 5. find_rvc_base_dependents + DELETE /profiles/{id} → 409 block
# ---------------------------------------------------------------------------


def test_find_rvc_base_dependents_lists_name_and_id(test_db):
    base_id = _insert_base_profile(
        test_db, voice_type="designed", design_prompt="A base voice."
    )
    dep_id = _insert_base_profile(
        test_db, voice_type="rvc", rvc_base_voice=f"profile:{base_id}"
    )
    # A preset-normalized rvc profile must NOT count as a dependent of the base.
    _insert_base_profile(test_db, voice_type="rvc", rvc_base_voice="kokoro:af_heart")

    dependents = find_rvc_base_dependents(base_id, test_db)
    assert dependents == [{"id": dep_id, "name": f"rvc-{dep_id[:8]}"}]
    # Nothing depends on the standalone rvc profile itself.
    assert find_rvc_base_dependents(dep_id, test_db) == []


@pytest.fixture
def client(tmp_path):
    """TestClient over the real app on a throwaway data dir (runs the lifespan)."""
    from starlette.testclient import TestClient

    from backend.app import app

    original = config.get_data_dir()
    config.set_data_dir(str(tmp_path))
    try:
        with TestClient(app) as c:
            yield c
    finally:
        config.set_data_dir(str(original))


def test_delete_profile_in_use_as_base_is_blocked_409(client):
    # A designed base needs no samples/model, so this exercises the delete-block
    # path without any audio upload.
    base_resp = client.post(
        "/profiles",
        json={
            "name": f"base-{uuid.uuid4().hex[:8]}",
            "voice_type": "designed",
            "design_prompt": "A warm base narrator voice.",
        },
    )
    assert base_resp.status_code == 200, base_resp.text
    base = base_resp.json()
    base_id = base["id"]

    rvc_resp = client.post(
        "/profiles",
        json={
            "name": f"rvc-{uuid.uuid4().hex[:8]}",
            "voice_type": "rvc",
            "rvc_base_voice": f"{RVC_PROFILE_BASE_PREFIX}{base_id}",
        },
    )
    assert rvc_resp.status_code == 200, rvc_resp.text
    rvc = rvc_resp.json()
    assert rvc["rvc_base_voice"] == f"{RVC_PROFILE_BASE_PREFIX}{base_id}"

    # Deleting the in-use base is blocked and names the dependent (id + name).
    blocked = client.delete(f"/profiles/{base_id}")
    assert blocked.status_code == 409, blocked.text
    detail = blocked.json()["detail"]
    assert detail["dependents"] == [{"id": rvc["id"], "name": rvc["name"]}]
    assert "1 RVC" in detail["message"]
    # The base is still present after a blocked delete.
    assert client.get(f"/profiles/{base_id}").status_code == 200

    # Unlink the dependent (repoint at a preset), then the delete succeeds.
    unlink = client.put(
        f"/profiles/{rvc['id']}",
        json={"name": rvc["name"], "voice_type": "rvc", "rvc_base_voice": "kokoro:af_heart"},
    )
    assert unlink.status_code == 200, unlink.text

    ok = client.delete(f"/profiles/{base_id}")
    assert ok.status_code == 200, ok.text
    assert client.get(f"/profiles/{base_id}").status_code == 404
