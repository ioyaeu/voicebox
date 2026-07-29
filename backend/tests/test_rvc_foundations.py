"""Foundation tests for the RVC voice-conversion feature (step 01).

Covers the three foundation surfaces that must hold before any RVC engine
work lands:

  * the ``profiles`` column migration that adds ``rvc_model_path`` /
    ``rvc_index_path`` to a *pre-migration* database without losing rows;
  * the ``rvc`` branch of profile field validation (no design_prompt, no
    non-``rvc`` default engine — the old ``kokoro`` allowance is gone);
  * the checkpoint security utility (``weights_only=True`` enforcement plus
    structure extraction).
"""

import os

import pytest
import torch
from sqlalchemy import create_engine, inspect, text

from backend.backends.rvc.checkpoint import (
    RVCCheckpointInfo,
    load_rvc_checkpoint,
    validate_rvc_checkpoint,
)
from backend.database.migrations import run_migrations
from backend.services.profiles import _validate_profile_fields

# Columns the ``profiles`` table had immediately before the RVC feature added
# its two path columns. Built by hand (raw SQL) on purpose: creating the table
# from current SQLAlchemy metadata would already include the RVC columns and
# make the migration assertion vacuous.
_PRE_RVC_PROFILES_DDL = """
    CREATE TABLE profiles (
        id VARCHAR PRIMARY KEY,
        name VARCHAR NOT NULL UNIQUE,
        description TEXT,
        language VARCHAR DEFAULT 'en',
        avatar_path VARCHAR,
        effects_chain TEXT,
        voice_type VARCHAR DEFAULT 'cloned',
        preset_engine VARCHAR,
        preset_voice_id VARCHAR,
        design_prompt TEXT,
        default_engine VARCHAR,
        personality TEXT,
        created_at DATETIME,
        updated_at DATETIME
    )
"""


# -- migration -------------------------------------------------------------

def test_migration_adds_rvc_columns_and_preserves_rows(tmp_path):
    db_path = tmp_path / "pre_rvc.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        conn.execute(text(_PRE_RVC_PROFILES_DDL))
        conn.execute(
            text(
                "INSERT INTO profiles (id, name, voice_type, default_engine) "
                "VALUES (:id, :name, :vt, :de)"
            ),
            {"id": "legacy-1", "name": "Legacy Voice", "vt": "cloned", "de": "qwen"},
        )

    columns_before = {c["name"] for c in inspect(engine).get_columns("profiles")}
    assert "rvc_model_path" not in columns_before
    assert "rvc_index_path" not in columns_before

    run_migrations(engine)

    columns_after = {c["name"] for c in inspect(engine).get_columns("profiles")}
    assert "rvc_model_path" in columns_after
    assert "rvc_index_path" in columns_after

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT id, name, default_engine, rvc_model_path, rvc_index_path "
                "FROM profiles WHERE id = :id"
            ),
            {"id": "legacy-1"},
        ).one()
    assert row.id == "legacy-1"
    assert row.name == "Legacy Voice"
    assert row.default_engine == "qwen"
    # Freshly added nullable columns backfill to NULL on the existing row.
    assert row.rvc_model_path is None
    assert row.rvc_index_path is None


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "pre_rvc_idempotent.db"
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.begin() as conn:
        conn.execute(text(_PRE_RVC_PROFILES_DDL))
        conn.execute(
            text("INSERT INTO profiles (id, name) VALUES (:id, :name)"),
            {"id": "legacy-2", "name": "Idempotent Voice"},
        )

    run_migrations(engine)
    # Second run must be a no-op, not an "duplicate column" error.
    run_migrations(engine)

    columns = {c["name"] for c in inspect(engine).get_columns("profiles")}
    assert {"rvc_model_path", "rvc_index_path"} <= columns


# -- profile field validation ---------------------------------------------

def test_validate_accepts_bare_rvc_profile():
    error = _validate_profile_fields(
        voice_type="rvc",
        preset_engine=None,
        preset_voice_id=None,
        design_prompt=None,
        default_engine=None,
    )
    assert error is None


def test_validate_accepts_rvc_profile_with_rvc_default_engine():
    error = _validate_profile_fields(
        voice_type="rvc",
        preset_engine=None,
        preset_voice_id=None,
        design_prompt=None,
        default_engine="rvc",
    )
    assert error is None


def test_validate_rejects_rvc_profile_with_design_prompt():
    error = _validate_profile_fields(
        voice_type="rvc",
        preset_engine=None,
        preset_voice_id=None,
        design_prompt="a warm narrator",
        default_engine=None,
    )
    assert error is not None
    assert "design_prompt" in error


def test_validate_rejects_rvc_profile_with_kokoro_default_engine():
    error = _validate_profile_fields(
        voice_type="rvc",
        preset_engine=None,
        preset_voice_id=None,
        design_prompt=None,
        default_engine="kokoro",
    )
    assert error is not None
    assert "kokoro" in error


# -- checkpoint security ---------------------------------------------------


def _valid_v2_config() -> list:
    """The full 18-hyperparameter v2/40k config (``spec_channels``..``sr``).

    ``validate_rvc_checkpoint`` reads the arity (must be 18) and the trailing
    sample rate; the interior values only matter when a real synthesizer is built.
    """
    return [
        1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
        [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000,
    ]


def _valid_v2_ckpt() -> dict:
    """A structurally valid v2/40k/f0 checkpoint dict.

    Carries ``emb_g.weight`` (the speaker embedding strictness now demands) and a
    full 18-hyperparameter ``config``; enough to pass ``validate_rvc_checkpoint``
    without being a real synthesizer.
    """
    return {
        "weight": {
            "emb_g.weight": torch.zeros(109, 256),
            "enc_p.emb_phone.weight": torch.zeros(2, 2),
        },
        "config": _valid_v2_config(),
        "f0": 1,
        "version": "v2",
        "sr": 40000,
    }


def test_load_rvc_checkpoint_refuses_malicious_pickle(tmp_path):
    """A checkpoint whose ``__reduce__`` runs code must be refused, and the
    payload must never execute (proving ``weights_only=True`` is enforced)."""
    sentinel = tmp_path / "pwned.txt"

    class Exploit:
        def __reduce__(self):
            # Serialized as a reference to ``os.system`` + args; the weights-only
            # unpickler must reject the global before it can run.
            return (os.system, (f'touch "{sentinel}"',))

    malicious_path = tmp_path / "malicious.pth"
    torch.save({"weight": Exploit()}, malicious_path)

    with pytest.raises(Exception):  # noqa: B017,PT011 - any refusal is acceptable
        load_rvc_checkpoint(str(malicious_path))

    assert not sentinel.exists(), "weights_only load executed the pickle payload"


def test_validate_rvc_checkpoint_extracts_v2_metadata(tmp_path):
    ckpt = _valid_v2_ckpt()
    path = tmp_path / "v2.pth"
    torch.save(ckpt, path)

    loaded = load_rvc_checkpoint(str(path))
    info = validate_rvc_checkpoint(loaded)

    assert isinstance(info, RVCCheckpointInfo)
    assert info.version == "v2"
    assert info.sample_rate == 40000
    assert info.if_f0 == 1
    assert info.embedder_dim == 768


def test_validate_rvc_checkpoint_v1_reads_sample_rate_from_config(tmp_path):
    # No top-level "sr": the extractor must fall back to the config tail, and a
    # v1 checkpoint must resolve to a 256-wide embedder with f0 disabled. The
    # config still carries the full 18 hyperparameters (strictness); only its
    # trailing sample rate differs from the v2 fixture.
    config = _valid_v2_config()
    config[-1] = 32000
    ckpt = {
        "weight": {"emb_g.weight": torch.zeros(109, 256)},
        "config": config,
        "f0": 0,
        "version": "v1",
    }
    path = tmp_path / "v1.pth"
    torch.save(ckpt, path)

    info = validate_rvc_checkpoint(load_rvc_checkpoint(str(path)))

    assert info.version == "v1"
    assert info.sample_rate == 32000
    assert info.if_f0 == 0
    assert info.embedder_dim == 256


# -- checkpoint strictness (malformed fixtures die at validation / build) ---

def test_validate_rejects_missing_emb_g_weight():
    """A state dict without ``emb_g.weight`` is rejected at validation.

    ``build_synthesizer`` sizes the speaker embedding from
    ``weight['emb_g.weight']``; without it conversion would die with a raw
    ``KeyError`` (a 500). Validation turns it into a user-facing message (a 400 at
    upload).
    """
    ckpt = _valid_v2_ckpt()
    del ckpt["weight"]["emb_g.weight"]

    with pytest.raises(ValueError, match="emb_g.weight"):
        validate_rvc_checkpoint(ckpt)


def test_validate_rejects_wrong_config_arity():
    """A ``config`` that is not the required 18 hyperparameters is rejected.

    ``build_synthesizer`` splats ``config`` as ``cls(*config)`` and indexes
    ``config[-3]``; a shorter list crashes conversion with a raw
    ``TypeError``/``IndexError``. Validation catches it up-front (a 400 at upload).
    """
    ckpt = _valid_v2_ckpt()
    ckpt["config"] = _valid_v2_config()[:11]

    with pytest.raises(ValueError, match="18"):
        validate_rvc_checkpoint(ckpt)


def test_build_synthesizer_raises_naming_missing_weights():
    """A checkpoint that passes structure validation but under-fills the
    synthesizer (missing weight tensors) is rejected by ``build_synthesizer``,
    which names the missing keys instead of running inference on garbage.

    This is the "silent garbage audio" class the plan bans: ``load_state_dict``
    here uses ``strict=False`` (a full generator checkpoint carries extra
    training-only ``enc_q`` tensors as *unexpected* keys), so *missing* keys must
    be asserted explicitly rather than discarded.
    """
    from backend.backends.rvc.synthesizer import build_synthesizer

    ckpt = _valid_v2_ckpt()
    # Keep only the speaker embedding; every other synthesizer tensor is absent,
    # so load_state_dict reports a large missing-keys set.
    ckpt["weight"] = {"emb_g.weight": torch.zeros(109, 256)}
    info = validate_rvc_checkpoint(ckpt)  # structure is still valid

    with pytest.raises(ValueError) as excinfo:
        build_synthesizer(ckpt, info)

    msg = str(excinfo.value)
    assert "missing" in msg.lower()
    # The message names concrete state-dict keys (dotted paths), not just a count.
    assert "." in msg
