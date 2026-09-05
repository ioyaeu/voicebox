"""Voice profile management module."""

import json as _json
import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path

from sqlalchemy import func
from sqlalchemy.orm import Session

from .. import config
from ..database import Generation as DBGeneration, ProfileSample as DBProfileSample, VoiceProfile as DBVoiceProfile
from ..models import (
    EffectConfig,
    ProfileSampleResponse,
    RVCParams,
    VoiceProfileCreate,
    VoiceProfileResponse,
)
from ..utils.audio import save_audio, validate_and_load_reference_audio
from ..utils.cache import _get_cache_dir, clear_profile_cache
from ..utils.images import process_avatar, validate_image

logger = logging.getLogger(__name__)

CLONING_ENGINES = {"qwen", "luxtts", "chatterbox", "chatterbox_turbo", "tada"}

RVC_MODEL_FILENAME = "model.pth"
RVC_INDEX_FILENAME = "model.index"
RVC_METADATA_FILENAME = "rvc_model.json"

# Default base TTS voice for the TTS->RVC chain: a cheap, CPU-friendly,
# cloning-free Kokoro preset (verified present in KOKORO_VOICES). Stored as
# "{engine}:{voice_id}" — see ``parse_rvc_base_voice``.
DEFAULT_RVC_BASE_VOICE = "kokoro:af_heart"

# Default conversion params, identical to step 03's ConvertRequest defaults.
DEFAULT_RVC_PARAMS: dict = {
    "f0_up_key": 0,
    "f0_method": "rmvpe",
    "index_rate": 0.75,
    "rms_mix_rate": 0.25,
    "protect": 0.33,
}


def parse_rvc_base_voice(raw: str | None) -> tuple[str, str]:
    """Split a stored ``"{engine}:{voice_id}"`` base-voice string.

    Falls back to :data:`DEFAULT_RVC_BASE_VOICE` when the value is missing or
    malformed (no colon, or an empty half), so a bad/absent setting can never
    crash the chain — it just uses the default preset voice.
    """
    if raw and ":" in raw:
        engine, voice_id = raw.split(":", 1)
        engine, voice_id = engine.strip(), voice_id.strip()
        if engine and voice_id:
            return engine, voice_id
    engine, voice_id = DEFAULT_RVC_BASE_VOICE.split(":", 1)
    return engine, voice_id


def load_rvc_params(raw: str | None) -> dict:
    """Deserialize the profile's ``rvc_params`` JSON, merged over the defaults.

    Returns a dict with all five keys always present. Unknown/invalid stored
    JSON degrades to the defaults (logged) rather than raising.
    """
    params = dict(DEFAULT_RVC_PARAMS)
    if not raw:
        return params
    try:
        stored = _json.loads(raw)
    except (ValueError, TypeError) as e:
        logger.warning("Invalid rvc_params JSON, using defaults: %s", e)
        return params
    if isinstance(stored, dict):
        for key in DEFAULT_RVC_PARAMS:
            if key in stored and stored[key] is not None:
                params[key] = stored[key]
    return params


def _rvc_metadata_path(profile_id: str) -> Path:
    return config.get_profiles_dir() / profile_id / RVC_METADATA_FILENAME


def _read_rvc_metadata(profile_id: str) -> dict:
    """Read the RVC checkpoint metadata sidecar; empty dict if absent/unreadable."""
    path = _rvc_metadata_path(profile_id)
    if not path.is_file():
        return {}
    try:
        data = _json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError) as e:
        logger.warning("Failed to read RVC metadata for profile %s: %s", profile_id, e)
        return {}


def _profile_to_response(
    profile: DBVoiceProfile,
    generation_count: int = 0,
    sample_count: int = 0,
) -> VoiceProfileResponse:
    """Convert a DB profile to a VoiceProfileResponse, deserializing effects_chain."""
    effects_chain = None
    if profile.effects_chain:
        try:
            raw = _json.loads(profile.effects_chain)
            effects_chain = [EffectConfig(**e) for e in raw]
        except Exception as e:
            import logging

            logging.warning(f"Failed to parse effects_chain for profile {profile.id}: {e}")

    rvc_model_stored = getattr(profile, "rvc_model_path", None)
    # rvc_has_model reflects an actually-present validated checkpoint on disk, not
    # merely a non-null column: a moved/pruned data dir can strand the column.
    rvc_has_model = False
    if rvc_model_stored:
        resolved_model = config.resolve_storage_path(rvc_model_stored)
        rvc_has_model = resolved_model is not None and resolved_model.exists()
    rvc_meta = _read_rvc_metadata(profile.id) if rvc_model_stored else {}

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    rvc_base_voice = getattr(profile, "rvc_base_voice", None)
    rvc_params = None
    if voice_type == "rvc":
        # Surface sensible defaults for rvc profiles even when the columns are
        # NULL (profiles created before the chain, or via the create endpoint
        # without these fields) so the editor always renders real values.
        rvc_base_voice = rvc_base_voice or DEFAULT_RVC_BASE_VOICE
        rvc_params = RVCParams(**load_rvc_params(getattr(profile, "rvc_params", None)))

    return VoiceProfileResponse(
        id=profile.id,
        name=profile.name,
        description=profile.description,
        language=profile.language,
        avatar_path=profile.avatar_path,
        effects_chain=effects_chain,
        voice_type=voice_type,
        preset_engine=getattr(profile, "preset_engine", None),
        preset_voice_id=getattr(profile, "preset_voice_id", None),
        design_prompt=getattr(profile, "design_prompt", None),
        default_engine=getattr(profile, "default_engine", None),
        rvc_has_model=rvc_has_model,
        rvc_base_voice=rvc_base_voice,
        rvc_params=rvc_params,
        rvc_version=rvc_meta.get("version"),
        rvc_sample_rate=rvc_meta.get("sample_rate"),
        rvc_f0=rvc_meta.get("if_f0"),
        personality=getattr(profile, "personality", None),
        generation_count=generation_count,
        sample_count=sample_count,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
    )


# Engines whose preset voices are cloning-free and thus valid base voices for
# the TTS->RVC chain (mirrors ``parse_rvc_base_voice`` / DEFAULT_RVC_BASE_VOICE).
RVC_BASE_VOICE_ENGINES = {"kokoro", "qwen_custom_voice", "voxtral"}

# Sentinel prefix distinguishing a profile-base (``"profile:{profile_id}"``) from
# a preset engine base (``"{engine}:{voice_id}"``). No preset engine is named
# "profile", so ``startswith`` disambiguates the two grammars unambiguously.
RVC_PROFILE_BASE_PREFIX = "profile:"


def rvc_base_is_profile_ref(rvc_base_voice: str | None) -> bool:
    """True when a stored base voice names a Voicebox profile, not a preset."""
    return bool(rvc_base_voice) and rvc_base_voice.startswith(RVC_PROFILE_BASE_PREFIX)


def _preset_voice_language(engine: str, voice_id: str) -> str | None:
    """Language code a preset ``engine:voice_id`` base voice speaks, or None.

    Reads the same per-voice tables ``routes/profiles.list_preset_voices`` serves,
    so the language shown in the picker and the language defaulted onto an rvc
    profile come from one source of truth.
    """
    if engine == "kokoro":
        from ..backends.kokoro_backend import KOKORO_VOICES

        for vid, _name, _gender, lang in KOKORO_VOICES:
            if vid == voice_id:
                return lang
    elif engine == "qwen_custom_voice":
        from ..backends.qwen_custom_voice_backend import QWEN_CUSTOM_VOICES

        for vid, _name, _gender, lang, _desc in QWEN_CUSTOM_VOICES:
            if vid == voice_id:
                return lang
    elif engine == "voxtral":
        from ..backends.voxtral_backend import VOXTRAL_VOICES

        for vid, _name, _gender, lang in VOXTRAL_VOICES:
            if vid == voice_id:
                return lang
    return None


def _validate_rvc_base_voice(
    rvc_base_voice: str, db: Session | None = None
) -> tuple[str | None, str, str | None]:
    """Validate and normalize an rvc base-voice string.

    Two grammars are accepted:

    * ``"{engine}:{voice_id}"`` — a cloning-free preset voice
      (``kokoro``/``qwen_custom_voice``/``voxtral``) plus a voice id that
      engine offers.
    * ``"profile:{profile_id}"`` — an existing non-rvc Voicebox profile used as
      the base. The profile must exist, must **not** itself be an rvc profile
      (one-level recursion guard, no chain-of-chains), and must be
      generation-capable for its type (cloned → at least one sample; designed →
      a design prompt). A **preset** profile is legal and is normalized to its
      ``"{engine}:{voice_id}"`` form so the chain has a single resolution path.

    Returns ``(error, normalized_value, base_language)``. ``error`` is a
    user-facing message (turns a garbage value into a clear 400 rather than a
    silent bad setting that only fails at generation) or None when valid.
    ``normalized_value`` is what should be persisted — identical to the input for
    every case except a preset-profile base, which collapses to its engine voice.
    ``base_language`` is the language code the base voice speaks (used to default
    the rvc profile's ``language`` column) or None when undeterminable.
    """
    # ── profile:{id} grammar ──────────────────────────────────────────────
    if rvc_base_is_profile_ref(rvc_base_voice):
        base_id = rvc_base_voice[len(RVC_PROFILE_BASE_PREFIX):].strip()
        if not base_id:
            return (
                f"Invalid rvc_base_voice '{rvc_base_voice}': missing profile id.",
                rvc_base_voice,
                None,
            )
        if db is None:
            return (
                "Validating a profile base voice requires a database session.",
                rvc_base_voice,
                None,
            )
        base = db.query(DBVoiceProfile).filter_by(id=base_id).first()
        if base is None:
            return (f"Base profile '{base_id}' does not exist.", rvc_base_voice, None)
        base_type = getattr(base, "voice_type", None) or "cloned"
        # Recursion guard — one level only.
        if base_type == "rvc":
            return (
                "An RVC profile cannot be used as the base voice for another RVC "
                "profile (no chain-of-chains).",
                rvc_base_voice,
                None,
            )
        # Preset profiles collapse to the engine:voice_id path.
        if base_type == "preset":
            if not base.preset_engine or not base.preset_voice_id:
                return (
                    f"Base profile '{base.name}' is a preset profile missing its "
                    "engine metadata.",
                    rvc_base_voice,
                    None,
                )
            return None, f"{base.preset_engine}:{base.preset_voice_id}", base.language
        # Designed profiles need a design prompt to be generation-capable.
        if base_type == "designed":
            if not (getattr(base, "design_prompt", None) or "").strip():
                return (
                    f"Base profile '{base.name}' is a designed profile without a "
                    "design prompt, so it cannot generate audio.",
                    rvc_base_voice,
                    None,
                )
            return None, rvc_base_voice, base.language
        # Cloned (default): needs at least one reference sample.
        sample_count = db.query(DBProfileSample).filter_by(profile_id=base_id).count()
        if sample_count == 0:
            return (
                f"Base profile '{base.name}' has no voice samples, so it cannot "
                "generate audio.",
                rvc_base_voice,
                None,
            )
        return None, rvc_base_voice, base.language

    # ── engine:voice_id grammar ───────────────────────────────────────────
    if ":" not in rvc_base_voice:
        return (
            f"Invalid rvc_base_voice '{rvc_base_voice}': expected "
            "'engine:voice_id' (e.g. 'kokoro:af_heart') or 'profile:{id}'.",
            rvc_base_voice,
            None,
        )
    engine, voice_id = rvc_base_voice.split(":", 1)
    engine, voice_id = engine.strip(), voice_id.strip()
    if engine not in RVC_BASE_VOICE_ENGINES:
        return (
            f"Invalid rvc_base_voice engine '{engine}': must be one of "
            f"{sorted(RVC_BASE_VOICE_ENGINES)}.",
            rvc_base_voice,
            None,
        )
    if not voice_id:
        return (
            f"Invalid rvc_base_voice '{rvc_base_voice}': missing voice id.",
            rvc_base_voice,
            None,
        )
    available_voice_ids = _get_preset_voice_ids(engine)
    if available_voice_ids and voice_id not in available_voice_ids:
        return (f"Voice '{voice_id}' is not a valid {engine} voice.", rvc_base_voice, None)
    return None, rvc_base_voice, _preset_voice_language(engine, voice_id)


def find_rvc_base_dependents(profile_id: str, db: Session) -> list[dict]:
    """RVC profiles that reference *profile_id* as their base voice.

    A profile base is stored as ``"profile:{id}"`` (preset bases are normalized to
    ``engine:voice_id`` at write time and never point at a profile), so an exact
    match on that string finds every dependent. Returned as ``{"id", "name"}``
    dicts, ordered by name, for the delete-block 409 payload.
    """
    ref = f"{RVC_PROFILE_BASE_PREFIX}{profile_id}"
    rows = (
        db.query(DBVoiceProfile.id, DBVoiceProfile.name)
        .filter(DBVoiceProfile.voice_type == "rvc")
        .filter(DBVoiceProfile.rvc_base_voice == ref)
        .order_by(DBVoiceProfile.name)
        .all()
    )
    return [{"id": rid, "name": rname} for rid, rname in rows]


def _get_preset_voice_ids(engine: str) -> set[str]:
    if engine == "kokoro":
        from ..backends.kokoro_backend import KOKORO_VOICES

        return {voice_id for voice_id, _name, _gender, _lang in KOKORO_VOICES}

    if engine == "qwen_custom_voice":
        from ..backends.qwen_custom_voice_backend import QWEN_CUSTOM_VOICES

        return {voice_id for voice_id, _name, _gender, _lang, _desc in QWEN_CUSTOM_VOICES}

    if engine == "voxtral":
        from ..backends.voxtral_backend import VOXTRAL_VOICES

        return {voice_id for voice_id, _name, _gender, _lang in VOXTRAL_VOICES}

    return set()


def _validate_profile_fields(
    *,
    voice_type: str,
    preset_engine: str | None,
    preset_voice_id: str | None,
    design_prompt: str | None,
    default_engine: str | None,
) -> str | None:
    if voice_type == "preset":
        if not preset_engine or not preset_voice_id:
            return "Preset profiles require both preset_engine and preset_voice_id"
        if default_engine and default_engine != preset_engine:
            return "Preset profiles must use their preset_engine as default_engine"

        available_voice_ids = _get_preset_voice_ids(preset_engine)
        if available_voice_ids and preset_voice_id not in available_voice_ids:
            return f"Preset voice '{preset_voice_id}' is not valid for engine '{preset_engine}'"
        return None

    if voice_type == "designed":
        if not design_prompt or not design_prompt.strip():
            return "Designed profiles require a design_prompt"
        if preset_engine or preset_voice_id:
            return "Designed profiles cannot set preset_engine or preset_voice_id"
        return None

    if voice_type == "rvc":
        if preset_engine or preset_voice_id:
            return "RVC profiles cannot set preset_engine or preset_voice_id"
        if design_prompt:
            return "RVC profiles cannot set design_prompt"
        if default_engine and default_engine != "rvc":
            return f"RVC profiles cannot use default engine '{default_engine}'"
        # The base voice (``rvc_base_voice``) is validated + normalized separately
        # in create_profile/update_profile, where a DB session is available to
        # resolve a ``profile:{id}`` base. (rvc_params bounds — f0_up_key, rates,
        # f0_method — are already enforced by the RVCParams pydantic model.)
        return None

    if preset_engine or preset_voice_id:
        return "Cloned profiles cannot set preset_engine or preset_voice_id"
    if design_prompt:
        return "Cloned profiles cannot set design_prompt"
    if default_engine and default_engine not in CLONING_ENGINES:
        return f"Cloned profiles cannot use default engine '{default_engine}'"
    return None


def validate_profile_engine(profile, engine: str) -> None:
    voice_type = getattr(profile, "voice_type", None) or "cloned"

    if voice_type == "preset":
        preset_engine = getattr(profile, "preset_engine", None)
        preset_voice_id = getattr(profile, "preset_voice_id", None)
        if not preset_engine or not preset_voice_id:
            raise ValueError(f"Preset profile {profile.id} is missing preset engine metadata")
        if preset_engine != engine:
            raise ValueError(
                f"Preset profile {profile.id} only supports engine '{preset_engine}', not '{engine}'"
            )
        return

    if voice_type == "designed":
        design_prompt = getattr(profile, "design_prompt", None)
        if not design_prompt or not design_prompt.strip():
            raise ValueError(f"Designed profile {profile.id} is missing design_prompt")
        return

    if voice_type == "rvc":
        if engine != "rvc":
            raise ValueError(f"Engine '{engine}' is not supported by RVC profiles")
        return

    if engine not in CLONING_ENGINES:
        raise ValueError(f"Engine '{engine}' does not support cloned voice profiles")


async def create_profile(
    data: VoiceProfileCreate,
    db: Session,
) -> VoiceProfileResponse:
    """
    Create a new voice profile.

    Args:
        data: Profile creation data
        db: Database session

    Returns:
        Created profile

    Raises:
        ValueError: If a profile with the same name already exists
    """
    existing_profile = db.query(DBVoiceProfile).filter_by(name=data.name).first()
    if existing_profile:
        raise ValueError(f"A profile with the name '{data.name}' already exists. Please choose a different name.")

    # Auto-set default_engine for preset profiles
    default_engine = data.default_engine
    voice_type = data.voice_type or "cloned"
    if voice_type == "preset" and data.preset_engine and not default_engine:
        default_engine = data.preset_engine
    # RVC profiles resolve their base engine internally; the external contract
    # still requires engine == "rvc", reached through default_engine.
    if voice_type == "rvc" and not default_engine:
        default_engine = "rvc"

    # Persist chain settings for rvc profiles (with defaults); leave NULL otherwise.
    rvc_base_voice = None
    rvc_params_json = None
    rvc_base_language = None
    if voice_type == "rvc":
        raw_base_voice = data.rvc_base_voice or DEFAULT_RVC_BASE_VOICE
        base_error, rvc_base_voice, rvc_base_language = _validate_rvc_base_voice(
            raw_base_voice, db
        )
        if base_error:
            raise ValueError(base_error)
        params = data.rvc_params.model_dump() if data.rvc_params is not None else DEFAULT_RVC_PARAMS
        rvc_params_json = _json.dumps(params)

    validation_error = _validate_profile_fields(
        voice_type=voice_type,
        preset_engine=data.preset_engine,
        preset_voice_id=data.preset_voice_id,
        design_prompt=data.design_prompt,
        default_engine=default_engine,
    )
    if validation_error:
        raise ValueError(validation_error)

    # Effective language: an rvc profile's spoken language is its base voice's.
    # Default from the base when the caller left the language at the schema
    # default ("en"); an explicit non-default language is always honored.
    language = data.language
    if voice_type == "rvc" and rvc_base_language and language == "en":
        language = rvc_base_language

    db_profile = DBVoiceProfile(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        language=language,
        voice_type=voice_type,
        preset_engine=data.preset_engine,
        preset_voice_id=data.preset_voice_id,
        design_prompt=data.design_prompt,
        default_engine=default_engine,
        rvc_base_voice=rvc_base_voice,
        rvc_params=rvc_params_json,
        personality=data.personality,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )

    db.add(db_profile)
    db.commit()
    db.refresh(db_profile)

    profile_dir = config.get_profiles_dir() / db_profile.id
    profile_dir.mkdir(parents=True, exist_ok=True)

    return _profile_to_response(db_profile)


async def add_profile_sample(
    profile_id: str,
    audio_path: str,
    reference_text: str,
    db: Session,
) -> ProfileSampleResponse:
    """
    Add a sample to a voice profile.

    Args:
        profile_id: Profile ID
        audio_path: Path to temporary audio file
        reference_text: Transcript of audio
        db: Database session

    Returns:
        Created sample
    """
    import asyncio

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    # Validate and load audio in a single pass, off the event loop
    is_valid, error_msg, audio, sr = await asyncio.to_thread(
        validate_and_load_reference_audio, audio_path
    )
    if not is_valid:
        raise ValueError(f"Invalid reference audio: {error_msg}")

    sample_id = str(uuid.uuid4())
    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    dest_path = profile_dir / f"{sample_id}.wav"
    await asyncio.to_thread(save_audio, audio, str(dest_path), sr)

    db_sample = DBProfileSample(
        id=sample_id,
        profile_id=profile_id,
        audio_path=config.to_storage_path(dest_path),
        reference_text=reference_text,
    )

    db.add(db_sample)

    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(db_sample)

    # Invalidate combined audio cache for this profile
    # Since a new sample was added, any cached combined audio is now stale
    clear_profile_cache(profile_id)

    return ProfileSampleResponse.model_validate(db_sample)


async def get_profile(
    profile_id: str,
    db: Session,
) -> VoiceProfileResponse | None:
    """
    Get a voice profile by ID.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        Profile or None if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None

    return _profile_to_response(profile)


def get_profile_orm_by_name_or_id(
    name_or_id: str,
    db: Session,
) -> DBVoiceProfile | None:
    """Resolve a profile from a user-supplied string that may be either id or name.

    Id is tried first (fast path, matches UUIDs). Name fallback is
    case-insensitive so agents can say "Morgan" regardless of casing.
    """
    if not name_or_id:
        return None
    row = db.query(DBVoiceProfile).filter(DBVoiceProfile.id == name_or_id).first()
    if row is not None:
        return row
    return (
        db.query(DBVoiceProfile)
        .filter(func.lower(DBVoiceProfile.name) == name_or_id.lower())
        .first()
    )


async def get_profile_samples(
    profile_id: str,
    db: Session,
) -> list[ProfileSampleResponse]:
    """
    Get all samples for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        List of samples
    """
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()
    return [ProfileSampleResponse.model_validate(s) for s in samples]


async def list_profiles(db: Session) -> list[VoiceProfileResponse]:
    """
    List all voice profiles with generation and sample counts.

    Args:
        db: Database session

    Returns:
        List of profiles
    """
    profiles = db.query(DBVoiceProfile).order_by(DBVoiceProfile.created_at.desc()).all()

    if not profiles:
        return []

    # Batch-fetch generation counts
    gen_counts_rows = (
        db.query(DBGeneration.profile_id, func.count(DBGeneration.id)).group_by(DBGeneration.profile_id).all()
    )
    gen_counts = {row[0]: row[1] for row in gen_counts_rows}

    # Batch-fetch sample counts
    sample_counts_rows = (
        db.query(DBProfileSample.profile_id, func.count(DBProfileSample.id)).group_by(DBProfileSample.profile_id).all()
    )
    sample_counts = {row[0]: row[1] for row in sample_counts_rows}

    return [
        _profile_to_response(
            p,
            generation_count=gen_counts.get(p.id, 0),
            sample_count=sample_counts.get(p.id, 0),
        )
        for p in profiles
    ]


async def update_profile(
    profile_id: str,
    data: VoiceProfileCreate,
    db: Session,
) -> VoiceProfileResponse | None:
    """
    Update a voice profile.

    Args:
        profile_id: Profile ID
        data: Updated profile data
        db: Database session

    Returns:
        Updated profile or None if not found

    Raises:
        ValueError: If a profile with the same name already exists (different profile)
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None

    if profile.name != data.name:
        existing_profile = db.query(DBVoiceProfile).filter_by(name=data.name).first()
        if existing_profile:
            raise ValueError(f"A profile with the name '{data.name}' already exists. Please choose a different name.")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    preset_engine = getattr(profile, "preset_engine", None)
    preset_voice_id = getattr(profile, "preset_voice_id", None)
    design_prompt = getattr(profile, "design_prompt", None)
    default_engine = data.default_engine if data.default_engine is not None else getattr(profile, "default_engine", None)

    validation_error = _validate_profile_fields(
        voice_type=voice_type,
        preset_engine=preset_engine,
        preset_voice_id=preset_voice_id,
        design_prompt=design_prompt,
        default_engine=default_engine,
    )
    if validation_error:
        raise ValueError(validation_error)

    # Validate + normalize the base voice up front (a ``profile:{id}`` base needs
    # the DB session) so we can also derive the effective language before writing.
    normalized_base_voice = None
    rvc_base_language = None
    if voice_type == "rvc" and data.rvc_base_voice is not None:
        raw_base_voice = data.rvc_base_voice or DEFAULT_RVC_BASE_VOICE
        base_error, normalized_base_voice, rvc_base_language = _validate_rvc_base_voice(
            raw_base_voice, db
        )
        if base_error:
            raise ValueError(base_error)

    profile.name = data.name
    profile.description = data.description
    # Default the language from the base voice when the caller left it at the
    # schema default ("en"); an explicit non-default language is honored.
    language = data.language
    if voice_type == "rvc" and rvc_base_language and language == "en":
        language = rvc_base_language
    profile.language = language
    profile.personality = data.personality
    if data.default_engine is not None:
        profile.default_engine = data.default_engine or None  # empty string → NULL
    # Chain settings are editable only on rvc profiles; empty base voice resets
    # to the default preset (normalized above).
    if voice_type == "rvc":
        if normalized_base_voice is not None:
            profile.rvc_base_voice = normalized_base_voice
        if data.rvc_params is not None:
            profile.rvc_params = _json.dumps(data.rvc_params.model_dump())
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


async def delete_profile(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete a voice profile and all associated data.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return False

    db.query(DBProfileSample).filter_by(profile_id=profile_id).delete()

    db.delete(profile)
    db.commit()

    profile_dir = config.get_profiles_dir() / profile_id
    if profile_dir.exists():
        shutil.rmtree(profile_dir)

    # Clean up combined audio cache files for this profile
    clear_profile_cache(profile_id)

    return True


async def delete_profile_sample(
    sample_id: str,
    db: Session,
) -> bool:
    """
    Delete a profile sample.

    Args:
        sample_id: Sample ID
        db: Database session

    Returns:
        True if deleted, False if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return False

    # Store profile_id before deleting
    profile_id = sample.profile_id

    audio_path = config.resolve_storage_path(sample.audio_path)
    if audio_path is not None and audio_path.exists():
        audio_path.unlink()

    db.delete(sample)
    db.commit()

    # Invalidate combined audio cache for this profile
    # Since the sample set changed, any cached combined audio is now stale
    clear_profile_cache(profile_id)

    return True


async def update_profile_sample(
    sample_id: str,
    reference_text: str,
    db: Session,
) -> ProfileSampleResponse | None:
    """
    Update a profile sample's reference text.

    Args:
        sample_id: Sample ID
        reference_text: Updated reference text
        db: Database session

    Returns:
        Updated sample or None if not found
    """
    sample = db.query(DBProfileSample).filter_by(id=sample_id).first()
    if not sample:
        return None

    # Store profile_id before updating
    profile_id = sample.profile_id

    sample.reference_text = reference_text
    db.commit()
    db.refresh(sample)

    # Invalidate combined audio cache for this profile
    # Since the reference text changed, cache keys and combined text are now stale
    clear_profile_cache(profile_id)

    return ProfileSampleResponse.model_validate(sample)


async def create_voice_prompt_for_profile(
    profile_id: str,
    db: Session,
    use_cache: bool = True,
    engine: str = "qwen",
    language: str | None = None,
) -> dict:
    """
    Create a voice prompt from a profile.

    For cloned profiles: combines all audio samples into a voice prompt.
    For preset profiles: returns the engine-specific preset voice reference.
    For designed profiles: returns the text design prompt (future).

    Args:
        profile_id: Profile ID
        db: Database session
        use_cache: Whether to use cached prompts
        engine: TTS engine to create prompt for

    Returns:
        Voice prompt dictionary
    """
    from ..backends import get_tts_backend_for_engine

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile not found: {profile_id}")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    prompt_language = language or getattr(profile, "language", None) or "en"
    validate_profile_engine(profile, engine)

    # ── Preset profiles: return engine-specific voice reference ──
    if voice_type == "preset":
        if not profile.preset_engine or not profile.preset_voice_id:
            raise ValueError(f"Preset profile {profile_id} is missing preset engine metadata")
        if profile.preset_engine != engine:
            raise ValueError(
                f"Preset profile {profile_id} only supports engine '{profile.preset_engine}', not '{engine}'"
            )
        return {
            "voice_type": "preset",
            "preset_engine": profile.preset_engine,
            "preset_voice_id": profile.preset_voice_id,
        }

    # ── Designed profiles: return text description (future) ──
    if voice_type == "designed":
        if not profile.design_prompt or not profile.design_prompt.strip():
            raise ValueError(f"Designed profile {profile_id} is missing design_prompt")
        return {
            "voice_type": "designed",
            "design_prompt": profile.design_prompt,
        }

    if engine not in CLONING_ENGINES:
        raise ValueError(f"Engine '{engine}' does not support cloned voice profiles")

    # ── Cloned profiles: create from audio samples ──
    samples = db.query(DBProfileSample).filter_by(profile_id=profile_id).all()

    if not samples:
        raise ValueError(f"No samples found for profile {profile_id}")

    tts_model = get_tts_backend_for_engine(engine)

    if len(samples) == 1:
        sample = samples[0]
        sample_audio_path = config.resolve_storage_path(sample.audio_path)
        if sample_audio_path is None:
            raise ValueError(f"Sample audio not found for profile {profile_id}")
        voice_prompt, _ = await tts_model.create_voice_prompt(
            str(sample_audio_path),
            sample.reference_text,
            use_cache=use_cache,
            language=prompt_language,
        )
        return voice_prompt

    audio_paths = []
    for sample in samples:
        sample_audio_path = config.resolve_storage_path(sample.audio_path)
        if sample_audio_path is None:
            raise ValueError(f"Sample audio not found for profile {profile_id}")
        audio_paths.append(str(sample_audio_path))
    reference_texts = [s.reference_text for s in samples]

    combined_audio, combined_text = await tts_model.combine_voice_prompts(
        audio_paths,
        reference_texts,
    )

    # Save combined audio to cache directory (persistent)
    # Create a hash of sample IDs to identify this specific combination
    import hashlib

    sample_ids_str = "-".join(sorted([s.id for s in samples]))
    combination_hash = hashlib.md5(sample_ids_str.encode()).hexdigest()[:12]

    cache_dir = _get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    combined_path = cache_dir / f"combined_{profile_id}_{combination_hash}.wav"

    save_audio(combined_audio, str(combined_path), 24000)

    voice_prompt, _ = await tts_model.create_voice_prompt(
        str(combined_path),
        combined_text,
        use_cache=use_cache,
        language=prompt_language,
    )
    return voice_prompt


async def upload_avatar(
    profile_id: str,
    image_path: str,
    db: Session,
) -> VoiceProfileResponse:
    """
    Upload and process avatar image for a profile.

    Args:
        profile_id: Profile ID
        image_path: Path to uploaded image file
        db: Database session

    Returns:
        Updated profile
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    is_valid, error_msg = validate_image(image_path)
    if not is_valid:
        raise ValueError(error_msg)

    if profile.avatar_path:
        old_avatar = config.resolve_storage_path(profile.avatar_path)
        if old_avatar is not None and old_avatar.exists():
            old_avatar.unlink()

    # Determine file extension from uploaded file
    from PIL import Image

    with Image.open(image_path) as img:
        # Normalize JPEG variants (MPO is multi-picture format from some cameras)
        img_format = img.format
        if img_format in ("MPO", "JPG"):
            img_format = "JPEG"

        ext_map = {"PNG": ".png", "JPEG": ".jpg", "WEBP": ".webp"}
        ext = ext_map.get(img_format, ".png")

    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    output_path = profile_dir / f"avatar{ext}"

    process_avatar(image_path, str(output_path))

    profile.avatar_path = config.to_storage_path(output_path)
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


async def delete_avatar(
    profile_id: str,
    db: Session,
) -> bool:
    """
    Delete avatar image for a profile.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        True if deleted, False if not found or no avatar
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile or not profile.avatar_path:
        return False

    avatar_path = config.resolve_storage_path(profile.avatar_path)
    if avatar_path is not None and avatar_path.exists():
        avatar_path.unlink()

    profile.avatar_path = None
    profile.updated_at = datetime.utcnow()

    db.commit()

    return True


def _validate_and_store_rvc_files(
    profile_id: str,
    model_tmp_path: str,
    index_tmp_path: str | None,
) -> tuple[str, str | None]:
    """Validate the streamed RVC artifacts, then atomically move them into place.

    Runs off the event loop (``torch.load`` on a large checkpoint blocks).
    Validation happens *before* any file is moved, so a failure raises
    ``ValueError`` without touching a previously stored model. Returns the
    storage paths to persist on the profile.
    """
    import os

    from ..backends.rvc.checkpoint import (
        load_rvc_checkpoint,
        validate_faiss_index,
        validate_rvc_checkpoint,
    )

    try:
        ckpt = load_rvc_checkpoint(model_tmp_path)
    except ValueError:
        # Size / not-a-dict rejections already carry a user-facing message.
        raise
    except Exception as e:
        # torch.load(weights_only=True) blocks malicious pickles by raising
        # UnpicklingError (and corrupt archives raise other library errors);
        # translate any such untrusted-load failure into a 400-mapped ValueError.
        logger.warning("Rejected RVC checkpoint upload for profile %s: %s", profile_id, e)
        raise ValueError(
            "The uploaded file is not a valid RVC checkpoint (it could not be safely deserialized)."
        ) from e

    info = validate_rvc_checkpoint(ckpt)
    if index_tmp_path is not None:
        try:
            validate_faiss_index(index_tmp_path, info.embedder_dim)
        except ValueError:
            raise
        except Exception as e:
            logger.warning("Rejected RVC index upload for profile %s: %s", profile_id, e)
            raise ValueError("The uploaded file is not a valid FAISS index.") from e

    profile_dir = config.get_profiles_dir() / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)

    model_dest = profile_dir / RVC_MODEL_FILENAME
    os.replace(model_tmp_path, model_dest)

    index_dest = profile_dir / RVC_INDEX_FILENAME
    if index_tmp_path is not None:
        os.replace(index_tmp_path, index_dest)
        index_stored = config.to_storage_path(index_dest)
    else:
        # A previously uploaded index is derived from a specific model; a fresh
        # model without an index makes the old one a stale, mismatched pair.
        index_dest.unlink(missing_ok=True)
        index_stored = None

    _rvc_metadata_path(profile_id).write_text(
        _json.dumps(
            {
                "version": info.version,
                "sample_rate": info.sample_rate,
                "if_f0": info.if_f0,
            }
        )
    )

    return config.to_storage_path(model_dest), index_stored


async def upload_rvc_model(
    profile_id: str,
    model_tmp_path: str,
    index_tmp_path: str | None,
    db: Session,
) -> VoiceProfileResponse:
    """Validate and store an uploaded RVC checkpoint (and optional index).

    Args:
        profile_id: Profile ID
        model_tmp_path: Temp path of the streamed ``.pth`` checkpoint
        index_tmp_path: Temp path of the streamed ``.index`` file, or None
        db: Database session

    Returns:
        Updated profile

    Raises:
        ValueError: If the profile is missing, is not an RVC profile, or the
            uploaded artifacts fail checkpoint/index validation.
    """
    import asyncio

    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        raise ValueError(f"Profile {profile_id} not found")

    voice_type = getattr(profile, "voice_type", None) or "cloned"
    if voice_type != "rvc":
        raise ValueError("RVC model upload is only supported for RVC voice profiles")

    model_stored, index_stored = await asyncio.to_thread(
        _validate_and_store_rvc_files, profile_id, model_tmp_path, index_tmp_path
    )

    profile.rvc_model_path = model_stored
    profile.rvc_index_path = index_stored
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)


async def delete_rvc_model(
    profile_id: str,
    db: Session,
) -> VoiceProfileResponse | None:
    """Remove a profile's RVC model, index and metadata sidecar.

    Args:
        profile_id: Profile ID
        db: Database session

    Returns:
        Updated profile, or None if the profile does not exist.
    """
    profile = db.query(DBVoiceProfile).filter_by(id=profile_id).first()
    if not profile:
        return None

    for stored in (profile.rvc_model_path, profile.rvc_index_path):
        resolved = config.resolve_storage_path(stored)
        if resolved is not None and resolved.exists():
            resolved.unlink()

    _rvc_metadata_path(profile_id).unlink(missing_ok=True)

    profile.rvc_model_path = None
    profile.rvc_index_path = None
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)

    return _profile_to_response(profile)
