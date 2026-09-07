"""Voice profile resolution for MCP tool calls.

Precedence:
  1. Explicit tool arg (profile name or id)
  2. Per-client MCPClientBinding.profile_id
  3. CaptureSettings.default_playback_voice_id (global default)
  4. None — caller raises a helpful error
"""

from sqlalchemy.orm import Session

from ..database import VoiceProfile as DBVoiceProfile, get_db
from ..database.models import CaptureSettings
from ..services.profiles import get_profile_orm_by_name_or_id as _lookup_profile


def resolve_profile(
    explicit: str | None,
    client_id: str | None,
    db: Session,
) -> DBVoiceProfile | None:
    """Apply the full precedence chain and return the profile ORM row (or None)."""
    if explicit:
        profile = _lookup_profile(explicit, db)
        if profile is not None:
            return profile
        # Explicit but not found — return None so the caller can report it.
        return None

    if client_id:
        # Per-client binding. Imported lazily so this module stays importable
        # even before the migration adds the table on first boot.
        from ..database.models import MCPClientBinding  # noqa: WPS433

        binding = (
            db.query(MCPClientBinding)
            .filter(MCPClientBinding.client_id == client_id)
            .first()
        )
        if binding and binding.profile_id:
            profile = _lookup_profile(binding.profile_id, db)
            if profile is not None:
                return profile

    # Global default from capture settings.
    settings = db.query(CaptureSettings).filter(CaptureSettings.id == 1).first()
    if settings and settings.default_playback_voice_id:
        profile = _lookup_profile(settings.default_playback_voice_id, db)
        if profile is not None:
            return profile

    return None


def resolve_bound_engine(explicit_engine: str | None, binding_engine: str | None, profile) -> str | None:
    """Resolve an agent engine without overriding profile-owned engines.

    Preset profiles and RVC profiles carry their engine in the profile itself.
    A binding default is useful for cloned/designed profiles, but applying it
    to a selected preset or RVC profile creates an invalid voice/engine pair.
    An explicit tool/request engine remains authoritative so the generation
    route can return its normal validation error for an intentional conflict.
    """
    if explicit_engine is not None:
        return explicit_engine
    if getattr(profile, "voice_type", None) in {"preset", "rvc"}:
        return None
    return binding_engine


def with_db() -> Session:
    """Utility for tool handlers that aren't managed by FastAPI's Depends."""
    return next(get_db())
