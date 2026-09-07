"""Migration tests for the speech-friendly per-client binding defaults.

``mcp_client_bindings`` gained ``default_plain_text`` / ``default_max_chars``
so agents that hand over raw markdown (Claude Code's Stop hook) can have it
stripped and capped without passing flags on every ``voicebox.speak`` call.
Existing databases pick the columns up on the next start.
"""

from sqlalchemy import create_engine, inspect, text

from backend.database.migrations import run_migrations

# The table exactly as it stood before this feature. Written by hand on
# purpose: building it from current SQLAlchemy metadata would already include
# the new columns and make the migration assertion vacuous.
_PRE_SPEECH_DEFAULTS_DDL = """
    CREATE TABLE mcp_client_bindings (
        client_id VARCHAR PRIMARY KEY,
        label VARCHAR,
        profile_id VARCHAR,
        default_engine VARCHAR,
        default_personality BOOLEAN NOT NULL DEFAULT 0,
        last_seen_at DATETIME,
        created_at DATETIME,
        updated_at DATETIME
    )
"""


def _legacy_engine(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'bindings.db'}")
    with engine.begin() as conn:
        conn.execute(text(_PRE_SPEECH_DEFAULTS_DDL))
        conn.execute(
            text(
                "INSERT INTO mcp_client_bindings (client_id, label, default_personality) "
                "VALUES ('claude-code', 'Claude Code', 1)"
            )
        )
    return engine


def test_migration_adds_speech_default_columns(tmp_path):
    engine = _legacy_engine(tmp_path)
    before = {c["name"] for c in inspect(engine).get_columns("mcp_client_bindings")}
    assert "default_plain_text" not in before
    assert "default_max_chars" not in before

    run_migrations(engine)

    after = {c["name"] for c in inspect(engine).get_columns("mcp_client_bindings")}
    assert {"default_plain_text", "default_max_chars"} <= after

    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT default_personality, default_plain_text, default_max_chars "
                "FROM mcp_client_bindings WHERE client_id = 'claude-code'"
            )
        ).one()
    # Existing rows keep what they had and get the conservative defaults:
    # markdown passes through untouched, no cap.
    assert row.default_personality == 1
    assert row.default_plain_text == 0
    assert row.default_max_chars is None


def test_migration_is_idempotent(tmp_path):
    engine = _legacy_engine(tmp_path)
    run_migrations(engine)
    run_migrations(engine)  # second pass must be a no-op, not a duplicate-column error
    columns = [c["name"] for c in inspect(engine).get_columns("mcp_client_bindings")]
    assert columns.count("default_plain_text") == 1
    assert columns.count("default_max_chars") == 1
