"""``plain_text`` / ``max_chars`` on ``voicebox.speak`` and ``POST /speak``.

Both surfaces resolve the two knobs the same way as ``personality`` and
``engine``: explicit argument → per-client binding default → off. These
tests run the real resolution code against a throwaway SQLite database and
stub the generation step, so they pin what text reaches TTS without loading
a model.
"""

import pytest
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from starlette.requests import Request

from backend import config, models
from backend.database import get_db
from backend.database.models import MCPClientBinding, VoiceProfile
from backend.mcp_server import events as mcp_events, tools
from backend.mcp_server.context import current_client_id
from backend.routes import generations
from backend.routes.speak import speak

CLIENT = "claude-code"
MARKDOWN = (
    "## Verdict\n\nVoici la **réponse finale**.\n\n```bash\nrm -rf /never-read\n```\n\n"
    "| a | b |\n|---|---|\n| 1 | 2 |\n\nVoir [.mcp.json](.mcp.json) pour la suite. "
    + "Encore une phrase de remplissage qui allonge le texte. "
    * 4
)


@pytest.fixture
def db(tmp_path, monkeypatch):
    """Fresh schema in a temp data dir, one profile, one binding with speech defaults on."""
    from backend.database import session as db_session

    original = config.get_data_dir()
    config.set_data_dir(str(tmp_path))
    db_session.init_db()
    monkeypatch.setattr(mcp_events, "publish", lambda *a, **k: None)
    session = next(get_db())
    session.add(VoiceProfile(id="p1", name="Siwis", language="fr"))
    session.add(
        MCPClientBinding(
            client_id=CLIENT,
            profile_id="p1",
            default_plain_text=True,
            default_max_chars=120,
        )
    )
    session.commit()
    try:
        yield session
    finally:
        session.close()
        config.set_data_dir(str(original))


@pytest.fixture
def captured_generation(monkeypatch):
    """Stub the model-backed generate_speech; both surfaces import it lazily from routes.generations."""
    captured = {}

    class _FakeGeneration:
        id = "gen-test"

        def model_dump(self, mode="json"):
            return {"id": self.id, "status": "generating"}

    async def fake_generate_speech(req, db):
        captured["req"] = req
        return _FakeGeneration()

    monkeypatch.setattr(generations, "generate_speech", fake_generate_speech)
    return captured


def _rest_request() -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/speak",
            "headers": [(b"x-voicebox-client-id", CLIENT.encode())],
        }
    )


# ── REST /speak ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_rest_applies_binding_defaults(db, captured_generation):
    await speak(models.SpeakRequest(text=MARKDOWN), _rest_request(), db)
    spoken = captured_generation["req"].text
    assert "rm -rf" not in spoken
    assert "|" not in spoken
    assert "**" not in spoken
    assert spoken.startswith("Verdict\nVoici la réponse finale.\nVoir .mcp.json")
    assert len(spoken) <= 120
    assert spoken.endswith(".")  # cut on a sentence end, not mid-word


@pytest.mark.asyncio
async def test_rest_explicit_args_override_binding(db, captured_generation):
    await speak(
        models.SpeakRequest(text=MARKDOWN, plain_text=False, max_chars=10000),
        _rest_request(),
        db,
    )
    assert captured_generation["req"].text == MARKDOWN.strip()


@pytest.mark.asyncio
async def test_rest_rejects_text_that_strips_to_nothing(db, captured_generation):
    from fastapi import HTTPException

    with pytest.raises(HTTPException, match="Nothing left to speak") as exc:
        await speak(models.SpeakRequest(text="```py\nprint(1)\n```"), _rest_request(), db)
    assert exc.value.status_code == 400
    assert "req" not in captured_generation


def test_rest_schema_bounds_max_chars():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="max_chars"):
        models.SpeakRequest(text="x", max_chars=10)


# ── MCP voicebox.speak ─────────────────────────────────────────────────────


@pytest.fixture
def captured_speak(monkeypatch):
    """Capture what the tool hands to ``_speak`` (the closure resolves it as a module global)."""
    captured = {}

    async def fake_speak(**kwargs):
        captured.update(kwargs)
        return {"generation_id": "gen-test", "status": "generating"}

    monkeypatch.setattr(tools, "_speak", fake_speak)
    return captured


@pytest.fixture
def mcp():
    server = FastMCP("test")
    tools.register_tools(server)
    return server


@pytest.fixture
def as_client():
    token = current_client_id.set(CLIENT)
    try:
        yield
    finally:
        current_client_id.reset(token)


@pytest.mark.asyncio
async def test_tool_applies_binding_defaults(db, mcp, captured_speak, as_client):
    await mcp.call_tool("voicebox.speak", {"text": MARKDOWN})
    assert "rm -rf" not in captured_speak["text"]
    assert len(captured_speak["text"]) <= 120


@pytest.mark.asyncio
async def test_tool_explicit_args_override_binding(db, mcp, captured_speak, as_client):
    await mcp.call_tool("voicebox.speak", {"text": MARKDOWN, "plain_text": False, "max_chars": 10000})
    assert captured_speak["text"] == MARKDOWN.strip()


@pytest.mark.asyncio
async def test_tool_rejects_small_max_chars(db, mcp, captured_speak, as_client):
    with pytest.raises(ToolError, match="between 50 and 10000"):
        await mcp.call_tool("voicebox.speak", {"text": "hello there", "max_chars": 10})
    assert "text" not in captured_speak


@pytest.mark.asyncio
async def test_tool_rejects_large_max_chars(db, mcp, captured_speak, as_client):
    with pytest.raises(ToolError, match="between 50 and 10000"):
        await mcp.call_tool("voicebox.speak", {"text": "hello there", "max_chars": 10001})
    assert "text" not in captured_speak


@pytest.mark.asyncio
async def test_tool_rejects_text_that_strips_to_nothing(db, mcp, captured_speak, as_client):
    with pytest.raises(ToolError, match="Nothing left to speak"):
        await mcp.call_tool("voicebox.speak", {"text": "```\ncode\n```"})
