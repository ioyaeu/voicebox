"""Exercise real voice resolution, REST and MCP against isolated SQLite."""

import httpx
import pytest
from fastapi import FastAPI
from fastmcp import FastMCP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.models import Base, Generation, MCPClientBinding, VoiceProfile
from backend.mcp_server.context import current_client_id
from backend.mcp_server.tools import register_tools
from backend.routes.speech_sessions import router
from backend.services import speech_sessions as service


@pytest.fixture
def storage(tmp_path, monkeypatch):
    from backend.database import session

    engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    monkeypatch.setattr(session, "SessionLocal", factory)
    with factory() as db:
        db.add(
            VoiceProfile(
                id="voice",
                name="Voxtral",
                language="fr",
                voice_type="preset",
                preset_engine="voxtral",
                default_engine="voxtral",
                preset_voice_id="fr_female",
            )
        )
        db.add(MCPClientBinding(client_id="agent", profile_id="voice", default_engine="kokoro"))
        db.commit()
    yield factory
    engine.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("engine", ["voxtral", "rvc"])
async def test_rest_and_mcp_share_session_and_profile_owned_engine(storage, monkeypatch, engine):
    if engine == "rvc":
        with storage() as db:
            voice = db.get(VoiceProfile, "voice")
            voice.voice_type = "rvc"
            voice.default_engine = "rvc"
            voice.rvc_base_voice = "voxtral:fr_female"
            db.commit()
    manager = service.SpeechSessions()
    monkeypatch.setattr(service, "sessions", manager)
    # routes imported their singleton before the monkeypatch
    monkeypatch.setattr("backend.routes.speech_sessions.sessions", manager)
    server = FastMCP("test")
    register_tools(server)
    app = FastAPI()
    app.include_router(router)
    token = current_client_id.set("agent")
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/speak/sessions", json={"language": "invalid"})).status_code == 422
            response = await client.post("/speak/sessions", json={"keep_audio": True})
            assert response.status_code == 200, response.text
            payload = response.json()
            s = manager.get(payload["session_id"], "agent")
            assert payload["engine"] == engine
            assert s.options["language"] == "fr"
            assert s.options["keep_audio"] is True
            await server.call_tool("voicebox.speech_append", {"session_id": s.id, "sequence": 0, "text": " "})
            assert (await client.get(f"/speak/sessions/{s.id}")).json()["next_sequence"] == 1
            assert (
                await client.post(f"/speak/sessions/{s.id}/append", json={"sequence": -1, "text": "x"})
            ).status_code == 422
            await server.call_tool("voicebox.speech_finish", {"session_id": s.id})
            await s.task
            assert (await client.get(f"/speak/sessions/{s.id}")).json()["state"] == "completed"
    finally:
        current_client_id.reset(token)
        await manager.shutdown()


@pytest.mark.asyncio
async def test_startup_removes_only_temporary_stream_rows_and_files(storage, tmp_path):
    with storage() as db:
        for key, source, keep in [
            ("temporary", "speech_stream", False),
            ("kept", "speech_stream", True),
            ("manual", "manual", False),
        ]:
            path = tmp_path / f"{key}.wav"
            path.write_bytes(b"test wave")
            db.add(
                Generation(
                    id=key,
                    profile_id="voice",
                    text="hello",
                    language="fr",
                    source=source,
                    keep_audio=keep,
                    audio_path=str(path),
                    status="completed",
                )
            )
        db.commit()
        await service.cleanup_interrupted_sessions(db)
        assert db.get(Generation, "temporary") is None
        assert not (tmp_path / "temporary.wav").exists()
        assert db.get(Generation, "kept") is not None
        assert (tmp_path / "kept.wav").exists()
        assert db.get(Generation, "manual") is not None


@pytest.mark.asyncio
async def test_stream_submission_marks_source_and_reuses_generation_pipeline(storage, monkeypatch):
    from backend.routes import generations

    captured = {}

    async def submit(request, db, *, source):
        captured.update(request=request, source=source)
        return type("Response", (), {"id": "generated"})()

    monkeypatch.setattr(generations, "submit_speech", submit)
    s = service.SpeechSession(
        "session",
        "agent",
        dict(profile_id="voice", language="fr", engine="voxtral", keep_audio=False, personality=False),
        "Voxtral",
    )
    assert await service.GenerationSink().submit(s, "Test speech.") == "generated"
    assert captured["source"] == "speech_stream"
    assert captured["request"].engine == "voxtral"
    assert captured["request"].keep_audio is False
