"""Incremental speech producer API and single-sink playback acknowledgements."""

from typing import Annotated

from fastapi import APIRouter
from pydantic import BaseModel, Field

from ..mcp_server.context import current_client_id
from ..services.speech_sessions import sessions, start_session

router = APIRouter(prefix="/speak/sessions", tags=["speech sessions"])


class Start(BaseModel):
    profile: str | None = None
    language: str | None = None
    keep_audio: bool = False


class Append(BaseModel):
    sequence: Annotated[int, Field(ge=0)]
    text: Annotated[str, Field(min_length=1, max_length=4096)]


class Playback(BaseModel):
    renderer: Annotated[str, Field(min_length=16, max_length=100)]
    acknowledged: Annotated[int | None, Field(ge=0)] = None
    error: Annotated[str | None, Field(max_length=300)] = None
    stopped: bool = False


def owned(session_id):
    return sessions.get(session_id, current_client_id.get())


@router.post("")
async def start(data: Start):
    return await start_session(**data.model_dump())


@router.get("/{session_id}")
async def status(session_id: str):
    return sessions.snapshot(owned(session_id))


@router.post("/{session_id}/append")
async def append(session_id: str, data: Append):
    return sessions.append(owned(session_id), data.sequence, data.text)


@router.post("/{session_id}/finish")
async def finish(session_id: str):
    return sessions.finish(owned(session_id))


@router.post("/{session_id}/cancel")
async def cancel(session_id: str):
    return sessions.cancel(owned(session_id))


@router.post("/{session_id}/playback")
async def playback(session_id: str, data: Playback):
    # The opaque session id is the local sink capability. Client-id headers
    # route producer voices, not authentication; never expose this API publicly.
    return await sessions.playback(sessions.get(session_id, player=True), **data.model_dump())
