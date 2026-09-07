"""Bounded text-in / ordered WAV-out sessions, independent of MCP transport.

Generation completion is not playback completion. Only the claimed sink may
acknowledge a segment; temporary generations are deleted after that ack.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

from fastapi import HTTPException

logger = logging.getLogger(__name__)
TERMINAL = {"completed", "cancelled", "failed"}
BUFFER_LIMIT = 4096
TOTAL_LIMIT = 100_000
SEGMENT_LIMIT = 320


def split_ready(text: str, final: bool = False) -> tuple[list[str], str]:
    """Conservative prose boundaries; hold trailing punctuation until a delta
    confirms whitespace. Long unpunctuated input is bounded as well.
    """
    from ..utils.chunked_tts import _find_last_sentence_end

    segments = []
    while text:
        # Reuse the batch splitter's abbreviation, decimal and tag handling.
        candidate = _find_last_sentence_end(text[: min(SEGMENT_LIMIT, len(text) - 1)])
        boundary = candidate + 1 if candidate >= 39 and text[candidate + 1 : candidate + 2].isspace() else None
        if boundary is None and len(text) > SEGMENT_LIMIT:
            boundary = text.rfind(" ", 0, SEGMENT_LIMIT + 1)
            if boundary <= 0:
                boundary = SEGMENT_LIMIT
        if boundary is None:
            if final:
                segments.append(text)
                text = ""
            break
        segments.append(text[:boundary])
        text = text[boundary:]
    return segments, text


@dataclass
class SpeechSession:
    id: str
    owner: str | None
    options: dict
    profile_name: str
    state: str = "accepting"
    buffer: str = ""
    pending: deque = field(default_factory=deque)
    ready: deque = field(default_factory=deque)
    receipts: dict = field(default_factory=dict)
    next_sequence: int = 0
    next_segment: int = 0
    acknowledged: int = -1
    total_chars: int = 0
    outstanding_chars: int = 0
    renderer: str | None = None
    touched: float = field(default_factory=time.monotonic)
    heartbeat: float = field(default_factory=time.monotonic)
    error: str | None = None
    task: asyncio.Task | None = None
    rendering: str | None = None
    cleanup: set = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    profile_revision: str | None = None


class GenerationSink:
    """Adapter to the existing serial TTS/effects/RVC pipeline."""

    async def submit(self, session, text):
        from ..database.session import SessionLocal
        from ..models import GenerationRequest
        from ..routes.generations import submit_speech

        with SessionLocal() as db:
            if session.profile_revision is not None:
                from ..database import VoiceProfile

                profile = db.get(VoiceProfile, session.options["profile_id"])
                if profile is None or str(profile.updated_at) != session.profile_revision:
                    raise ValueError("Voice profile changed during speech; start a new session")
            result = await submit_speech(
                GenerationRequest(text=text, **session.options),
                db,
                source="speech_stream",
            )
            return result.id

    async def status(self, generation_id):
        from ..database import Generation
        from ..database.session import SessionLocal

        with SessionLocal() as db:
            gen = db.get(Generation, generation_id)
            if gen is None:
                return "failed", "Generation was removed"
            return gen.status, gen.error

    async def delete(self, generation_id):
        from ..database.session import SessionLocal
        from .history import delete_generation

        with SessionLocal() as db:
            await delete_generation(generation_id, db)


class SpeechSessions:
    def __init__(self, sink=None, *, idle_timeout=300, playback_timeout=120, poll_interval=0.25):
        self.sink = sink or GenerationSink()
        self.sessions: dict[str, SpeechSession] = {}
        self.idle_timeout = idle_timeout
        self.playback_timeout = playback_timeout
        self.poll_interval = poll_interval

    def active(self):
        return next((s for s in self.sessions.values() if s.state not in TERMINAL), None)

    def require_idle(self):
        if any(s.task and not s.task.done() for s in self.sessions.values()):
            raise HTTPException(409, "A speech session is active or draining cleanup; finish or cancel it first")

    def get(self, session_id, owner=None, *, player=False):
        session = self.sessions.get(session_id)
        if session is None or (not player and session.owner != owner):
            raise HTTPException(404, "Speech session not found")
        return session

    def snapshot(self, s):
        return {
            "session_id": s.id,
            "state": s.state,
            "profile": s.profile_name,
            "engine": s.options["engine"],
            "keep_audio": s.options["keep_audio"],
            "next_sequence": s.next_sequence,
            "acknowledged": s.acknowledged,
            "available_chars": BUFFER_LIMIT - s.outstanding_chars,
            "ready": list(s.ready),
            "error": s.error,
        }

    def notify(self, s):
        from ..mcp_server.events import publish

        publish("speech-session", {"session_id": s.id, "state": s.state})

    def start(self, owner, options, profile_name, profile_revision=None):
        self.require_idle()
        # Keep only a small, bounded terminal status cache for retries.
        while len(self.sessions) >= 32:
            old = next((key for key, s in self.sessions.items() if s.task is None or s.task.done()), None)
            if old is None:
                raise HTTPException(503, "Speech cleanup is still draining")
            del self.sessions[old]
        s = SpeechSession(str(uuid.uuid4()), owner, dict(options), profile_name)
        s.profile_revision = profile_revision
        self.sessions[s.id] = s
        s.task = asyncio.create_task(self._run(s))
        self.notify(s)
        return self.snapshot(s)

    def append(self, s, sequence, text):
        digest = hashlib.sha256(text.encode()).hexdigest()
        if sequence in s.receipts:
            if s.receipts[sequence] != digest:
                raise HTTPException(409, "Sequence was already used for different text")
            return self.snapshot(s)
        if s.state != "accepting":
            raise HTTPException(409, "Session no longer accepts text")
        if sequence != s.next_sequence:
            raise HTTPException(409, f"Expected sequence {s.next_sequence}")
        if not text or len(text) > BUFFER_LIMIT:
            raise HTTPException(422, f"Text must contain 1..{BUFFER_LIMIT} characters")
        if s.total_chars + len(text) > TOTAL_LIMIT or s.next_sequence >= 10_000:
            raise HTTPException(413, "Session input limit reached; finish this session")
        if s.outstanding_chars + len(text) > BUFFER_LIMIT:
            raise HTTPException(
                429, "Speech buffer full; retry the SAME sequence after playback", headers={"Retry-After": "1"}
            )
        s.receipts[sequence] = digest
        s.next_sequence += 1
        s.total_chars += len(text)
        s.outstanding_chars += len(text)
        s.touched = time.monotonic()
        parts, s.buffer = split_ready(s.buffer + text)
        s.pending.extend(parts)
        return self.snapshot(s)

    def finish(self, s):
        if s.state == "accepting":
            parts, s.buffer = split_ready(s.buffer, final=True)
            s.pending.extend(parts)
            s.state = "draining"
            self.notify(s)
        return self.snapshot(s)

    def cancel(self, s, error=None):
        if s.state not in TERMINAL:
            s.state = "failed" if error else "cancelled"
            s.error = error
            s.buffer = ""
            s.pending.clear()
            self.notify(s)
        return self.snapshot(s)

    async def playback(self, s, renderer, acknowledged=None, error=None, stopped=False):
        async with s.lock:
            if s.renderer is not None and s.renderer != renderer:
                raise HTTPException(409, "Another renderer owns this session")
            if s.state in TERMINAL:
                return self.snapshot(s)
            s.renderer = renderer
            s.heartbeat = time.monotonic()
            if stopped:
                return self.cancel(s)
            if error:
                return self.cancel(s, f"Playback failed: {error[:300]}")
            if acknowledged is not None and acknowledged > s.acknowledged:
                if not s.ready or s.ready[0]["sequence"] != acknowledged:
                    raise HTTPException(409, "Only the next ready segment may be acknowledged")
                segment = s.ready.popleft()
                s.acknowledged = acknowledged
                s.outstanding_chars -= segment["chars"]
                # Mark before I/O so a retry can never replay already-heard text.
                if not s.options["keep_audio"]:
                    s.cleanup.add(segment["generation_id"])
                self.notify(s)
            return self.snapshot(s)

    async def _clean(self, s):
        for generation_id in list(s.cleanup):
            try:
                await self.sink.delete(generation_id)
                s.cleanup.discard(generation_id)
            except Exception:
                logger.exception("Could not clean speech generation %s", generation_id)

    async def _run(self, s):
        try:
            rendering_chars = 0
            render_started = 0.0
            while s.state not in TERMINAL:
                await self._clean(s)
                now = time.monotonic()
                if now - s.heartbeat > self.playback_timeout:
                    self.cancel(s, "Playback renderer missing or disconnected")
                    break
                if s.state == "accepting" and now - s.touched > self.idle_timeout:
                    self.cancel(s, "Text producer timed out; finish was not received")
                    break
                if s.rendering:
                    status, error = await self.sink.status(s.rendering)
                    if status == "failed":
                        raise RuntimeError(error or "Speech generation failed")
                    if status == "completed":
                        s.ready.append(
                            {"sequence": s.next_segment, "generation_id": s.rendering, "chars": rendering_chars}
                        )
                        s.next_segment += 1
                        s.rendering = None
                        self.notify(s)
                    elif now - render_started > 600:
                        raise RuntimeError("Speech generation timed out")
                if not s.rendering and s.pending and len(s.ready) < 2:
                    text = s.pending.popleft()
                    if not text.strip():
                        s.outstanding_chars -= len(text)
                        continue
                    rendering_chars = len(text)
                    s.rendering = await self.sink.submit(s, text)
                    render_started = time.monotonic()
                if s.state == "draining" and not s.pending and not s.rendering and not s.ready:
                    s.state = "completed"
                    self.notify(s)
                    break
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            self.cancel(s)
            raise
        except Exception as exc:
            logger.exception("Speech session failed")
            self.cancel(s, str(exc))
        finally:
            if not asyncio.current_task().cancelling():
                await self._drain(s)

    async def _drain(self, s):
        # Do not cancel an in-flight GPU thread and unload under it. Drain
        # its serial job before deleting files it may still be writing.
        if s.rendering:
            while True:
                status, _ = await self.sink.status(s.rendering)
                if status in {"completed", "failed"}:
                    break
                await asyncio.sleep(self.poll_interval)
            if not s.options["keep_audio"]:
                s.cleanup.add(s.rendering)
            s.rendering = None
        if not s.options["keep_audio"]:
            s.cleanup.update(item["generation_id"] for item in s.ready)
        s.ready.clear()
        await self._clean(s)

    async def shutdown(self):
        for s in self.sessions.values():
            self.cancel(s)
        tasks = [s.task for s in self.sessions.values() if s.task]
        if not tasks:
            return True
        _, pending = await asyncio.wait(tasks, timeout=15)
        for task in pending:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return not pending


sessions = SpeechSessions()


async def start_session(profile=None, language=None, keep_audio=False):
    """Resolve binding once. No personality rewrite or engine override per delta."""
    from ..backends import resolve_model_size_for_engine
    from ..database import MCPClientBinding
    from ..database.session import SessionLocal
    from ..mcp_server.context import current_client_id
    from ..mcp_server.resolve import resolve_bound_engine, resolve_profile
    from ..models import GenerationRequest
    from ..routes.generations import _raise_if_realtime_stream_active, _resolve_generation_engine
    from .profiles import validate_profile_engine

    owner = current_client_id.get()
    sessions.require_idle()
    with SessionLocal() as db:
        vp = resolve_profile(profile, owner, db)
        if vp is None:
            raise HTTPException(404, "No voice profile resolved; configure an MCP binding or pass profile")
        binding = db.query(MCPClientBinding).filter_by(client_id=owner).first() if owner else None
        engine = resolve_bound_engine(None, binding.default_engine if binding else None, vp)
        lang = language or vp.language or "en"
        try:
            engine = _resolve_generation_engine(
                GenerationRequest(profile_id=vp.id, text=".", engine=engine, language=lang), vp
            )
            validate_profile_engine(vp, engine)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        _raise_if_realtime_stream_active()
        options = dict(
            profile_id=vp.id,
            engine=engine,
            language=lang,
            model_size=resolve_model_size_for_engine(engine, None, lang),
            personality=False,
            keep_audio=keep_audio,
            normalize=True,
        )
        return sessions.start(owner, options, vp.name, str(vp.updated_at))


async def cleanup_interrupted_sessions(db):
    """Run only at process startup: no session from the previous process survives."""
    from ..database import Generation
    from .history import delete_generation

    rows = db.query(Generation).filter_by(source="speech_stream", keep_audio=False).all()
    for row in rows:
        await delete_generation(row.id, db)
