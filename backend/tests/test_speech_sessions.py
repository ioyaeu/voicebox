"""Protocol and lifecycle checks, with no model download or speaker access."""

import asyncio

import pytest
from fastapi import HTTPException

from backend.services.speech_sessions import BUFFER_LIMIT, SpeechSessions, split_ready


class Sink:
    def __init__(self):
        self.jobs = {}
        self.deleted = []
        self.texts = []

    async def submit(self, session, text):
        key = str(len(self.jobs))
        self.jobs[key] = "generating"
        self.texts.append(text)
        return key

    async def status(self, key):
        return self.jobs[key], "model failure" if self.jobs[key] == "failed" else None

    async def delete(self, key):
        self.deleted.append(key)


async def until(predicate):
    async with asyncio.timeout(2):
        while not predicate():
            await asyncio.sleep(0.002)


def start(manager, keep=False):
    result = manager.start("agent", {"engine": "voxtral", "keep_audio": keep}, "Voice")
    return manager.get(result["session_id"], "agent")


@pytest.mark.parametrize("width", [1, 3, 17, 400])
def test_delta_boundaries_preserve_text(width):
    text = (
        "The doctor is Dr. Smith, who can help with the next step. "
        "The measured value is 3.14, and the text has trailing spaces. "
    ) * 9
    remaining, spoken = "", []
    for i in range(0, len(text), width):
        parts, remaining = split_ready(remaining + text[i : i + width])
        spoken.extend(parts)
    parts, remaining = split_ready(remaining, final=True)
    spoken.extend(parts)
    assert "".join(spoken) == text
    assert remaining == ""
    assert all(len(part) <= 320 for part in spoken)
    assert not any(part.rstrip().endswith("Dr.") for part in spoken)


@pytest.mark.asyncio
async def test_retry_order_credit_and_busy():
    manager = SpeechSessions(Sink())
    s = start(manager)
    try:
        with pytest.raises(HTTPException) as busy:
            start(manager)
        assert busy.value.status_code == 409
        with pytest.raises(HTTPException) as wrong_owner:
            manager.get(s.id, "other")
        assert wrong_owner.value.status_code == 404
        text = "hello " * (BUFFER_LIMIT // 6)
        first = manager.append(s, 0, text)
        assert manager.append(s, 0, text) == first
        for seq, value, code in [(0, "changed", 409), (2, "gap", 409), (1, "overflow", 429)]:
            with pytest.raises(HTTPException) as exc:
                manager.append(s, seq, value)
            assert exc.value.status_code == code
        assert s.next_sequence == 1
    finally:
        manager.cancel(s)
        await s.task


@pytest.mark.asyncio
@pytest.mark.parametrize("keep", [False, True])
async def test_finish_waits_for_playback_and_ack_is_idempotent(keep):
    sink = Sink()
    manager = SpeechSessions(sink, poll_interval=0.001)
    s = start(manager, keep)
    manager.append(s, 0, "A short sentence, deliberately left without punctuation")
    manager.finish(s)
    await manager.playback(s, "renderer-one")
    await until(lambda: bool(sink.jobs))
    sink.jobs["0"] = "completed"
    await until(lambda: bool(s.ready))
    assert s.state == "draining"
    # Repeated heartbeats during pause must not delete, advance, or complete.
    for _ in range(4):
        response = await manager.playback(s, "renderer-one")
        assert response["ready"][0]["sequence"] == 0
    assert sink.deleted == []
    with pytest.raises(HTTPException):
        await manager.playback(s, "renderer-two", acknowledged=0)
    with pytest.raises(HTTPException):
        await manager.playback(s, "renderer-one", acknowledged=1)
    await manager.playback(s, "renderer-one", acknowledged=0)
    await manager.playback(s, "renderer-one", acknowledged=0)
    await s.task
    assert s.state == "completed"
    assert sink.deleted == ([] if keep else ["0"])
    assert s.outstanding_chars == 0


@pytest.mark.asyncio
async def test_prefetch_is_bounded_and_cancel_drains_inflight():
    sink = Sink()
    manager = SpeechSessions(sink, poll_interval=0.001)
    s = start(manager)
    manager.append(s, 0, "a " * 1500)
    manager.finish(s)
    await until(lambda: "0" in sink.jobs)
    sink.jobs["0"] = "completed"
    await until(lambda: "1" in sink.jobs)
    sink.jobs["1"] = "completed"
    await until(lambda: len(s.ready) == 2)
    await asyncio.sleep(0.02)
    assert len(sink.jobs) == 2
    await manager.playback(s, "renderer", acknowledged=0)
    await until(lambda: "2" in sink.jobs)
    manager.cancel(s)
    assert s.state == "cancelled"
    assert "2" not in sink.deleted
    sink.jobs["2"] = "completed"
    await s.task
    assert set(sink.deleted) == {"0", "1", "2"}
    assert len(sink.jobs) == 3


@pytest.mark.asyncio
async def test_disconnected_sink_and_failed_generation_are_not_completed():
    manager = SpeechSessions(Sink(), playback_timeout=0.01, poll_interval=0.001)
    s = start(manager)
    await s.task
    assert s.state == "failed"
    assert "disconnected" in s.error
    sink = Sink()
    manager = SpeechSessions(sink, poll_interval=0.001)
    s = start(manager)
    manager.append(s, 0, "This model request will fail.")
    manager.finish(s)
    await until(lambda: bool(sink.jobs))
    sink.jobs["0"] = "failed"
    await s.task
    assert s.state == "failed"
    assert sink.deleted == ["0"]


@pytest.mark.asyncio
async def test_whitespace_finishes_without_inference():
    sink = Sink()
    manager = SpeechSessions(sink, poll_interval=0.001)
    s = start(manager)
    manager.append(s, 0, "  \n  ")
    manager.finish(s)
    await s.task
    assert s.state == "completed"
    assert s.outstanding_chars == 0
    assert sink.jobs == {}
