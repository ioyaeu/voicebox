"""No models or private transcripts: synthetic public protocol fixtures only."""

import json
import subprocess
import sys
import threading
import time

import pytest

from backend.agent_bridge.events import Events, Lines, Prose, wants_keep
from backend.agent_bridge.narrator import HTTPError, Narrator
from backend.agent_bridge.proxy import resolve_codex


class Recorder:
    def __init__(self):
        self.calls = []

    def begin(self, keep=False):
        self.calls.append(("begin", keep))

    def append(self, text):
        if text:
            self.calls.append(("text", text))

    def finish(self):
        self.calls.append(("finish",))

    def cancel(self):
        self.calls.append(("cancel",))


def claude(event):
    return {"type": "stream_event", "parent_tool_use_id": None, "event": event}


def start_claude(events, prompt="Explain this"):
    events.accept("in", {"type": "user", "message": {"role": "user", "content": prompt}})
    events.accept("out", claude({"type": "message_start", "message": {"id": "m1"}}))
    events.accept(
        "out", claude({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}})
    )


def test_claude_stream_no_reasoning_tool_subagent_or_complete_duplicate():
    recorder = Recorder()
    events = Events("claude", recorder)
    events.accept("out", {"type": "assistant", "message": {"id": "old", "content": "restored history"}})
    start_claude(events, "keep audio, please explain")
    for event in [
        {"type": "content_block_start", "index": 1, "content_block": {"type": "thinking", "thinking": "secret"}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "thinking_delta", "thinking": "secret"}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Bonjour **Igor**.\n"}},
        {
            "type": "content_block_delta",
            "index": 2,
            "delta": {"type": "input_json_delta", "partial_json": "tool secret"},
        },
        {"type": "message_stop"},
    ]:
        events.accept("out", claude(event))
    events.accept("out", {"type": "assistant", "message": {"id": "m1", "content": "Bonjour Igor."}})
    events.accept(
        "out", {"type": "assistant", "parent_tool_use_id": "child", "message": {"id": "child", "content": "private"}}
    )
    events.accept("out", {"type": "result", "is_error": False, "result": "Bonjour Igor."})
    assert recorder.calls == [("begin", True), ("text", "Bonjour Igor.\n"), ("finish",)]


def test_claude_complete_message_fallback_and_interrupt():
    recorder = Recorder()
    events = Events("claude", recorder)
    events.accept("in", {"type": "user", "message": {"role": "user", "content": "Hi"}})
    events.accept(
        "out",
        {
            "type": "assistant",
            "message": {
                "id": "m",
                "content": [{"type": "thinking", "thinking": "hidden"}, {"type": "text", "text": "Hello.\n"}],
            },
        },
    )
    events.accept("in", {"type": "control_request", "request": {"subtype": "interrupt"}})
    events.accept("out", {"type": "assistant", "message": {"id": "late", "content": "Too late"}})
    assert recorder.calls == [("begin", False), ("text", "Hello.\n"), ("cancel",)]


def test_claude_queued_prompt_does_not_speak_late_previous_answer():
    recorder = Recorder()
    events = Events("claude", recorder)
    start_claude(events)
    events.accept("in", {"type": "user", "message": {"role": "user", "content": "keep audio\nNext question"}})
    events.accept(
        "out",
        claude({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Old answer.\n"}}),
    )
    events.accept("out", {"type": "result", "is_error": False})
    events.accept("out", {"type": "assistant", "message": {"id": "new", "content": "New answer.\n"}})
    events.accept("out", {"type": "result", "is_error": False})
    assert recorder.calls == [("begin", False), ("cancel",), ("begin", True), ("text", "New answer.\n"), ("finish",)]


def codex(method, **params):
    return {"method": method, "params": {"threadId": "thread", "turnId": "turn", **params}}


def test_codex_filters_thread_turn_and_item_types():
    recorder = Recorder()
    events = Events("codex", recorder)
    events.accept("in", codex("turn/start", input=[{"type": "text", "text": "Explain"}]))
    events.accept("out", codex("turn/started", turn={"id": "turn"}))
    events.accept("out", codex("item/started", item={"id": "answer", "type": "agentMessage", "phase": "final_answer"}))
    events.accept("out", codex("item/reasoning/textDelta", delta="secret", itemId="answer"))
    events.accept("out", codex("item/agentMessage/delta", threadId="child", itemId="answer", delta="child"))
    events.accept("out", codex("item/agentMessage/delta", turnId="old", itemId="answer", delta="old"))
    events.accept("out", codex("item/agentMessage/delta", itemId="answer", delta="The answer.\n"))
    events.accept("out", codex("item/completed", item={"id": "answer", "type": "agentMessage", "text": "The answer."}))
    events.accept("out", codex("turn/completed", turn={"id": "turn", "status": "completed"}))
    assert recorder.calls == [("begin", False), ("text", "The answer.\n"), ("finish",)]


def test_explicit_speech_tool_yields_ownership_to_mcp():
    recorder = Recorder()
    events = Events("claude", recorder)
    start_claude(events)
    events.accept(
        "out",
        claude(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {"type": "tool_use", "name": "mcp__voicebox__voicebox.speak"},
            }
        ),
    )
    events.accept(
        "out",
        claude({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Duplicate.\n"}}),
    )
    assert recorder.calls == [("begin", False), ("cancel",)]


@pytest.mark.parametrize("width", [1, 3, 13, 500])
def test_markdown_across_arbitrary_deltas(width):
    prose = Prose()
    text = "## Bonjour\nUne **phrase** avec [un lien](https://example.test).\n```sh\nsecret command\n```\n~~~py\nother secret\n~~~\n| A | B |\n|---|---|\n    indented code\nLa suite."
    result = "".join(prose.feed(text[i : i + width]) for i in range(0, len(text), width)) + prose.feed("", final=True)
    assert result == "Bonjour\nUne phrase avec un lien.\nLa suite.\n"


def test_unclosed_fence_and_long_line_are_not_spoken():
    prose = Prose()
    assert prose.feed("```\nsecret" + "x" * 20_000 + "\n", final=True) == ""
    assert prose.feed("still secret\n", final=True) == ""


def test_quoted_and_list_fences_are_not_spoken():
    prose = Prose()
    assert prose.feed("> ```sh\n> secret\n> ```\n- ~~~py\n  secret\n  ~~~\nDone.\n") == "Done.\n"


def test_json_line_parser_bounds_memory_and_preserves_utf8():
    seen = []
    lines = Lines(seen.append, limit=100)
    payload = json.dumps({"text": "Bonjour, été."}, ensure_ascii=False).encode() + b"\n"
    for byte in payload:
        lines.feed(bytes([byte]))
    lines.feed(b"x" * 500)
    assert len(lines.buffer) == 0
    lines.feed(b'\n{"ok": true}\n')
    assert seen == [{"text": "Bonjour, été."}, {"ok": True}]


def test_keep_requires_user_directive():
    assert wants_keep("keep audio, on teste")
    assert wants_keep("Question\nKEEP AUDIO")
    assert not wants_keep('Do not keep audio. The docs say "keep audio".')


def wait_for(check):
    deadline = time.monotonic() + 3
    while not check():
        assert time.monotonic() < deadline
        time.sleep(0.005)


class FakeClient:
    def __init__(self):
        self.calls = []
        self.full_once = True

    def request(self, path, data=None):
        self.calls.append((path, data))
        if path == "":
            return {"session_id": "session"}
        if path.endswith("/append") and self.full_once:
            self.full_once = False
            raise HTTPError(429)
        return {"state": "draining"}


def test_narrator_retries_same_sequence_and_keep_without_blocking_producer():
    client = FakeClient()
    narrator = Narrator(client, lambda message: None)
    try:
        narrator.begin(True)
        narrator.append("First phrase.")
        narrator.finish()
        wait_for(lambda: any(path.endswith("/finish") for path, _ in client.calls))
        appends = [data for path, data in client.calls if path.endswith("/append")]
        assert appends == [{"sequence": 0, "text": "First phrase."}] * 2
        assert client.calls[0] == ("", {"keep_audio": True, "request_id": narrator.current.request_id})
        narrator.cancel()
        wait_for(lambda: any(path.endswith("/cancel") for path, _ in client.calls))
    finally:
        narrator.close()


def test_narrator_bounds_unconsumed_text():
    client = FakeClient()
    narrator = Narrator(client, lambda message: None, limit=10)
    try:
        narrator.begin()
        narrator.append("far more than ten characters")
        assert narrator.current.disabled
        assert not narrator.current.pending
        time.sleep(0.01)
        assert client.calls == []
    finally:
        narrator.close()


def test_interrupt_during_slow_start_cancels_late_session():
    entered, release = threading.Event(), threading.Event()
    client = FakeClient()
    request = client.request

    def slow(path, data=None):
        if path == "":
            entered.set()
            release.wait(2)
        return request(path, data)

    client.request = slow
    narrator = Narrator(client, lambda message: None)
    try:
        narrator.begin()
        narrator.append("Hello.")
        assert entered.wait(1)
        narrator.cancel()
        release.set()
        wait_for(lambda: any(path.endswith("/cancel") for path, _ in client.calls))
        assert not any(path.endswith("/append") for path, _ in client.calls)
    finally:
        release.set()
        narrator.close()


def test_process_relay_preserves_bytes_stderr_and_exit_code():
    child = "import sys; data=sys.stdin.buffer.read(); sys.stdout.buffer.write(data+b'\\xff\\x00\\n'); sys.stderr.write('child stderr\\n'); sys.exit(7)"
    parent = (
        "import sys; from backend.agent_bridge.proxy import relay; from backend.agent_bridge.narrator import Client; "
        f"sys.exit(relay([sys.executable, '-c', {child!r}], 'claude', Client('http://127.0.0.1:1')))"
    )
    payload = b'{"type":"control_request","request":{"subtype":"permission","secret":"not speech"}}\n'
    result = subprocess.run([sys.executable, "-c", parent], input=payload, capture_output=True, timeout=8)
    assert result.returncode == 7
    assert result.stdout == payload + b"\xff\x00\n"
    assert b"child stderr" in result.stderr
    assert b"not speech" not in result.stderr


def test_cancel_after_lost_start_cleans_known_session_without_sending_text():
    entered, release = threading.Event(), threading.Event()
    calls = []

    class LostStartClient:
        def request(self, path, data=None):
            calls.append((path, data))
            if path == "":
                entered.set()
                assert release.wait(2)
                raise TimeoutError("response lost after session creation")
            return {"state": "cancelled"}

    narrator = Narrator(LostStartClient(), lambda _: None)
    try:
        narrator.begin()
        narrator.append("Hello.")
        assert entered.wait(1)
        request_id = calls[0][1]["request_id"]
        narrator.cancel()
        release.set()
        wait_for(lambda: any(path.endswith("/cancel") for path, _ in calls))
        assert calls[-1] == (f"/{request_id}/cancel", {})
        assert not any(path.endswith("/append") for path, _ in calls)
    finally:
        release.set()
        narrator.close()


def test_codex_binary_follows_extension_registry(tmp_path):
    binary = tmp_path / "new-version/bin/codex"
    binary.parent.mkdir(parents=True)
    binary.touch()
    (tmp_path / "extensions.json").write_text(
        json.dumps([{"identifier": {"id": "openai.chatgpt"}, "location": {"path": str(tmp_path / "new-version")}}])
    )
    assert resolve_codex(
        {
            "extensions_root": str(tmp_path),
            "codex_relative_binary": "bin/codex",
            "codex_executable": "/missing-old-version",
        }
    ) == str(binary)
