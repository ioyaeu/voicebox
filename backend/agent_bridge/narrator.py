"""Best-effort audio consumer: bounded memory, no backpressure into the IDE."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field


class HTTPError(Exception):
    def __init__(self, status):
        self.status = status
        super().__init__(f"HTTP {status}")


class Client:
    def __init__(self, base="http://127.0.0.1:17493", client="codex", timeout=2):
        url = urllib.parse.urlsplit(base)
        if (
            url.scheme != "http"
            or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username
            or url.password
        ):
            raise ValueError("The agent bridge only supports a trusted loopback HTTP server")
        self.base = base.rstrip("/") + "/speak/sessions"
        self.client = client
        self.timeout = timeout

        # Do not route local answer text through an environment proxy or a redirect.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None

        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())

    def request(self, path, data=None):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json", "X-Voicebox-Client-Id": self.client},
        )
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.loads(response.read(65_537))
        except urllib.error.HTTPError as exc:
            raise HTTPError(exc.code) from exc


@dataclass
class Turn:
    keep: bool
    pending: deque = field(default_factory=deque)
    chars: int = 0
    finished: bool = False
    disabled: bool = False


class Narrator:
    def __init__(self, client, log, limit=65_536):
        self.client = client
        self.log = log
        self.limit = limit
        self.condition = threading.Condition()
        self.current = None
        self.closed = False
        self.worker = threading.Thread(target=self._run, name="voicebox-narrator", daemon=True)
        self.worker.start()

    def begin(self, keep=False):
        with self.condition:
            self.current = Turn(keep)
            self.condition.notify_all()

    def append(self, text):
        if not text:
            return
        with self.condition:
            turn = self.current
            if turn is None or turn.disabled or turn.finished:
                return
            if turn.chars + len(text) > self.limit:
                turn.disabled = True
                turn.pending.clear()
                self.log("audio queue full; this turn's narration was cancelled")
            else:
                turn.pending.append(text)
                turn.chars += len(text)
            self.condition.notify_all()

    def finish(self):
        with self.condition:
            if self.current:
                self.current.finished = True
            self.condition.notify_all()

    def cancel(self):
        with self.condition:
            self.current = None
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.closed = True
            self.current = None
            self.condition.notify_all()
        self.worker.join(timeout=5)

    def _alive(self, turn):
        return not self.closed and self.current is turn and not turn.disabled

    def _wait(self, seconds=0.25):
        with self.condition:
            self.condition.wait(seconds)

    def _request(self, turn, path, data=None):
        deadline = time.monotonic() + 120
        network_deadline = time.monotonic() + 10
        while self._alive(turn):
            try:
                return self.client.request(path, data)
            except HTTPError as exc:
                if exc.status != 429:
                    raise
            except (OSError, ValueError):
                # Safe only for append/finish/status: the same sequence is retried.
                if time.monotonic() > network_deadline:
                    raise
            if time.monotonic() > deadline:
                raise TimeoutError("Audio did not consume input credit")
            self._wait(0.5)
        return None

    def _cancel_session(self, session_id):
        if session_id:
            try:
                self.client.request(f"/{session_id}/cancel", {})
            except Exception:
                self.log("could not cancel audio; server lease expiry will clean it up")

    def _run(self):
        turn = None
        session_id = None
        sequence = 0
        finished = False
        while True:
            if turn is not None and not self._alive(turn):
                self._cancel_session(session_id)
                turn, session_id = None, None
            if self.closed:
                return
            if turn is None:
                with self.condition:
                    candidate = self.current
                    if candidate and not candidate.disabled and candidate.pending:
                        turn = candidate
                        sequence, finished = 0, False
                if turn is None:
                    self._wait()
                    continue
            try:
                if session_id is None:
                    # A lost start response is NOT retried: that could create a
                    # second reader. The orphan, if any, expires on the server.
                    response = self.client.request("", {"keep_audio": turn.keep})
                    session_id = response["session_id"]
                if not self._alive(turn):
                    continue
                with self.condition:
                    text = turn.pending.popleft() if turn.pending else ""
                if text:
                    for offset in range(0, len(text), 256):
                        part = text[offset : offset + 256]
                        response = self._request(turn, f"/{session_id}/append", {"sequence": sequence, "text": part})
                        if response is None:
                            break
                        sequence += 1
                        with self.condition:
                            turn.chars -= len(part)
                    continue
                if turn.finished and not finished:
                    self._request(turn, f"/{session_id}/finish", {})
                    finished = True
                response = self._request(turn, f"/{session_id}")
                if response and response["state"] in {"completed", "cancelled", "failed"}:
                    if response["state"] == "failed":
                        self.log("Voicebox reported failed playback; this turn is silent")
                    turn.disabled = True
                    session_id = None
                self._wait(0.5)
            except Exception as exc:
                # No prompt, answer, tool data or backend error body is logged.
                self.log(
                    f"narration unavailable ({type(exc).__name__}: {getattr(exc, 'status', 'local error')}); IDE unaffected"
                )
                turn.disabled = True
                turn.pending.clear()
