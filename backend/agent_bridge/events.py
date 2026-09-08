"""Observe public agent events only; never inspect transcripts or reasoning."""

from __future__ import annotations

import json
import re
from collections import deque
from contextlib import suppress

from ..utils.speech_text import strip_markdown


class Prose:
    """Bounded line-oriented Markdown filter, independent of token boundaries.

    Plain prose can flush at sentence boundaries. Marked-up lines wait until
    newline/message end; unclosed fences and oversized lines are discarded.
    """

    def __init__(self):
        self.buffer = ""
        self.fence = None
        self.discard_line = False

    def feed(self, text, final=False):
        output = []
        for part in text.splitlines(keepends=True):
            complete = part.endswith(("\n", "\r"))
            if not self.discard_line:
                self.buffer += part
                if len(self.buffer) > 16_384:
                    self.buffer = ""
                    self.discard_line = True
            if complete:
                if not self.discard_line:
                    output.append(self._line(self.buffer))
                self.buffer = ""
                self.discard_line = False
        if final:
            if not self.discard_line:
                output.append(self._line(self.buffer))
            self.buffer = ""
            self.discard_line = False
        elif self.fence is None and not self.discard_line:
            # No partial inline markup or indented code may cross this fast path.
            if not re.search(r"[`~*_|\[\]<>#]", self.buffer) and not self.buffer.startswith(("    ", "\t")):
                matches = list(re.finditer(r"[.!?]\s+", self.buffer))
                if matches and matches[-1].end() >= 60:
                    end = matches[-1].end()
                    output.append(self._line(self.buffer[:end]))
                    self.buffer = self.buffer[end:]
        return "".join(output)

    def _line(self, line):
        fence = re.match(r"^\s*(?:>\s*)*(?:(?:[-*+]|\d+[.)])\s+)?(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if self.fence is None:
                self.fence = marker
            elif marker[0] == self.fence[0] and len(marker) >= len(self.fence):
                self.fence = None
            return ""
        if self.fence or line.startswith(("    ", "\t")) or "|" in line:
            return ""
        value = strip_markdown(line)
        return value + "\n" if value else ""


def user_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") in {"text", "input_text"}
        )
    return ""


def wants_keep(text):
    # A directive at the beginning of a line, not a match in tool output or a
    # quoted explanation such as 'do not use keep audio'.
    return bool(re.search(r"(?im)^\s*keep audio\b", text))


def is_speech_tool(name):
    name = str(name).lower().replace(".", "_")
    return "voicebox" in name and name.endswith(("_speak", "_speech_start"))


class Events:
    def __init__(self, protocol, narrator):
        self.protocol = protocol
        self.narrator = narrator
        self.active = False
        self.thread = None
        self.turn = None
        self.items = set()
        self.message = None
        self.streamed = set()
        self.blocks = set()
        self.prose = Prose()
        self.epoch = 0
        self.in_flight = False
        self.pending_prompts = deque()

    def begin(self, keep=False):
        self.epoch += 1
        self.active = True
        self.items.clear()
        self.streamed.clear()
        self.blocks.clear()
        self.message = None
        self.prose = Prose()
        self.narrator.begin(keep)

    def text(self, text):
        if self.active and isinstance(text, str):
            self.narrator.append(self.prose.feed(text))

    def flush(self):
        self.narrator.append(self.prose.feed("", final=True))
        self.prose = Prose()

    def cancel(self):
        self.active = False
        self.narrator.cancel()

    def accept(self, direction, record):
        if not isinstance(record, dict):
            return
        if len(self.streamed) > 4096 or len(self.items) > 4096:
            self.cancel()
            self.streamed.clear()
            self.items.clear()
        if self.protocol == "claude":
            self._claude(direction, record)
        else:
            self._codex(direction, record)

    def _claude(self, direction, record):
        kind = record.get("type")
        if direction == "in":
            if kind == "user" and not record.get("parent_tool_use_id"):
                message = record.get("message", {})
                if message.get("role") == "user":
                    keep = wants_keep(user_text(message.get("content")))
                    if self.in_flight:
                        self.cancel()
                        if len(self.pending_prompts) < 16:
                            self.pending_prompts.append(keep)
                    else:
                        self.begin(keep)
                        self.in_flight = True
            elif kind == "control_request" and record.get("request", {}).get("subtype") == "interrupt":
                self.cancel()
            return
        if kind == "result" and self.in_flight and not record.get("parent_tool_use_id"):
            if self.active:
                if record.get("is_error"):
                    self.cancel()
                else:
                    self.flush()
                    self.narrator.finish()
                    self.active = False
            self.in_flight = False
            if self.pending_prompts:
                self.begin(self.pending_prompts.popleft())
                self.in_flight = True
            return
        if not self.active or record.get("parent_tool_use_id"):
            return
        if kind == "stream_event":
            event = record.get("event", {})
            event_type = event.get("type")
            if event_type == "message_start":
                self.message = event.get("message", {}).get("id")
                self.blocks.clear()
                if self.message:
                    self.streamed.add(self.message)
            elif event_type == "content_block_start":
                block = event.get("content_block", {})
                if block.get("type") == "tool_use" and is_speech_tool(block.get("name")):
                    self.cancel()
                    return
                if block.get("type") == "text":
                    self.blocks.add(event.get("index"))
                    self.text(block.get("text", ""))
            elif event_type == "content_block_delta" and event.get("index") in self.blocks:
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    self.text(delta.get("text", ""))
            elif event_type == "message_stop":
                self.flush()
                self.blocks.clear()
        elif kind == "assistant":
            message = record.get("message", {})
            # Claude also sends each complete message after its deltas.
            if message.get("id") and message["id"] not in self.streamed:
                self.streamed.add(message["id"])
                self.text(user_text(message.get("content")))
                self.flush()

    def _codex(self, direction, record):
        method = record.get("method")
        params = record.get("params", {})
        if direction == "in":
            if method in {"turn/start", "turn/steer"} and params.get("threadId"):
                self.thread = params["threadId"]
                self.turn = params.get("expectedTurnId") if method == "turn/steer" else None
                self.begin(wants_keep(user_text(params.get("input"))))
            elif method == "turn/interrupt" and params.get("threadId") == self.thread:
                self.cancel()
            return
        if not self.active or params.get("threadId") != self.thread:
            return
        if method == "turn/started":
            self.turn = params.get("turn", {}).get("id")
            return
        if method == "turn/completed":
            turn = params.get("turn", {})
            if turn.get("id") != self.turn:
                return
            if turn.get("status") == "completed":
                self.flush()
                self.narrator.finish()
                self.active = False
            else:
                self.cancel()
            return
        if not self.turn or params.get("turnId") != self.turn:
            return
        item = params.get("item", {})
        if (
            method == "item/started"
            and item.get("type") == "mcpToolCall"
            and is_speech_tool(str(item.get("server", "")) + "_" + str(item.get("tool", "")))
        ):
            self.cancel()
            return
        if method == "item/started" and item.get("type") == "agentMessage":
            if item.get("phase") not in {None, "commentary", "final_answer"}:
                return
            self.items.add(item.get("id"))
        elif method == "item/agentMessage/delta" and params.get("itemId") in self.items:
            self.text(params.get("delta", ""))
            self.streamed.add(params["itemId"])
        elif method == "item/completed" and item.get("id") in self.items:
            if item.get("id") not in self.streamed:
                self.text(item.get("text", ""))
            self.flush()
            self.items.discard(item["id"])
            self.streamed.discard(item["id"])
        # No generic recursive walk: tool results, approvals, plans, reasoning
        # and other threads must never accidentally become spoken text.


class Lines:
    """Bound memory even when a tool emits enormous JSON or non-JSON lines."""

    def __init__(self, callback, limit=1_048_576):
        self.callback = callback
        self.limit = limit
        self.buffer = bytearray()
        self.discard = False

    def feed(self, chunk):
        for part in chunk.splitlines(keepends=True):
            ended = part.endswith(b"\n")
            if not self.discard:
                self.buffer.extend(part)
                if len(self.buffer) > self.limit:
                    self.buffer.clear()
                    self.discard = True
            if ended:
                if not self.discard:
                    with suppress(ValueError, TypeError, AttributeError, KeyError):
                        self.callback(json.loads(self.buffer))
                self.buffer.clear()
                self.discard = False
