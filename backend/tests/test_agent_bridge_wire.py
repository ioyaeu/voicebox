"""Loopback integration test: real relay process, synthetic agent, fake sink.

No agent model, private prompt, playback device or production database is used.
"""

import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def test_live_agent_pipe_to_session_with_ide_bytes_unchanged():
    calls = []
    finished = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            calls.append((self.path, data, self.headers.get("X-Voicebox-Client-Id")))
            if self.path.endswith("/finish"):
                finished.set()
            self.reply({"session_id": "s", "state": "draining"})

        def do_GET(self):
            self.reply({"state": "completed" if finished.is_set() else "accepting"})

        def reply(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    records = [
        {"type": "stream_event", "event": {"type": "message_start", "message": {"id": "message"}}},
        {
            "type": "stream_event",
            "event": {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Une réponse publique.\n"},
            },
        },
        {
            "type": "stream_event",
            "event": {
                "type": "content_block_delta",
                "index": 1,
                "delta": {"type": "thinking_delta", "thinking": "never send this to Voicebox"},
            },
        },
        {"type": "stream_event", "event": {"type": "message_stop"}},
        {"type": "assistant", "message": {"id": "message", "content": "Une réponse publique."}},
        {"type": "result", "is_error": False},
    ]
    raw = "".join(json.dumps(value, ensure_ascii=False) + "\n" for value in records).encode()
    child = f"import sys; sys.stdin.buffer.readline(); sys.stdout.buffer.write({raw!r}); sys.stdout.buffer.flush(); sys.stdin.buffer.readline()"
    parent = (
        "import sys; from backend.agent_bridge.proxy import relay; from backend.agent_bridge.narrator import Client; "
        f"sys.exit(relay([sys.executable, '-c', {child!r}], 'claude', Client('http://127.0.0.1:{server.server_port}', 'claude-code')))"
    )
    process = subprocess.Popen(
        [sys.executable, "-c", parent], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output = bytearray()

    def read_output():
        while data := process.stdout.read(1):
            output.extend(data)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    try:
        prompt = {"type": "user", "message": {"role": "user", "content": "keep audio, test the bridge"}}
        process.stdin.write(json.dumps(prompt).encode() + b"\n")
        process.stdin.flush()
        assert finished.wait(5), calls
        deadline = time.monotonic() + 2
        while len(output) < len(raw) and time.monotonic() < deadline:
            time.sleep(0.01)
        assert bytes(output) == raw
        assert calls[0][1] == {"keep_audio": True}
        assert all(client == "claude-code" for _, _, client in calls)
        spoken = "".join(data["text"] for endpoint, data, _ in calls if endpoint.endswith("/append"))
        assert spoken == "Une réponse publique.\n"
        process.stdin.write(b"QUIT\n")
        process.stdin.flush()
        assert process.wait(timeout=6) == 0
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=6)
        server.shutdown()
        server.server_close()
