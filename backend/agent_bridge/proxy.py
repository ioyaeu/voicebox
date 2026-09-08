"""Byte-preserving process relay with an optional, non-blocking speech tap."""

from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
from contextlib import suppress
from pathlib import Path

from .events import Events, Lines
from .narrator import Client, Narrator


def log(message):
    print(f"[voicebox bridge] {message}", file=sys.stderr, flush=True)


def relay(command, protocol, client):
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None, bufsize=0)
    chunks = queue.Queue(maxsize=128)
    disabled = threading.Event()
    stopping = threading.Event()
    narrator = Narrator(client, log)
    events = Events(protocol, narrator)

    def observe():
        readers = {
            direction: Lines(lambda record, direction=direction: events.accept(direction, record))
            for direction in ("in", "out")
        }
        while not stopping.is_set():
            if disabled.is_set():
                narrator.cancel()
                return
            try:
                direction, data = chunks.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                readers[direction].feed(data)
            except Exception:
                disabled.set()
                narrator.cancel()
                log("unrecognized event shape; speech disabled, relay continues")

    def copy(source, destination, direction):
        try:
            while True:
                data = os.read(source, 8192)
                if not data:
                    break
                # Observe input before the child can answer it; output is still
                # byte-identical. Never wait for the audio consumer.
                if not disabled.is_set():
                    try:
                        chunks.put_nowait((direction, data))
                    except queue.Full:
                        disabled.set()
                        log("event mirror full; speech disabled, relay continues")
                view = memoryview(data)
                while view:
                    count = os.write(destination, view)
                    view = view[count:]
        except (OSError, ValueError):
            if direction == "out" and process.poll() is None:
                process.terminate()
        finally:
            if direction == "in":
                with suppress(OSError):
                    process.stdin.close()

    observer = threading.Thread(target=observe, daemon=True)
    incoming = threading.Thread(target=copy, args=(sys.stdin.fileno(), process.stdin.fileno(), "in"), daemon=True)
    outgoing = threading.Thread(target=copy, args=(process.stdout.fileno(), sys.stdout.fileno(), "out"), daemon=True)
    observer.start()
    incoming.start()
    outgoing.start()
    handlers = {}

    def forward(signum, _frame):
        narrator.cancel()
        if process.poll() is None:
            process.send_signal(signum)

    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        handlers[signum] = signal.signal(signum, forward)
    try:
        code = process.wait()
        outgoing.join(timeout=2)
    finally:
        stopping.set()
        narrator.close()
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    return code if code >= 0 else 128 - code


def main(protocol, argv, config_path):
    # The Claude extension passes its bundled binary as argv[0]. Codex's
    # development override does not, so the installer records the real binary.
    try:
        with open(config_path, encoding="utf-8") as file:
            config = json.load(file)
    except (OSError, ValueError):
        config = {"enabled": False}
    if protocol == "claude":
        if not argv or not os.path.isfile(argv[0]):
            raise SystemExit("Claude wrapper requires the bundled executable as its first argument")
        command = argv
        streaming = "stream-json" in argv and "--output-format" in argv
    else:
        command = [resolve_codex(config), *argv]
        streaming = "app-server" in argv and not any(
            flag in argv for flag in ("--help", "generate-ts", "generate-json-schema")
        )
        if "--listen" in argv and argv[argv.index("--listen") + 1 : argv.index("--listen") + 2] != ["stdio://"]:
            streaming = False
        if any(arg.startswith("--listen=") and arg != "--listen=stdio://" for arg in argv):
            streaming = False
    if os.path.realpath(command[0]) == os.path.realpath(sys.argv[0]):
        raise SystemExit("Recursive voice bridge executable configuration")
    if not streaming or not config.get("enabled", True):
        os.execv(command[0], command)
    try:
        client = Client(config.get("url", "http://127.0.0.1:17493"), config.get("clients", {}).get(protocol, protocol))
    except Exception:
        log("invalid audio configuration; launching agent without speech")
        os.execv(command[0], command)
    return relay(command, protocol, client)


def resolve_codex(config):
    """Follow VS Code's extension registry across normal extension updates."""
    root = Path(config.get("extensions_root", Path.home() / ".vscode/extensions"))
    relative = config.get("codex_relative_binary", "bin/macos-aarch64/codex")
    try:
        entries = json.loads((root / "extensions.json").read_text())
        for entry in entries:
            if entry.get("identifier", {}).get("id") == "openai.chatgpt":
                path = Path(entry["location"]["path"]) / relative
                if path.is_file():
                    return str(path)
    except (OSError, ValueError, KeyError):
        pass
    fallback = config.get("codex_executable")
    if fallback and os.path.isfile(fallback):
        return fallback
    raise SystemExit("Codex binary not found; reinstall the bridge or remove chatgpt.cliExecutable")
