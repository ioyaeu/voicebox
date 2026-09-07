#!/usr/bin/env python3
"""Pipe spoken text or JSONL {"text": delta} into local Voicebox.

Standard library only. The desktop app owns playback; this adapter never
plays audio itself and never scrapes agent logs or enables a Stop hook.
"""

import argparse
import codecs
import json
import sys
import time
import urllib.error
import urllib.request


def request(base, client, path, data=None, *, retry=False):
    deadline = time.monotonic() + 120
    while True:
        req = urllib.request.Request(
            base.rstrip("/") + "/speak/sessions" + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={
                "Content-Type": "application/json",
                "X-Voicebox-Client-Id": client,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            message = exc.read().decode(errors="replace")
            if not retry or exc.code != 429 or time.monotonic() >= deadline:
                raise RuntimeError(f"Voicebox HTTP {exc.code}: {message}") from exc
        except (urllib.error.URLError, TimeoutError):
            if not retry or time.monotonic() >= deadline:
                raise
        time.sleep(1)


def deltas(stream, format):
    if format == "jsonl":
        for line in stream:
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("text"), str):
                raise ValueError('Each JSONL record must contain a string "text" field')
            yield value["text"]
    else:
        decoder = codecs.getincrementaldecoder("utf-8")()
        while chunk := stream.read1(256):
            yield decoder.decode(chunk)
        yield decoder.decode(b"", final=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:17493")
    parser.add_argument("--client", default="codex")
    parser.add_argument("--profile")
    parser.add_argument("--language")
    parser.add_argument("--keep-audio", action="store_true")
    parser.add_argument("--format", choices=["text", "jsonl"], default="text")
    args = parser.parse_args()
    session_id = None
    try:
        result = request(
            args.url,
            args.client,
            "",
            {
                "profile": args.profile,
                "language": args.language,
                "keep_audio": args.keep_audio,
            },
        )
        session_id = result["session_id"]
        print(json.dumps(result), flush=True)
        sequence = 0
        for delta in deltas(sys.stdin.buffer, args.format):
            # Small requests allow useful progress even near the buffer ceiling.
            for offset in range(0, len(delta), 256):
                request(
                    args.url,
                    args.client,
                    f"/{session_id}/append",
                    {
                        "sequence": sequence,
                        "text": delta[offset : offset + 256],
                    },
                    retry=True,
                )
                sequence += 1
        request(args.url, args.client, f"/{session_id}/finish", {}, retry=True)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            result = request(args.url, args.client, f"/{session_id}", retry=True)
            if result["state"] in {"completed", "cancelled", "failed"}:
                print(json.dumps(result), flush=True)
                return 0 if result["state"] == "completed" else 1
            time.sleep(1)
        raise TimeoutError("Playback did not finish within 30 minutes")
    except (Exception, KeyboardInterrupt) as exc:
        if session_id:
            try:
                request(args.url, args.client, f"/{session_id}/cancel", {})
            except Exception:
                pass
        print(f"Speech stream stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
