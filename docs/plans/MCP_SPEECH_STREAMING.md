# Incremental Agent Speech

Status: first implementation in the working tree, September 2026. Packaged
desktop listening tests remain a release gate, not an assumed success.

This extends [MCP_SERVER](MCP_SERVER.md) and phase 5 of [VOICE_IO](VOICE_IO.md).
It does not replace `voicebox.speak` or turn Voicebox into an agent orchestrator.

## First Implementation

Pipeline: agent spoken-prose deltas -> bounded session -> existing serial
generation queue -> TTS / optional RVC / effects / normalization -> WAV segments
-> one desktop renderer -> playback acknowledgement -> temporary file and row
deletion. Python owns segmentation and inference; Rust forwards session events;
the existing React pill owns playback and pause/stop controls.

This is **incremental text input with phrase-level audio output**, not native
PCM streaming. The model still finishes each phrase before it can be played.
The existing loudness stabilization remains in the normal generation path;
naturalness and level consistency across independently generated phrases must
still be checked by listening, especially with Voxtral and RVC.

## Producer Contract

| MCP tool | REST equivalent | Meaning |
| --- | --- | --- |
| `voicebox.speech_start(profile?, language?, keep_audio=false)` | `POST /speak/sessions` | Resolve voice and create session |
| `voicebox.speech_append(session_id, sequence, text)` | `POST /speak/sessions/{id}/append` | Append exact spoken-prose delta |
| `voicebox.speech_finish(session_id)` | `POST /speak/sessions/{id}/finish` | Close input and flush final phrase |
| `voicebox.speech_status(session_id)` | `GET /speak/sessions/{id}` | Observe credit and playback state |
| `voicebox.speech_cancel(session_id)` | `POST /speak/sessions/{id}/cancel` | Stop playback and discard pending text |

Sequences start at zero and are contiguous. Retrying an accepted sequence with
identical text is idempotent; different text or a gap is rejected. Whitespace
between deltas is significant. `available_chars` is producer credit; REST 429
means retry the **same** sequence and text later. Other validation errors must
not be retried blindly. The tools report these service errors through MCP's
normal tool-error surface, not HTTP status codes in JSON-RPC.

Send spoken prose only, not raw Markdown, tool output, code, or reasoning.
`speak` retains its full-text Markdown cleanup, truncation and personality
rewrite options. Streaming deliberately does not rewrite every fragment or
truncate a reply silently. Speak and stream must not narrate the same answer.

Voice resolution is shared with speak: explicit profile, then client binding,
then global voice. Preset/RVC profile engines take precedence over unrelated
binding defaults. Profile id, engine, language and model size are resolved at
start. Changes to profile metadata are detected before subsequent submissions;
do not edit the profile or its samples during a session. RVC uses the existing
TTS-to-RVC chain per phrase; this is compatible, but not a promise of realtime
latency or permanently resident base and conversion models.

Only one session is admitted, including its cleanup drain. Up to two rendered
or rendering phrases are prefetched, plus 4096 characters of outstanding text.
A session accepts at most 100,000 characters / 10,000 append calls. Phrase size
is bounded at 320 characters, using the batch splitter's sentence rules. Short
partial sentences wait for further input or finish. Independent manual jobs
still use the existing serial queue and may delay speech or cause model swaps;
there is no new global GPU scheduler or reservation of every inference API.

## Sink And Lifecycle

`POST /speak/sessions/{id}/playback` accepts a stable random `renderer` id,
optional `acknowledged` segment sequence, `stopped`, or `error`. The first
renderer claims the session. A second renderer is rejected instead of playing
the same audio. Heartbeat replies contain ordered ready generation ids, served
by the existing `/audio/{generation_id}` route.

The pill acknowledges only `audio.ended`. Pause sends heartbeats but no ack,
and neither consumes credit nor deletes files. Failed playback is a failure,
not a successful listen. Lost ack replies retry the ack without replaying the
segment. Events are hints: current state is polled, and active session events
are replayed on SSE reconnect and heartbeat. Backend completion events from
individual generations do not control stream playback.

States: `accepting`, `draining`, `completed`, `cancelled`, `failed`. `finish`
returns `draining`; only acknowledgement of all audio leads to `completed`.
Missing producer activity for 5 minutes or renderer heartbeat for 2 minutes
fails the session. Paused playback can stay alive through renderer heartbeats.
Renderer reload does not silently transfer ownership or replay speech; cancel
and start a new session. Sessions are in-memory, with 32 terminal snapshots for
short-term inspection/retries, and are not resumable across backend restart.

Stop halts local audio and pending text; an already submitted inference job
drains through the serial worker before its files are removed. Cancellation
does not interrupt a GPU thread and unload a model underneath it. Shutdown
allows a short drain, then leaves remaining model teardown to process exit.
Orphaned temporary stream rows and files are removed on next startup.

Temporary is the default. An acknowledged segment's audio **and generation
row** are removed, including versions. On cancellation/failure, unplayed
temporary segments are also discarded after inference drains. Cleanup failures
are logged; startup cleanup is the fallback. With `keep_audio=true`, generated
phrases remain as separate normal History entries, including on cancellation.
There is no joined export or persistent session transcript in this version.

Local trust boundary: client-id headers are routing identity, not credentials.
Producer operations check that identity; the opaque session id is the playback
capability. This extends the existing trusted-local API, not remote multi-user
authentication. Keep Voicebox on loopback; do not expose these routes publicly.

## Daily Use

Rebuild backend and Tauri frontend, restart the app, then refresh the MCP tool
list in the client. No hook or user configuration is installed by this feature.

For an agent that can issue tools during its answer: start once, append spoken
sentences in order, finish once, and inspect status on errors. Set `keep_audio`
at start when the user says "keep audio". Use the exact client id configured in
Voicebox for that agent's voice; do not hardcode Voxtral in agent settings.

For an external agent harness, forward its public answer-text events through
the standard-library adapter:

```sh
your-agent-text-command | python3 scripts/speech_stream.py --client codex
your-agent-jsonl-command | python3 scripts/speech_stream.py --client claude-code --format jsonl --keep-audio
```

The first command consumes UTF-8 text; the second expects one `{"text":"delta"}`
JSON object per line. The commands on the left are placeholders for the host's
actual text event source, not bundled executables. The adapter handles input
credit, retries, finish, completion polling and Ctrl-C cancellation. It never
plays audio itself. Its stdout contains start and terminal session snapshots.

MCP availability alone does **not** subscribe to an IDE's response tokens.
Automatic token forwarding needs a host adapter exposing those public deltas.
A Stop hook runs too late for that purpose. Do not re-enable it alongside the
session player. This implementation changes neither Codex nor Claude settings.

## Verification

Implementation checks: 74 targeted backend tests passed (one skipped and the
real-model RVC smoke test deselected), four player tests passed, TypeScript,
Vite production build and offline `cargo check` passed. The initial real-model
RVC smoke run was interrupted while retrying an unavailable Hugging Face
request. No end-to-end speaker or packaged-app streaming claim is made.

```sh
backend/venv/bin/python -m pytest backend/tests/test_speech_sessions.py backend/tests/test_speech_session_api.py backend/tests/test_mcp_speak.py backend/tests/test_speak_speech_options.py -q
bun test app/tests/speechSessionPlayer.test.ts
npm run typecheck --workspace=@voicebox/app
```

Before release, use at least a two-minute Voxtral passage and an RVC passage:
pause mid-phrase, wait beyond generation completion, resume, stop, retry a
delta, disconnect the backend, and repeat with keep-audio. Verify no automatic
double reader, stable levels, no deleted paused audio, no temporary History
rows after cleanup, and kept files still playable. Check packaged macOS and
the supported desktop platforms; unit tests do not validate speakers or GPU
latency. Main-window manual playback remains independent of the pill.

## Next Phase

1. Add an example host adapter for an actual public agent text-event API.
   Forward answer text only, preserve turn ids, propagate stop/barging-in, and
   test reconnect without duplicate speech. Keep host-specific policy outside
   the core Voicebox service and avoid private conversation-log scraping.
2. Measure cold/warm time to first audio, realtime factor, underruns, memory,
   disk usage and level variation for Voxtral, Kokoro and TTS-to-RVC. Tune
   phrase size and prefetch from evidence. Record machine/model/build details.
3. Introduce a backend capability for native progressive PCM, with a bounded
   native playback buffer, sample-rate/channel metadata, cancellation, and
   continuous loudness control. Preserve WAV fallback for engines/effects
   without that capability. Benchmark RVC buffering separately.
4. Add optional session-level kept export, using actual segment order, and
   explicit recoverability semantics. Design authenticated remote sinks only
   with a separate security review and resource budget.

Upstream contribution boundaries: session service and tests; REST/MCP tools;
desktop single-sink renderer; adapter and documentation. No new runtime
dependency, parallel model manager, or IDE-specific hook is required.
