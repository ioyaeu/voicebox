# Step 05 — Phase B: real-time streaming (SOLA engine, WebSocket, realtime UI, virtual device output)

**Prerequisite reading:** `00_OVERVIEW.md`. **Depends on:** Phase A merged (steps 01–04, 06).
**This is the hardest step of the feature. Budget accordingly and do not start it as a side quest of Phase A.**

## Goal

Live microphone → converted voice with usable latency on GPU, correct (click-free) chunk stitching, and output routable to any device **including virtual audio devices (BlackHole on macOS, VB-Cable on Windows)** so the converted voice can feed Discord/games/calls (overview decision 8).

## Architecture

```
Mic ──AudioWorklet(capture, 16k mono f32)──► WS /convert/stream ──► RVCStreamSession
                                                                       │  sliding context + SOLA
Selected output device (incl. BlackHole) ◄──AudioWorklet(playback)◄── converted PCM chunks
```

Latency budget target on CUDA: block 250–350 ms, total mouth-to-ear < 700 ms. On CPU: measure, warn, allow anyway (user may accept it for testing).

## Deliverables

### 1. `backend/backends/rvc/streaming.py` — `RVCStreamSession`

The core. Wraps a loaded `RVCPipeline` model for frame-based inference:

- **Ring buffer of input audio** at 16 kHz. Each inference call processes `[extra_context | crossfade | block | lookahead]` — the block is the new audio; extra context (~0.5–1.0 s of past input) is re-inferred every call and discarded, giving the nets the context they need; converted output is stitched to the previous output via **SOLA**: search ±`sola_search_ms` (~10 ms) for the max-correlation offset in the overlap zone, then equal-power crossfade (~80 ms). This mirrors w-okada / RVC realtime GUI semantics — implement from the description here and upstream reading, tune constants by measurement.
- Pitch: RMVPE on the same windowed input; f0 continuity across calls comes from the shared context window (no cross-call f0 smoothing state in v1).
- Config: `block_ms` (client-negotiated, default 300), `extra_ms=800`, `crossfade_ms=80`, plus the step 03 conversion params.
- Session state is per-connection, single-threaded per session; inference runs in a worker thread (`asyncio.to_thread`) so the event loop never blocks.
- Reports per-block `infer_ms` alongside audio so the client can show a latency gauge and the server can detect underrun (infer_ms > block_ms → send an overload flag).

### 2. `WS /convert/stream` (`backend/routes/convert.py`)

- FastAPI WebSocket endpoint. Handshake: client sends JSON `{profile_id, block_ms, f0_up_key, ...}`; server loads/locks the model (reuse the loaded pipeline if it's the same profile), replies `{ready, model_sr}`.
- Then binary frames: client sends f32le PCM 16 kHz mono blocks; server replies binary converted blocks (f32le at model_sr) with a small JSON header frame or a fixed prefix carrying `infer_ms`/flags — pick the simplest framing and document it in the module docstring.
- Clean teardown on disconnect (buffer freed, model stays loaded). One active stream session at a time (409 on second connect) — this is a local single-user app.
- Backpressure: if the client outruns inference, drop oldest pending input and set the overload flag rather than growing the queue unboundedly.

### 3. Frontend realtime mode (`VoiceChangerTab`)

- Enable the `Real-time` mode switch reserved in step 04.
- Capture: `getUserMedia` + `AudioWorklet` downsampling to 16 kHz mono f32, sent over the WS. Echo cancellation/noise suppression constraints **off** (they fight voice conversion).
- Playback: `AudioWorklet` fed by received chunks with a small jitter buffer (~1 block).
- **Output device selection**: enumerate outputs via `navigator.mediaDevices.enumerateDevices()`, route with `HTMLMediaElement.setSinkId`/`AudioContext.setSinkId` (verify support in the Tauri webview on macOS/Windows early — **this is the step's riskiest unknown**; if the webview lacks `setSinkId`, fall back to a Tauri-side rust audio output using an existing crate, and flag the scope change to the coordinator before building it).
- **Virtual device guidance**: detect BlackHole (macOS) / VB-Cable (Windows) in the device list; when absent, show a setup card linking to install instructions ("route output to BlackHole 2ch, select it as mic in Discord"). Voicebox does not bundle the drivers.
- UI: input/output device selectors, monitor toggle (hear yourself) with a headphones-required warning against feedback, input/output level meters (reuse `AudioBars.tsx`), latency gauge from `infer_ms` + block size, overload indicator, the pitch/advanced params shared with file mode.

### 4. Tests & benches

- `backend/tests/test_rvc_streaming.py`: SOLA unit tests **with synthetic signals, no model needed** — stitch a sine wave processed identity-wise through the session windowing and assert no discontinuity above a threshold at block joins (sample-to-sample delta bounded), correct output length, overload flag on artificially slow inference (monkeypatched).
- WS contract test with a monkeypatched pipeline (identity conversion): handshake, binary round-trip, second-connection 409, disconnect cleanup.
- Real-model latency bench as a manual script (`backend/tests/bench_rvc_stream.py`, not collected by CI): feeds a WAV in block_ms chunks, prints p50/p95 infer_ms per device.

## Subagent plan (1 lead + 2 build, 3 verify)

| Agent | Owns | Task |
|-------|------|------|
| B-lead `sola-engine` | `streaming.py` | Deliverable 1 — sequential, single-owner; the ring buffer/SOLA/windowing logic must be one mind's work |
| B1 `ws-route` | `routes/convert.py` (WS section), related schemas | Deliverable 2, after B-lead defines the session API (B-lead publishes the class signature first) |
| B2 `realtime-ui` | `VoiceChangerTab/` realtime components, worklets, device logic + i18n | Deliverable 3, in parallel with B1. **First task: verify `setSinkId` works in the Tauri webview and report before building the rest** |
| V1 `boundary-test` | `test_rvc_streaming.py` | Synthetic SOLA/WS tests as specced; run them |
| V2 `latency-bench` | `bench_rvc_stream.py` | Real-model bench on the local machine; report p50/p95 vs block size on available devices |
| V3 `adversarial-review` | read-only | Refute: event loop never blocked, queue bounded, teardown leak-free, worklets don't allocate per-frame, feedback warning present, echoCancellation disabled, no second streaming implementation diverging from `RVCPipeline` internals (streaming must reuse pipeline components, not fork them) |

## Step-specific Opus guardrails

- **Naive chunking is the trap this step exists to avoid.** Any implementation that converts blocks independently without SOLA + re-inferred context is an automatic reject, even if the demo "sounds okay" on one sample.
- Opus will be tempted to add cross-call f0 smoothing, adaptive block sizing, or a WebRTC stack. No. Fixed windowing + SOLA first; measure; iterate only if V2's bench shows a problem.
- Don't buffer PCM as base64 JSON — binary WS frames.
- The synthetic-signal tests are mandatory *because* audio bugs are silent; "I listened and it sounds fine" is not verification.
- `asyncio.to_thread` for inference — Opus habitually runs torch inside the event loop and everything "works" until the UI socket starves.
- If `setSinkId` is unsupported in the webview, **stop and report** (B2's first task) — do not silently ship speaker-only output, and do not build a rust audio stack without coordinator sign-off.

## Acceptance criteria

1. Synthetic SOLA tests + WS contract tests green in CI (no model, no GPU needed).
2. Bench transcript: p50/p95 infer_ms per device; on CUDA (or the best local device), p95 infer_ms < block_ms — or the report documents the measured floor and the CPU warning path is shown in the UI.
3. Manual e2e: live mic conversion audible, mode switch works, output device selector lists and routes to BlackHole when installed, setup card appears when it isn't, monitor-off by default with headphone warning.
4. Kill-switch behavior: closing the tab / toggling off tears down the WS and the server logs a clean session end; a second session immediately connects fine.
