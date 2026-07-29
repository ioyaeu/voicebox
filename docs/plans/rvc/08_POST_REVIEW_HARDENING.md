# Step 08 — Post-review hardening: fixes + six-month-regret prevention

**Prerequisite reading:** `00_OVERVIEW.md`. **Depends on:** Phases A/B/C implemented on `feature/rvc-voice-conversion`.
**Source:** consolidated QA review of 2026-07-03 (6 parallel reviewers over engine/streaming/API/frontend/chain/packaging, findings cross-verified). Every item below carries its verified location — agents must re-verify the line still matches before editing, not re-litigate whether the finding is real.

Work is organized in **three waves**. Wave 1 gates any release. Waves 2–3 can follow in separate runs.

---

## Wave 1 — Release gate (blockers + cheap majors)

### 1.1 Frozen build: `rvc.streaming` hidden-import (BLOCKER)

`routes/convert.py:354` imports `RVCStreamSession` lazily; `backend.backends.rvc.streaming` is missing from the hidden-import block in `build_binary.py:334-344` and from `voicebox-server.spec` — every frozen binary raises `ModuleNotFoundError` on the first WS handshake. Add the module in **both** files, then re-run the frozen smoke (step 06 deliverable 2) **extended with a WS handshake check** (connect, handshake, one silent block round-trip, clean close). This extension becomes a permanent part of the frozen smoke.

### 1.2 Engine identity + arbitration (BLOCKER — the architectural fix)

Three consumers share `get_rvc_engine()` with no coordination: WS stream, offline `/convert` (task queue), TTS→RVC chain. Confirmed failures: offline job swaps `net_g`/`_tgt_sr` under a live stream (voice flips mid-stream; sr mismatch corrupts blocks); route-global `_loaded_model_key` (`routes/convert.py:109`) is only written by the WS path, so a reconnecting stream silently reuses another profile's weights; re-uploads to the fixed `model.pth` path defeat the key entirely.

Design (settled, do not re-debate):

- **Model identity lives in `RVCPipeline`**, not in route globals: `load()` records `(resolved_path, index_path, mtime_ns)` and becomes a no-op when identity matches; delete `_loaded_model_key` from `routes/convert.py`. This also fixes the chain's per-generation reload cost (`pipeline.py:111` has no same-path short-circuit today) and the re-upload staleness (mtime changes).
- **Arbitration = exclusive lease.** Add `acquire(owner: str)` / `release(owner)` on the engine wrapper (or a tiny module-level lease around it): the stream session holds the lease for its lifetime; offline conversion jobs and chained generations try-acquire per job. On contention: offline/chain jobs **fail fast with a user-readable error** ("Voice Changer live session is active — stop it to run conversions"), surfaced through the existing task error path; the stream returns the existing 1008/409 when a job holds the lease. No queuing/preemption in v1 — this is a single-user local app; fail-fast is honest and simple.
- Concurrency tests (extend `test_rvc_streaming.py` / `test_convert_api.py` with monkeypatched pipeline): stream active → enqueue convert → task errors cleanly, stream blocks unaffected; convert running → WS connect → 1008; stream stop → convert immediately acquires; reconnect after another profile's conversion reloads the right model (identity test with two fake checkpoints, distinct mtimes).

### 1.3 Checkpoint load strictness (major)

`synthesizer.py:1640` discards the `load_state_dict(strict=False)` return. After `del net.enc_q` (upstream-expected unexpected keys), assert `missing_keys == []` and raise `ValueError` with the first few missing key names — this is the "silent garbage" class the plan bans. While there: `validate_rvc_checkpoint` (`checkpoint.py:63-97`) must also check `config` arity (18 hparams) and `weight["emb_g.weight"]` presence so malformed checkpoints die at upload with a 400, not at conversion with a raw 500 (`build_synthesizer` dereferences both today). Add both to the malicious/malformed fixtures in `test_rvc_foundations.py`.

### 1.4 Chain lifecycle: unload + `converting` status (majors)

- **RVC stack unload:** after a chained generation completes (success or failure), unload the RVC stack (`pipeline.unload()` — which must also unload the Crepe backbone, see 1.5) so the next plain TTS generation doesn't coexist with a resident synthesizer+ContentVec+RMVPE. Keep it simple: unload in a `finally` at the chain call site in `services/generation.py` (~line 285). Loading cost returns via 1.2's identity no-op when the next generation is also chained on the same profile. Do NOT build a VRAM manager.
- **`converting` status wiring:** `services/generation.py:268` emits `"converting"` but `routes/generations.py:250` (cancel active-check) and `HistoryTable.tsx:468` (`isInProgress`) only know `loading_model`/`generating`, and `app/src/lib/api/types.ts:131` misses the literal. Register it in all three + any status-badge switch. Result: chained generations cancellable during their slowest stage; history row keeps its spinner until `audio_path` is real.
- `base_model.unload_model()` between stages runs on the event loop (`generation.py:266`) — wrap in `asyncio.to_thread` like the convert step next to it.

### 1.5 Engine memory leaks (majors)

- `pipeline.unload()` (`pipeline.py:175-183`) must call `pitch.unload_crepe()` (the ~85 MB torchcrepe backbone is pinned forever via `torchcrepe.infer.model` today).
- `empty_device_cache` (`backends/base.py:265-277`) gains an `mps` branch (`torch.mps.empty_cache()`) — it is a no-op on Apple Silicon while every RVC unload path relies on it.

### 1.6 Realtime quick fixes (majors)

- **DeviceSelect crash pre-permission:** `DeviceSelect.tsx:46` renders `SelectItem value={device.deviceId}` — Chromium/WebView2 returns `deviceId: ""` for every device before mic permission, and Radix Select throws on empty values (panel crashes on first open). Filter empty-id devices into a single disabled "grant microphone access" placeholder (labels are already fallback-handled in `useAudioDevices.ts:67-70`), and never use `""` as an item value or React key.
- **`session.close()` off the event loop:** `routes/convert.py:396` calls it synchronously while the abandoned inference thread may hold the session lock — on CPU this freezes the whole backend up to one window inference and leaves `_stream_active` true. Run close via `asyncio.to_thread`, and reset `_stream_active` in a `finally` that does not depend on close completing.

### Wave 1 subagent plan (4 build + 2 verify)

| Agent | Owns | Items |
|-------|------|-------|
| B1 `packaging` | `build_binary.py`, `voicebox-server.spec` | 1.1 |
| B2 `engine-lease` | `backends/rvc/pipeline.py`, `backends/rvc/__init__.py`, `routes/convert.py` (lease + key removal + close fix), `services/convert.py`, chain call site in `services/generation.py` | 1.2, 1.4 (unload+to_thread), 1.6 (close) |
| B3 `engine-strictness` | `backends/rvc/synthesizer.py`, `checkpoint.py`, `pitch.py` (unload_crepe), `backends/base.py` (mps branch) | 1.3, 1.5 |
| B4 `status-and-select` | `routes/generations.py`, `HistoryTable.tsx`, `lib/api/types.ts`, `DeviceSelect.tsx` | 1.4 (status), 1.6 (select) |
| V1 `concurrency-tests` | test files | 1.2's concurrency matrix + 1.3 fixtures; run full RVC suite |
| V2 `frozen-smoke` | run-only | Rebuild, frozen smoke **including WS handshake**, paste transcript |

B2 and B3 both touch `backends/rvc/` — file ownership above is disjoint at file level except none overlap; keep it that way. B1/B4 fully parallel.

---

## Wave 2 — Robustness (realtime + packaging + CI)

### 2.1 Realtime latency & audio-path correctness

- **Jitter buffer drain:** `rvc-playback-processor.js:48-56,90-116` re-primes on underrun but never trims over-fill — one transient overload adds up to ~2 s of *permanent* latency. Add a drain policy (when buffered > target + 1 block for N consecutive callbacks, drop/time-compress down to target) and report actual buffered ms to the UI.
- **Measured latency gauge:** `useRvcStream.ts:512` shows a fixed formula. Replace with measurement: capture-timestamp → playback-dequeue delta (the 8-byte frame header has a reserved uint16 — see 2.3 — or add a client-side block sequence map).
- **Monitor/larsen:** `useRvcStream.ts:263-285` — with default output the `<audio>` element already plays converted audio (monitor-off is a no-op) and monitor-on plays a second copy via `playbackCtx.destination`. Make monitor-off actually mute the default-sink path, single playback route when on, and show the headphones warning whenever mic is open on a non-virtual output device, not only on the monitor toggle (`RealtimeConversionPanel.tsx:238-247`).
- **Handshake timeout + mid-stream error frame:** `convert.py:331` awaits `receive_json()` forever (half-open socket = permanent 1008 lockout — add `asyncio.wait_for`, ~10 s); add a JSON error frame so inference failures reach the client with a reason instead of a bare close (`convert.py:279-283`).

### 2.2 Protocol versioning

Add `version: 1` to the WS handshake request/response (`backend/models.py:865-896`); server rejects unknown major. Fix the drifted header doc: `useRvcStream.ts:23` calls bytes 5–8 "reserved" while the server defines byte 5 as `dropped` (`convert.py:38-40`) — document the real layout in ONE place (the server module docstring) and make the client doc point to it. Fix the client fallback rounding mismatch (`computeBlockFrames` `Math.round` vs Python banker's rounding, `useRvcStream.ts:100-106` vs `streaming.py:72`) — trivial while touching the file; also correct the window-layout comment in `streaming.py:153-155` (actual order is `[extra | crossfade | sola_search | block]`).

### 2.3 Structural single-libomp + CI

- **Build-time exclusion:** exclude faiss's `libomp.dylib` copy from collection (spec-level filter) so the frozen bundle structurally contains one libomp; keep `rth_faiss_libomp.py` as belt-and-braces but make it **log a warning** instead of `except OSError: pass` (`rth_faiss_libomp.py:34-36`). Add a post-build assertion to the build script: `find <bundle> -name "libomp*" | wc -l == 1`.
- **Release workflow:** the macOS job in `.github/workflows/release.yml` never runs `scripts/fix-macos-openmp.sh` — invoke it after pip install, before build.
- **Backend CI job:** add a backend test job to `.github/workflows/ci.yml` (fixture-less: RVC suites skip cleanly, 37 passed/2 skipped baseline). Exclude/fix the two **pre-existing main failures** first (`test_profile_duplicate_names.py` collection ImportError, `test_progress.py::test_hf_progress_tracker`) — fix if <30 min each, otherwise `xfail` with a linked issue; do not let them block the job's introduction.
- **Dev venv regression guard:** `pip install --upgrade faiss-cpu` silently restores the dual-libomp crash. Add a cheap startup check (macOS, dev mode only): if `faiss/.dylibs/libomp.dylib` exists and is not a symlink, log a prominent warning naming `just setup-python`. Also make `scripts/fix-macos-openmp.sh:24` resolve the interpreter instead of hardcoding `backend/venv/bin/python`, and exit non-zero on "python not found" (it currently reports success while skipping).

### Wave 2 subagent plan (3 build + 2 verify)

| Agent | Owns | Items |
|-------|------|-------|
| B1 `realtime-client` | worklets, `useRvcStream.ts`, `RealtimeConversionPanel.tsx` | 2.1 client side, 2.2 client side |
| B2 `realtime-server` | `routes/convert.py` (WS), `backend/models.py`, `streaming.py` (comment) | 2.1 server side, 2.2 server side |
| B3 `packaging-ci` | spec/build script, `rth_faiss_libomp.py`, workflows, `fix-macos-openmp.sh`, startup check | 2.3 |
| V1 `latency-verify` | run-only | Re-run synthetic SOLA suite + a scripted overload scenario proving the buffer drains; manual latency gauge sanity |
| V2 `ci-verify` | run-only | CI job green on a fixture-less checkout; frozen bundle libomp count == 1; paste outputs |

---

## Wave 3 — Six-month-regret prevention

### 3.1 API contract: stop leaking storage paths (do this before anyone codes against it)

Remove `rvc_model_path`/`rvc_index_path` from `VoiceProfileResponse` (`backend/models.py:65-66`, `services/profiles.py::_profile_to_response`) in favor of `rvc_has_model: bool` (+ the already-returned `rvc_version`/`rvc_sample_rate`/`rvc_f0` metadata). Grep the frontend for every consumer of those fields and migrate (`app/src/lib/api/types.ts` + panels). The DB columns stay (internal); add them to `_normalize_storage_paths` (`database/migrations.py:325-330`) so a moved data dir doesn't strand them. **One agent owns backend+frontend for this change** — it's a breaking contract edit that must land atomically.

### 3.2 Single source of truth for RVC defaults & helpers

- Frontend: one `RVC_DEFAULTS` module (e.g. `app/src/lib/api/constants.ts`) replacing the three hand-synced copies (`FileConversionPanel.tsx:25-31`, `RvcModelPanel.tsx:34-43`, `DEFAULT_RVC_BASE_VOICE`).
- Backend: extract the duplicated chunked temp-upload loop (`routes/profiles.py::_stream_upload_to_temp` vs inline loop `routes/convert.py:153-164`) and the copy-pasted crepe-availability check (`services/convert.py:71-79` = `routes/convert.py:211-219`) into shared helpers.

### 3.3 API behavior polish

- Explicit conflicting `engine` on an RVC profile → 400, not silent override (`routes/generations.py:53-59`); omitted engine keeps resolving to the chain.
- `POST /convert`: run the rvc/model-uploaded checks **before** consuming the multipart body (`routes/convert.py:145-167`), and add a `Content-Length` pre-check upper bound so a 5 GB body is rejected at header time.
- Orphan hygiene: startup sweep for stranded `tmp*.pth` in profile dirs and `{id}_source*` in the generations dir (findings: enqueue-failure and crash windows leak files today).
- Stage attribution for chained failures: wrap TTS and RVC stages so history `error` reads `"TTS stage: …"` / `"Conversion stage: …"` (`run_generation` except block).
- Base-model download check: stop hardcoding `"1.7B"` (`services/generation.py:238`) — resolve the size from what's installed / the profile's base voice, and let the frontend show the normal download dialog for RVC profiles instead of skipping it (`useGenerationForm.ts`).

### 3.4 Frontend hygiene

- Gate profile save on a resolved base-voice id — never submit `"{engine}:"` (`RvcModelPanel.tsx:511-528`); disable Save while the voice list loads/errs.
- Upload cancel: abort support + timeout on `uploadWithProgress` (`client.ts:103-149`); abort on dialog close; add a "validating…" state after 100%.
- Model delete behind the standard `AlertDialog` confirmation (`RvcModelPanel.tsx:410-421`).
- Extract the duplicated paired file-picker block (`RvcModelPanel.tsx:352-397` vs `427-467`).
- `SourceFilePicker`: surface rejected-file errors inline; align the drag-drop regex with the `accept` attribute.
- `FileConversionPanel`: render an error state on `useProfiles` `isError` (not the empty state); poll the active `['history', taskId]` detail during conversion so the stage label isn't frozen.
- Crepe weights out of the HF cache root: move `HF_HUB_CACHE/crepe-full/` to the app's own model dir (`backends/base.py:94` + `pitch.py` + `routes/models.py:263-267` unwind the corrupted-repo workaround) with a one-shot migration of the existing file. Pin/assert `torchcrepe` internals used by the monkeypatch (`pitch.py:557-598`) — upper-bound the dep or assert attributes at import.

### Wave 3 subagent plan (3 build + 1 verify)

| Agent | Owns | Items |
|-------|------|-------|
| B1 `api-contract` | backend response models/services + ALL frontend consumers of rvc paths | 3.1 (atomic), 3.3 (400 + pre-checks + sweep + stage attribution) |
| B2 `dedupe` | constants/helpers both sides, crepe relocation, torchcrepe pin | 3.2, 3.4 (crepe/pin) |
| B3 `frontend-hygiene` | `RvcModelPanel.tsx`, `client.ts`, `SourceFilePicker.tsx`, `FileConversionPanel.tsx`, `useGenerationForm.ts` | 3.4, 3.3 (download dialog) |
| V1 `regression` | run-only | Full RVC suite + typecheck/lint/build + manual pass: profile CRUD, upload/cancel, convert, chain generate |

---

## Step-specific Opus guardrails

- **Findings are pre-verified — do not re-litigate them, but do re-verify line numbers** before editing (the branch may have moved).
- **1.2 is the only architectural change.** The lease is a dozen lines, not a framework: no queueing, no priorities, no async context-manager hierarchy. If the diff for arbitration exceeds ~150 lines, you're over-engineering it.
- **3.1 is a breaking API change** — it must be one atomic commit touching both sides; do not ship the backend half "for later".
- The pre-existing main test failures (Wave 2.3) are NOT license to touch unrelated test files beyond fixing/xfailing those two.
- Every wave ends with the full RVC suite + the relevant smoke actually executed and pasted. The frozen smoke now permanently includes the WS handshake — a packaging change that skips it repeats blocker 1.1.
- Do not "improve" SOLA math, DSP constants, or the security layer while passing through those files — they are verified-correct; fidelity changes are out of scope for this step.

## Acceptance criteria

**Wave 1 (release gate):**
1. Frozen binary: WS handshake + one block round-trip succeeds (transcript pasted); `find <bundle> -name "libomp*"` count reported.
2. Concurrency matrix green: stream-vs-convert, convert-vs-stream, reconnect-after-other-profile-conversion (identity), all with pasted pytest output.
3. Truncated-checkpoint upload → 400 naming missing keys; malformed config/emb_g → 400.
4. Chained generation: cancellable mid-conversion; history row stays in-progress until audio exists; second consecutive plain generation shows no resident RVC stack (assert via unload call or memory probe in test).
5. Realtime panel opens pre-permission without crash (manual, WebView + browser).

**Wave 2:** overload scenario drains back to target latency (scripted proof); handshake `version` round-trips; backend CI job green; release workflow runs the openmp fix; bundle libomp count == 1 asserted at build time.

**Wave 3:** `grep -rn "rvc_model_path\|rvc_index_path" app/src` → no matches; frontend builds; defaults exist in exactly one frontend module; crepe weights outside `HF_HUB_CACHE`; full manual pass transcript.
