# Step 09 — Chain extension: cloned profiles as base voices, language visibility

**Prerequisite reading:** `00_OVERVIEW.md`. **Depends on:** step 08 (all waves merged).
**Scope decision (settled with the user, 2026-07-03):** the extension applies to the **offline chain only** (TTS→RVC text generation, step 07). Realtime streaming keeps its current constraints for performance — it has no base-voice concept and must not grow one in this step. Sequential two-stage cost (cloning-engine generation, then RVC conversion) is acceptable offline.

## Why

- Base voices are currently limited to `RVC_BASE_VOICE_ENGINES = {"kokoro", "qwen_custom_voice"}` (`services/profiles.py:166`) because the `engine:voice_id` format presupposes preset voices. All other Voicebox engines are cloning engines — their natural unit is an existing **profile**, deliberately deferred by step 07's "no nested profile picker" guardrail. This step lifts that deferral, scoped to offline.
- **Language coverage:** the preset engines cover no Russian (and other gaps). Cloned profiles support the full profile language list (`backend/models.py` pattern includes `ru`), so profile-as-base closes the gap without adding engines.
- **Language visibility:** users can't tell today whether a base voice is Chinese, English, or French. The backend already returns `language` per preset voice (`routes/profiles.py::list_preset_voices` — kokoro and qwen_custom_voice both include it); the selector just doesn't show it.

## Wave 0 — carried-over release blockers (NOT in step 08's implementation; verify before skipping)

These were found after step 08 was written and are confirmed still live on the branch:

1. **Worklets missing from every packaged frontend build.** `useRvcStream.ts:40-41` loads worklets by URL from `public/worklets/`; the `web/` frontend (Docker) shares `app/src` via a Vite alias but has **no `public/` dir**, and `tauri/dist/worklets` is absent — `addModule()` receives the SPA's `index.html` ("text/html is not valid JavaScript") and realtime is dead in web and desktop builds. Fix: move the two worklet files under `app/src/` (e.g. `app/src/lib/worklets/`) and reference them with `new URL('./rvc-capture-processor.js', import.meta.url)` so Vite emits them as hashed assets in every consuming build. Do NOT fix by copying files into each `public/` — that re-diverges. Acceptance: `grep -rn "BASE_URL.*worklets" app/src` → no matches; build `web/` and `app/` and assert the worklet assets exist in both dists; realtime smoke against the backend-served web build.
2. **Device list redacted on relaunch (BlackHole "disappears").** `enumerateDevices()` only returns full labels after mic permission is granted *in the current session*; the BlackHole detection tests labels (`useAudioDevices.ts:27`), so on a fresh launch the panel shows only the default device until a stream is started. Fix: when `labelsAvailable` is false, show a "grant microphone access to list devices" CTA that runs a `getUserMedia({audio:true})` preflight (stop tracks immediately) then `refresh()`; **persist the selected output sink** (settings store) so a returning user keeps BlackHole selected before enumeration completes. Warn (don't clear) if the persisted sink is not in the current list.

## Deliverables

### 1. `rvc_base_voice` format extension: `profile:{profile_id}`

- Grammar becomes `engine:voice_id` **or** `profile:{profile_id}`. Extend `_validate_rvc_base_voice` (`services/profiles.py:169`):
  - referenced profile must exist and must **not** be `voice_type == "rvc"` (recursion guard — one level only, no chain-of-chains);
  - it must be generation-capable for its type (cloned → has at least one sample; designed → has `design_prompt`); reuse the existing per-type validation rather than re-implementing;
  - preset-type profiles are legal too (they resolve to `engine:voice_id` internally — normalize at write time so the chain has one resolution path).
- **Deletion semantics: block.** `DELETE /profiles/{id}` returns 409 with the list of RVC profiles using it as base (name + id) when any exist. Blocking is honest and simple; silent fallback re-introduces the "why did my voice change" bug class step 08 just killed.

### 2. Chain execution with a profile base

In the chain branch of `services/generation.py`: when the base is `profile:{id}`, resolve the base profile and generate through the **existing profile generation path** (same code the base profile would use if targeted directly — voice prompt build, engine dispatch, language), then feed the audio to `convert_audio` exactly as today. Constraints:

- No re-entry: the resolved base is guaranteed non-RVC by validation, but assert it at the call site anyway (`ValueError`, not a silent skip) — validation and execution are separated by time and DB edits.
- VRAM lifecycle: identical pattern to step 07/08 — base engine generates, unloads, then RVC loads/converts (the step 08 lease + identity no-op already cover the RVC side). Cloning engines are heavier than kokoro; measure once and record the sequential wall-clock in the completion report so expectations are documented, but do not optimize in this step.
- Task progress: the existing two-stage reporting from step 08 (1.4) applies unchanged; the first stage's label should name the base profile.

### 3. Language visibility

- **Selector:** show each preset voice's `language` (already in the `/profiles/presets/{engine}` payload) in the base-voice dropdown items, and the base profile's `language` field when the base is a profile. Plain text label (e.g. "Serena — zh"), reusing however the app displays language elsewhere (check `VoicesTab`/profile cards for the existing convention); no flag-icon system.
- **Effective language on the RVC profile:** an RVC profile's spoken language is its base voice's language. On create/update, when `rvc_base_voice` is set, default the profile's `language` column from the base (preset table or base profile's `language`), keep it user-overridable; show it on the profile card like other profiles.
- **Coverage note in the picker:** when the user's UI language or profile language has no preset coverage (e.g. `ru`), the selector's empty/gap state should point to "use a cloned profile as base" — one i18n string, not a wizard.

### 4. Frontend: base-voice selector extension

In `RvcModelPanel.tsx` chain settings: a second source option "Voicebox profile" alongside the engine presets — a simple select over existing non-RVC profiles (name + language + type badge), reusing the existing profile-list query. Keep it a flat select; the step 07 guardrail against a nested profile-picker UI still applies — one dropdown, no search/preview/tabs unless an existing component already provides it for free.

### 5. Tests

- Validation: `profile:{id}` accepted for cloned/designed/preset bases; rejected for RVC bases, missing profiles, sample-less cloned profiles; delete-blocking 409 with dependent listing.
- Chain execution with a monkeypatched cloning engine: correct dispatch order, base profile's language passed through, audio handed to `convert_audio`, unload called between stages.
- Fixture-gated e2e (skip-if-absent, same pattern as `test_rvc_chain.py`): text → cloned-base chain → non-silent output at checkpoint sr.
- Wave 0: worklet asset presence asserted in both built dists (a small build-output test or CI step, so the regression class is pinned).

## Subagent plan (3 build + 2 verify)

| Agent | Owns | Task |
|-------|------|------|
| B0 `wave0-fixes` | worklet files + `useRvcStream.ts` URL refs; `useAudioDevices.ts`, `RealtimeConversionPanel.tsx` (permission CTA + persisted sink) | Wave 0 (runs first, independent) |
| B1 `chain-backend` | `services/profiles.py` (validation + delete-block), `services/generation.py` (profile-base execution), `backend/models.py`, migrations if any | Deliverables 1–2, 3 (language column default) |
| B2 `chain-frontend` | `RvcModelPanel.tsx`, profile-card language display, i18n | Deliverables 3 (UI) – 4, after B1 publishes the schema |
| V1 `tests` | test files | Deliverable 5, full RVC suite run with output |
| V2 `e2e` | run-only | Live transcript: create RVC profile with a cloned base, generate text, verify language display; packaged-web realtime smoke (Wave 0 proof); delete-block 409 demo |

## Step-specific Opus guardrails

- **Realtime is out of scope.** Any diff touching `streaming.py`, the WS route, or the realtime panel beyond Wave 0's device-permission UX is a scope violation.
- The recursion guard is validation + runtime assert, not a graph traversal framework — RVC-based-on-RVC is simply forbidden, one `if`.
- Do not re-implement voice-prompt building for cloned bases — the generation service already does it; the chain must call through the same path (step 07's single-choke-point rule still rules).
- Language display reuses existing conventions; no new badge/flag component system.
- Wave 0's worklet fix must be the `import.meta.url` pattern, not `public/` copies — a copy "works" in whatever build the agent tests and silently misses the others, which is exactly how the bug was born.

## Acceptance criteria

1. Wave 0: worklet assets present in `app` and `web` dists (command output); realtime works against the backend-served web build; relaunch shows the permission CTA and the persisted sink.
2. RVC profile with a cloned (`ru`-language) base generates Russian speech end-to-end (fixture-gated; monkeypatched variant in CI).
3. Base-voice selector shows languages for every option; RVC profile card shows its effective language.
4. Deleting a profile used as a base → 409 naming the dependent RVC profiles; deleting after unlinking succeeds.
5. Full RVC suite green; realtime suite untouched (`git diff --stat` shows no streaming-path changes outside Wave 0's files).
