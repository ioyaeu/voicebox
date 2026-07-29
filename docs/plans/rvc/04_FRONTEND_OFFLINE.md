# Step 04 — Frontend: RVC profiles + Voice Changer tab (offline file mode)

**Prerequisite reading:** `00_OVERVIEW.md`. **Depends on:** step 03 (API live).
**Blocks:** step 06 sign-off of Phase A. Real-time UI is **step 05**, not here — this step ships the tab with file-conversion mode only, laid out so the realtime mode slots in later.

## Where things live (verified paths — do not create parallel structures)

- API layer: `app/src/lib/api/` (`client.ts`, `services/`, `types.ts`, `schemas/`). Read two existing services (e.g. profiles, generations) and copy their conventions — fetch wrapper, error shape, react-query usage via `app/src/lib/queryClient.ts` and hooks in `app/src/lib/hooks` or `app/src/hooks`.
- Routes: `app/src/router.tsx` (paths like `/voices`, `/effects` — see the route table starting ~line 98).
- Navigation: `app/src/components/Sidebar.tsx`.
- Voice profile UI: `app/src/components/VoicesTab/` and `app/src/components/VoiceProfiles/`.
- Strings: `app/src/i18n` — every user-facing string keyed there, all supported locales (check how existing tabs add keys; missing-locale fallback is fine if that's the established pattern).
- Reusable pieces to check before building anything new: `AudioPlayer/`, `AudioBars.tsx`, drag-drop handling in `CapturesTab`/`VoicesTab` sample upload, task-progress display used by generation UI, `components/ui` primitives.

## Deliverables

### 1. API client + types (`app/src/lib/api/`)

- Types for the RVC fields on the profile response, upload endpoints, and convert task endpoints (match step 03's schemas exactly — read `backend/models.py`, don't guess).
- Service functions: `uploadRvcModel(profileId, modelFile, indexFile?)` (multipart with upload progress callback), `deleteRvcModel(profileId)`, `startConversion(file, profileId, params)`, plus task polling via the **existing** task/generation polling hook if one exists — check before writing a new poller.

### 2. RVC profile creation (`VoicesTab` / `VoiceProfiles`)

- Add "RVC Voice Profile" to the existing voice-type choice UI (alongside cloned/preset/designed — find where `voice_type` branches in the creation flow and extend it, matching the existing option-card/selector pattern).
- RVC-specific config panel: `.pth` file picker (required) + `.index` picker (optional), upload progress, display of validated model metadata (version, sample rate, f0) returned by the API, clear inline error surface for the 400 validator messages (these are user-actionable: wrong file, wrong dims).
- Profile detail view: show model status, replace/delete model actions.

### 3. Voice Changer tab

- `app/src/components/VoiceChangerTab/` (folder with an index component, matching sibling tab structure), route `/voice-changer` in `router.tsx`, Sidebar item with an appropriate icon from the icon set already in use.
- File mode UI: drop zone / file picker → RVC profile selector (only profiles with an uploaded model; empty-state links to profile creation) → parameter controls (pitch shift ±24 semitones as the prominent one; index_rate / rms_mix / protect / f0_method behind an "Advanced" disclosure with the step 03 defaults) → convert button → task progress → result in the existing audio player component with download/export.
- Layout reserves a mode switch (`File | Real-time`) with Real-time disabled + "coming soon" tooltip, so step 05 doesn't restructure the tab.

### 4. Frontend verification

Whatever the repo's checks are (`package.json` scripts in `app/`): typecheck, lint, build. Plus a manual end-to-end pass against the running backend (create profile → upload model → convert file → play result) with screenshots or a described transcript.

## Subagent plan (3 build + 2 verify)

| Agent | Owns (exclusive) | Task |
|-------|------------------|------|
| B1 `api-client` | `app/src/lib/api/**` (new RVC files + minimal type additions) | Deliverable 1 |
| B2 `profiles-ui` | `VoicesTab/`, `VoiceProfiles/`, its i18n keys | Deliverable 2 |
| B3 `changer-tab` | `VoiceChangerTab/` (new), `router.tsx`, `Sidebar.tsx`, its i18n keys | Deliverable 3 |
| V1 `ui-review` | read-only | Refute: conventions match sibling tabs (structure, query hooks, i18n coverage, no hardcoded strings, no new fetch pattern, empty/error/loading states handled) |
| V2 `build-gate` | read-only + run | Typecheck/lint/build + manual e2e against the live backend; paste outputs |

B2 and B3 both depend on B1's types → run B1 first (it's small), then B2 ∥ B3. B2 and B3 must not both touch shared i18n files simultaneously — if locale files are monolithic JSON, the coordinator serializes those specific edits.

## Step-specific Opus guardrails

- **Read before writing.** The dominant failure mode here is inventing a component style instead of using `components/ui` primitives and sibling-tab layout. Each build agent's first action: read one full sibling implementation (B2: the designed-voice creation flow; B3: `EffectsTab` or `CapturesTab` top to bottom).
- No new state libraries, no context providers, no CSS frameworks; use the existing stores/hooks/styling approach.
- No `any`-typed API responses; types mirror `backend/models.py`.
- Do not build a custom audio player or progress bar — both exist.
- react-query cache invalidation after upload/delete must follow the existing query-key conventions (find where profile queries are keyed).
- Don't gold-plate: no waveform visualizer, no batch conversion, no drag-reorder — file mode as specced.

## Acceptance criteria

1. Typecheck + lint + production build green (paste output).
2. Manual e2e transcript: RVC profile created, model uploaded with visible progress, invalid file shows the backend's message inline, conversion runs with progress, result plays and downloads.
3. `/voice-changer` reachable from Sidebar; direct navigation works.
4. i18n: no literal user-facing strings in the new TSX (spot-check by V1).
