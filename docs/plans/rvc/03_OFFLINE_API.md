# Step 03 — Offline conversion API: model upload + POST /convert

**Prerequisite reading:** `00_OVERVIEW.md`. **Depends on:** steps 01–02.
**Blocks:** step 04.

## Deliverables

### 1. RVC model upload endpoints (`backend/routes/profiles.py` + `backend/services/profiles.py`)

Follow the exact shape of the existing avatar upload (`routes/profiles.py:228` `upload_profile_avatar`): route stays thin, logic in the service layer.

- `POST /profiles/{profile_id}/rvc-model` — multipart with `model: UploadFile` (required, `.pth`) and `index: UploadFile | None` (optional, `.index`).
  - 404 if profile missing; 400 if `profile.voice_type != "rvc"`.
  - Stream to `data/profiles/{profile_id}/model.pth` / `model.index` via a temp file + atomic rename (don't buffer whole file in RAM; enforce the caps from step 01 (`MAX_CHECKPOINT_BYTES` 500 MB / `MAX_INDEX_BYTES` 1 GB) **during** streaming, not after).
  - Validate with `load_rvc_checkpoint` + `validate_rvc_checkpoint` + `validate_faiss_index` (step 01). On validation failure: delete the temp file, return 400 with the validator's message, leave any previously valid model untouched.
  - On success: persist `rvc_model_path`/`rvc_index_path` on the profile, return the updated `VoiceProfileResponse`. Store checkpoint metadata (version/sr/f0) — piggyback on the response or a small JSON sidecar in the profile dir, whichever the frontend step needs; keep it simple.
- `DELETE /profiles/{profile_id}/rvc-model` — remove files, null the columns.

### 2. Conversion route (`backend/routes/convert.py`, new)

- `POST /convert` — multipart: `file: UploadFile` (source audio), `profile_id: str`, optional form fields `f0_up_key: int = 0`, `f0_method: str = "rmvpe"`, `index_rate: float = 0.75`, `rms_mix_rate: float = 0.25`, `protect: float = 0.33`.
  - Validate profile is `rvc` with a model uploaded; validate source audio is decodable (reuse the audio validation used by `add_profile_sample`).
  - **Run through the existing task queue** (`services/task_queue.py` — read how `routes/generations.py`/`services/generation.py` enqueue TTS jobs and mirror that): the route returns a task id immediately; progress/completion flow through the existing task endpoints the frontend already polls. A synchronous fast-path for clips < ~30 s is optional — only add it if the task-queue round-trip proves awkward for the UI, and say so in the report.
  - Result WAV goes under the same storage layout other generated audio uses (read `services/generation.py` for where outputs live and how they're served; mirror it, including the download/serve route).
- `GET /convert/{task_id}` only if the existing generic `routes/tasks.py` endpoints don't already cover result retrieval — check first, do not duplicate.
- New service module `backend/services/convert.py` holding the logic; route file stays thin like every other route.
- Register the router in `backend/routes/__init__.py::register_routers` (both the import block and the `include_router` block).

### 3. Pydantic schemas (`backend/models.py`)

`ConvertRequest`-equivalent form validation (bounds: `f0_up_key` ∈ [-24, 24], rates ∈ [0, 1], `f0_method` ∈ {rmvpe, crepe}), and a `ConvertResponse`/task-status shape consistent with existing generation task responses.

### 4. Tests (`backend/tests/test_convert_api.py`)

Using the existing FastAPI test client patterns from `backend/tests/`:
- Upload rejection: non-`rvc` profile → 400; malicious pickle → 400 with validator message and **no file persisted**; oversized file → 413/400 cut off during streaming.
- Upload success with the minimal fake checkpoint from step 01's tests → columns set, files on disk.
- `POST /convert` against a profile without a model → 400.
- Full conversion path with the real fixture checkpoint (skip-if-absent, like step 02): enqueue, poll to completion, response WAV parses with correct sr.

## Subagent plan (2 build + 2 verify)

| Agent | Owns | Task |
|-------|------|------|
| B1 `upload` | `routes/profiles.py`, `services/profiles.py` (RVC sections only), `backend/models.py` (upload-related schemas) | Deliverable 1 + its schemas |
| B2 `convert-route` | `routes/convert.py`, `services/convert.py`, `routes/__init__.py`, `backend/models.py` (convert schemas) | Deliverables 2–3. `backend/models.py` is shared with B1 — B2 appends its own schema block only after B1 lands, or the coordinator serializes the two edits |
| V1 `api-review` | read-only | Refute: streaming upload really streams (no `await file.read()` of the whole body); failed validation cannot leave a half-written model; task-queue integration matches the generation flow; router registered; no duplicated task-status endpoint |
| V2 `e2e-runner` | `backend/tests/test_convert_api.py` | Deliverable 4; run it + full suite; then boot the real server and exercise upload + convert with `curl`, paste transcripts |

## Step-specific Opus guardrails

- **Do not invent a new job system.** The task queue exists (`services/task_queue.py`); the frontend already knows how to poll it. Mirror the generation flow.
- **Do not return the WAV inline from `POST /convert`** as the default path — conversion of long files takes minutes on CPU; the plan's contract is task-based.
- Opus habitually writes `contents = await file.read()` for uploads — that loads a 500 MB pickle into RAM. Chunked streaming to a temp path is required.
- Check `routes/tasks.py` before writing any status/result endpoint; duplication is a defect.
- No auth/rate-limiting scope creep: this is a localhost app; match the security posture of existing routes exactly.

## Acceptance criteria

1. `curl` transcript: create rvc profile → upload real `.pth` → `POST /convert` with a WAV → poll task → download result; all against a locally running server.
2. Malicious-pickle upload returns 400, `data/profiles/{id}/` contains no model file afterwards.
3. `pytest backend/tests -x -q` green.
4. `grep -n "convert" backend/routes/__init__.py` shows registration.
