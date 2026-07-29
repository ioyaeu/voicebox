# Step 01 — Foundations: schema, migration, dependencies, model registry, checkpoint security

**Prerequisite reading:** `00_OVERVIEW.md` (decisions + guardrails).
**Depends on:** nothing. **Blocks:** all other steps.

## Starting state (important)

The working tree already contains an **uncommitted partial diff** for this feature touching `backend/database/models.py`, `backend/models.py`, `backend/services/profiles.py`, `backend/requirements.txt`, and the **root** `requirements.txt`. This step reconciles and completes that diff. Do not assume a clean slate; read the current state of each file first.

## Deliverables

### 1. Dependencies — fix placement and versions

- **Remove** `torchcrepe` and `fairseq` from the root `requirements.txt` (fairseq is banned entirely; see overview decision 1 — it does not install on Python ≥3.12, which `backend/pyproject.toml` requires).
- In `backend/requirements.txt`, keep/adjust the RVC block to exactly:
  ```
  # RVC voice conversion (see docs/plans/rvc/)
  faiss-cpu>=1.8.0
  pyworld>=0.3.5
  torchcrepe>=0.0.23
  ```
  `pyworld` 0.3.5 is the latest PyPI release (an earlier revision of this doc required `>=0.3.6`, which was never published — verified against PyPI on 2026-07-02). 0.3.5 has no cp312 wheel; it builds from sdist on macOS/Linux (verified working in `backend/venv` on Python 3.12.13), which requires a C++ toolchain at install time — step 06 must confirm this doesn't break the frozen-build or CI environments, and setup docs should mention the compiler requirement if the repo's install docs don't already imply one. Verify each pin actually resolves: run `pip index versions <pkg>` or a dry `pip install --dry-run` in the backend venv and paste the output in the completion report.
- Revert the stray whitespace-only edits to `docs/PROJECT_STATUS.md` from the partial diff.
- **macOS dual-OpenMP fixup (mandatory).** `faiss-cpu` wheels bundle their own `libomp.dylib`; torch bundles another. Two OpenMP runtimes in one process segfault under concurrent load (`SIGSEGV` in `__kmp_suspend_initialize_thread` — diagnosed and reproduced 2026-07-02; `OMP: Error #15` on import + work). Fix: replace `site-packages/faiss/.dylibs/libomp.dylib` with a **symlink** to `site-packages/torch/lib/libomp.dylib` (verified compatible: both LLVM libomp, interface 5.0.0; a file *copy* does NOT work — dyld dedupes by resolved path, so only a symlink yields a single loaded image). Automate this as an idempotent post-install step in the backend setup script (the one handling the existing `--no-deps` installs), macOS-only, keeping a `.bak` of the original. `KMP_DUPLICATE_LIB_OK=TRUE` is explicitly rejected (unsupported, silent audio corruption risk).

### 2. Database schema + migration

- `backend/database/models.py`: keep the two new columns on `VoiceProfile` (`rvc_model_path`, `rvc_index_path`), both `Column(String, nullable=True)`. Remove the trailing-whitespace line introduced by the partial diff.
- `backend/database/migrations.py`: in `_migrate_profiles`, add:
  ```python
  _add_column(engine, "profiles", "rvc_model_path VARCHAR", "rvc_model_path")
  _add_column(engine, "profiles", "rvc_index_path VARCHAR", "rvc_index_path")
  ```
  following the exact pattern of the existing `personality` migration at `migrations.py:161`.

### 3. Pydantic schemas + profile validation

- `backend/models.py`: keep the `voice_type` pattern extension (`^(cloned|preset|designed|rvc)$`) and the two optional path fields on `VoiceProfileResponse` (already in the partial diff). Remove the stray blank line the diff added.
- `backend/services/profiles.py::_validate_profile_fields`, `rvc` branch — replace the partial diff's version with:
  - RVC profiles cannot set `preset_engine`, `preset_voice_id`, or `design_prompt`.
  - `default_engine`, if set, must be exactly `"rvc"`. **Delete the `kokoro` allowance** (overview decision 7).
- `backend/services/profiles.py::validate_profile_engine`, `rvc` branch: only `engine == "rvc"` is valid; anything else raises `ValueError` (delete the CLONING_ENGINES/kokoro allowances from the partial diff).

### 4. Model registry entries

In `backend/backends/__init__.py`, extend `get_all_model_configs()` (currently at line ~462) with two system models, following the existing `ModelConfig` dataclass (line ~48):

- `contentvec` — feature extractor. `engine="rvc"`, HF repo hosting a **transformers-format** ContentVec/HuBERT checkpoint (e.g. `lengyue233/content-vec-best`, which loads via `HubertModel.from_pretrained` with its custom final-projection config — the engine step consumes it; here we only register it for download). `size_mb≈360`.
- `rmvpe` — pitch estimator. `engine="rvc"`, repo `lj1995/VoiceConversionWebUI`, single file `rmvpe.pt`. Use the `required_files` capability of `is_model_cached` (`backends/base.py:24`) for cache detection; downloading a single file uses `huggingface_hub.hf_hub_download`, consistent with existing flows. `size_mb≈180`.

Check how `routes/models.py:243` consumes the registry (download/status endpoints) and make sure both entries download and report status through the existing Models tab flow without special-casing.

### 5. Checkpoint security utility

New file `backend/backends/rvc/checkpoint.py` (create the `backend/backends/rvc/` package with `__init__.py`; the engine of step 02 lives in the same package):

```python
def load_rvc_checkpoint(path: str) -> dict: ...
def validate_rvc_checkpoint(ckpt: dict) -> RVCCheckpointInfo: ...
```

- `load_rvc_checkpoint` calls `torch.load(path, map_location="cpu", weights_only=True)`. **Explicitly pass `weights_only=True`** — the repo floor is torch 2.2 where it is not the default. Reject files > 500 MB before loading.
- `validate_rvc_checkpoint` verifies the community-RVC checkpoint shape: top-level keys `weight` (state dict), `config` (list of hparams), `f0` (0/1), `version` (`"v1"`/`"v2"`), `sr` (or sample rate embedded in config). Return a small dataclass `RVCCheckpointInfo(version, sample_rate, if_f0, embedder_dim)` where `embedder_dim` is 256 (v1) or 768 (v2), derived from `config`. Raise `ValueError` with a user-readable message on any mismatch — this message surfaces in the upload endpoint (step 03).
- Same file: `validate_faiss_index(path)` — cap size (1 GB — raised from the original 200 MB on 2026-07-03: real-world indexes from large training sets reach 450 MB+; RAM cost is ~2× file size while loaded, see `checkpoint.py` comment), load with `faiss.read_index`, check `index.d == embedder_dim`.

### 6. Tests

New `backend/tests/test_rvc_foundations.py`:
- Migration: copy a fixture SQLite DB **without** the new columns (build it in the test via raw SQL matching the pre-migration `profiles` schema), run `run_migrations`, assert both columns exist and existing rows survive.
- Validation: `_validate_profile_fields` accepts a bare `rvc` profile, rejects `rvc` + `design_prompt`, rejects `default_engine="kokoro"`.
- Checkpoint: `weights_only` enforcement — build a malicious pickle containing a reduce payload, assert `load_rvc_checkpoint` refuses it; build a minimal well-formed fake checkpoint dict saved with `torch.save`, assert validation extracts version/sr/f0 correctly.

## Subagent plan (3 build + 2 verify)

| Agent | Owns (exclusive) | Task |
|-------|------------------|------|
| B1 `deps-and-schema` | both `requirements.txt`, `database/models.py`, `database/migrations.py`, `docs/PROJECT_STATUS.md` | Deliverables 1–2 |
| B2 `validation` | `backend/models.py`, `backend/services/profiles.py` | Deliverable 3 |
| B3 `registry-and-security` | `backend/backends/__init__.py`, `backend/backends/rvc/` | Deliverables 4–5 |
| V1 `adversarial-review` | read-only | Try to refute: fairseq truly gone from both requirement files; kokoro allowance gone; migration matches `_add_column` pattern; `weights_only=True` present; no guardrail violations (overview list) |
| V2 `test-runner` | `backend/tests/test_rvc_foundations.py` | Write deliverable 6, run `pytest backend/tests/test_rvc_foundations.py -v` plus the existing suite (`pytest backend/tests -x -q`), paste output |

B1–B3 run in parallel (disjoint files). V1/V2 run after all three complete.

## Step-specific Opus guardrails

- Do **not** install fairseq "just to check". It is banned (overview decision 1).
- Do not invent a `ModelConfig` field (e.g. `required_files=`) — the dataclass at `backends/__init__.py:48` has a fixed field list; single-file cache detection goes through `is_model_cached(..., required_files=[...])` at call sites, not through new dataclass fields, unless you first verify how existing configs handle it and follow that exact mechanism.
- The migration test must use a **pre-migration schema DB**, not a DB created by current SQLAlchemy metadata (which already has the columns — the test would be vacuous).
- Do not "helpfully" bump unrelated pins in requirements.txt.

## Acceptance criteria

1. `grep -rn "fairseq" requirements.txt backend/requirements.txt` → no matches.
2. Fresh venv check or `pip install --dry-run -r backend/requirements.txt` resolves on Python 3.12 (paste output).
3. `pytest backend/tests/test_rvc_foundations.py -v` green; `pytest backend/tests -x -q` no new failures vs. `main`.
4. Server boots (`python -m backend.main` or the documented dev command) against a copied pre-existing `data/` DB with no migration errors in the log.
5. `GET /models` (or the Models-tab endpoint in `routes/models.py`) lists `contentvec` and `rmvpe` with correct cached/not-cached status.
6. On macOS: `python -c "import torch, faiss"` followed by concurrent torch matmul + faiss search does not abort or segfault (single-libomp fixup applied by the setup script; paste the repro command output).
