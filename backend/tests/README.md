# Backend Tests

Manual test scripts for debugging and validating backend functionality.

## Test Files

### `test_generation_progress.py`
Tests TTS generation with SSE progress monitoring to identify UX issues where users see download progress even when the model is already cached.

**Usage:**
```bash
cd backend
python tests/test_generation_progress.py
```

**Prerequisites:**
- Server must be running (`python main.py`)
- At least one voice profile must exist

### `test_real_download.py`
Tests real model download with SSE progress monitoring.

**Usage:**
```bash
cd backend
# Delete cache first to force fresh download
rm -rf ~/.cache/huggingface/hub/models--openai--whisper-base
python tests/test_real_download.py
```

**Prerequisites:**
- Server must be running (`python main.py`)

### `test_progress.py`
Unit tests for ProgressManager and HFProgressTracker functionality.

**Usage:**
```bash
cd backend
python tests/test_progress.py
```

### `test_check_progress_state.py`
Debugging script to inspect the internal state of ProgressManager and TaskManager.

**Usage:**
```bash
cd backend
python tests/test_check_progress_state.py
```

## Notes

These are manual test scripts, not automated unit tests. They're designed for:
- Debugging progress tracking issues
- Validating SSE event streams
- Monitoring real-time download behavior
- Inspecting internal state during development

## RVC Voice-Conversion Tests

`test_rvc_foundations.py`, `test_rvc_engine.py`, and `test_convert_api.py` are
automated `pytest` tests (unlike the manual scripts above). They run under the
normal suite:

```bash
cd backend
venv/bin/python -m pytest tests -q
```

Most of the coverage — migrations, profile validation, checkpoint security, and
the upload/convert API guardrails — runs with no network, GPU, or model files.
The two **full-fidelity** tests that drive a real conversion
(`test_rvc_engine.py::test_offline_conversion_produces_wellformed_audio` and
`test_convert_api.py::test_full_conversion_roundtrip`) **skip** unless you place a
checkpoint locally, so the suite stays green on a clean checkout.

### Placing an RVC checkpoint

Drop a community RVC checkpoint into `backend/tests/fixtures/rvc/`. It must be an
**extracted inference** checkpoint — one that passes `validate_rvc_checkpoint`
(carries `weight`/`config`/`f0`/`version`/`sr`). Raw training generators like
`f0G40k.pth` do **not** qualify. A small 40k v2 model (~55 MB) is ideal:

```bash
cd backend
venv/bin/python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    "trojblue/rvc-kanade-voice",
    "_weights_unsorted/keruanv2.pth",
    local_dir="tests/fixtures/rvc",
)
PY
```

The first `*.pth` (sorted) is used; an optional matching `*.index` FAISS file in
the same directory is picked up to exercise the retrieval blend. Don't commit
checkpoint binaries — keep them local. See
[`fixtures/rvc/README.md`](fixtures/rvc/README.md) for the same instructions
alongside the fixture directory.

**Prerequisites for the full-fidelity run:** the shared ContentVec
(`lengyue233/content-vec-best`) and RMVPE (`lj1995/VoiceConversionWebUI`,
`rmvpe.pt`) backbones. When they're already in the HuggingFace cache the test
downloads nothing; when they're absent and there's no network, the test skips
rather than fails.
