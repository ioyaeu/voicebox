# Step 02 — RVC inference engine (offline pipeline)

**Prerequisite reading:** `00_OVERVIEW.md`. **Depends on:** step 01 merged.
**Blocks:** steps 03–06.

## Goal

A working offline (file-in → file-out) RVC conversion engine in `backend/backends/rvc/`, loading community `.pth` models (v1 & v2, f0 & non-f0, 32k/40k/48k), with RMVPE pitch extraction by default and ContentVec features via `transformers`. **No streaming in this step** — the pipeline is designed so step 05 can add SOLA streaming on top, but nothing here should block on that.

## Package layout

```
backend/backends/rvc/
├── __init__.py          # get_rvc_engine() thread-safe factory (mirrors existing backend factories in backends/__init__.py)
├── checkpoint.py        # from step 01
├── synthesizer.py       # vendored generator architectures (see below)
├── pitch.py             # RMVPE + torchcrepe f0 extraction
├── features.py          # ContentVec feature extraction via transformers.HubertModel
├── pipeline.py          # RVCPipeline: end-to-end offline conversion
└── LICENSE_RVC.md       # MIT attribution for vendored code
```

## Deliverables

### 1. `synthesizer.py` — vendored generator (the only vendored code)

Vendor the minimal net definitions needed to load community checkpoints, from RVC WebUI (`RVC-Project/Retrieval-based-Voice-Conversion-WebUI`, MIT — reproduce the license header in `LICENSE_RVC.md`): `SynthesizerTrnMs256NSFsid`, `SynthesizerTrnMs768NSFsid`, and their `_nono` (non-f0) variants, plus the modules they pull in (NSF-HiFiGAN generator, residual coupling blocks, text/feature encoder). Rules:

- Strip training-only code paths (discriminators, losses, DDP) — inference only (`infer()` path).
- Keep tensor math **identical** to upstream; renaming internals breaks `state_dict` loading. Verify with an actual community checkpoint: `model.load_state_dict(ckpt["weight"], strict=False)` must report **zero missing keys** (unexpected keys from enc_q are OK — document this in a comment).
- Selection logic lives in `checkpoint.py`'s `RVCCheckpointInfo`: (version, if_f0) → class.

### 2. `features.py` — ContentVec via transformers

- Load with `HubertModel.from_pretrained` from the `contentvec` registry entry of step 01 (the `lengyue233/content-vec-best` conversion exposes the needed final-proj behavior; confirm at implementation time which layer output RVC expects: **v1 uses layer 9 output projected to 256 dims via the checkpoint's trained projection; v2 uses layer 12 output, 768 dims raw**). Match upstream RVC's `extract_features` semantics exactly — this is the highest-fidelity-risk point of the whole port; compare against upstream `vc/pipeline.py` line by line.
- Input: 16 kHz mono float32. Use `load_audio` from `backend/utils/audio.py` for file loading/resampling — do not re-implement.
- Wrap model download/load in `model_load_progress` + `is_model_cached` from `backends/base.py` like every other backend.

### 3. `pitch.py` — f0 extraction

- `extract_f0_rmvpe(audio_16k, ...)`: port the minimal RMVPE inference wrapper (model file from the `rmvpe` registry entry). The RMVPE net definition is also vendored here (it is small, same MIT source).
- `extract_f0_crepe(...)` via `torchcrepe` as the optional alternative.
- Both return the (f0_coarse, f0) pair quantized to RVC's 1-based 255-bin mel scale — copy upstream's quantization exactly (off-by-one here produces monotone robot voice, a classic silent failure).
- Pitch shift parameter `f0_up_key` (semitones) applied as `f0 *= 2 ** (n / 12)` before quantization.

### 4. `pipeline.py` — `RVCPipeline`

```python
class RVCPipeline:
    def load(self, model_path: str, index_path: str | None) -> None
    def unload(self) -> None
    def is_loaded(self) -> bool
    def convert_file(self, input_path: str, *, f0_up_key: int = 0,
                     f0_method: str = "rmvpe", index_rate: float = 0.75,
                     rms_mix_rate: float = 0.25, protect: float = 0.33) -> tuple[np.ndarray, int]
```

- Device from `get_torch_device(allow_mps=True)` (`backends/base.py:80`); fp16 on CUDA, **fp32 on MPS and CPU** (MPS half-precision is unreliable for these nets).
- faiss index retrieval (`index_rate` blend) when an index is provided — faiss runs on CPU regardless of torch device.
- Follow upstream's chunked processing of long files (silence-based segmentation with `t_pad` context padding) so multi-minute files don't OOM; this chunking is *quality-neutral* (padded + trimmed), unrelated to step 05's real-time streaming.
- Output at the checkpoint's native sample rate; return `(audio, sr)` and let callers resample.
- Thread-safety: module-level factory `get_rvc_engine()` with a lock, one loaded model at a time (mirrors how `services/tts.py` manages engines — read it first and follow the same lifecycle, including `empty_device_cache` on unload).

### 5. Engine smoke test + golden fixture

`backend/tests/test_rvc_engine.py`:
- Marked `@pytest.mark.skipif` when no RVC test checkpoint is available; document in the test docstring how to place one under `backend/tests/fixtures/rvc/` (a small community model, e.g. any 40k v2 model).
- With fixture present: convert a bundled 3-second WAV; assert output duration within 5% of input, sample rate == checkpoint sr, non-silent RMS, no NaNs.
- Without network/GPU, the test suite must still pass (skip, not fail).

## Subagent plan (1 lead + 2 support build, 3 verify)

The pipeline has tight cross-module contracts (quantization scales, feature dims, padding). **Do not split `pipeline.py` across agents.**

| Agent | Owns | Task |
|-------|------|------|
| B1 `vendor-synthesizer` | `synthesizer.py`, `LICENSE_RVC.md` | Deliverable 1; proves state_dict loads with zero missing keys against a real checkpoint |
| B2 `pitch` | `pitch.py` | Deliverable 3 |
| B-lead `pipeline` | `features.py`, `pipeline.py`, `__init__.py` | Deliverables 2 & 4, **after** B1/B2 hand off; integrates and runs the first end-to-end conversion |
| V1 `fidelity-review` | read-only | Line-by-line diff of feature extraction, f0 quantization, and index blending against upstream RVC pipeline semantics; refute equivalence |
| V2 `security-review` | read-only | Checkpoint loading paths all go through step 01's `load_rvc_checkpoint`; no `torch.load` without `weights_only=True`; no `eval`/`exec`; vendored code contains no network calls |
| V3 `audio-smoke` | `backend/tests/test_rvc_engine.py` | Deliverable 5; runs it with a real checkpoint if one is available locally, otherwise verifies the skip path and runs full suite |

B1 and B2 run in parallel; B-lead starts when both finish. V1–V3 in parallel after B-lead.

## Step-specific Opus guardrails

- **The #1 risk in this step is hallucinated equivalence.** Opus tends to write plausible-looking mel/f0/feature code from memory that is subtly wrong (wrong hop size, wrong layer index, 0- vs 1-based f0 bins) and *runs without error* while producing garbage audio. Mitigation: V1's mandate is textual comparison against the actual upstream source (fetch it read-only), not "does it look right".
- Do not pull in upstream's `config.py`, CLI, or i18n machinery when vendoring — inference classes only.
- No `fairseq`, no `rvc-python`, no `infer_rvc` pip packages (guardrail 1 in overview). The vendored code + listed deps are sufficient.
- Do not add a new device-detection helper, progress system, or audio loader (overview guardrail 5).
- If a real test checkpoint cannot be obtained in the execution environment, say so explicitly in the completion report and mark exactly which acceptance criteria were verified vs. deferred — do not fake a conversion log.

## Acceptance criteria

1. `python -c "from backend.backends.rvc import get_rvc_engine"` succeeds in the backend venv.
2. With a real community checkpoint: end-to-end conversion of a 3 s clip completes on CPU; output audible/non-silent; state_dict load reports zero missing keys. Command + output pasted.
3. `pytest backend/tests -x -q` green (engine test skips cleanly without fixture).
4. V1 review finds no semantic divergence from upstream on: feature layer selection, f0 mel quantization, index blend, padding/trim logic — or all found divergences are fixed.
5. No new dependency beyond step 01's list (verify `git diff backend/requirements.txt` is empty in this step).
