# RVC Voice Conversion — Master Implementation Plan

**Status:** All three phases **implemented on `feature/rvc-voice-conversion`**. Phase A (offline file conversion — steps 01–04 + 06 packaging/QA gate), Phase B (realtime streaming, step 05), and Phase C (TTS→RVC chain, step 07) are all shipped on this branch. A consolidated QA review (2026-07-03, 6 parallel reviewers) found 2 blockers + ~10 majors — addressed by **step 08 (post-review hardening, waves 1–3 implemented)**. **Step 09** extends the chain (cloned profiles as base voices — offline only, realtime untouched — plus base-voice language visibility) and carries two post-08 release blockers in its Wave 0 (worklets missing from packaged frontend builds; device-list redaction on relaunch). Wave 0 of step 09 gates the next release.
**Target executor:** Claude Opus 4.8 in ultra-workflow (multi-agent) mode
**Approved decisions (do not re-litigate):**

1. **No fairseq.** HuBERT/ContentVec features are extracted with `transformers.HubertModel` (already a dependency, see `backend/requirements.txt`). Any plan or code that imports `fairseq` is wrong and must be rejected.
2. **Three delivery phases.** Phase A = offline file conversion (steps 01–04, 06). Phase B = real-time streaming with SOLA crossfade (step 05). Phase C = TTS→RVC chain (step 07). Phases B and C both require Phase A merged and verified; they are independent of each other and can run in either order.
3. **WebSocket** for real-time streaming (Phase B). Not HTTP chunking.
4. **RMVPE is the default pitch estimator**, `torchcrepe` optional, no parselmouth/PM. Crepe's `full.pth` (~85 MB) is **not bundled** in the frozen binary and torchcrepe has no built-in download fallback — the model is registered in the `ModelConfig` registry and downloaded on demand like contentvec/rmvpe (commit-pinned GitHub source + sha256; see step 06 correction note). `f0_method=crepe` without the model returns a clear 400, never a crash.
5. **`hubert`/`contentvec` and `rmvpe` are registered in the `ModelConfig` registry** (`backend/backends/__init__.py::get_all_model_configs`) so they download via the existing Models tab flow.
6. **Security:** uploaded `.pth` files are loaded with `torch.load(..., weights_only=True)` + checkpoint structure validation + size limits. Non-negotiable.
7. The `default_engine == "kokoro"` allowance for RVC profiles in the current uncommitted diff is an ad-hoc hack — **remove it in step 01**. The underlying idea (speak text through an RVC voice) is a real feature and is designed properly as the **TTS→RVC chain, step 07 (Phase C)**: base TTS engine renders the text, RVC converts it, integrated at the single generation-service choke point.
8. **Virtual audio device output is in scope for Phase B.** The real-time mode must support routing converted audio to a selectable output device — including virtual devices (BlackHole on macOS, VB-Cable on Windows) — so the converted voice can feed Discord/games/calls. Device enumeration and output selection happen client-side (Tauri/browser `setSinkId`); the app detects whether BlackHole/VB-Cable is installed and shows setup guidance when it is not. Voicebox does **not** bundle or auto-install the virtual driver.

## Step sequence and dependency graph

```
01_FOUNDATIONS  (DB migration, schemas, deps, model registry, checkpoint security)
      │
02_RVC_ENGINE   (vendored synthesizer, pitch + feature extraction, offline pipeline)
      │
03_OFFLINE_API  (upload endpoints, POST /convert via task queue)
      │
04_FRONTEND_OFFLINE (RVC profile creation UI, Voice Changer tab — file mode)
      │
06_PACKAGING_QA (PyInstaller, tests consolidation, docs)   ← end of Phase A [SHIPPED 2026-07-02]
      │
      ├── 05_REALTIME_STREAMING (SOLA engine, WS endpoint, realtime UI,   ← Phase B [SHIPPED]
      │                          BlackHole/VB-Cable output routing)
      └── 07_TTS_RVC_CHAIN (speak text through an RVC voice)              ← Phase C [SHIPPED]
              │
08_POST_REVIEW_HARDENING (QA-review fixes: Wave 1 = release gate;
                          Wave 2 = robustness; Wave 3 = regret prevention)  ← [IMPLEMENTED]
              │
09_CLONED_BASE_VOICES (Wave 0 = worklet packaging + device-permission UX [RELEASE GATE];
                       cloned profiles as chain base voices, language visibility)  ← NEXT
```

Steps 01→06 are **strictly sequential** (each consumes the previous step's artifacts). Steps 05 and 07 are independent of each other. Parallelism happens *inside* a step via subagents, per the budgets below.

## Subagent budget per step

| Step | Doc | Build agents | Review/verify agents | Isolation notes |
|------|-----|--------------|----------------------|-----------------|
| 01 | `01_FOUNDATIONS.md` | 3 parallel (disjoint files) | 2 (adversarial review + migration test on a real v0.4 DB) | No worktree needed — file sets are disjoint |
| 02 | `02_RVC_ENGINE.md` | 1 lead + 2 support (vendor import, pitch module) | 3 (correctness, security, audio-quality smoke) | Support agents feed the lead; do NOT parallel-edit the pipeline file |
| 03 | `03_OFFLINE_API.md` | 2 parallel (profiles upload / convert route) | 2 (API contract review + end-to-end run) | Disjoint files |
| 04 | `04_FRONTEND_OFFLINE.md` | 3 parallel (API client+types / VoicesTab / VoiceChangerTab+router+sidebar) | 2 (UI review + typecheck/build gate) | Disjoint files except `router.tsx`/`Sidebar.tsx` — owned by agent 3 only |
| 06 | `06_PACKAGING_QA.md` | 2 parallel (build config / test suite) | 1 (frozen-build smoke on the local platform) | Disjoint |
| 05 | `05_REALTIME_STREAMING.md` | 1 lead (SOLA engine) then 2 parallel (WS route / frontend realtime) | 3 (chunk-boundary artifact test, latency bench, adversarial review) | SOLA engine is sequential by nature; do not fan out its internals |
| 07 | `07_TTS_RVC_CHAIN.md` | 2 sequential-ish (chain backend, then UI) | 2 (adversarial review + e2e with regression on non-RVC generation) | Backend first — UI depends on its schema |
| 08 | `08_POST_REVIEW_HARDENING.md` | Wave 1: 4, Wave 2: 3, Wave 3: 3 (one wave per workflow run) | Wave 1: 2, Wave 2: 2, Wave 3: 1 | Findings are pre-verified with file:line — agents re-verify locations, not conclusions |
| 09 | `09_CLONED_BASE_VOICES.md` | 3 (wave-0 fixes ∥ chain backend → chain frontend) | 2 (tests + live e2e incl. packaged-web realtime smoke) | Realtime is out of scope beyond Wave 0 — any `streaming.py`/WS diff is a violation |

Total: ~24 build agents + ~20 verify agents across the feature. Do not exceed a step's budget to "go faster" — the file-ownership boundaries above exist to prevent merge conflicts and half-duplicated helpers.

## Orchestration rules for the coordinating agent

- Run one step per workflow invocation. Read the step doc **in full** before spawning agents; pass each agent the doc path plus its exact file-ownership list.
- Every build fan-out is followed by a verify fan-out (adversarial: reviewers are prompted to *refute* that the step's acceptance criteria are met, not to confirm them).
- A step is done only when its **Acceptance criteria** section passes with commands actually executed and outputs captured. "Should work" is a failure state.
- If an agent reports a blocker that contradicts this plan (e.g. a dependency that won't resolve), stop the workflow and surface it — do not improvise a substitute architecture mid-run.

## Known Opus 4.8 failure modes — global guardrails

Every subagent prompt must include the guardrails relevant to its task. These are recurring, observed habits; treat them as lint rules.

1. **Hallucinated imports.** Opus will confidently `import fairseq`, `from rvc import ...`, or `import rvc_python`. None of these are dependencies. The only allowed new imports are: `faiss` (from `faiss-cpu`), `pyworld`, `torchcrepe`, plus existing deps (`torch`, `torchaudio`, `transformers`, `librosa`, `soundfile`, `numpy`). Anything else → reject.
2. **Placeholder code.** No `# In a real implementation...`, no `raise NotImplementedError` left behind, no mocked inference paths outside tests. If a piece can't be finished, the agent must say so instead of stubbing it silently.
3. **Unverified "it works".** Claims require executed commands: `pytest` output, an actual `curl`/httpx call against a running server, `pnpm typecheck` output. Reviewers must reject any completion report without pasted command output.
4. **Drive-by refactors.** Agents touch only the files in their ownership list. Reformatting, renaming, or "improving" adjacent code is out of scope and must be reverted.
5. **Duplicate utilities.** `backend/backends/base.py` already provides `get_torch_device`, `is_model_cached`, `model_load_progress`, `empty_device_cache`; `backend/utils/audio.py` provides `load_audio`/`normalize_audio`. Re-implementing any of these is a defect.
6. **Over-engineering.** No plugin systems, no abstract base class hierarchies, no config dataclasses beyond what the step doc specifies. Copy the shape of the closest existing module (e.g. `kokoro_backend.py` for backends, `routes/profiles.py` for routes).
7. **Narrating comments.** No `# Added for RVC`, `# New endpoint`, changelog-style comments. Comments only for non-obvious constraints (matching `backend/STYLE_GUIDE.md`).
8. **Swallowed exceptions.** No bare `except Exception: pass`. Errors propagate to the route layer / task queue, which already handles reporting.
9. **Skipped DB migration testing.** Schema changes must be tested against a copy of a real pre-existing SQLite DB, not just a freshly created one.
10. **Frontend pattern drift.** API calls go through `app/src/lib/api/` (existing client/services structure); user-facing strings go through `app/src/i18n`; no new state-management patterns beyond the existing stores/hooks.

## Reference implementations (for reading, not copy-pasting wholesale)

- RVC WebUI (`RVC-Project/Retrieval-based-Voice-Conversion-WebUI`, MIT) — canonical synthesizer architecture (`infer/lib/infer_pack/models.py`) and offline pipeline (`infer/modules/vc/pipeline.py`). We vendor a minimal subset (step 02) with license attribution.
- w-okada voice changer — reference for SOLA crossfade logic (step 05 only).
