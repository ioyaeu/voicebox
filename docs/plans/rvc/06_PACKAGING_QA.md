# Step 06 — Packaging, test consolidation, docs (Phase A gate)

**Status:** **shipped 2026-07-02** — Phase A gate passed (frozen-build smoke, test-suite CI hygiene, user-facing docs).

**Prerequisite reading:** `00_OVERVIEW.md`. **Depends on:** steps 01–04. Gates the Phase A release; step 05 re-runs the packaging checks for its additions afterwards.

## Deliverables

### 1. PyInstaller / frozen build (`backend/build_binary.py`, `backend/voicebox-server.spec`, `backend/pyi_hooks/`)

- Add hidden-import / collect rules for the new deps, following the existing long `--hidden-import` list pattern in `build_binary.py` (starts ~line 87):
  - `faiss` — needs its native lib collected; check whether a `collect-all faiss` or a custom hook in `backend/pyi_hooks/` (see existing hooks there for the pattern) is required for `libfaiss` + `swigfaiss`.
  - **Single `libomp` in the frozen bundle (macOS).** torch and faiss each ship a `libomp.dylib`; loading both segfaults (see step 01, dual-OpenMP fixup). The dev venv is fixed by symlinking faiss's copy to torch's, but PyInstaller collection can re-introduce the duplicate. The frozen app must contain exactly one `libomp.dylib` with both consumers resolving to it — exclude faiss's copy in the spec/hook and verify with `find <bundle> -name "libomp*"` plus the concurrent torch+faiss repro from step 01 run against the frozen binary.
  - `pyworld` (compiled extension), `torchcrepe` (ships `.pth` assets inside the package — needs data collection), and the vendored `backend.backends.rvc` package itself if module discovery is dynamic anywhere.
  - `pyworld` 0.3.5 ships **no cp312 wheel** — it builds from sdist, needing a C++ toolchain (verified on macOS in step 01). Confirm the CI and release-build environments can build it, **especially Windows** (MSVC); if a Windows builder lacks the toolchain, resolve it here (toolchain in the build image, or a prebuilt wheel cache) and document the choice.
- Windows/macOS specifics only if the existing script already branches per-OS — extend those branches, don't restructure the script.
- **Measure and report the binary size delta** vs. a pre-RVC build. If the delta exceeds ~300 MB, investigate what got over-collected before accepting.
- **torchcrepe assets — CORRECTION (2026-07-02).** An earlier revision of this doc claimed torchcrepe "lazily downloads models at runtime" — **false**. torchcrepe ships its weights inside the package (`torchcrepe/assets/full.pth` ~85 MB, `tiny.pth` ~2 MB) and has **no download fallback**; excluding the assets makes `f0_method=crepe` crash in the frozen binary (observed in the shipped Phase A build). Resolution: exclude the assets from the bundle **and** register the crepe model in the `ModelConfig` registry for download-on-demand (like contentvec/rmvpe), sourced from a commit-pinned GitHub URL with sha256 verification; `pitch.py` loads the weights from the app's model directory instead of the package assets dir. Missing model → clear 400 from `/convert`, never a crash.

### 2. Frozen-build smoke test

On the local platform: build, run the frozen binary, and against it execute: server boots → migrations run → `GET /models` lists contentvec/rmvpe → RVC profile create + model upload → `POST /convert` on a short clip completes. This catches the class of failure PyInstaller creates (missing dynamic imports crash at *call* time, not import time) — an unfrozen pytest pass proves nothing here.

### 3. Test-suite + CI consolidation

- Ensure all RVC tests (steps 01–03) skip cleanly in CI (no network, no GPU, no big fixture): `pytest backend/tests -x -q` from a clean checkout without fixtures must be green.
- Add fixture documentation to `backend/tests/README.md`: how to place an RVC checkpoint locally to run the full-fidelity tests.
- If the repo has a CI workflow running backend tests, confirm nothing new is collected that needs the fixture.

### 4. Documentation

- `docs/plans/rvc/` docs get a status line update (Phase A shipped).
- User-facing: whichever doc surface documents features (check `docs/` structure and `docs/PROJECT_STATUS.md` conventions) gets an RVC section: what it is, model compatibility (RVC v1/v2 community `.pth` + optional `.index`), where to get models, security note (checkpoints are loaded weights-only), CPU-vs-GPU expectations, and Phase B (realtime + BlackHole/VB-Cable routing) marked as upcoming.
- `README` mention only if other engines are listed there (match existing granularity).

## Subagent plan (2 build + 1 verify)

| Agent | Owns | Task |
|-------|------|------|
| B1 `packaging` | `build_binary.py`, `voicebox-server.spec`, `pyi_hooks/` | Deliverable 1 |
| B2 `tests-docs` | `backend/tests/README.md`, docs surfaces | Deliverables 3–4 |
| V1 `frozen-smoke` | run-only | Deliverable 2 against B1's build; paste the full transcript including binary size |

## Step-specific Opus guardrails

- **The frozen smoke test is the whole point.** Opus reliably reports packaging steps as done after editing the spec file without ever building. V1's transcript (build log tail + runtime `curl` transcript + `ls -lh` of the binary) is the only acceptable evidence.
- Don't `--collect-all torch` or other blunt-instrument collection to make errors go away — that's how binaries gain a gigabyte. Diagnose the specific missing module from the frozen traceback.
- Don't touch signing/notarization config (`docs/plans/MACOS_NOTARIZATION.md` territory) — out of scope.
- Docs match repository voice and existing structure; no new marketing pages.

## Acceptance criteria

1. Frozen binary transcript per deliverable 2, on at least the local platform, with binary size delta reported.
2. `pytest backend/tests -x -q` green from a fixture-less clean checkout.
3. Docs updated; `git diff --stat` for this step touches only owned files.
