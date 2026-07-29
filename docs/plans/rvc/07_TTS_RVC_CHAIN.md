# Step 07 — Phase C: TTS → RVC chain (speak any text in an RVC voice)

**Prerequisite reading:** `00_OVERVIEW.md`. **Depends on:** Phase A merged (01–04, 06). Independent of step 05 — Phases B and C can run in either order.

## Goal

Let an RVC profile *generate speech from text*: a base TTS engine renders the text, then the RVC pipeline converts the result into the profile's voice. This turns RVC profiles into first-class citizens of the whole app — generation tab, stories, `/speak`, MCP — not just the Voice Changer tab. This replaces (properly) the `kokoro` allowance hack that step 01 removed.

## Design decisions (settled here, not re-debated by agents)

1. **Base voice is a profile setting.** `VoiceProfile.rvc_base_voice` (new nullable String column + migration, same `_add_column` pattern as step 01) stores `"{engine}:{voice_or_profile_id}"` — default `kokoro:af_heart`-style preset (cheap, CPU-friendly, no cloning needed). The UI exposes it as "Base voice" in the RVC profile config with a sensible default; power users can pick any preset voice.
2. **Chain lives in the generation service, not in a new endpoint.** `services/generation.py` (read it first — it's the single choke point the routes/task queue already call): when the target profile is `voice_type == "rvc"`, generate with the base engine/voice, then pass the audio through `RVCPipeline.convert` (in-memory array variant — add `convert_audio(audio, sr, **params)` to the pipeline if step 02 only shipped `convert_file`; it's the same code path minus file I/O) before the normal post-processing (effects chain, storage) runs. Everything downstream (history, versions, stories, export) works untouched because the chain output enters the existing flow at the same point engine output does.
3. **Conversion params on the profile.** Reuse step 03's param set; persist per-profile as JSON in a new nullable `rvc_params` Text column (defaults identical to step 03). The Voice Changer tab and TTS chain read the same defaults.
4. **Latency accounting.** Chained generation ≈ TTS time + conversion time; task progress must reflect two stages (the task/progress system supports staged updates — check how multi-stage generation reports today; if it's single-stage, a simple "converting…" progress message suffices, don't rebuild progress plumbing).
5. **Engine dispatch guard rails.** `validate_profile_engine` (step 01 version) currently rejects everything except `"rvc"` for rvc profiles. It now must accept the resolved base engine when the caller is the chain itself — implement by resolving the base engine *inside* the generation service (the external API surface still only ever sees the rvc profile; callers never pass the base engine).

## Deliverables

1. Schema: `rvc_base_voice`, `rvc_params` columns + migration + pydantic exposure (mirror step 01 mechanics exactly).
2. `services/generation.py` chain integration per decision 2, including model lifecycle: base TTS engine and RVC pipeline may not fit in VRAM together on small GPUs — generate, then load/convert sequentially, reusing the existing engine load/unload management in `services/tts.py` rather than inventing a VRAM manager.
3. UI: base-voice selector + params in the RVC profile editor (`VoiceProfiles`/`VoicesTab`); RVC profiles become selectable in the generation tab / stories / anywhere profiles are listed for TTS (they were presumably filtered out — find the filter and lift it).
4. `/speak` + MCP: verify the speak path (routes/speak.py, mcp_server) goes through the same generation service and therefore works for free; test it, fix only if it bypasses the choke point.
5. Tests (`backend/tests/test_rvc_chain.py`): chain path with monkeypatched TTS + identity-patched RVC (assert order of calls, params passed, output enters post-processing); real-fixture end-to-end (skip-if-absent) — generate short text on an rvc profile, output is non-silent, correct sr; regression: non-rvc profiles' generation path byte-identical behavior (no accidental chain).

## Subagent plan (2 build + 2 verify)

| Agent | Owns | Task |
|-------|------|------|
| B1 `chain-backend` | schema/migration files, `services/generation.py`, `services/profiles.py`, `backend/models.py` | Deliverables 1–2, 4 |
| B2 `chain-ui` | `VoiceProfiles/`, `VoicesTab/`, generation-tab profile filters, i18n | Deliverable 3, after B1 publishes the schema |
| V1 `adversarial-review` | read-only | Refute: external API never accepts a base engine directly; non-rvc generation untouched; VRAM lifecycle sane; params defaults consistent across tab and chain |
| V2 `e2e-runner` | `test_rvc_chain.py` | Deliverable 5 + full suite + live-server transcript of text→rvc-voice generation and `/speak` |

## Step-specific Opus guardrails

- **One choke point.** The chain goes in the generation service only. Opus will want to add chain logic in routes, `/speak`, stories, and MCP separately — each such copy is a defect.
- No "universal post-processor pipeline" abstraction. It's one `if voice_type == "rvc"` stage at one call site.
- Don't let the base-voice selector grow into a full nested profile picker — preset voices of cloning-free engines are enough for v1.
- The regression test for non-rvc profiles is mandatory; this step touches the hottest path in the app.

## Acceptance criteria

1. Live transcript: text generated on an RVC profile from the generation tab plays in the converted voice; same via `POST /speak`.
2. Task progress shows both stages (or the documented single-message fallback).
3. Full suite green; regression test proves non-rvc generation unchanged.
4. RVC profiles appear in stories/generation pickers; base voice editable in profile UI.
