"""Real-model latency bench for the RVC real-time streaming session.

Manual script for deliverable 4 / acceptance criterion 2 of
``docs/plans/rvc/05_REALTIME_STREAMING.md``. It is **not** a pytest module: the
filename does not match ``test_*.py`` so pytest never collects it, and the
runnable body is guarded under ``__main__``. Nothing here runs in CI (it needs a
real checkpoint and the shared backbones cached, plus a GPU/MPS to be meaningful).

What it measures
----------------
It drives a real :class:`~backend.backends.rvc.streaming.RVCStreamSession` -- the
same class the WebSocket route uses -- with a loaded ``RVCPipeline`` and feeds a
WAV through it in ``block_ms`` chunks, exactly as the live path would. For each
block it records the session's own ``infer_ms`` (conversion + SOLA stitch, the
number the server reports and uses for its overload flag) and prints p50/p95 per
device and block size. The realtime constraint is ``p95 infer_ms < block_ms`` on
the best local device: if it holds the stream keeps up in real time; if not, the
printed floor is the honest latency this machine can sustain (05 allows CPU/MPS
to be slower with a UI warning path -- only CUDA is held to the hard budget).

It does not fork any conversion math: it constructs the shipped session against a
real pipeline and times ``session.process`` end to end.

Warmup
------
The first few blocks on MPS pay one-time costs (device init, kernel compilation,
lazy backbone load) that do not reflect steady state, so a small number of
leading blocks are excluded from the percentiles and reported separately as the
cold-start cost.

Running
-------
From ``backend/`` with the venv python::

    venv/bin/python tests/bench_rvc_stream.py
    venv/bin/python tests/bench_rvc_stream.py --block-ms 250,300,350 --devices mps,cpu
    venv/bin/python tests/bench_rvc_stream.py --wav /path/to/voice.wav --seconds 15

The checkpoint is resolved from the local HuggingFace cache
(``trojblue/rvc-kanade-voice`` ``_weights_unsorted/keruanv2.pth`` -- a real
v2/40k/f0 model) or, failing that, the first ``*.pth`` under
``tests/fixtures/rvc/``. ContentVec and RMVPE must already be cached; this script
does not hit the network.
"""

import argparse
import contextlib
import sys
import time
from pathlib import Path

import numpy as np

# Repo-root imports (the project convention: run from backend/, import backend.*).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.backends.rvc import pipeline as _pipeline_mod
from backend.backends.rvc.pipeline import RVCPipeline
from backend.backends.rvc.streaming import (
    DEFAULT_CROSSFADE_MS,
    DEFAULT_EXTRA_MS,
    DEFAULT_SOLA_SEARCH_MS,
    RVCStreamSession,
)
from backend.utils.audio import load_audio

_SR_16K = 16000
_REPO = "trojblue/rvc-kanade-voice"
_CKPT_FILENAME = "_weights_unsorted/keruanv2.pth"
_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rvc"


def resolve_checkpoint() -> tuple[str, str | None]:
    """Return ``(checkpoint_path, index_path_or_None)`` from local caches only.

    Prefers the commit-pinned community model in the HuggingFace cache; falls
    back to the first ``*.pth`` (and matching ``*.index``) under the fixtures
    directory. Never downloads -- a missing model is a setup error, not a fetch.
    """
    try:
        from huggingface_hub import hf_hub_download

        ckpt = hf_hub_download(_REPO, _CKPT_FILENAME, local_files_only=True)
        return ckpt, None
    except Exception as exc:
        print(f"  (HF cache miss for {_REPO}: {exc}; trying fixtures/)")

    if _FIXTURE_DIR.is_dir():
        pths = sorted(_FIXTURE_DIR.glob("*.pth"))
        if pths:
            idx = sorted(_FIXTURE_DIR.glob("*.index"))
            return str(pths[0]), (str(idx[0]) if idx else None)

    raise SystemExit(
        f"No checkpoint: {_REPO}/{_CKPT_FILENAME} is not in the HF cache and no "
        f"*.pth is under {_FIXTURE_DIR}. Cache the model first (see the module "
        "docstring in tests/test_rvc_engine.py)."
    )


def synth_voiced(seconds: float, sr: int = _SR_16K) -> np.ndarray:
    """Synthesize a voiced tone (fundamental + harmonics + light vibrato).

    RMVPE only emits f0 on voiced frames, so the bench must feed real voiced
    content -- silence/noise would exercise nothing. Mirrors the generator in
    tests/test_rvc_engine.py so the bench signal matches the smoke test's.
    """
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    f0 = 150.0 * (2.0 ** (0.05 * np.sin(2 * np.pi * 5.0 * t)))  # ~150 Hz + vibrato
    phase = 2 * np.pi * np.cumsum(f0) / sr
    sig = np.zeros_like(t)
    for harmonic, amp in ((1, 1.0), (2, 0.5), (3, 0.25), (4, 0.125)):
        sig += amp * np.sin(harmonic * phase)
    sig = sig.astype(np.float32)
    sig /= np.abs(sig).max() + 1e-9
    return (sig * 0.8).astype(np.float32)


@contextlib.contextmanager
def force_device(device: str):
    """Force ``RVCPipeline.load`` onto ``device`` by swapping its device probe.

    ``load`` picks the device via ``get_torch_device`` imported into the pipeline
    module; overriding that symbol for the duration of the load is how the bench
    exercises CPU and MPS from one process without touching pipeline.py.
    """
    original = _pipeline_mod.get_torch_device
    _pipeline_mod.get_torch_device = lambda **_kw: device
    try:
        yield
    finally:
        _pipeline_mod.get_torch_device = original


def bench_one(
    pipeline: RVCPipeline,
    audio_16k: np.ndarray,
    *,
    block_ms: int,
    warmup: int,
    f0_up_key: int,
    index_rate: float,
) -> dict | None:
    """Run one (device, block_ms) sweep; return per-block timing stats.

    Feeds ``audio_16k`` through a freshly-constructed session in exact
    ``block_frame_16k`` slices, collecting the session's own ``infer_ms`` per
    block. Returns ``None`` if the clip is too short for a post-warmup sample.
    """
    session = RVCStreamSession(
        pipeline,
        block_ms=block_ms,
        extra_ms=DEFAULT_EXTRA_MS,
        crossfade_ms=DEFAULT_CROSSFADE_MS,
        sola_search_ms=DEFAULT_SOLA_SEARCH_MS,
        f0_up_key=f0_up_key,
        f0_method="rmvpe",
        index_rate=index_rate,
    )
    nb = session.block_frame_16k
    n_blocks = audio_16k.shape[0] // nb
    if n_blocks <= warmup:
        return None

    infer_ms: list[float] = []
    overloads = 0
    for i in range(n_blocks):
        block = audio_16k[i * nb : (i + 1) * nb]
        _out, ms, overload = session.process(block)
        infer_ms.append(ms)
        if overload:
            overloads += 1
    session.close()

    cold = infer_ms[:warmup]
    steady = np.asarray(infer_ms[warmup:], dtype=np.float64)
    block_ms_actual = nb / _SR_16K * 1000.0
    return {
        "block_ms": block_ms,
        "block_ms_actual": block_ms_actual,
        "window_16k": session._window_16k,
        "window_ms": session._window_16k / _SR_16K * 1000.0,
        "n_blocks": n_blocks,
        "warmup": warmup,
        "steady_n": steady.size,
        "cold_max_ms": max(cold) if cold else float("nan"),
        "p50": float(np.percentile(steady, 50)),
        "p95": float(np.percentile(steady, 95)),
        "min": float(steady.min()),
        "max": float(steady.max()),
        "mean": float(steady.mean()),
        "overloads": overloads,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", default=None, help="Input WAV (default: synth ~voiced clip)")
    ap.add_argument("--seconds", type=float, default=15.0, help="Synth clip length")
    ap.add_argument("--block-ms", default="300", help="Comma list, e.g. 250,300,350")
    ap.add_argument("--devices", default=None, help="Comma list; default: mps (if avail),cpu")
    ap.add_argument("--warmup", type=int, default=3, help="Leading blocks excluded from stats")
    ap.add_argument("--f0-up-key", type=int, default=0)
    ap.add_argument("--index-rate", type=float, default=0.0)
    args = ap.parse_args()

    block_sizes = [int(x) for x in args.block_ms.split(",") if x.strip()]

    if args.devices:
        devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    else:
        import torch

        devices = []
        if torch.cuda.is_available():
            devices.append("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            devices.append("mps")
        devices.append("cpu")

    ckpt, index = resolve_checkpoint()
    print(f"Checkpoint: {ckpt}")
    print(f"Index:      {index or '(none — running index-free)'}")

    if args.wav:
        audio_16k, _ = load_audio(args.wav, sample_rate=_SR_16K)
        audio_16k = np.ascontiguousarray(audio_16k, dtype=np.float32)
        # Match the offline pipeline's pre-normalisation so levels are realistic.
        peak = np.abs(audio_16k).max() / 0.95
        if peak > 1:
            audio_16k = audio_16k / peak
        print(f"Input:      {args.wav} ({audio_16k.shape[0] / _SR_16K:.1f}s @ 16k mono)")
    else:
        audio_16k = synth_voiced(args.seconds)
        print(f"Input:      synth voiced tone ({args.seconds:.1f}s @ 16k mono)")

    print(f"Devices:    {devices}   block_ms: {block_sizes}   warmup: {args.warmup}")
    print()

    rows: list[dict] = []
    for device in devices:
        print(f"[{device}] loading checkpoint...", flush=True)
        pipeline = RVCPipeline()
        with force_device(device):
            t0 = time.perf_counter()
            pipeline.load(ckpt, index)
            load_ms = (time.perf_counter() - t0) * 1000.0
        assert pipeline._device == device, f"expected {device}, got {pipeline._device}"
        print(
            f"[{device}] loaded on {pipeline._device} "
            f"(sr={pipeline._tgt_sr}, if_f0={pipeline._if_f0}, ver={pipeline._version}, "
            f"is_half={pipeline._is_half}) in {load_ms:.0f} ms",
            flush=True,
        )
        try:
            for block_ms in block_sizes:
                stats = bench_one(
                    pipeline,
                    audio_16k,
                    block_ms=block_ms,
                    warmup=args.warmup,
                    f0_up_key=args.f0_up_key,
                    index_rate=args.index_rate if index else 0.0,
                )
                if stats is None:
                    print(f"[{device}] block_ms={block_ms}: clip too short, skipped")
                    continue
                stats["device"] = device
                rows.append(stats)
                print(
                    f"[{device}] block_ms={block_ms}: "
                    f"p50={stats['p50']:.0f} p95={stats['p95']:.0f} "
                    f"(steady n={stats['steady_n']}, cold={stats['cold_max_ms']:.0f})",
                    flush=True,
                )
        finally:
            pipeline.unload()

    print()
    print("=" * 92)
    print("RVC streaming latency — per-block infer_ms (conversion + SOLA stitch)")
    print("=" * 92)
    hdr = (
        f"{'device':<7} {'block_ms':>8} {'window_ms':>9} {'blocks':>6} "
        f"{'p50':>7} {'p95':>7} {'max':>7} {'cold':>7} {'p95<blk':>8} {'overld':>6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        realtime = "YES" if r["p95"] < r["block_ms_actual"] else "NO"
        print(
            f"{r['device']:<7} {r['block_ms']:>8} {r['window_ms']:>9.0f} "
            f"{r['steady_n']:>6} {r['p50']:>7.0f} {r['p95']:>7.0f} {r['max']:>7.0f} "
            f"{r['cold_max_ms']:>7.0f} {realtime:>8} {r['overloads']:>6}"
        )
    print("-" * len(hdr))
    print("p50/p95/max/cold in ms. 'p95<blk' = p95 infer_ms below the block duration")
    print("(the real-time keep-up condition). 'overld' = blocks the session flagged")
    print("as overloaded (infer_ms > block_ms), warmup included.")

    # Verdict on the best available device (first in the probed list).
    best = devices[0]
    best_rows = [r for r in rows if r["device"] == best]
    if best_rows:
        print()
        ok = [r for r in best_rows if r["p95"] < r["block_ms_actual"]]
        if ok:
            print(
                f"Best device '{best}': real-time at block_ms "
                f"{sorted(r['block_ms'] for r in ok)} (p95 < block_ms)."
            )
        else:
            floor = min(r["p95"] for r in best_rows)
            print(
                f"Best device '{best}': p95 infer_ms does NOT beat block_ms at any tested "
                f"size — measured floor p95≈{floor:.0f} ms. No CUDA on this machine; per 05 "
                "this is allowed with the CPU/MPS warning path shown in the UI."
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
