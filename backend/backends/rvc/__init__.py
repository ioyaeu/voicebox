"""RVC voice-conversion backend package.

Step 01 shipped checkpoint validation (``checkpoint.py``). Step 02 adds the
vendored synthesizer, pitch and feature extractors, the offline pipeline, and
the ``get_rvc_engine()`` factory exported here.
"""

import threading
from typing import Optional

from .pipeline import RVCPipeline

_rvc_engine: Optional[RVCPipeline] = None
_rvc_engine_lock = threading.Lock()


def get_rvc_engine() -> RVCPipeline:
    """Return the process-wide RVC pipeline (one loaded model at a time).

    Mirrors the lazily-created, lock-guarded singleton factories in
    ``backends/__init__.py`` and the engine lifecycle in ``services/tts.py``:
    the returned :class:`RVCPipeline` owns load/unload of the checkpoint, and
    its ``unload`` frees the model plus shared backbones and calls
    ``empty_device_cache``.
    """
    global _rvc_engine
    # Fast path: no lock once created.
    if _rvc_engine is not None:
        return _rvc_engine
    with _rvc_engine_lock:
        if _rvc_engine is None:
            _rvc_engine = RVCPipeline()
        return _rvc_engine


# ── Exclusive engine lease ──────────────────────────────────────────────────
# The single shared RVC engine has three would-be users (the WS stream, offline
# /convert jobs, and the TTS->RVC chain). Only one may drive it at a time or an
# offline job swaps the synthesizer/sample-rate under a live stream. The lease is
# a plain holder string behind a lock: the stream holds it for its session; jobs
# try-acquire per job and fail fast on contention. No queueing/preemption — this
# is a single-user local app, so honest fail-fast beats a scheduler.
_lease_lock = threading.Lock()
_lease_holder: Optional[str] = None


def acquire(owner: str) -> bool:
    """Take the exclusive engine lease for ``owner``.

    Returns ``True`` if acquired (or already held by the same ``owner``),
    ``False`` if a different owner holds it.
    """
    global _lease_holder
    with _lease_lock:
        if _lease_holder is None or _lease_holder == owner:
            _lease_holder = owner
            return True
        return False


def release(owner: str) -> None:
    """Release the lease if ``owner`` holds it; a no-op otherwise."""
    global _lease_holder
    with _lease_lock:
        if _lease_holder == owner:
            _lease_holder = None


def lease_holder() -> Optional[str]:
    """Return the current lease holder, or ``None`` if the lease is free."""
    with _lease_lock:
        return _lease_holder


def realtime_stream_active() -> bool:
    """True when the realtime WebSocket stream currently owns the RVC engine."""
    holder = lease_holder()
    return bool(holder and holder.startswith("stream:"))


__all__ = [
    "get_rvc_engine",
    "RVCPipeline",
    "acquire",
    "release",
    "lease_holder",
    "realtime_stream_active",
]
