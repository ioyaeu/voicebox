"""RVC voice-conversion endpoints: offline file conversion and realtime streaming.

``POST /convert`` enqueues an offline file conversion on the shared generation
queue (see :mod:`backend.services.convert`) and returns a task id.

``WS /convert/stream`` is the realtime, low-latency path (step 05). It is a
**single-session** endpoint — only one live conversion stream may be connected
at a time, because this is a local single-user app and the process holds one RVC
model. A second concurrent connection is accepted and then immediately closed
with WebSocket close code ``1008`` (policy violation) plus a reason string: a
WebSocket cannot return an HTTP 409 after the upgrade, so ``1008`` is the
post-upgrade equivalent.

This module docstring is the **single authoritative definition** of the wire
protocol — the handshake schemas and the binary audio-frame header layout. The
frontend client (``app/src/lib/hooks/useRvcStream.ts``) documents the frame by
pointing here rather than re-describing it; keep this the one place the byte
layout is spelled out so the two sides cannot drift.

Wire protocol
-------------
Protocol version: :data:`~backend.models.STREAM_PROTOCOL_VERSION` (currently 1).
The client sends ``version`` in the handshake; the server rejects any major
version it does not implement (handshake error + ``1008`` close) *before*
loading a model. An omitted ``version`` is treated as the current version.

1. **Handshake** (JSON text frames):
   - client -> server: :class:`~backend.models.StreamHandshakeRequest`
     ``{version, profile_id, block_ms, f0_up_key, f0_method, index_rate,
     rms_mix_rate, protect}``.
   - server -> client on success: :class:`~backend.models.StreamHandshakeReply`
     ``{ready: true, version, model_sr, block_frame_16k, block_frame_sr}``. The
     client MUST send exactly ``block_frame_16k`` float32 samples per input frame.
   - server -> client on rejection:
     :class:`~backend.models.StreamHandshakeError` ``{ready: false, error}``
     followed by a ``1008`` close.

2. **Audio** (binary frames), after a successful handshake:
   - client -> server: one binary frame per block = exactly ``block_frame_16k``
     **little-endian float32** samples of mono 16 kHz PCM
     (``block_frame_16k * 4`` bytes). No JSON, no base64.
   - server -> client: one binary frame per converted block = an **8-byte
     little-endian header** followed by ``block_frame_sr`` little-endian float32
     samples of mono PCM at ``model_sr``. The header is ``struct '<fBBH'``
     (this is the canonical layout — 8 bytes total):

       * offset 0, ``float32 infer_ms`` — wall-clock ms this block took;
       * offset 4, ``uint8  overload``  — 1 when the stream cannot keep up
         (inference slower than one block, or input was dropped for backpressure);
       * offset 5, ``uint8  dropped``   — 1 when >= 1 pending input block was
         dropped for backpressure since the previous output frame;
       * offset 6, ``uint16 spare``     — reserved, currently always 0.

     The 8-byte header keeps the PCM payload 4-byte aligned, so the client can
     wrap it directly in a ``Float32Array``.

3. **Mid-stream error** (JSON text frame): if conversion fails after the
   handshake, the server sends :class:`~backend.models.StreamErrorFrame`
   ``{type: "error", error}`` and then closes (``1011``), so the client learns
   the reason instead of seeing a bare disconnect.

Backpressure: inbound blocks are buffered in a bounded queue
(``_MAX_PENDING_INPUT_BLOCKS``). If the client outruns inference the **oldest**
pending block is dropped (the queue never grows unboundedly) and the next output
frame carries ``overload``/``dropped`` = 1.

Threading: :meth:`RVCStreamSession.process` is a blocking torch call and is
always run via ``asyncio.to_thread`` so the event loop — and the WebSocket —
never block. Inbound receive and outbound send run as two cooperating tasks, so
the receive side keeps draining (and dropping) while a block is being converted.

Teardown: on disconnect the per-session buffers are freed
(``session.close()``); the shared RVC model **stays loaded** so a reconnecting
stream (or an offline conversion) reuses it.
"""

import asyncio
import contextlib
import json
import logging
import struct
from pathlib import Path

import numpy as np
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import ValidationError
from sqlalchemy.orm import Session
from starlette.datastructures import UploadFile as StarletteUploadFile

from .. import config, models
from ..database import VoiceProfile as DBVoiceProfile, get_db
from ..services import convert

logger = logging.getLogger(__name__)

router = APIRouter()

CONVERT_SOURCE_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".aac", ".webm", ".opus"}
CONVERT_SOURCE_MAX_BYTES = 200 * 1024 * 1024  # 200 MB
# Coarse upper bound on the whole multipart request body, checked against the
# Content-Length header before any bytes are read (so a 5 GB body is rejected at
# header time instead of streamed). Headroom above the per-file cap covers the
# multipart boundaries and small form fields; the exact per-file cap is still
# enforced chunk-by-chunk while streaming.
CONVERT_REQUEST_MAX_BYTES = CONVERT_SOURCE_MAX_BYTES + 16 * 1024 * 1024  # 216 MB

# Max input blocks buffered before drop-oldest backpressure kicks in. Small so
# the added mouth-to-ear latency stays bounded (~this many blocks) when the
# client briefly outruns inference; beyond it we drop rather than grow the queue.
_MAX_PENDING_INPUT_BLOCKS = 4

# Upper bound on how long the server waits for the handshake JSON frame after the
# WebSocket upgrade. A half-open socket that never sends one would otherwise keep
# ``_stream_active`` set forever and lock out every future stream (permanent
# 1008), so the wait is bounded: on timeout we reject cleanly and free the guard.
_HANDSHAKE_TIMEOUT_S = 10.0

# Per-output-frame header prepended to the converted PCM. The module docstring
# is the authoritative layout: float32 infer_ms | uint8 overload | uint8 dropped
# | uint16 spare == 8 bytes, little-endian, keeping the trailing PCM 4-byte
# aligned.
_STREAM_OUT_HEADER = struct.Struct("<fBBH")

# Single-session guard, read/written only on the event loop (one stream at a
# time), so no lock is needed. Which checkpoint is resident is tracked inside the
# pipeline (identity no-op), not here — a reconnecting stream just calls load()
# and the pipeline skips the reload when the model is unchanged.
_stream_active = False


@router.post("/convert", response_model=models.ConvertResponse)
async def convert_audio(request: Request, db: Session = Depends(get_db)):
    """Convert an uploaded audio clip to an RVC profile's voice.

    The conversion is enqueued on the shared generation queue and a task id is
    returned immediately; status is polled via ``GET /generate/{id}/status``
    (or ``GET /history/{id}``) and the result WAV served by ``GET /audio/{id}``,
    mirroring the TTS generation flow.

    This endpoint takes the raw ``Request`` (rather than ``File``/``Form``
    parameters) on purpose: FastAPI would otherwise parse — and spool — the
    entire multipart body *before* the function runs, defeating a header-time
    size guard. Instead the checks run in the cheapest-first order:

    1. **Content-Length upper bound** — a wildly oversized body (e.g. 5 GB) is
       rejected straight from the header, before the multipart parser reads a
       single byte.
    2. **Parameter + profile validation** — the profile must be a
       conversion-ready RVC profile *before* the upload is streamed to our own
       temp file and decoded, so a non-rvc / model-less profile never costs a
       source copy or an enqueued job. (``profile_id`` is a multipart field, so
       the form must be parsed to read it; that parse is gated behind step 1.)
    """
    # 1. Header-time size guard.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            declared_bytes = int(content_length)
        except ValueError:
            declared_bytes = None
        if declared_bytes is not None and declared_bytes > CONVERT_REQUEST_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Request body too large (max {CONVERT_REQUEST_MAX_BYTES // (1024 * 1024)} MB)",
            )

    form = await request.form()
    upload = form.get("file")
    if not isinstance(upload, StarletteUploadFile):
        raise HTTPException(status_code=422, detail="Missing 'file' upload")

    try:
        params = models.ConvertRequest(
            profile_id=str(form.get("profile_id") or ""),
            f0_up_key=form.get("f0_up_key", 0),
            f0_method=form.get("f0_method", "rmvpe"),
            index_rate=form.get("index_rate", 0.75),
            rms_mix_rate=form.get("rms_mix_rate", 0.25),
            protect=form.get("protect", 0.33),
        )
    except ValidationError as e:
        detail = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
        )
        raise HTTPException(status_code=400, detail=detail or "Invalid conversion parameters")

    # 2. Profile must be a conversion-ready rvc profile before we stream the
    #    upload to our conversion temp (and before start_conversion decodes it).
    profile_row = db.query(DBVoiceProfile).filter_by(id=params.profile_id).first()
    if not profile_row:
        raise HTTPException(status_code=404, detail="Profile not found")
    try:
        convert.resolve_conversion_target(profile_row, params.f0_method)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    uploaded_ext = Path(upload.filename or "").suffix.lower()
    source_ext = uploaded_ext if uploaded_ext in CONVERT_SOURCE_EXTENSIONS else ".wav"

    # Single shared chunked-upload-to-temp helper (see routes/profiles.py), so
    # the byte-capped streaming loop is not copy-pasted per endpoint. Imported
    # lazily: routes/profiles.py imports backend.app, and app.py builds the app
    # at import time, so a module-top import here would form a startup cycle.
    from .profiles import _stream_upload_to_temp

    # Temp lives in the generations dir (same filesystem) so start_conversion can
    # commit it with an atomic rename.
    tmp_path = await _stream_upload_to_temp(
        upload, config.get_generations_dir(), source_ext, CONVERT_SOURCE_MAX_BYTES
    )
    try:
        return await convert.start_conversion(
            params=params,
            source_tmp_path=tmp_path,
            source_ext=source_ext,
            source_filename=upload.filename or "source",
            db=db,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# ── Realtime streaming (WS /convert/stream) ────────────────────────────────


async def _resolve_stream_target(handshake: models.StreamHandshakeRequest) -> tuple[str, str | None]:
    """Validate the handshake's profile and return ``(model_path, index_path)``.

    Reads the RVC storage paths from the ORM row directly — they are internal and
    no longer exposed on ``VoiceProfileResponse``. Opens and closes its own
    short-lived DB session so the stream itself never holds a connection open.
    Raises ``ValueError`` with a user-facing message on any validation failure;
    the caller turns that into a handshake error frame. The crepe check is the
    shared :func:`backend.services.convert.ensure_crepe_available`.
    """
    db = next(get_db())
    try:
        profile = db.query(DBVoiceProfile).filter_by(id=handshake.profile_id).first()
        if not profile:
            raise ValueError("Profile not found")
        if (getattr(profile, "voice_type", None) or "cloned") != "rvc":
            raise ValueError("Real-time conversion is only supported for RVC voice profiles")

        model_stored = getattr(profile, "rvc_model_path", None)
        if not model_stored:
            raise ValueError("This RVC profile has no uploaded model. Upload a .pth checkpoint first.")
        model_path = config.resolve_storage_path(model_stored)
        if model_path is None or not model_path.exists():
            raise ValueError("The profile's RVC model file is missing on disk.")

        # crepe pitch needs an on-demand model the frozen build does not bundle;
        # fail at handshake so a missing model is a clean rejection, not a
        # mid-stream crash.
        convert.ensure_crepe_available(handshake.f0_method)

        index_path = config.resolve_storage_path(getattr(profile, "rvc_index_path", None))
        index_path_str = str(index_path) if index_path is not None and index_path.exists() else None
        return str(model_path), index_path_str
    finally:
        db.close()


async def _reject_handshake(websocket: WebSocket, message: str) -> None:
    """Send an error frame then close with a policy-violation code (best effort)."""
    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
        await websocket.send_json(models.StreamHandshakeError(error=message).model_dump())
    with contextlib.suppress(WebSocketDisconnect, RuntimeError):
        await websocket.close(code=1008, reason="handshake rejected")


async def _run_stream(websocket: WebSocket, session) -> None:
    """Pump binary PCM frames through ``session`` until the client disconnects.

    The receive loop below drains inbound frames into a bounded queue, dropping
    the oldest under backpressure so the socket never stalls. A single child
    task, ``consume``, pulls from the queue, runs inference off the event loop,
    and sends converted blocks back. The loop waits on the receive **and** the
    consumer at once, so a consumer that dies (e.g. an inference error) is
    noticed even while the client is momentarily quiet — the client is then told
    *why* via a :class:`~backend.models.StreamErrorFrame` before a clean close,
    rather than seeing a bare disconnect. Returns on client disconnect or after
    surfacing a consumer failure.
    """
    queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_MAX_PENDING_INPUT_BLOCKS)
    state = {"dropped": 0}
    expected_bytes = session.block_frame_16k * 4

    async def consume() -> None:
        while True:
            data = await queue.get()
            if len(data) != expected_bytes:
                logger.warning(
                    "convert stream: bad frame size %d bytes (expected %d); closing",
                    len(data),
                    expected_bytes,
                )
                with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                    await websocket.close(code=1003, reason="Unexpected PCM frame size")
                return
            block = np.frombuffer(data, dtype="<f4")
            converted, infer_ms, overload = await asyncio.to_thread(session.process, block)

            dropped = state["dropped"]
            state["dropped"] = 0
            header = _STREAM_OUT_HEADER.pack(
                float(infer_ms),
                1 if (overload or dropped) else 0,
                1 if dropped else 0,
                0,
            )
            pcm = np.ascontiguousarray(converted, dtype="<f4").tobytes()
            try:
                await websocket.send_bytes(header + pcm)
            except (WebSocketDisconnect, RuntimeError):
                return

    consumer = asyncio.create_task(consume())
    receiver: asyncio.Task | None = None
    try:
        while not consumer.done():
            if receiver is None:
                receiver = asyncio.create_task(websocket.receive())
            # Wake on whichever finishes first: the next inbound frame or the
            # consumer ending (so a dead consumer is not missed while we block
            # waiting for a client that has gone quiet).
            await asyncio.wait({receiver, consumer}, return_when=asyncio.FIRST_COMPLETED)
            if not receiver.done():
                break  # only the consumer finished; surface its outcome below
            message = receiver.result()
            receiver = None
            if message["type"] == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is None:
                # Text/control frames are not part of the audio protocol; ignore.
                continue
            while True:
                try:
                    queue.put_nowait(data)
                    break
                except asyncio.QueueFull:
                    try:
                        queue.get_nowait()  # drop oldest, keep the queue bounded
                        state["dropped"] += 1
                    except asyncio.QueueEmpty:
                        break
    finally:
        if receiver is not None and not receiver.done():
            receiver.cancel()
        if not consumer.done():
            consumer.cancel()
        # Retrieve both tasks' outcomes here without propagating (avoids "exception
        # never retrieved" warnings); the consumer's failure, if any, is surfaced
        # to the client below instead.
        pending = [t for t in (receiver, consumer) if t is not None]
        await asyncio.gather(*pending, return_exceptions=True)

    # A genuine consumer failure (torch error, etc.) reaches the client as a JSON
    # error frame before a clean close, so it learns why. A clean return or a
    # teardown cancellation is not an error.
    if not consumer.cancelled():
        exc = consumer.exception()
        if exc is not None and not isinstance(exc, WebSocketDisconnect):
            reason = f"Conversion failed: {exc}"
            logger.warning("convert stream: consumer failed: %s", exc, exc_info=exc)
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await websocket.send_json(models.StreamErrorFrame(error=reason).model_dump())
            with contextlib.suppress(WebSocketDisconnect, RuntimeError):
                await websocket.close(code=1011, reason="conversion failed")


@router.websocket("/convert/stream")
async def convert_stream(websocket: WebSocket) -> None:
    """Realtime mic -> converted-voice WebSocket. See the module docstring."""
    global _stream_active

    # Single-session guard: this check-and-set runs with no intervening await, so
    # it is atomic on the single-threaded event loop.
    if _stream_active:
        await websocket.accept()
        await websocket.close(code=1008, reason="A voice-conversion stream is already active")
        logger.info("convert stream: rejected second concurrent connection")
        return
    _stream_active = True

    lease_owner = f"stream:{id(websocket)}"
    lease_held = False
    session = None
    try:
        await websocket.accept()

        try:
            raw = await asyncio.wait_for(
                websocket.receive_json(), timeout=_HANDSHAKE_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            # Half-open socket: never send it, and the single-session guard (reset
            # in the finally below) would stay set forever, locking out all future
            # streams. Reject cleanly so the slot frees for the next connection.
            logger.info(
                "convert stream: handshake timed out after %.0fs; closing",
                _HANDSHAKE_TIMEOUT_S,
            )
            await _reject_handshake(websocket, "Handshake timed out; no handshake frame received")
            return
        except json.JSONDecodeError:
            await _reject_handshake(websocket, "Handshake must be a valid JSON object")
            return
        if not isinstance(raw, dict):
            await _reject_handshake(websocket, "Handshake must be a JSON object")
            return
        try:
            handshake = models.StreamHandshakeRequest(**raw)
        except ValidationError as e:
            detail = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in e.errors()
            )
            await _reject_handshake(websocket, detail or "Invalid handshake parameters")
            return

        # Reject an unrecognised protocol major version before loading anything.
        if handshake.version != models.STREAM_PROTOCOL_VERSION:
            await _reject_handshake(
                websocket,
                f"Unsupported protocol version {handshake.version}; this server "
                f"speaks version {models.STREAM_PROTOCOL_VERSION}. Update the client.",
            )
            return

        try:
            model_path, index_path = await _resolve_stream_target(handshake)
        except ValueError as e:
            await _reject_handshake(websocket, str(e))
            return

        from ..backends.rvc import (  # lazy: heavy import (torch)
            acquire as rvc_acquire,
            get_rvc_engine,
            lease_holder,
            release as rvc_release,
        )
        from ..backends.rvc.streaming import RVCStreamSession  # lazy: heavy import

        # Hold the exclusive engine lease for the whole session so an offline
        # conversion or chained generation cannot swap the model mid-stream. If a
        # job already holds it, reject like a second stream (1008 post-upgrade).
        if not rvc_acquire(lease_owner):
            logger.info("convert stream: rejected — engine busy (%s)", lease_holder())
            await _reject_handshake(
                websocket,
                "A voice conversion job is running — wait for it to finish, then reconnect.",
            )
            return
        lease_held = True

        engine = get_rvc_engine()
        try:
            # The pipeline no-ops the reload when this model is already resident.
            await asyncio.to_thread(engine.load, model_path, index_path)
            session = RVCStreamSession(
                engine,
                block_ms=handshake.block_ms,
                f0_up_key=handshake.f0_up_key,
                f0_method=handshake.f0_method,
                index_rate=handshake.index_rate,
                rms_mix_rate=handshake.rms_mix_rate,
                protect=handshake.protect,
            )
        except (ValueError, RuntimeError, OSError) as e:
            logger.warning("convert stream: failed to load RVC model: %s", e)
            await _reject_handshake(websocket, f"Failed to load RVC model: {e}")
            return

        await websocket.send_json(
            models.StreamHandshakeReply(
                model_sr=session.model_sr,
                block_frame_16k=session.block_frame_16k,
                block_frame_sr=session.block_frame_sr,
            ).model_dump()
        )
        logger.info(
            "convert stream: session started (profile=%s, block_ms=%d, model_sr=%d)",
            handshake.profile_id,
            handshake.block_ms,
            session.model_sr,
        )

        await _run_stream(websocket, session)
        logger.info("convert stream: session ended (client disconnected)")
    except WebSocketDisconnect:
        logger.info("convert stream: client disconnected during handshake")
    finally:
        # Reset the single-session guard first, independent of close(): an
        # abandoned inference thread may still hold the session lock, and close()
        # blocks on it (up to one window). Run close off the event loop so the
        # loop is not frozen, and only release the engine lease afterwards — while
        # that thread may still touch the engine, no job may swap the model.
        _stream_active = False
        try:
            if session is not None:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(session.close)
        finally:
            if lease_held:
                rvc_release(lease_owner)
