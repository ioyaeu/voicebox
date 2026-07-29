"""Real-time RVC streaming tests: SOLA/windowing math and the WS wire contract.

These are **CI-safe**: no RVC checkpoint, no ContentVec/RMVPE backbones, no GPU
and no torch inference are required. The one piece a real stream needs from a
model -- turning a 16 kHz input window into model-rate audio -- is replaced by an
*identity* transform so the parts this step actually adds can be exercised in
isolation:

* the sliding-context ring buffer, block/crossfade/search region arithmetic, and
  the SOLA correlation-search + equal-power crossfade in
  :class:`~backend.backends.rvc.streaming.RVCStreamSession`;
* the ``WS /convert/stream`` handshake, binary framing, single-session guard and
  disconnect teardown in :mod:`backend.routes.convert`.

The identity substitution is the exact seam the production code documents for
this purpose: ``RVCStreamSession._convert_window`` is called out as "Overridden
in the synthetic SOLA test with an identity transform" and the constructor
accepts ``pipeline=None`` together with an explicit ``model_sr`` for precisely
this test. The real ``process`` / ``_stitch`` / ``_sola_offset`` code runs
unchanged; only the neural net is stubbed.

Why synthetic signals rather than "listen to a clip": audio chunk-boundary bugs
are silent -- a naive chunker that converts blocks independently can sound fine
on one sample and click on the next. A continuous sine has a known,
bounded sample-to-sample slope, so a discontinuity at a block join shows up as a
delta far above that slope. The negative control
(:func:`test_click_metric_discriminates_sola_from_naive_chunking`) demonstrates
the metric flags exactly the failure mode this step forbids (independent-block
conversion without SOLA), so a green continuity assertion is meaningful and not
merely a loose bound.
"""

import io
import os
import struct
import time
import uuid

import numpy as np
import pytest
import soundfile as sf
import torch
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import config
from backend.backends.rvc import (
    acquire as rvc_acquire,
    lease_holder,
    release as rvc_release,
)
from backend.backends.rvc.streaming import DEFAULT_BLOCK_MS, RVCStreamSession, _SR_16K
from backend.routes import convert as convert_route


def _wait_lease_free(timeout: float = 5.0) -> None:
    """Wait for an async engine-lease release to land.

    The WS route releases the lease in a ``finally`` that runs on the app's event
    loop *after* the client's ``websocket_connect`` context exits, so a following
    test can otherwise start while a prior session's release is still in flight.
    """
    deadline = time.time() + timeout
    while lease_holder() is not None and time.time() < deadline:
        time.sleep(0.02)

# A continuous sine has max |x[n+1]-x[n]| ~= A * 2*pi*f / sr. A click-free stitch
# only adds the equal-power crossfade's in-phase boost (up to sqrt(2)), so its
# max delta stays ~1.4x that slope; a phase-discontinuity click is an order of
# magnitude larger (~11x, see the negative-control test). 3x sits cleanly
# between the two -- loose enough to absorb the sqrt(2) boost and SOLA's
# integer-sample jitter, tight enough that any real click fails it.
MAX_DELTA_FACTOR = 3.0

# Per-output-frame header the WS route prepends to converted PCM
# (backend/routes/convert.py: struct '<fBBH' == infer_ms|overload|dropped|resv).
_STREAM_OUT_HEADER = struct.Struct("<fBBH")


def _pure_sine_step(amp: float, freq: float, sr: int) -> float:
    """Largest sample-to-sample delta of a continuous sine at these params."""
    return amp * 2.0 * np.pi * freq / sr


class _IdentityStreamSession(RVCStreamSession):
    """Real session windowing/SOLA with the neural conversion replaced by identity.

    Overrides only :meth:`_convert_window` (the documented test seam). At
    ``model_sr == 16 kHz`` the window is returned untouched, so the emitted
    stream is the SOLA-stitched original signal and any block-join discontinuity
    is the session's own doing. At a higher ``model_sr`` the window is length-
    resampled (linear) so the input->output ratio arithmetic is exercised for
    real, without pulling in a model.
    """

    def _convert_window(self, window_16k: np.ndarray) -> np.ndarray:
        w = np.ascontiguousarray(window_16k, dtype=np.float32).reshape(-1)
        if self.model_sr == _SR_16K:
            return w.copy()
        n_out = round(w.shape[0] * self.model_sr / _SR_16K)
        x_old = np.linspace(0.0, 1.0, w.shape[0], dtype=np.float64)
        x_new = np.linspace(0.0, 1.0, n_out, dtype=np.float64)
        return np.interp(x_new, x_old, w).astype(np.float32)


class _SlowIdentityStreamSession(_IdentityStreamSession):
    """Identity session whose conversion sleeps, to force the underrun path."""

    def __init__(self, *args, sleep_s: float = 0.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._sleep_s = float(sleep_s)

    def _convert_window(self, window_16k: np.ndarray) -> np.ndarray:
        if self._sleep_s:
            time.sleep(self._sleep_s)
        return super()._convert_window(window_16k)


def _run_sine(session: RVCStreamSession, freq: float, amp: float, n_blocks: int):
    """Feed ``n_blocks`` of one continuous sine; return (outputs_list, infer_ms_list).

    The sine is indexed by a single global sample counter across blocks, so
    successive input blocks are the genuine continuation of one another (a real
    mic stream), not independent segments. Starting phase is 0 (sin(0)=0) so the
    fill from the initially-silent ring buffer is itself continuous.
    """
    nb = session.block_frame_16k
    outs: list[np.ndarray] = []
    infer_ms: list[float] = []
    overloads: list[bool] = []
    for k in range(n_blocks):
        n = np.arange(k * nb, (k + 1) * nb)
        block = (amp * np.sin(2.0 * np.pi * freq * n / _SR_16K)).astype(np.float32)
        out, ms, overload = session.process(block)
        outs.append(out)
        infer_ms.append(ms)
        overloads.append(overload)
    return outs, infer_ms, overloads


# ── 1. SOLA / windowing on synthetic signals (no model) ─────────────────────


@pytest.mark.parametrize("freq", [311.13, 440.0, 523.25])
def test_sola_stitch_is_click_free_and_correct_length(freq):
    """A continuous sine, identity-converted, stitches back click-free.

    Asserts three things the step's acceptance criteria call out: every emitted
    block is exactly ``block_frame_sr`` long (and the concatenation is
    ``n_blocks * block_frame_sr``), the whole stitched output has no
    sample-to-sample discontinuity above the click threshold, and the
    equal-power crossfade does not push the amplitude past its sqrt(2) ceiling
    (i.e. no runaway / clipping at seams).
    """
    amp = 0.2
    n_blocks = 64
    session = _IdentityStreamSession(None, model_sr=_SR_16K, block_ms=DEFAULT_BLOCK_MS)

    outs, _, _ = _run_sine(session, freq, amp, n_blocks)

    # correct output length, every block and in total
    assert all(o.shape[0] == session.block_frame_sr for o in outs)
    stitched = np.concatenate(outs)
    assert stitched.shape[0] == n_blocks * session.block_frame_sr

    # no discontinuity at any join (measured over the ENTIRE output, warmup
    # included -- the sine starts at 0 so the ring-buffer fill is smooth too)
    ref_step = _pure_sine_step(amp, freq, _SR_16K)
    max_delta = float(np.abs(np.diff(stitched)).max())
    ratio = max_delta / ref_step
    assert ratio < MAX_DELTA_FACTOR, (
        f"click at a block join: max |delta|={max_delta:.5f} is {ratio:.2f}x the "
        f"pure-sine step {ref_step:.5f} (threshold {MAX_DELTA_FACTOR}x)"
    )

    # equal-power crossfade of in-phase sines peaks at A*sqrt(2); never clips
    peak = float(np.abs(stitched).max())
    assert peak <= amp * np.sqrt(2.0) * 1.05, f"amplitude runaway at seam: {peak:.4f}"
    assert peak < 1.0


def test_stable_across_many_consecutive_blocks():
    """Stitching stays click-free and memory-bounded over many blocks.

    Guards against slow drift and unbounded per-session growth: after 200 blocks
    the ring buffer is still exactly one window long and the retained SOLA tail
    is exactly one crossfade long, while continuity holds throughout.
    """
    amp, freq, n_blocks = 0.2, 443.0, 200
    session = _IdentityStreamSession(None, model_sr=_SR_16K, block_ms=DEFAULT_BLOCK_MS)

    outs, _, _ = _run_sine(session, freq, amp, n_blocks)

    assert all(o.shape[0] == session.block_frame_sr for o in outs)
    # per-session state is bounded, not accumulating with block count
    assert session._input_buffer.shape[0] == session._window_16k
    assert session._sola_buffer is not None
    assert session._sola_buffer.shape[0] == session._crossfade_sr

    stitched = np.concatenate(outs)
    ref_step = _pure_sine_step(amp, freq, _SR_16K)
    ratio = float(np.abs(np.diff(stitched)).max()) / ref_step
    assert ratio < MAX_DELTA_FACTOR, f"continuity degraded over {n_blocks} blocks: {ratio:.2f}x"


def test_click_metric_discriminates_sola_from_naive_chunking():
    """Negative control: the same continuity check that passes for SOLA fails for
    the failure mode this step exists to prevent.

    On identical per-block input, the real SOLA session emits a click-free
    stream, whereas naive chunking -- converting blocks independently, with no
    cross-block phase continuity (modelled here as a phase flip per block) --
    produces an order-of-magnitude larger discontinuity at every join. If this
    assertion did not separate the two, the continuity tests above would be
    meaningless.
    """
    amp, freq, n_blocks = 0.2, 443.0, 64
    ref_step = _pure_sine_step(amp, freq, _SR_16K)

    session = _IdentityStreamSession(None, model_sr=_SR_16K, block_ms=DEFAULT_BLOCK_MS)
    outs, _, _ = _run_sine(session, freq, amp, n_blocks)
    sola = np.concatenate(outs)
    sola_ratio = float(np.abs(np.diff(sola)).max()) / ref_step

    bf = session.block_frame_sr
    naive_blocks = []
    for k in range(n_blocks):
        n = np.arange(k * bf, (k + 1) * bf)
        seg = (amp * np.sin(2.0 * np.pi * freq * n / _SR_16K)).astype(np.float32)
        naive_blocks.append(seg if k % 2 == 0 else -seg)  # independent-block phase flip
    naive = np.concatenate(naive_blocks)
    naive_ratio = float(np.abs(np.diff(naive)).max()) / ref_step

    assert sola_ratio < MAX_DELTA_FACTOR, f"real SOLA output should be smooth, got {sola_ratio:.2f}x"
    assert naive_ratio > MAX_DELTA_FACTOR, (
        f"metric failed to flag naive chunking ({naive_ratio:.2f}x) -- the continuity "
        f"test would not catch a click"
    )
    # the two regimes must be clearly separated, not marginally so
    assert naive_ratio > 3.0 * sola_ratio


def test_output_length_tracks_model_sample_rate_ratio():
    """Output block length follows the model/16k ratio, exercised at 48 kHz.

    The offline pipeline emits at the checkpoint's native rate; streaming must
    keep the same relationship. With a 48 kHz model each emitted block is 3x the
    16 kHz input block, and the value is stable call-to-call.
    """
    model_sr = 48000
    session = _IdentityStreamSession(None, model_sr=model_sr, block_ms=DEFAULT_BLOCK_MS)
    assert session.block_frame_sr == round(session.block_frame_16k * model_sr / _SR_16K)
    assert session.block_frame_sr == session.block_frame_16k * 3  # 48000/16000

    outs, _, _ = _run_sine(session, freq=220.0, amp=0.2, n_blocks=8)
    assert all(o.shape[0] == session.block_frame_sr for o in outs)


# ── 2. Overload flag on artificially slow inference ─────────────────────────


def test_overload_flag_set_when_inference_slower_than_block():
    """``overload`` is True iff a block took longer than its real-time budget.

    A 100 ms block that takes ~150 ms to convert cannot keep up and must flag
    overload; the same session shape with instant conversion must not. Testing
    both directions proves the flag is derived from timing, not hard-wired.
    """
    block_ms = 100  # -> 100 ms real-time budget per block
    amp, freq = 0.2, 220.0

    slow = _SlowIdentityStreamSession(None, model_sr=_SR_16K, block_ms=block_ms, sleep_s=0.15)
    n = np.arange(slow.block_frame_16k)
    block = (amp * np.sin(2.0 * np.pi * freq * n / _SR_16K)).astype(np.float32)
    _, infer_ms, overload = slow.process(block)
    assert overload is True
    assert infer_ms > block_ms, f"slow block infer_ms {infer_ms:.1f} should exceed {block_ms} ms"

    fast = _IdentityStreamSession(None, model_sr=_SR_16K, block_ms=block_ms)
    _, infer_ms_fast, overload_fast = fast.process(block)
    assert overload_fast is False
    assert infer_ms_fast < block_ms, f"identity block should be well under {block_ms} ms"


# ── 3. WebSocket wire contract (FastAPI TestClient, no model) ────────────────


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    """TestClient over the real app pinned to a throwaway data dir (as convert_api)."""
    from backend.app import app

    data_dir = tmp_path_factory.mktemp("rvc_stream_data")
    original = config.get_data_dir()
    config.set_data_dir(str(data_dir))
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        config.set_data_dir(str(original))


def _fake_checkpoint_bytes() -> bytes:
    """A minimal *valid* extracted v2/40k/f0 checkpoint (matches test_convert_api).

    Enough for ``validate_rvc_checkpoint`` to accept the upload and set
    ``rvc_model_path`` so the stream handshake's profile resolution passes; it is
    never actually loaded (``get_rvc_engine`` is stubbed for these tests).
    Carries ``emb_g.weight`` (strictness requires it) and a full 18-arity config.
    """
    ckpt = {
        "weight": {
            "emb_g.weight": torch.zeros(109, 256),
            "enc_p.emb_phone.weight": torch.zeros(2, 2),
        },
        "config": [
            1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
            [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            [10, 10, 2, 2], 512, [16, 16, 4, 4], 109, 256, 40000,
        ],
        "f0": 1,
        "version": "v2",
        "sr": 40000,
    }
    buf = io.BytesIO()
    torch.save(ckpt, buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def rvc_profile_id(client) -> str:
    """An RVC profile with a valid (unused) checkpoint so the handshake resolves."""
    resp = client.post(
        "/profiles",
        json={"name": f"stream-{uuid.uuid4().hex[:8]}", "voice_type": "rvc"},
    )
    assert resp.status_code == 200, resp.text
    profile_id = resp.json()["id"]

    up = client.post(
        f"/profiles/{profile_id}/rvc-model",
        files={"model": ("keruanv2.pth", _fake_checkpoint_bytes(), "application/octet-stream")},
    )
    assert up.status_code == 200, up.text
    assert up.json()["rvc_has_model"], "handshake needs a model present"
    return profile_id


class _StubEngine:
    """Stands in for the process-wide RVCPipeline: never loads a real model."""

    def is_loaded(self) -> bool:
        return False

    def load(self, *args, **kwargs) -> None:  # pragma: no cover - trivial no-op
        return None


class _WSIdentitySession(_IdentityStreamSession):
    """Route-constructed identity session: ignores the (stub) engine, forces 16k.

    The route calls ``RVCStreamSession(engine, block_ms=..., f0_up_key=..., ...)``;
    this subclass swallows the engine, pins ``model_sr`` to 16 kHz for a true
    identity, and otherwise runs the real session so the WS test drives the real
    ``process``/SOLA path with only the model faked.
    """

    def __init__(self, pipeline, **kwargs) -> None:
        super().__init__(None, model_sr=_SR_16K, **kwargs)


@pytest.fixture
def patched_stream(monkeypatch):
    """Fake the model out of the WS path and start from a clean single-session flag."""
    # A prior WS test releases the engine lease asynchronously; wait for it so this
    # test starts from a free lease rather than racing the previous teardown.
    _wait_lease_free()
    monkeypatch.setattr("backend.backends.rvc.get_rvc_engine", lambda: _StubEngine())
    monkeypatch.setattr("backend.backends.rvc.streaming.RVCStreamSession", _WSIdentitySession)
    monkeypatch.setattr(convert_route, "_stream_active", False)


def _handshake_reply(ws, profile_id: str, **overrides) -> dict:
    payload = {"profile_id": profile_id}
    payload.update(overrides)
    ws.send_json(payload)
    return ws.receive_json()


def test_ws_handshake_returns_ready_and_model_sr(client, rvc_profile_id, patched_stream):
    """The handshake replies with the success schema: ready + sizing fields."""
    with client.websocket_connect("/convert/stream") as ws:
        reply = _handshake_reply(ws, rvc_profile_id)

    assert reply["ready"] is True
    assert reply["model_sr"] == _SR_16K
    assert reply["block_frame_16k"] > 0
    # identity session pins model_sr to 16k, so in == out block size here
    assert reply["block_frame_sr"] == reply["block_frame_16k"]


def test_ws_binary_roundtrip_returns_converted_block_with_header(client, rvc_profile_id, patched_stream):
    """One f32 PCM input frame yields a bounded, non-silent audio output frame."""
    with client.websocket_connect("/convert/stream") as ws:
        reply = _handshake_reply(ws, rvc_profile_id)
        n_in = reply["block_frame_16k"]
        n_out = reply["block_frame_sr"]

        pcm_in = (0.2 * np.sin(2 * np.pi * 220.0 * np.arange(n_in) / _SR_16K)).astype("<f4")
        ws.send_bytes(pcm_in.tobytes())
        msg = ws.receive_bytes()

    assert len(msg) == _STREAM_OUT_HEADER.size + n_out * 4
    infer_ms, overload, dropped, reserved = _STREAM_OUT_HEADER.unpack(
        msg[: _STREAM_OUT_HEADER.size]
    )
    assert infer_ms >= 0.0
    assert overload in (0, 1)
    assert dropped == 0  # single in-order block, nothing to drop
    assert reserved == 0

    out = np.frombuffer(msg[_STREAM_OUT_HEADER.size :], dtype="<f4")
    assert out.shape[0] == n_out
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) <= 1.0

    # The identity session should produce an audible but safe block: not silence,
    # not clipping, and roughly in the energy range implied by the 0.2-amplitude
    # input sine after the initial ring-buffer warmup.
    peak = float(np.max(np.abs(out)))
    rms = float(np.sqrt(np.mean(np.square(out, dtype=np.float64))))
    assert 0.05 <= peak <= 0.35
    assert 0.03 <= rms <= 0.20


def test_stream_active_rejects_new_tts_generation(client, rvc_profile_id, patched_stream):
    """A live real-time stream gets priority over starting new TTS work."""
    preset_resp = client.post(
        "/profiles",
        json={
            "name": f"stream-guard-{uuid.uuid4().hex[:8]}",
            "language": "en",
            "voice_type": "preset",
            "preset_engine": "kokoro",
            "preset_voice_id": "af_heart",
            "default_engine": "kokoro",
        },
    )
    assert preset_resp.status_code == 200, preset_resp.text
    preset_id = preset_resp.json()["id"]

    with client.websocket_connect("/convert/stream") as ws:
        reply = _handshake_reply(ws, rvc_profile_id)
        assert reply["ready"] is True
        assert (lease_holder() or "").startswith("stream:")

        blocked = client.post(
            "/generate",
            json={
                "profile_id": preset_id,
                "text": "This should wait until the live stream stops.",
                "language": "en",
                "engine": "kokoro",
            },
        )

    assert blocked.status_code == 409
    assert "real-time streaming is active" in blocked.json()["detail"]


def test_ws_second_concurrent_connection_is_rejected(client, rvc_profile_id, patched_stream):
    """Only one live stream at a time: a second connect is closed with 1008."""
    with client.websocket_connect("/convert/stream") as ws1:
        assert _handshake_reply(ws1, rvc_profile_id)["ready"] is True

        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/convert/stream") as ws2:
                # server accepts then immediately closes; the close surfaces here
                ws2.receive_json()
        assert excinfo.value.code == 1008


def test_ws_fresh_session_connects_after_disconnect(client, rvc_profile_id, patched_stream):
    """After a clean disconnect the single-session slot frees and a new stream connects."""
    with client.websocket_connect("/convert/stream") as ws1:
        assert _handshake_reply(ws1, rvc_profile_id)["ready"] is True
    # teardown runs on the app's event-loop thread; wait for the slot to clear
    deadline = time.time() + 5.0
    while convert_route._stream_active and time.time() < deadline:
        time.sleep(0.02)
    assert convert_route._stream_active is False, "stream slot not released on disconnect"

    with client.websocket_connect("/convert/stream") as ws3:
        reply = _handshake_reply(ws3, rvc_profile_id)
    assert reply["ready"] is True
    assert reply["model_sr"] == _SR_16K


# ── 3b. Wave 2 robustness: protocol version, handshake timeout, error frame ──


def test_ws_handshake_version_one_is_accepted_and_echoed(client, rvc_profile_id, patched_stream):
    """An explicit ``version: 1`` handshake succeeds and the reply echoes it."""
    from backend.models import STREAM_PROTOCOL_VERSION

    with client.websocket_connect("/convert/stream") as ws:
        reply = _handshake_reply(ws, rvc_profile_id, version=1)

    assert reply["ready"] is True
    assert reply["version"] == STREAM_PROTOCOL_VERSION == 1


def test_ws_handshake_unknown_version_is_rejected(client, rvc_profile_id, patched_stream):
    """A future major (``version: 2``) is rejected with a clear reason + 1008,
    before any model is loaded or the engine lease is taken."""
    with client.websocket_connect("/convert/stream") as ws:
        reply = _handshake_reply(ws, rvc_profile_id, version=2)
        assert reply["ready"] is False
        assert "version" in reply["error"].lower()
        assert "2" in reply["error"]
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()  # the error frame is followed by a 1008 close
    assert excinfo.value.code == 1008
    # rejection happened before the lease was taken; nothing to release
    _wait_lease_free()
    assert lease_holder() is None


def test_ws_handshake_timeout_frees_the_single_session_slot(
    client, rvc_profile_id, patched_stream, monkeypatch
):
    """A half-open socket that never sends a handshake must not lock the endpoint
    forever: it times out, is rejected cleanly, and the single-session slot frees
    so the *next* connection handshakes normally."""
    # Shrink the wait so the test is fast; the route reads this at call time.
    monkeypatch.setattr(convert_route, "_HANDSHAKE_TIMEOUT_S", 0.3)

    with client.websocket_connect("/convert/stream") as ws:
        # never send a handshake; the server must time out and reject
        reply = ws.receive_json()
        assert reply["ready"] is False
        assert "timed out" in reply["error"].lower()
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
    assert excinfo.value.code == 1008

    # the slot must be released even though no handshake ever arrived
    deadline = time.time() + 5.0
    while convert_route._stream_active and time.time() < deadline:
        time.sleep(0.02)
    assert convert_route._stream_active is False, "timed-out handshake left the slot locked"

    # the next connection is unaffected
    with client.websocket_connect("/convert/stream") as ws2:
        reply2 = _handshake_reply(ws2, rvc_profile_id)
    assert reply2["ready"] is True


class _FailingWSSession(_WSIdentitySession):
    """Route-constructed identity session whose conversion raises, to drive the
    mid-stream error-frame path (an inference failure after a good handshake)."""

    def _convert_window(self, window_16k: np.ndarray) -> np.ndarray:
        raise RuntimeError("synthetic inference failure")


def test_ws_inference_error_sends_json_error_frame_before_close(
    client, rvc_profile_id, patched_stream, monkeypatch
):
    """When conversion fails mid-stream the client receives a JSON error frame
    (``{type: "error", error}``) explaining why, then a clean 1011 close — not a
    bare disconnect."""
    # override patched_stream's identity session with one that fails in inference
    monkeypatch.setattr(
        "backend.backends.rvc.streaming.RVCStreamSession", _FailingWSSession
    )

    with client.websocket_connect("/convert/stream") as ws:
        reply = _handshake_reply(ws, rvc_profile_id)
        assert reply["ready"] is True
        n_in = reply["block_frame_16k"]

        pcm = (0.2 * np.sin(2 * np.pi * 220.0 * np.arange(n_in) / _SR_16K)).astype("<f4")
        ws.send_bytes(pcm.tobytes())

        err = ws.receive_json()  # mid-stream error frame, before the close
        assert err["type"] == "error"
        assert "conversion failed" in err["error"].lower()
        with pytest.raises(WebSocketDisconnect) as excinfo:
            ws.receive_json()
    assert excinfo.value.code == 1011

    # the failed session still releases the engine lease on teardown
    _wait_lease_free()
    assert lease_holder() is None


# ── 4. Engine-lease arbitration (single shared RVC engine) ──────────────────
#
# The WS stream, offline /convert, and TTS->RVC chain share ONE process-wide RVC
# engine. Only one may drive it at a time or an offline job swaps net_g/sample
# rate under a live stream. Arbitration is an exclusive lease: the stream holds it
# for its session; jobs try-acquire per job and fail fast on contention.


def _convert_source_wav(seconds: float = 1.0, sr: int = _SR_16K) -> bytes:
    """A short decodable mono WAV for the offline ``/convert`` request body."""
    t = np.linspace(0.0, seconds, int(sr * seconds), endpoint=False)
    sig = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, sig, sr, format="WAV")
    return buf.getvalue()


def test_lease_is_exclusive_reentrant_and_released():
    """The lease primitive the three consumers depend on: exclusive, reentrant for
    the same owner, and freed on release (releasing as a non-holder is a no-op)."""
    _wait_lease_free()
    a, b = "owner-a", "owner-b"
    try:
        assert lease_holder() is None
        assert rvc_acquire(a) is True          # free -> acquired
        assert lease_holder() == a
        assert rvc_acquire(a) is True          # same owner is reentrant
        assert rvc_acquire(b) is False         # a different owner is refused
        assert lease_holder() == a             # refusal does not change the holder
        rvc_release(b)                         # releasing as a non-holder is a no-op
        assert lease_holder() == a
        rvc_release(a)
        assert lease_holder() is None
        assert rvc_acquire(b) is True          # now free for the next owner
    finally:
        rvc_release(a)
        rvc_release(b)


def test_stream_active_blocks_offline_convert_and_stream_unaffected(
    client, rvc_profile_id, patched_stream
):
    """Matrix (stream vs convert): a live stream holds the lease; an offline
    ``/convert`` enqueued during it fails fast with the user-readable message,
    while the stream keeps converting blocks unaffected.
    """
    with client.websocket_connect("/convert/stream") as ws:
        reply = _handshake_reply(ws, rvc_profile_id)
        assert reply["ready"] is True
        # the live session now owns the engine lease
        assert (lease_holder() or "").startswith("stream:")

        resp = client.post(
            "/convert",
            data={"profile_id": rvc_profile_id},
            files={"file": ("source.wav", _convert_source_wav(), "audio/wav")},
        )
        assert resp.status_code == 200, resp.text  # accepted + enqueued
        task_id = resp.json()["task_id"]

        deadline = time.time() + 30
        status = None
        error = None
        while time.time() < deadline:
            payload = client.get(f"/history/{task_id}").json()
            status = payload["status"]
            error = payload.get("error")
            if status in ("completed", "failed"):
                break
            time.sleep(0.1)
        assert status == "failed", f"offline convert should fail while stream holds lease (last={status})"
        assert "Voice Changer live session is active" in (error or "")

        # The stream is unaffected: a block still round-trips through the session.
        n_in = reply["block_frame_16k"]
        pcm = (0.2 * np.sin(2 * np.pi * 220.0 * np.arange(n_in) / _SR_16K)).astype("<f4")
        ws.send_bytes(pcm.tobytes())
        msg = ws.receive_bytes()
        assert len(msg) == _STREAM_OUT_HEADER.size + reply["block_frame_sr"] * 4
        # the live session still holds the lease after the failed convert
        assert (lease_holder() or "").startswith("stream:")

    # stopping the stream releases the lease
    _wait_lease_free()
    assert lease_holder() is None


def test_offline_convert_lease_rejects_ws_connect(client, rvc_profile_id, patched_stream):
    """Matrix (convert vs stream): an offline conversion (or chain) holds the
    engine lease; a WS connect is rejected at handshake with the 'job is running'
    message and a 1008 close — never a model swap under the running job.
    """
    job_owner = "convert:test-running-job"
    assert rvc_acquire(job_owner) is True
    try:
        with pytest.raises(WebSocketDisconnect) as excinfo:
            with client.websocket_connect("/convert/stream") as ws:
                reply = _handshake_reply(ws, rvc_profile_id)
                assert reply["ready"] is False
                assert "job is running" in reply["error"].lower()
                # the error frame is followed by a 1008 policy-violation close
                ws.receive_json()
        assert excinfo.value.code == 1008
        # the running job still owns the lease; the rejected stream never took it
        assert lease_holder() == job_owner
    finally:
        rvc_release(job_owner)


def test_stream_stop_releases_lease_for_convert(client, rvc_profile_id, patched_stream):
    """Matrix (stream stop -> convert acquires): stopping the stream frees the
    lease so a conversion can immediately acquire it."""
    with client.websocket_connect("/convert/stream") as ws:
        assert _handshake_reply(ws, rvc_profile_id)["ready"] is True
        assert (lease_holder() or "").startswith("stream:")

    _wait_lease_free()
    assert lease_holder() is None, "stream did not release the engine lease on stop"

    job_owner = "convert:after-stream"
    assert rvc_acquire(job_owner) is True  # a conversion takes the lease straight away
    rvc_release(job_owner)


def test_pipeline_load_identity_switches_and_no_ops(monkeypatch, tmp_path):
    """Matrix (identity): ``RVCPipeline.load()`` reloads only when the checkpoint
    identity (resolved path, index, mtime) changes and no-ops when it is
    unchanged — so a stream reconnecting after *another* profile's conversion
    loads the RIGHT model, a same-model reconnect skips the multi-second reload,
    and a re-upload to the same path (new mtime) is correctly seen as a new model.

    The neural build is monkeypatched out (this asserts the identity arithmetic,
    not synthesizer construction), and two on-disk checkpoints with distinct
    mtimes stand in for two profiles' models.
    """
    import torch.nn as nn

    from backend.backends.rvc import pipeline as pipeline_mod
    from backend.backends.rvc.checkpoint import RVCCheckpointInfo
    from backend.backends.rvc.pipeline import RVCPipeline

    builds: list[str] = []

    def fake_build(ckpt, info, *, is_half=False):
        builds.append("build")
        return nn.Module()  # supports .to()/.half()/.float(); no real net needed

    monkeypatch.setattr(pipeline_mod, "build_synthesizer", fake_build)
    monkeypatch.setattr(pipeline_mod, "load_rvc_checkpoint", lambda p: {"fake": True})
    monkeypatch.setattr(
        pipeline_mod,
        "validate_rvc_checkpoint",
        lambda ckpt: RVCCheckpointInfo(
            version="v2", sample_rate=40000, if_f0=1, embedder_dim=768
        ),
    )

    model_a = tmp_path / "profile_a.pth"
    model_a.write_bytes(b"A")
    model_b = tmp_path / "profile_b.pth"
    model_b.write_bytes(b"B")

    eng = RVCPipeline()

    eng.load(str(model_a))
    assert len(builds) == 1 and eng.is_loaded()   # first load builds

    eng.load(str(model_a))
    assert len(builds) == 1                        # identical identity -> no-op

    eng.load(str(model_b))
    assert len(builds) == 2                        # different profile -> switch

    eng.load(str(model_a))
    assert len(builds) == 3                        # back to A -> switch (not stale B)

    # Re-upload to the SAME path: bump the mtime and expect a reload, not a no-op.
    st = os.stat(model_a)
    bumped = st.st_mtime_ns + 1_000_000_000
    os.utime(model_a, ns=(bumped, bumped))
    eng.load(str(model_a))
    assert len(builds) == 4                        # mtime change defeats the no-op
