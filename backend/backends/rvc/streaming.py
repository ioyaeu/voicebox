"""Real-time (frame-by-frame) RVC voice conversion with SOLA stitching.

This is the streaming counterpart to the offline :class:`RVCPipeline`. It does
**not** fork the conversion math: feature extraction, RMVPE f0 and the loaded
synthesizer are reused through the pipeline's own components
(``RVCPipeline._vc`` / ``_get_f0``, ``features.extract_contentvec_features``,
``pitch.extract_f0_rmvpe``). The only new logic here is the sliding-context
windowing and the click-free stitching of independently-converted blocks.

Why not just convert each incoming block on its own: a neural vocoder started
cold on a 300 ms block has no left/right context, so successive blocks disagree
on phase and level at their seams and you hear a click every block. Two
mechanisms fix that, both mandatory:

1. **Re-inferred sliding context.** Every call converts a window
   ``[extra | crossfade | block | sola_search]`` of *past + new* input, not the
   bare block. The ``extra`` region (``extra_ms`` of real past audio, ~0.8 s) is
   converted every call purely to warm the nets, then discarded. Only the tail
   of the converted window is kept.
2. **SOLA (Synchronised OverLap-Add).** The kept tail is aligned to the
   previously-emitted output by searching ``±sola_search_ms`` for the integer
   offset that maximises normalised cross-correlation in the overlap zone, then
   the two are joined with an equal-power crossfade over ``crossfade_ms``. The
   correlation search absorbs the block-to-block phase drift; the crossfade
   removes the residual seam.

Everything after conversion (search, crossfade, emit) happens at the model's
native sample rate. Per-call timing is returned so the caller can drive a
latency gauge and detect underrun (``infer_ms > block_ms``).

Concurrency: one session per connection, single-threaded. :meth:`process` is a
blocking CPU/GPU call meant to be run off the event loop via
``asyncio.to_thread``; an internal lock only guards against a concurrent
:meth:`reset`/:meth:`close`.
"""

import logging
import threading
import time

import numpy as np
import torch
from scipy import signal

from .pipeline import (
    _HP_A,
    _HP_B,
    _SR_16K,
    _WINDOW,
    RVCPipeline,
    _change_rms,
)

logger = logging.getLogger(__name__)

# Defaults mirror 05_REALTIME_STREAMING.md. block_ms is client-negotiated; the
# rest are fixed windowing constants tuned to w-okada / RVC-realtime practice.
DEFAULT_BLOCK_MS = 300
DEFAULT_EXTRA_MS = 800
DEFAULT_CROSSFADE_MS = 80
DEFAULT_SOLA_SEARCH_MS = 10


def _ms_to_frames(ms: float, sr: int) -> int:
    """Convert milliseconds to a whole number of ``_WINDOW`` (160-sample) hops.

    Rounding to the 160-sample f0/feature hop keeps every region an integer
    number of frames at 16 kHz, so the frame arithmetic that ``_vc`` and the f0
    extractor depend on never drifts by a partial hop.
    """
    samples = ms / 1000.0 * sr
    hops = max(1, round(samples / _WINDOW))
    return hops * _WINDOW


class RVCStreamSession:
    """Frame-based RVC conversion for one live stream, with SOLA stitching.

    Wraps a **loaded** :class:`RVCPipeline`. Feed fixed-size 16 kHz mono blocks
    to :meth:`process`; get back converted mono blocks at :attr:`model_sr`, one
    per call, that concatenate into a click-free stream.

    The conversion parameters (``f0_up_key`` etc.) match the offline
    ``convert_file`` API so file mode and real-time mode sound the same.

    Args:
        pipeline: A loaded ``RVCPipeline``. May be ``None`` **only** when
            ``model_sr`` is given and :meth:`_convert_window` is overridden — the
            seam the synthetic SOLA test uses to exercise the windowing/stitching
            math without a model.
        block_ms: Size of each input/output block. Client-negotiated; the
            emitted block advances the stream by exactly this much.
        extra_ms: Past-audio context re-converted every call and discarded.
        crossfade_ms: Equal-power overlap-add length at block joins.
        sola_search_ms: Forward correlation search range for SOLA alignment.
        f0_up_key: Pitch shift in semitones (f0 checkpoints only).
        f0_method: ``"rmvpe"`` (default) or ``"crepe"``.
        index_rate: FAISS retrieval blend weight in ``[0, 1]``.
        rms_mix_rate: Loudness-envelope blend (see ``RVCPipeline.convert_file``).
        protect: Consonant/breath protection in ``[0, 0.5]``.
        model_sr: Override for the model sample rate; defaults to the loaded
            checkpoint's rate. Only pass this in tests without a pipeline.

    Attributes:
        model_sr: Output sample rate (the checkpoint's native rate).
        block_frame_16k: Exact input block length in samples the caller must
            send to :meth:`process` (16 kHz mono f32).
        block_frame_sr: Exact output block length returned per call.
    """

    def __init__(
        self,
        pipeline: RVCPipeline | None,
        *,
        block_ms: int = DEFAULT_BLOCK_MS,
        extra_ms: int = DEFAULT_EXTRA_MS,
        crossfade_ms: int = DEFAULT_CROSSFADE_MS,
        sola_search_ms: int = DEFAULT_SOLA_SEARCH_MS,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        model_sr: int | None = None,
    ) -> None:
        if model_sr is None:
            if pipeline is None or not pipeline.is_loaded():
                raise RuntimeError(
                    "RVCStreamSession requires a loaded RVCPipeline "
                    "(or an explicit model_sr for the no-model SOLA test)."
                )
            model_sr = int(pipeline._tgt_sr)
        if model_sr <= 0:
            raise ValueError(f"model_sr must be positive, got {model_sr}")
        if block_ms <= 0:
            raise ValueError(f"block_ms must be positive, got {block_ms}")

        self._pipeline = pipeline
        self.model_sr = int(model_sr)

        self._f0_up_key = int(f0_up_key)
        self._f0_method = str(f0_method)
        self._index_rate = float(index_rate)
        self._rms_mix_rate = float(rms_mix_rate)
        self._protect = float(protect)

        # Region sizes at the 16 kHz input rate (whole 160-sample hops).
        self.block_frame_16k = _ms_to_frames(block_ms, _SR_16K)
        self._crossfade_16k = _ms_to_frames(crossfade_ms, _SR_16K)
        self._sola_search_16k = _ms_to_frames(sola_search_ms, _SR_16K)
        self._extra_16k = _ms_to_frames(extra_ms, _SR_16K)

        # Full input window fed to the converter each call, laid out from oldest
        # to newest as [extra | crossfade | sola_search | block]: the newest
        # block sits at the end (``process`` writes it to ``buf[-nb:]``), with the
        # crossfade+sola_search context immediately before it and the discarded
        # ``extra`` conversion warm-up at the front. (The sum below is order-
        # independent; the order is defined by how ``process`` fills the buffer.)
        self._window_16k = self._extra_16k + self._crossfade_16k + self.block_frame_16k + self._sola_search_16k

        # Matching region sizes at the output (model) rate. Derived from the
        # 16 kHz sizes by the fixed ratio so SOLA arithmetic stays integer.
        self._ratio = self.model_sr / _SR_16K
        self.block_frame_sr = round(self.block_frame_16k * self._ratio)
        self._crossfade_sr = round(self._crossfade_16k * self._ratio)
        self._sola_search_sr = round(self._sola_search_16k * self._ratio)
        # Tail of each converted window we keep (extra context is discarded).
        self._keep_sr = self._crossfade_sr + self._sola_search_sr + self.block_frame_sr

        # Equal-power crossfade windows: fade_in^2 + fade_out^2 == 1, and the
        # endpoints (fade_in[0]=0, fade_out[0]=1 / fade_in[-1]=1, fade_out[-1]=0)
        # make each join C0-continuous regardless of the SOLA offset.
        t = np.linspace(0.0, 1.0, self._crossfade_sr, dtype=np.float64)
        self._fade_in = np.sin(0.5 * np.pi * t).astype(np.float32)
        self._fade_out = np.cos(0.5 * np.pi * t).astype(np.float32)

        self._lock = threading.Lock()
        self._input_buffer = np.zeros(self._window_16k, dtype=np.float32)
        # Tail of the last emitted output, held for the next crossfade. None
        # until the first block has been emitted.
        self._sola_buffer: np.ndarray | None = None

        logger.info(
            "RVCStreamSession: model_sr=%d block=%d/%d extra=%d crossfade=%d "
            "search=%d (16k samples), window=%d, keep_sr=%d",
            self.model_sr,
            self.block_frame_16k,
            self.block_frame_sr,
            self._extra_16k,
            self._crossfade_16k,
            self._sola_search_16k,
            self._window_16k,
            self._keep_sr,
        )

    # -- public API ---------------------------------------------------------

    def process(self, block: np.ndarray) -> tuple[np.ndarray, float, bool]:
        """Convert one input block and stitch it onto the running output.

        Args:
            block: Mono 16 kHz f32 PCM of length exactly
                :attr:`block_frame_16k`.

        Returns:
            ``(converted, infer_ms, overload)`` where ``converted`` is mono f32
            PCM at :attr:`model_sr` of length :attr:`block_frame_sr`,
            ``infer_ms`` is the wall-clock time this call took (conversion +
            stitching), and ``overload`` is ``True`` when ``infer_ms`` exceeded
            the block duration (the stream cannot keep up in real time).
        """
        block = np.ascontiguousarray(block, dtype=np.float32).reshape(-1)
        if block.shape[0] != self.block_frame_16k:
            raise ValueError(
                f"block must be {self.block_frame_16k} samples "
                f"(got {block.shape[0]}); block_ms is fixed for the session."
            )

        start = time.perf_counter()
        with self._lock:
            nb = self.block_frame_16k
            buf = self._input_buffer
            buf[:-nb] = buf[nb:]
            buf[-nb:] = block

            converted = self._convert_window(buf)
            converted = np.ascontiguousarray(converted, dtype=np.float32).reshape(-1)

            # Keep only the tail matching [crossfade | block | sola_search];
            # the leading extra-context conversion is thrown away. Front-pad
            # with silence in the unlikely case the vocoder returns short.
            if converted.shape[0] < self._keep_sr:
                pad = self._keep_sr - converted.shape[0]
                converted = np.concatenate([np.zeros(pad, dtype=np.float32), converted])
            conv = converted[-self._keep_sr :]

            out_block = self._stitch(conv)

        infer_ms = (time.perf_counter() - start) * 1000.0
        block_ms = self.block_frame_16k / _SR_16K * 1000.0
        overload = infer_ms > block_ms
        return out_block, infer_ms, overload

    def reset(self) -> None:
        """Clear the sliding context and stitch state (keeps the model loaded)."""
        with self._lock:
            self._input_buffer[:] = 0.0
            self._sola_buffer = None

    def close(self) -> None:
        """Release per-session buffers. The shared pipeline stays loaded."""
        with self._lock:
            self._input_buffer = np.zeros(0, dtype=np.float32)
            self._sola_buffer = None

    # -- SOLA stitching -----------------------------------------------------

    def _stitch(self, conv: np.ndarray) -> np.ndarray:
        """Align ``conv`` to the previous output and crossfade the seam.

        ``conv`` is the kept tail of one converted window, length
        ``crossfade + block + sola_search`` at the model rate. Returns the next
        ``block``-length slice to emit and updates the saved crossfade tail.
        """
        cf = self._crossfade_sr
        bf = self.block_frame_sr

        if self._sola_buffer is None:
            # First block: nothing to align to. Emit from the start and seed the
            # crossfade tail from the sample right after the emitted block.
            aligned = conv[: bf + cf].copy()
        else:
            offset = self._sola_offset(conv[: cf + self._sola_search_sr], self._sola_buffer)
            aligned = conv[offset : offset + bf + cf].copy()
            # Equal-power crossfade the new head against the saved tail. The join
            # at aligned[0] equals the saved tail (fade_in[0]=0, fade_out[0]=1),
            # the natural continuation of the previously emitted block.
            aligned[:cf] = aligned[:cf] * self._fade_in + self._sola_buffer * self._fade_out

        out_block = aligned[:bf].copy()
        self._sola_buffer = aligned[bf : bf + cf].copy()
        # Guard the DAC against equal-power overshoot on strongly-correlated
        # frames; a no-op for tanh-bounded generator output at normal levels.
        np.clip(out_block, -1.0, 1.0, out=out_block)
        return out_block

    def _sola_offset(self, search: np.ndarray, buffer: np.ndarray) -> int:
        """Integer offset in ``[0, sola_search]`` maximising normalised x-corr.

        ``search`` has length ``crossfade + sola_search``; ``buffer`` (the saved
        crossfade tail) has length ``crossfade``. Returns the shift where the
        buffer best matches the search window, normalising by the search energy
        so a louder region does not win purely on amplitude.
        """
        cf = buffer.shape[0]
        nom = np.correlate(search, buffer, mode="valid")  # length sola_search+1
        csum = np.concatenate(([0.0], np.cumsum(np.square(search, dtype=np.float64))))
        energy = csum[cf:] - csum[:-cf]  # sliding sum of squares, same length
        den = np.sqrt(energy + 1e-8)
        return int(np.argmax(nom / den))

    # -- model touchpoint (the only place the loaded net is used) -----------

    def _convert_window(self, window_16k: np.ndarray) -> np.ndarray:
        """Convert one 16 kHz input window to model-rate audio.

        Reuses the loaded pipeline's f0 and ``_vc`` path verbatim — no separate
        inference implementation. Unlike the offline pipeline this does **not**
        reflect-pad or silence-chunk: the real ``extra`` context in the window is
        the left padding, and the window is short enough to convert whole.

        Overridden in the synthetic SOLA test with an identity transform so the
        windowing/stitching math can be verified without a model.
        """
        p = self._pipeline
        if p is None:
            raise RuntimeError("No pipeline bound; _convert_window must be overridden.")
        with p._lock:
            if not p.is_loaded():
                raise RuntimeError("No RVC model loaded; call pipeline.load() first.")

            window = np.ascontiguousarray(window_16k, dtype=np.float32)
            # 48 Hz high-pass, verbatim with the offline pipeline (pipeline.py:318).
            window = signal.filtfilt(_HP_B, _HP_A, window).astype(np.float32)

            sid = torch.tensor(0, device=p._device).unsqueeze(0).long()
            if p._if_f0 == 1:
                pitch, pitchf = p._get_f0(window, self._f0_up_key, self._f0_method)
                p_len = window.shape[0] // _WINDOW
                pitch = pitch[:p_len]
                pitchf = pitchf[:p_len].astype(np.float32)
                pitch_t = torch.tensor(pitch, device=p._device).unsqueeze(0).long()
                pitchf_t = torch.tensor(pitchf, device=p._device).unsqueeze(0).float()
                out = p._vc(
                    sid,
                    window,
                    pitch_t,
                    pitchf_t,
                    p._index,
                    p._big_npy,
                    self._index_rate,
                    self._protect,
                )
            else:
                out = p._vc(
                    sid,
                    window,
                    None,
                    None,
                    p._index,
                    p._big_npy,
                    self._index_rate,
                    self._protect,
                )

            out = np.asarray(out, dtype=np.float32)
            if self._rms_mix_rate != 1:
                out = _change_rms(window, _SR_16K, out, self.model_sr, self._rms_mix_rate)
            return out.astype(np.float32)
