"""End-to-end offline RVC voice conversion.

Ports the inference-only path of upstream RVC (``infer/modules/vc/pipeline.py``
``Pipeline.{get_f0,vc,pipeline}`` + ``change_rms`` and the pre-normalisation in
``infer/modules/vc/modules.py`` ``VC.vc_single``; RVC-Project, MIT — see
``LICENSE_RVC.md``). Only file-in -> array-out conversion is kept; training,
streaming, ``pm``/``harvest`` f0, and the int16 export are dropped.

Fidelity-sensitive constants and tensor ops are annotated with their upstream
line (``pipeline.py:NN``) because a plausible-but-wrong hop size, feature layer,
or f0 bin here yields robotic/garbage audio that still runs without error.

The long-file handling is upstream's silence-based chunking with ``t_pad``
context padding: each chunk is padded, converted, then trimmed by ``t_pad_tgt``,
so the concatenation is quality-neutral and only bounds peak memory. It is
unrelated to (and not a substitute for) the real-time SOLA streaming of step 05.
"""

import logging
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from ..base import empty_device_cache, get_torch_device
from ...utils.audio import load_audio
from .checkpoint import (
    load_rvc_checkpoint,
    validate_faiss_index,
    validate_rvc_checkpoint,
)
from .features import extract_contentvec_features, unload_contentvec
from .pitch import extract_f0_crepe, extract_f0_rmvpe, unload_rmvpe
from .synthesizer import build_synthesizer

logger = logging.getLogger(__name__)

_SR_16K = 16000  # pipeline.py:74 — ContentVec/RMVPE input rate
_WINDOW = 160  # pipeline.py:75 — samples per f0 frame (hop)

# 5th-order 48 Hz high-pass applied to the 16 kHz input, verbatim pipeline.py:24.
_HP_B, _HP_A = signal.butter(N=5, Wn=48, btype="high", fs=_SR_16K)

# (x_pad, x_query, x_center, x_max) chunk-context params from configs/config.py
# device_config(): the fp16 branch (CUDA) and the fp32 branch (CPU/MPS). The
# <=4 GB low-VRAM branch is not reproduced — Voicebox only splits on precision.
_CHUNK_PARAMS_FP16 = (3, 10, 60, 65)
_CHUNK_PARAMS_FP32 = (1, 6, 38, 41)


def _resize_features_to_frame_count(feats: torch.Tensor, frame_count: int) -> torch.Tensor:
    """Align ContentVec frames to the synthesizer's 100 Hz hop grid."""
    if feats.shape[1] == frame_count:
        return feats
    return F.interpolate(feats.permute(0, 2, 1), size=frame_count).permute(0, 2, 1)


def _change_rms(
    data1: np.ndarray, sr1: int, data2: np.ndarray, sr2: int, rate: float
) -> np.ndarray:
    """Blend the source loudness envelope into the output (pipeline.py:43-62).

    ``rate`` is the weight of the *converted* signal's own envelope; ``1 - rate``
    pulls it back toward the source loudness contour.
    """
    rms1 = librosa.feature.rms(
        y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2
    )  # one point per half second
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(
        rms1.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(
        rms2.unsqueeze(0), size=data2.shape[0], mode="linear"
    ).squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= (
        torch.pow(rms1, torch.tensor(1 - rate))
        * torch.pow(rms2, torch.tensor(rate - 1))
    ).numpy()
    return data2


class RVCPipeline:
    """Offline (file-in -> array-out) RVC voice-conversion engine.

    Holds one loaded checkpoint at a time. Instances are created through the
    process-wide ``get_rvc_engine()`` factory; a re-entrant lock serialises
    ``load``/``unload``/``convert_file`` so a conversion never races a model
    swap on the single shared synthesizer.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reset_state()

    def _reset_state(self) -> None:
        self._net_g = None
        self._info = None
        self._index = None
        self._big_npy = None
        self._device: Optional[str] = None
        self._is_half = False
        self._tgt_sr: Optional[int] = None
        self._version: Optional[str] = None
        self._if_f0: Optional[int] = None
        self._chunk = _CHUNK_PARAMS_FP32
        # Identity of the resident checkpoint: (resolved_model_path,
        # resolved_index_path | None, model_mtime_ns). Lets load() short-circuit
        # a redundant reload while still catching a re-upload to the same path.
        self._identity: Optional[Tuple[str, Optional[str], int]] = None

    def is_loaded(self) -> bool:
        return self._net_g is not None

    def load(self, model_path: str, index_path: Optional[str] = None) -> None:
        """Load an RVC ``.pth`` checkpoint (and optional FAISS index).

        Checkpoints are read only through step 01's ``load_rvc_checkpoint`` +
        ``validate_rvc_checkpoint`` (``weights_only=True`` + structure checks);
        the index is validated with ``validate_faiss_index`` before use. fp16 is
        used on CUDA only; MPS and CPU stay fp32 (MPS half precision is
        unreliable for these nets).
        """
        with self._lock:
            resolved_model = str(Path(model_path).resolve())
            resolved_index = str(Path(index_path).resolve()) if index_path else None
            try:
                model_mtime_ns = os.stat(resolved_model).st_mtime_ns
            except OSError:
                # Missing/unreadable: fall through so load_rvc_checkpoint raises
                # the real error instead of a stale short-circuit.
                model_mtime_ns = None
            identity = (resolved_model, resolved_index, model_mtime_ns)

            # Already resident with the same path/index and on-disk mtime: skip
            # the multi-second reload. mtime keys the identity so a re-upload to
            # the same path is correctly seen as a new model.
            if (
                self._net_g is not None
                and model_mtime_ns is not None
                and self._identity == identity
            ):
                return

            ckpt = load_rvc_checkpoint(model_path)
            info = validate_rvc_checkpoint(ckpt)

            device = get_torch_device(allow_mps=True)
            is_half = device == "cuda"

            # Free any previously loaded model first — one model at a time.
            self._drop_net()

            net_g = build_synthesizer(ckpt, info, is_half=is_half)
            net_g = net_g.to(device)
            net_g = net_g.half() if is_half else net_g.float()

            index = None
            big_npy = None
            if index_path:
                validate_faiss_index(index_path, info.embedder_dim)
                import faiss  # lazy: matches checkpoint.py; faiss stays on CPU

                index = faiss.read_index(index_path)
                # big_npy is the reconstructed feature bank the retrieval blend
                # searches against (pipeline.py:310-312).
                big_npy = index.reconstruct_n(0, index.ntotal)

            self._net_g = net_g
            self._info = info
            self._index = index
            self._big_npy = big_npy
            self._device = device
            self._is_half = is_half
            self._tgt_sr = info.sample_rate
            self._version = info.version
            self._if_f0 = info.if_f0
            self._chunk = _CHUNK_PARAMS_FP16 if is_half else _CHUNK_PARAMS_FP32
            self._identity = identity

            logger.info(
                "Loaded RVC model (%s, sr=%d, if_f0=%d, index=%s) on %s (is_half=%s)",
                info.version,
                info.sample_rate,
                info.if_f0,
                index is not None,
                device,
                is_half,
            )

    def _drop_net(self) -> None:
        if self._net_g is not None:
            device = self._device
            self._net_g = None
            self._index = None
            self._big_npy = None
            if device:
                empty_device_cache(device)

    def unload(self) -> None:
        """Free the loaded model and the shared ContentVec / RMVPE / Crepe backbones."""
        from .pitch import unload_crepe

        with self._lock:
            device = self._device
            self._reset_state()
            unload_rmvpe()
            unload_contentvec()
            unload_crepe()
            if device:
                empty_device_cache(device)

    def convert_file(
        self,
        input_path: str,
        *,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ) -> Tuple[np.ndarray, int]:
        """Convert an audio file to the loaded voice.

        Args:
            input_path: Source audio (any format ``load_audio`` accepts; it is
                resampled to 16 kHz mono).
            f0_up_key: Pitch shift in semitones (f0 checkpoints only).
            f0_method: ``"rmvpe"`` (default) or ``"crepe"``.
            index_rate: FAISS retrieval blend weight in ``[0, 1]`` (ignored when
                no index is loaded).
            rms_mix_rate: Loudness-envelope blend; 1.0 keeps the converted
                envelope, lower values track the source.
            protect: Consonant/breath protection in ``[0, 0.5]``; ``>= 0.5``
                disables it.

        Returns:
            ``(audio, sample_rate)`` — float32 mono at the checkpoint's native
            sample rate. Peak is limited to 0.99; callers resample if needed.
        """
        with self._lock:
            if self._net_g is None:
                raise RuntimeError("No RVC model loaded; call load() first.")

            audio, _ = load_audio(input_path, sample_rate=_SR_16K)
            # load_audio already returns 16 kHz mono; convert_audio then runs the
            # identical pre-normalise + pipeline core (no resample happens here).
            return self.convert_audio(
                audio,
                _SR_16K,
                f0_up_key=f0_up_key,
                f0_method=f0_method,
                index_rate=index_rate,
                rms_mix_rate=rms_mix_rate,
                protect=protect,
            )

    def convert_audio(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        f0_up_key: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
    ) -> Tuple[np.ndarray, int]:
        """Convert an in-memory audio array to the loaded voice.

        The in-memory sibling of :meth:`convert_file`: it shares the exact
        pre-normalisation + ``_run_pipeline`` DSP core, only skipping file I/O.
        Used by the TTS->RVC generation chain, where the base TTS engine already
        produced ``audio`` (mono float32 at its own ``sr``); the array is
        resampled to 16 kHz here before conversion.

        Args:
            audio: Mono (or multi-channel) float array. Multi-channel input is
                averaged to mono.
            sr: Sample rate of ``audio``; resampled to 16 kHz if it differs.
            f0_up_key: Pitch shift in semitones (f0 checkpoints only).
            f0_method: ``"rmvpe"`` (default) or ``"crepe"``.
            index_rate: FAISS retrieval blend weight in ``[0, 1]`` (ignored when
                no index is loaded).
            rms_mix_rate: Loudness-envelope blend; 1.0 keeps the converted
                envelope, lower values track the source.
            protect: Consonant/breath protection in ``[0, 0.5]``; ``>= 0.5``
                disables it.

        Returns:
            ``(audio, sample_rate)`` — float32 mono at the checkpoint's native
            sample rate. Peak is limited to 0.99; callers resample if needed.
        """
        with self._lock:
            if self._net_g is None:
                raise RuntimeError("No RVC model loaded; call load() first.")

            audio = np.asarray(audio, dtype=np.float32)
            if audio.ndim > 1:
                audio = audio.mean(axis=1).astype(np.float32)
            if sr != _SR_16K:
                audio = librosa.resample(
                    audio, orig_sr=sr, target_sr=_SR_16K
                ).astype(np.float32)
            # Pre-normalise to 0.95 peak (modules.py:166-168).
            audio_max = np.abs(audio).max() / 0.95
            if audio_max > 1:
                audio /= audio_max

            audio_opt = self._run_pipeline(
                audio,
                int(f0_up_key),
                f0_method,
                float(index_rate),
                float(rms_mix_rate),
                float(protect),
            )
            return audio_opt, int(self._tgt_sr)

    def _get_f0(
        self, audio_pad: np.ndarray, f0_up_key: int, f0_method: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Dispatch to the f0 estimator (delegates quantization to ``pitch.py``)."""
        if f0_method == "rmvpe":
            return extract_f0_rmvpe(
                audio_pad, f0_up_key, device=self._device, is_half=self._is_half
            )
        if f0_method == "crepe":
            return extract_f0_crepe(audio_pad, f0_up_key, device=self._device)
        raise ValueError(f"Unsupported f0_method {f0_method!r}; use 'rmvpe' or 'crepe'.")

    def _run_pipeline(
        self,
        audio: np.ndarray,
        f0_up_key: int,
        f0_method: str,
        index_rate: float,
        rms_mix_rate: float,
        protect: float,
    ) -> np.ndarray:
        """Port of upstream ``Pipeline.pipeline`` (pipeline.py:281-457)."""
        x_pad, x_query, x_center, x_max = self._chunk
        window = _WINDOW
        sr = _SR_16K
        tgt_sr = self._tgt_sr
        device = self._device

        t_pad = sr * x_pad  # pipeline.py:76
        t_pad_tgt = tgt_sr * x_pad  # pipeline.py:77
        t_pad2 = t_pad * 2  # pipeline.py:78
        t_query = sr * x_query  # pipeline.py:79
        t_center = sr * x_center  # pipeline.py:80
        t_max = sr * x_max  # pipeline.py:81

        index = self._index
        big_npy = self._big_npy

        audio = signal.filtfilt(_HP_B, _HP_A, audio)  # pipeline.py:318
        audio_pad = np.pad(
            audio, (window // 2, window // 2), mode="reflect"
        )  # pipeline.py:319
        opt_ts = []
        if audio_pad.shape[0] > t_max:  # pipeline.py:321
            # Moving |sum| over one window; cut chunks at its local minima
            # (quietest points) to hide boundaries (pipeline.py:322-333).
            audio_sum = np.zeros_like(audio)
            for i in range(window):
                audio_sum += np.abs(audio_pad[i : i - window])
            for t in range(t_center, audio.shape[0], t_center):
                opt_ts.append(
                    t
                    - t_query
                    + np.where(
                        audio_sum[t - t_query : t + t_query]
                        == audio_sum[t - t_query : t + t_query].min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (t_pad, t_pad), mode="reflect")  # pipeline.py:338
        p_len = audio_pad.shape[0] // window  # pipeline.py:339
        sid = torch.tensor(0, device=device).unsqueeze(0).long()  # pipeline.py:351

        pitch = None
        pitchf = None
        if self._if_f0 == 1:  # pipeline.py:353
            pitch, pitchf = self._get_f0(audio_pad, f0_up_key, f0_method)
            pitch = pitch[:p_len]  # pipeline.py:363
            pitchf = pitchf[:p_len]  # pipeline.py:364
            # pipeline.py:365-366 casts pitchf to float32 (the guard there is
            # always true — mps and xpu are mutually exclusive).
            pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=device).unsqueeze(0).long()  # :367
            pitchf = torch.tensor(pitchf, device=device).unsqueeze(0).float()  # :368

        # Chunk loop (pipeline.py:371-441): each vc() output is trimmed by
        # t_pad_tgt on both ends to remove the context padding.
        for t in opt_ts:
            t = t // window * window
            if self._if_f0 == 1:
                audio_opt.append(
                    self._vc(
                        sid,
                        audio_pad[s : t + t_pad2 + window],
                        pitch[:, s // window : (t + t_pad2) // window],
                        pitchf[:, s // window : (t + t_pad2) // window],
                        index,
                        big_npy,
                        index_rate,
                        protect,
                    )[t_pad_tgt:-t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self._vc(
                        sid,
                        audio_pad[s : t + t_pad2 + window],
                        None,
                        None,
                        index,
                        big_npy,
                        index_rate,
                        protect,
                    )[t_pad_tgt:-t_pad_tgt]
                )
            s = t
        if self._if_f0 == 1:
            audio_opt.append(
                self._vc(
                    sid,
                    audio_pad[t:],
                    pitch[:, t // window :] if t is not None else pitch,
                    pitchf[:, t // window :] if t is not None else pitchf,
                    index,
                    big_npy,
                    index_rate,
                    protect,
                )[t_pad_tgt:-t_pad_tgt]
            )
        else:
            audio_opt.append(
                self._vc(
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    index,
                    big_npy,
                    index_rate,
                    protect,
                )[t_pad_tgt:-t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)  # pipeline.py:442
        if rms_mix_rate != 1:  # pipeline.py:443-444
            audio_opt = _change_rms(audio, sr, audio_opt, tgt_sr, rms_mix_rate)
        # pipeline.py:449-453 limits the peak to 0.99 before export; upstream
        # then casts to int16, but Voicebox returns float32 (its audio
        # convention) and lets callers resample.
        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max
        return audio_opt.astype(np.float32)

    def _vc(
        self,
        sid: torch.Tensor,
        audio0: np.ndarray,
        pitch: Optional[torch.Tensor],
        pitchf: Optional[torch.Tensor],
        index,
        big_npy,
        index_rate: float,
        protect: float,
    ) -> np.ndarray:
        """Convert one padded chunk (port of upstream ``Pipeline.vc``, :186-279)."""
        device = self._device
        is_half = self._is_half

        # pipeline.py:201-220 — ContentVec features (with v1 final_proj).
        feats = extract_contentvec_features(
            audio0, self._version, device=device, is_half=is_half
        )

        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()  # pipeline.py:221 (pre-retrieval copy)

        if index is not None and big_npy is not None and index_rate != 0:
            # FAISS retrieval blend (pipeline.py:228-245): weight the 8 nearest
            # trained features by inverse-square L2 distance, blend by index_rate.
            npy = feats[0].cpu().numpy()
            if is_half:
                npy = npy.astype("float32")
            score, ix = index.search(npy, k=8)  # pipeline.py:235
            weight = np.square(1 / score)  # pipeline.py:236
            weight /= weight.sum(axis=1, keepdims=True)  # pipeline.py:237
            npy = np.sum(
                big_npy[ix] * np.expand_dims(weight, axis=2), axis=1
            )  # pipeline.py:238
            if is_half:
                npy = npy.astype("float16")
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(device) * index_rate
                + (1 - index_rate) * feats
            )  # pipeline.py:242-245

        p_len = audio0.shape[0] // _WINDOW  # pipeline.py:253
        # Upstream assumes ContentVec emits exactly 50 Hz frames, so 2x
        # interpolation lands on the synthesizer's 100 Hz grid. The transformers
        # HuBERT path can be shorter/longer by model revision or framing rules;
        # matching the audio-derived frame count keeps conversion duration stable.
        feats = _resize_features_to_frame_count(feats, p_len)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = _resize_features_to_frame_count(feats0, p_len)
        if pitch is not None and pitchf is not None:
            pitch = pitch[:, :p_len]
            pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            # Restore unvoiced/near-silent frames from the pre-retrieval feats
            # so consonants and breaths are not over-smoothed (pipeline.py:260-266).
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)

        p_len = torch.tensor([p_len], device=device).long()  # pipeline.py:267
        with torch.no_grad():
            hasp = pitch is not None and pitchf is not None
            arg = (feats, p_len, pitch, pitchf, sid) if hasp else (feats, p_len, sid)
            audio1 = (self._net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg
        del feats, p_len
        empty_device_cache(device)  # no-op off CUDA/XPU (pipeline.py:274-275)
        return audio1
