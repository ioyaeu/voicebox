"""Fundamental-frequency (f0) extraction for RVC voice conversion.

Provides the two pitch estimators RVC uses at inference time:

* ``extract_f0_rmvpe`` — the default, backed by a vendored copy of the RMVPE
  network (RVC-Project/Retrieval-based-Voice-Conversion-WebUI, MIT; the license
  text lives in ``LICENSE_RVC.md``). Model weights (``rmvpe.pt``) download from
  the ``rmvpe`` registry entry (``lj1995/VoiceConversionWebUI``).
* ``extract_f0_crepe`` — the optional alternative, backed by ``torchcrepe``.

Both return the ``(f0_coarse, f0)`` pair the synthesizer expects, where
``f0_coarse`` is the pitch quantized to RVC's 1-based 255-bin mel scale. The
quantization math is copied verbatim from upstream ``infer/modules/vc/pipeline.py``
``Pipeline.get_f0`` — an off-by-one there yields a monotone robot voice that runs
without error, so every constant below is annotated with its upstream line.

The estimators return their natural frame count (the pipeline slices to ``p_len``,
matching upstream, which slices after ``get_f0`` returns rather than inside it).
"""

import logging
import threading
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as _librosa_mel

from ..base import (
    empty_device_cache,
    get_crepe_model_path,
    get_torch_device,
    is_model_cached,
    model_load_progress,
)

logger = logging.getLogger(__name__)

_RMVPE_REPO = "lj1995/VoiceConversionWebUI"
_RMVPE_FILENAME = "rmvpe.pt"


# ---------------------------------------------------------------------------
# f0 quantization — verbatim from upstream Pipeline.get_f0
# ---------------------------------------------------------------------------

# pipeline.py:96-99 — the mel-scale bounds RVC quantizes against.
_F0_MIN = 50
_F0_MAX = 1100
_F0_MEL_MIN = 1127 * np.log(1 + _F0_MIN / 700)
_F0_MEL_MAX = 1127 * np.log(1 + _F0_MAX / 700)


def _f0_to_coarse(f0: np.ndarray) -> np.ndarray:
    """Quantize continuous f0 (Hz) to RVC's 1-based 255-bin mel scale.

    Byte-for-byte port of upstream ``Pipeline.get_f0`` (pipeline.py:177-183).
    The factor ``254`` is ``f0_bin - 2`` with ``f0_bin = 256`` (the training-time
    bin count in ``infer_pack/models.py``); the ``+ 1`` makes the scale 1-based,
    so voiced frames land in ``[1, 255]`` and 0 is reserved for unvoiced.
    """
    f0_mel = 1127 * np.log(1 + f0 / 700)  # pipeline.py:177
    # pipeline.py:178-180
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - _F0_MEL_MIN) * 254 / (
        _F0_MEL_MAX - _F0_MEL_MIN
    ) + 1
    f0_mel[f0_mel <= 1] = 1  # pipeline.py:181
    f0_mel[f0_mel > 255] = 255  # pipeline.py:182
    f0_coarse = np.rint(f0_mel).astype(np.int32)  # pipeline.py:183
    return f0_coarse


# ---------------------------------------------------------------------------
# Vendored RMVPE network (RVC-Project ... /infer/lib/rmvpe.py, MIT)
#
# Inference-only subset: the DirectML ("privateuseone") STFT/ONNX path and the
# TorchScript (jit) path are dropped — this build targets CPU / CUDA / MPS only.
# Tensor shapes and module names are kept identical to upstream so rmvpe.pt loads
# with zero missing keys.
# ---------------------------------------------------------------------------


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, (1, 1))

    def forward(self, x: torch.Tensor):
        if not hasattr(self, "shortcut"):
            return self.conv(x) + x
        else:
            return self.conv(x) + self.shortcut(x)


class ResEncoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01
    ):
        super(ResEncoderBlock, self).__init__()
        self.n_blocks = n_blocks
        self.conv = nn.ModuleList()
        self.conv.append(ConvBlockRes(in_channels, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv.append(ConvBlockRes(out_channels, out_channels, momentum))
        self.kernel_size = kernel_size
        if self.kernel_size is not None:
            self.pool = nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, x):
        for i, conv in enumerate(self.conv):
            x = conv(x)
        if self.kernel_size is not None:
            return x, self.pool(x)
        else:
            return x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.n_encoders = n_encoders
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for i in range(self.n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x: torch.Tensor):
        concat_tensors: List[torch.Tensor] = []
        x = self.bn(x)
        for i, layer in enumerate(self.layers):
            t, x = layer(x)
            concat_tensors.append(t)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.n_inters = n_inters
        self.layers = nn.ModuleList()
        self.layers.append(
            ResEncoderBlock(in_channels, out_channels, None, n_blocks, momentum)
        )
        for i in range(self.n_inters - 1):
            self.layers.append(
                ResEncoderBlock(out_channels, out_channels, None, n_blocks, momentum)
            )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.n_blocks = n_blocks
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList()
        self.conv2.append(ConvBlockRes(out_channels * 2, out_channels, momentum))
        for i in range(n_blocks - 1):
            self.conv2.append(ConvBlockRes(out_channels, out_channels, momentum))

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for i, conv2 in enumerate(self.conv2):
            x = conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        self.n_decoders = n_decoders
        for i in range(self.n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x: torch.Tensor, concat_tensors: List[torch.Tensor]):
        for i, layer in enumerate(self.layers):
            x = layer(x, concat_tensors[-1 - i])
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        x = self.decoder(x, concat_tensors)
        return x


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E, self).__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        # rmvpe.pt is trained with n_gru=1; upstream's n_gru==0 FC head (which
        # references undefined nn.N_MELS/nn.N_CLASS) is dead for RVC and omitted.
        self.fc = nn.Sequential(
            BiGRU(3 * 128, 256, n_gru),
            nn.Linear(512, 360),
            nn.Dropout(0.25),
            nn.Sigmoid(),
        )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        is_half,
        n_mel_channels,
        sampling_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super().__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = _librosa_mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sampling_rate = sampling_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = str(keyshift) + "_" + str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )
        fft = torch.stft(
            audio,
            n_fft=n_fft_new,
            hop_length=hop_length_new,
            win_length=win_length_new,
            window=self.hann_window[keyshift_key],
            center=center,
            return_complex=True,
        )
        magnitude = torch.sqrt(fft.real.pow(2) + fft.imag.pow(2))
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half == True:
            mel_output = mel_output.half()
        log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        return log_mel_spec


class RMVPE:
    """Inference wrapper for the vendored RMVPE pitch estimator.

    Mirrors upstream ``infer/lib/rmvpe.py::RMVPE`` for CPU / CUDA / MPS. The
    weights file is loaded with ``weights_only=True`` — although ``rmvpe.pt`` is
    a first-party model (not an untrusted upload), this keeps every ``torch.load``
    in the RVC backend on the safe path.
    """

    def __init__(self, model_path: str, is_half: bool, device: Optional[str] = None):
        self.resample_kernel = {}
        self.is_half = is_half
        if device is None:
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mel_extractor = MelSpectrogram(
            is_half, 128, 16000, 1024, 160, None, 30, 8000
        ).to(device)
        if str(self.device) == "cuda":
            self.device = torch.device("cuda:0")
        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()
        if is_half:
            model = model.half()
        else:
            model = model.float()
        self.model = model.to(device)
        # cents_mapping maps each of the 360 output bins to a pitch in cents;
        # padded by 4 on each side to support the ±4-bin local weighted average.
        cents_mapping = 20 * np.arange(360) + 1997.3794084376191
        self.cents_mapping = np.pad(cents_mapping, (4, 4))  # 368

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            # DeepUnet halves the frame axis 5×, so pad up to a multiple of 32.
            n_pad = 32 * ((n_frames - 1) // 32 + 1) - n_frames
            if n_pad > 0:
                mel = F.pad(mel, (0, n_pad), mode="constant")
            mel = mel.half() if self.is_half else mel.float()
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        mel = self.mel_extractor(
            audio.float().to(self.device).unsqueeze(0), center=True
        )
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()
        if self.is_half == True:
            hidden = hidden.astype("float32")
        f0 = self.decode(hidden, thred=thred)
        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=1)  # frame-wise argmax bin index
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience = np.array(todo_salience)  # (n_frames, 9)
        todo_cents_mapping = np.array(todo_cents_mapping)  # (n_frames, 9)
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        devided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        devided[maxx <= thred] = 0
        return devided


# ---------------------------------------------------------------------------
# Module-level RMVPE cache (a shared backbone, not a per-voice model)
# ---------------------------------------------------------------------------

_rmvpe_lock = threading.Lock()
_rmvpe_model: Optional[RMVPE] = None
_rmvpe_key: Optional[Tuple[str, bool]] = None


def _get_rmvpe(*, device: Optional[str] = None, is_half: bool = False) -> RMVPE:
    """Return a cached RMVPE, downloading ``rmvpe.pt`` on first use.

    Cached per ``(device, is_half)`` so repeated conversions do not reload the
    ~180 MB network. Download goes through ``model_load_progress`` /
    ``is_model_cached`` like every other backend so the Models tab sees progress.
    """
    global _rmvpe_model, _rmvpe_key
    if device is None:
        device = get_torch_device(allow_mps=True)
    device = str(device)
    key = (device, bool(is_half))
    with _rmvpe_lock:
        if _rmvpe_model is not None and _rmvpe_key == key:
            return _rmvpe_model

        is_cached = is_model_cached(_RMVPE_REPO, required_files=[_RMVPE_FILENAME])
        with model_load_progress("rmvpe", is_cached):
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(repo_id=_RMVPE_REPO, filename=_RMVPE_FILENAME)

        logger.info("Loading RMVPE on %s (is_half=%s)", device, is_half)
        _rmvpe_model = RMVPE(model_path, is_half=is_half, device=device)
        _rmvpe_key = key
        return _rmvpe_model


def download_rmvpe() -> None:
    """Download ``rmvpe.pt`` into the HF cache without building the network.

    Used by the Models tab download flow (``get_model_load_func``): fetches the
    single weights file via ``hf_hub_download`` so
    ``is_model_cached(_RMVPE_REPO, required_files=[_RMVPE_FILENAME])`` reports it
    downloaded, without instantiating the torch model. Progress is reported
    through ``model_load_progress`` / ``is_model_cached`` under the ``"rmvpe"``
    key, exactly like ``_get_rmvpe``, so the Models tab observes it.
    """
    from huggingface_hub import hf_hub_download

    is_cached = is_model_cached(_RMVPE_REPO, required_files=[_RMVPE_FILENAME])
    with model_load_progress("rmvpe", is_cached):
        hf_hub_download(repo_id=_RMVPE_REPO, filename=_RMVPE_FILENAME)


def unload_rmvpe() -> None:
    """Drop the cached RMVPE and free its device memory."""
    global _rmvpe_model, _rmvpe_key
    with _rmvpe_lock:
        device = _rmvpe_key[0] if _rmvpe_key is not None else None
        _rmvpe_model = None
        _rmvpe_key = None
    if device:
        empty_device_cache(device)


# ---------------------------------------------------------------------------
# Crepe loader — loads OUR registry-downloaded weights, not the package assets
# ---------------------------------------------------------------------------

_crepe_lock = threading.Lock()
_crepe_key: Optional[Tuple[str, str]] = None  # (device, capacity)

# torchcrepe is imported lazily and its version is upper-bounded in
# backend/requirements.txt. This flag makes the API-surface assertion below run
# only once per process (it's checked under ``_crepe_lock``).
_torchcrepe_api_verified = False


def _assert_torchcrepe_api(torchcrepe) -> None:
    """Fail loudly if the torchcrepe internals our crepe loader relies on vanish.

    ``_ensure_crepe_loaded`` short-circuits torchcrepe's bundled-asset load by
    setting ``torchcrepe.infer.model`` / ``.capacity`` (attributes on the
    module-level ``infer`` function that ``torchcrepe.predict`` calls), and the
    extractor uses ``torchcrepe.Crepe`` / ``predict`` / ``filter.median|mean``.
    torchcrepe is pinned (``requirements.txt``: ``torchcrepe>=0.0.23,<0.1.0``);
    should a future version still slip in and rename/remove any of these, raise a
    clear, actionable error here instead of a cryptic ``AttributeError`` deep in
    inference or a silent wrong-weights load.
    """
    global _torchcrepe_api_verified
    if _torchcrepe_api_verified:
        return

    missing: List[str] = []
    # infer() is the function whose .model/.capacity attributes we set so its
    # "load weights if needed" branch (torchcrepe.load.model) is skipped.
    if not callable(getattr(torchcrepe, "infer", None)):
        missing.append("torchcrepe.infer (callable)")
    load = getattr(torchcrepe, "load", None)
    if load is None or not hasattr(load, "model"):
        missing.append("torchcrepe.load.model")
    if not callable(getattr(torchcrepe, "Crepe", None)):
        missing.append("torchcrepe.Crepe")
    if not callable(getattr(torchcrepe, "predict", None)):
        missing.append("torchcrepe.predict")
    filt = getattr(torchcrepe, "filter", None)
    if filt is None or not hasattr(filt, "median") or not hasattr(filt, "mean"):
        missing.append("torchcrepe.filter.median/.mean")

    if missing:
        try:
            import importlib.metadata as _md

            version = _md.version("torchcrepe")
        except Exception:
            version = "unknown"
        raise RuntimeError(
            f"Installed torchcrepe {version} is missing internals the RVC crepe "
            f"f0 loader depends on: {', '.join(missing)}. This backend pins "
            f"torchcrepe to a tested range (see backend/requirements.txt); a "
            f"newer/older torchcrepe changed its API. Reinstall the pinned "
            f"version to use f0_method='crepe' (rmvpe, the default, is unaffected)."
        )
    _torchcrepe_api_verified = True


def _ensure_crepe_loaded(device: str, capacity: str) -> None:
    """Populate ``torchcrepe.infer`` from the registry-downloaded weights.

    ``torchcrepe.predict`` lazily ``torch.load``s ``torchcrepe/assets/<capacity>.pth``,
    which the frozen build deliberately excludes (~85 MB, shipped as the on-demand
    ``crepe-full`` registry model instead). We load that registry file ourselves
    and set ``torchcrepe.infer.model`` / ``infer.capacity`` so ``infer``'s skip
    condition (``infer.capacity == model``) short-circuits the bundled-asset load.

    Cached per ``(device, capacity)`` like the RMVPE loader. Weights load with
    ``weights_only=True``. Raises a clear, user-actionable ``ValueError`` when the
    model has not been downloaded — never a raw ``FileNotFoundError`` from
    torchcrepe.
    """
    import torchcrepe

    global _crepe_key
    key = (device, capacity)
    with _crepe_lock:
        _assert_torchcrepe_api(torchcrepe)
        already = (
            _crepe_key == key
            and getattr(torchcrepe.infer, "model", None) is not None
            and getattr(torchcrepe.infer, "capacity", None) == capacity
        )
        if already:
            return

        model_path = get_crepe_model_path()
        if model_path is None or not model_path.exists():
            raise ValueError(
                'The Crepe pitch model is not downloaded. Download '
                '"Crepe (pitch, full)" in the Models tab to use f0_method="crepe".'
            )

        logger.info("Loading Crepe (%s) on %s from %s", capacity, device, model_path)
        model = torchcrepe.Crepe(capacity)
        state = torch.load(str(model_path), map_location=device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(device).eval()
        torchcrepe.infer.model = model
        torchcrepe.infer.capacity = capacity
        _crepe_key = key


def unload_crepe() -> None:
    """Drop the preloaded Crepe model and free its device memory."""
    import torchcrepe

    global _crepe_key
    with _crepe_lock:
        device = _crepe_key[0] if _crepe_key is not None else None
        _crepe_key = None
        if hasattr(torchcrepe.infer, "model"):
            del torchcrepe.infer.model
        if hasattr(torchcrepe.infer, "capacity"):
            del torchcrepe.infer.capacity
    if device:
        empty_device_cache(device)


# ---------------------------------------------------------------------------
# Public extractors
# ---------------------------------------------------------------------------


def extract_f0_rmvpe(
    audio_16k: np.ndarray,
    f0_up_key: int = 0,
    *,
    device: Optional[str] = None,
    is_half: bool = False,
    thred: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate f0 with RMVPE (RVC's default estimator).

    Args:
        audio_16k: Mono 16 kHz float audio (the rate ContentVec/RMVPE expect).
        f0_up_key: Pitch shift in semitones, applied before quantization.
        device: Torch device; defaults to ``get_torch_device(allow_mps=True)``.
        is_half: fp16 inference (CUDA only; keep False on CPU/MPS).
        thred: RMVPE salience threshold (upstream default 0.03).

    Returns:
        ``(f0_coarse, f0)`` — ``f0_coarse`` is int32 on the 1-based 255-bin mel
        scale (the synthesizer's pitch input); ``f0`` is the pitch-shifted
        continuous Hz (``pitchf``). Length is RMVPE's natural frame count; the
        pipeline slices both to ``p_len``.
    """
    model = _get_rmvpe(device=device, is_half=is_half)
    f0 = model.infer_from_audio(np.asarray(audio_16k), thred=thred)
    f0 *= 2 ** (f0_up_key / 12)  # pipeline.py:161 — shift before quantization
    f0_coarse = _f0_to_coarse(f0)
    return f0_coarse, f0


def extract_f0_crepe(
    audio_16k: np.ndarray,
    f0_up_key: int = 0,
    *,
    device: Optional[str] = None,
    crepe_model: str = "full",
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate f0 with torchcrepe (RVC's optional alternative estimator).

    Mirrors upstream ``Pipeline.get_f0``'s crepe branch (pipeline.py:121-141):
    16 kHz input, hop length 160, ``f0_min=50``/``f0_max=1100``, median-filtered
    periodicity and mean-filtered pitch, unvoiced frames (periodicity < 0.1)
    zeroed. Returns the same ``(f0_coarse, f0)`` contract as ``extract_f0_rmvpe``.
    """
    import torchcrepe  # declared in backend/requirements.txt; imported lazily

    if device is None:
        device = get_torch_device(allow_mps=True)
    device = str(device)

    # Load weights from the registry file (frozen build has no bundled assets).
    _ensure_crepe_loaded(device, crepe_model)

    audio = torch.tensor(np.copy(audio_16k))[None].float()  # pipeline.py:126
    f0, pd = torchcrepe.predict(
        audio,
        16000,  # self.sr — pipeline.py:127-137
        160,  # self.window (hop length)
        _F0_MIN,
        _F0_MAX,
        crepe_model,
        batch_size=batch_size,
        device=device,
        return_periodicity=True,
    )
    pd = torchcrepe.filter.median(pd, 3)  # pipeline.py:138
    f0 = torchcrepe.filter.mean(f0, 3)  # pipeline.py:139
    f0[pd < 0.1] = 0  # pipeline.py:140
    f0 = f0[0].cpu().numpy()  # pipeline.py:141

    f0 *= 2 ** (f0_up_key / 12)  # pipeline.py:161 — shift before quantization
    f0_coarse = _f0_to_coarse(f0)
    return f0_coarse, f0
