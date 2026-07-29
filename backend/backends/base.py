"""
Shared utilities for TTS/STT backend implementations.

Eliminates duplication of cache checking, device detection,
voice prompt combination, and model loading progress tracking.
"""

import logging
import platform
import shutil
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from . import ModelConfig

from ..utils.audio import normalize_audio, load_audio
from ..utils.progress import get_progress_manager
from ..utils.hf_progress import HFProgressTracker, create_hf_progress_callback
from ..utils.tasks import get_task_manager

logger = logging.getLogger(__name__)


def is_model_cached(
    hf_repo: str,
    *,
    weight_extensions: tuple[str, ...] = (".safetensors", ".bin"),
    required_files: Optional[list[str]] = None,
) -> bool:
    """
    Check if a HuggingFace model is fully cached locally.

    Args:
        hf_repo: HuggingFace repo ID (e.g. "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        weight_extensions: File extensions that count as model weights.
        required_files: If set, check that these specific filenames exist
                        in snapshots instead of checking by extension.

    Returns:
        True if model is fully cached, False if missing or incomplete.
    """
    try:
        from huggingface_hub import constants as hf_constants

        repo_cache = Path(hf_constants.HF_HUB_CACHE) / ("models--" + hf_repo.replace("/", "--"))

        if not repo_cache.exists():
            return False

        # Incomplete blobs mean a download is still in progress
        blobs_dir = repo_cache / "blobs"
        if blobs_dir.exists() and any(blobs_dir.glob("*.incomplete")):
            logger.debug(f"Found .incomplete files for {hf_repo}")
            return False

        snapshots_dir = repo_cache / "snapshots"
        if not snapshots_dir.exists():
            return False

        if required_files:
            # Check that every required filename exists somewhere in snapshots
            for fname in required_files:
                if not any(snapshots_dir.rglob(fname)):
                    return False
            return True

        # Check that at least one weight file exists
        for ext in weight_extensions:
            if any(snapshots_dir.rglob(f"*{ext}")):
                return True

        logger.debug(f"No model weights found for {hf_repo}")
        return False

    except Exception as e:
        logger.warning(f"Error checking cache for {hf_repo}: {e}")
        return False


# One-shot relocation of direct-download weights out of the HF cache root is
# serialized on this lock so concurrent callers (Models-tab status polls, the
# download flow, the pitch loader) can't race the move.
_legacy_download_migration_lock = threading.Lock()


def get_direct_download_path(config: "ModelConfig") -> Optional[Path]:
    """Local path for a non-HF registry model (one with ``download_url``).

    Returns ``None`` for HuggingFace-hosted configs. Direct-download models live
    under the app's own model directory (``config.get_models_dir()``) at
    ``<model_name>/<file_name>`` — deliberately OUTSIDE the HuggingFace cache
    root, so first-party downloads (only ``crepe-full`` today) neither pollute
    that cache nor get swept up by the cache-dir migration flow. A copy left at
    the legacy in-HF-cache location by an older build is migrated across on first
    access (one-shot; see ``_migrate_legacy_direct_download``).
    """
    if not getattr(config, "download_url", None) or not getattr(config, "file_name", None):
        return None
    from ..config import get_models_dir

    dest = get_models_dir() / config.model_name / config.file_name
    _migrate_legacy_direct_download(config, dest)
    return dest


def _migrate_legacy_direct_download(config: "ModelConfig", dest: Path) -> None:
    """One-shot move of a direct-download model into the app's model dir.

    Older builds stored direct-download weights inside the HuggingFace cache root
    at ``HF_HUB_CACHE/<model_name>/<file_name>`` (a workaround so the Models tab's
    single cache-dir surface covered them). They now live under the app's own
    model dir. If the file is still only at the legacy location, move it there so
    an already-downloaded model is never re-fetched. No-ops once ``dest`` exists.
    """
    if dest.exists():
        return
    from huggingface_hub import constants as hf_constants

    legacy = Path(hf_constants.HF_HUB_CACHE) / config.model_name / config.file_name
    if not legacy.exists() or legacy.resolve() == dest.resolve():
        return

    with _legacy_download_migration_lock:
        # Re-check under the lock: another caller may have migrated already.
        if dest.exists() or not legacy.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            legacy.replace(dest)  # atomic rename within one filesystem
        except OSError:
            # Cross-filesystem move: copy to a temp beside dest then swap in
            # atomically, so a crash mid-copy never leaves a partial dest.
            tmp = dest.parent / f".{dest.name}.migrating"
            tmp.unlink(missing_ok=True)
            shutil.copy2(str(legacy), str(tmp))
            tmp.replace(dest)
            legacy.unlink(missing_ok=True)
        logger.info(
            "Migrated %s weights out of the HF cache: %s -> %s",
            config.model_name,
            legacy,
            dest,
        )
        # Best-effort cleanup of the now-empty legacy model dir.
        try:
            legacy.parent.rmdir()
        except OSError:
            pass


def is_direct_download_cached(config: "ModelConfig") -> bool:
    """Whether a direct-download model's verified file already exists on disk."""
    path = get_direct_download_path(config)
    return path is not None and path.exists()


def get_crepe_model_path() -> Optional[Path]:
    """Single source of truth for the downloaded Crepe ``full.pth`` location.

    Resolved from the ``crepe-full`` registry entry so the download flow, the
    Models-tab cache check, and the pitch loader all agree on one path. Returns
    ``None`` only if the registry entry is somehow missing.
    """
    from . import get_model_config

    cfg = get_model_config("crepe-full")
    return get_direct_download_path(cfg) if cfg else None


async def download_direct_model(config: "ModelConfig") -> Path:
    """Download + sha256-verify a non-HF registry model, reporting progress.

    Streams ``config.download_url`` to ``get_direct_download_path(config)`` via a
    ``.incomplete`` temp file, checks ``config.sha256`` before committing with an
    atomic rename, and reports progress under ``config.model_name`` so the Models
    tab SSE tracks it like any HuggingFace download. A hash mismatch (or any
    error) deletes the partial file and raises — an unverified file is never kept.
    """
    import hashlib

    import httpx

    dest = get_direct_download_path(config)
    if dest is None:
        raise ValueError(f"{config.model_name} is not a direct-download model")
    if dest.exists():
        return dest

    progress_manager = get_progress_manager()
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.parent / f".{dest.name}.incomplete"
    tmp_path.unlink(missing_ok=True)

    hasher = hashlib.sha256()
    downloaded = 0
    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
            async with client.stream("GET", config.download_url) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0)) or config.size_mb * 1024 * 1024
                with open(tmp_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024 * 1024):
                        f.write(chunk)
                        hasher.update(chunk)
                        downloaded += len(chunk)
                        progress_manager.update_progress(
                            config.model_name,
                            downloaded,
                            total,
                            filename=f"Downloading {config.file_name}",
                            status="downloading",
                        )
        if config.sha256:
            actual = hasher.hexdigest()
            if actual != config.sha256:
                raise ValueError(
                    f"{config.model_name} integrity check failed: "
                    f"expected {config.sha256[:16]}..., got {actual[:16]}..."
                )
        tmp_path.replace(dest)
    except Exception as e:
        progress_manager.mark_error(config.model_name, str(e))
        raise
    else:
        progress_manager.mark_complete(config.model_name)
    finally:
        tmp_path.unlink(missing_ok=True)

    return dest


def get_torch_device(
    *,
    allow_xpu: bool = False,
    allow_directml: bool = False,
    allow_mps: bool = False,
    force_cpu_on_mac: bool = False,
) -> str:
    """
    Detect the best available torch device.

    Args:
        allow_xpu: Check for Intel XPU (IPEX) support.
        allow_directml: Check for DirectML (Windows) support.
        allow_mps: Allow MPS (Apple Silicon). If False, MPS falls back to CPU.
        force_cpu_on_mac: Force CPU on macOS regardless of GPU availability.
    """
    if force_cpu_on_mac and platform.system() == "Darwin":
        return "cpu"

    import torch

    if torch.cuda.is_available():
        return "cuda"

    if allow_xpu:
        try:
            import intel_extension_for_pytorch  # noqa: F401

            if hasattr(torch, "xpu") and torch.xpu.is_available():
                return "xpu"
        except ImportError:
            pass

    if allow_directml:
        try:
            import torch_directml

            if torch_directml.device_count() > 0:
                return torch_directml.device(0)
        except ImportError:
            pass

    if allow_mps:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

    return "cpu"


def check_cuda_compatibility() -> tuple[bool, str | None]:
    """Check if the installed PyTorch supports the current GPU's compute capability.

    Returns:
        (compatible, warning_message) — compatible is True if OK or no CUDA GPU,
        warning_message is a human-readable string if there's a problem.
    """
    import torch

    if not torch.cuda.is_available():
        return True, None

    # ROCm/HIP uses the cuda frontend but has different architecture names (gfx*).
    # Skip NVIDIA-specific compute capability checks on AMD hardware.
    if hasattr(torch.version, "hip") and torch.version.hip:
        return True, None

    major, minor = torch.cuda.get_device_capability(0)
    capability = f"{major}.{minor}"
    device_name = torch.cuda.get_device_name(0)
    sm_tag = f"sm_{major}{minor}"

    # torch.cuda._get_arch_list() returns the SM architectures this build
    # was compiled for (e.g. ["sm_50", "sm_60", ..., "sm_90"]).
    try:
        arch_list = torch.cuda._get_arch_list()
        if arch_list:
            # Check for both sm_XX and compute_XX (JIT-compiled) entries
            compute_tag = f"compute_{major}{minor}"
            if sm_tag not in arch_list and compute_tag not in arch_list:
                return False, (
                    f"{device_name} (compute capability {capability} / {sm_tag}) "
                    f"is not supported by this PyTorch build. "
                    f"Supported architectures: {', '.join(arch_list)}. "
                    f"Install PyTorch nightly (cu128) for newer GPU support: "
                    f"pip install torch --index-url https://download.pytorch.org/whl/nightly/cu128"
                )
    except AttributeError:
        pass

    return True, None


def empty_device_cache(device: str) -> None:
    """
    Free cached memory on the given device (CUDA, XPU, or MPS).

    Backends should call this after unloading models so VRAM is returned
    to the OS.
    """
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
    elif device == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def manual_seed(seed: int, device: str) -> None:
    """
    Set the random seed on both CPU and the active accelerator.

    Covers CUDA and Intel XPU so that generation is reproducible
    regardless of which GPU backend is in use.
    """
    import torch

    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    elif device == "xpu" and hasattr(torch, "xpu"):
        torch.xpu.manual_seed(seed)


async def combine_voice_prompts(
    audio_paths: List[str],
    reference_texts: List[str],
    *,
    sample_rate: Optional[int] = None,
) -> Tuple[np.ndarray, str]:
    """
    Combine multiple reference audio samples into one.

    Loads each audio file, normalizes, concatenates, and joins texts.

    Args:
        audio_paths: Paths to reference audio files.
        reference_texts: Corresponding transcripts.
        sample_rate: If set, resample audio to this rate during loading.
    """
    combined_audio = []

    for path in audio_paths:
        kwargs = {"sample_rate": sample_rate} if sample_rate else {}
        audio, _sr = load_audio(path, **kwargs)
        audio = normalize_audio(audio)
        combined_audio.append(audio)

    mixed = np.concatenate(combined_audio)
    mixed = normalize_audio(mixed)
    combined_text = " ".join(reference_texts)

    return mixed, combined_text


@contextmanager
def model_load_progress(
    model_name: str,
    is_cached: bool,
    filter_non_downloads: Optional[bool] = None,
):
    """
    Context manager for model loading with HF download progress tracking.

    Handles the tqdm patching, progress_manager/task_manager lifecycle,
    and error reporting that every backend duplicates.

    Args:
        model_name: Progress tracking key (e.g. "qwen-tts-1.7B", "whisper-base").
        is_cached: Whether the model is already downloaded.
        filter_non_downloads: Whether to filter non-download tqdm bars.
                              Defaults to `is_cached`.

    Yields:
        The tracker context (already entered). The caller loads the model
        inside the `with` block. The tqdm patch is torn down on exit.

    Usage:
        with model_load_progress("qwen-tts-1.7B", is_cached) as ctx:
            self.model = SomeModel.from_pretrained(...)
    """
    if filter_non_downloads is None:
        filter_non_downloads = is_cached

    progress_manager = get_progress_manager()
    task_manager = get_task_manager()

    progress_callback = create_hf_progress_callback(model_name, progress_manager)
    tracker = HFProgressTracker(progress_callback, filter_non_downloads=filter_non_downloads)

    tracker_context = tracker.patch_download()
    tracker_context.__enter__()

    if not is_cached:
        task_manager.start_download(model_name)
        progress_manager.update_progress(
            model_name=model_name,
            current=0,
            total=0,
            filename="Connecting to HuggingFace...",
            status="downloading",
        )

    try:
        yield tracker_context
    except Exception as e:
        # Report error to both managers
        progress_manager.mark_error(model_name, str(e))
        task_manager.error_download(model_name, str(e))
        raise
    else:
        # Only mark complete if we were tracking a download
        if not is_cached:
            progress_manager.mark_complete(model_name)
            task_manager.complete_download(model_name)
    finally:
        tracker_context.__exit__(None, None, None)


def patch_chatterbox_f32(model) -> None:
    """
    Patch float64 -> float32 dtype mismatches in upstream chatterbox.

    librosa.load returns float64 numpy arrays. Multiple upstream code paths
    convert these to torch tensors via torch.from_numpy() without casting,
    then matmul against float32 model weights. This patches the two known
    entry points:

    1. S3Tokenizer.log_mel_spectrogram — audio tensor hits _mel_filters (f32)
    2. VoiceEncoder.forward — float64 mel spectrograms hit LSTM weights (f32)
    """
    import types

    # Patch S3Tokenizer
    _tokzr = model.s3gen.tokenizer
    _orig_log_mel = _tokzr.log_mel_spectrogram.__func__

    def _f32_log_mel(self_tokzr, audio, padding=0):
        import torch as _torch

        if _torch.is_tensor(audio):
            audio = audio.float()
        return _orig_log_mel(self_tokzr, audio, padding)

    _tokzr.log_mel_spectrogram = types.MethodType(_f32_log_mel, _tokzr)

    # Patch VoiceEncoder
    _ve = model.ve
    _orig_ve_forward = _ve.forward.__func__

    def _f32_ve_forward(self_ve, mels):
        return _orig_ve_forward(self_ve, mels.float())

    _ve.forward = types.MethodType(_f32_ve_forward, _ve)
