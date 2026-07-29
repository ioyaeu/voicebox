"""ContentVec content-feature extraction for RVC voice conversion.

Ports upstream RVC's per-segment feature step (``infer/modules/vc/pipeline.py``
``Pipeline.vc`` lines 201-220) onto the ``transformers`` HuBERT API, since
Voicebox does not depend on ``fairseq``. Features come from the ``contentvec``
registry entry (``lengyue233/content-vec-best``), whose ``config.json`` declares
the architecture ``HubertModelWithFinalProj``: a ``HubertModel`` plus a trained
``Linear(hidden_size -> classifier_proj_size)`` == ``Linear(768 -> 256)`` final
projection. v1 checkpoints consume the layer-9 encoder output projected to 256
dims through that projection; v2 checkpoints consume the layer-12 output at 768
dims raw.

The ``(layer index, final_proj)`` mapping is not guessed. transformers
``hidden_states[N]`` is the output after ``N`` transformer layers
(``hidden_states[0]`` is the pre-encoder feature projection, and
``hidden_states[12] == last_hidden_state``). fairseq HuBERT's
``extract_features(..., output_layer=N)`` passes ``layer=N-1`` to its encoder,
which returns after the ``N``-th layer — so upstream's ``output_layer=9``/``12``
line up one-for-one with ``hidden_states[9]``/``[12]`` here. The final
projection is the trained ``Linear(768 -> 256)`` shipped in the checkpoint (the
subclass loads it with zero missing/unexpected keys), matching upstream's
``model.final_proj(...)`` for v1.
"""

import logging
import threading
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from transformers import HubertModel

from ..base import (
    empty_device_cache,
    get_torch_device,
    is_model_cached,
    model_load_progress,
)

logger = logging.getLogger(__name__)

_CONTENTVEC_REPO = "lengyue233/content-vec-best"


class HubertModelWithFinalProj(HubertModel):
    """``HubertModel`` plus the trained final projection of content-vec-best.

    The checkpoint's config names ``HubertModelWithFinalProj`` as its
    architecture and ships ``final_proj`` weights; a plain ``HubertModel`` would
    report them as unexpected and drop them, leaving v1's 768->256 projection
    untrained (silently wrong features). The subclass is copied from the repo's
    README / ``convert.py`` so the projection loads.
    """

    def __init__(self, config):
        super().__init__(config)
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)


# Module-level cache: ContentVec is a shared backbone (not a per-voice model),
# so it is loaded once and reused across conversions, keyed by (device, is_half).
_contentvec_lock = threading.Lock()
_contentvec_model: Optional[HubertModelWithFinalProj] = None
_contentvec_key: Optional[Tuple[str, bool]] = None


def _get_contentvec(
    *, device: Optional[str] = None, is_half: bool = False
) -> HubertModelWithFinalProj:
    """Return a cached ContentVec model, downloading it on first use.

    Download goes through ``model_load_progress`` / ``is_model_cached`` from
    ``backends/base.py`` so the Models tab observes progress, exactly like the
    RMVPE loader in ``pitch.py`` and every other backend.
    """
    global _contentvec_model, _contentvec_key
    if device is None:
        device = get_torch_device(allow_mps=True)
    device = str(device)
    key = (device, bool(is_half))
    with _contentvec_lock:
        if _contentvec_model is not None and _contentvec_key == key:
            return _contentvec_model

        is_cached = is_model_cached(_CONTENTVEC_REPO)
        with model_load_progress("contentvec", is_cached):
            model = HubertModelWithFinalProj.from_pretrained(_CONTENTVEC_REPO)

        model = model.to(device)
        model = model.half() if is_half else model.float()
        model.eval()
        logger.info("Loading ContentVec on %s (is_half=%s)", device, is_half)
        _contentvec_model = model
        _contentvec_key = key
        return model


def download_contentvec() -> None:
    """Download the ContentVec repo into the HF cache without loading it.

    Used by the Models tab download flow (``get_model_load_func``): fetching the
    ~360 MB repo via ``snapshot_download`` populates the cache so
    ``is_model_cached(_CONTENTVEC_REPO)`` reports it downloaded, without paying
    the cost of instantiating the torch model. Progress is reported through
    ``model_load_progress`` / ``is_model_cached`` under the ``"contentvec"`` key,
    exactly like ``_get_contentvec``, so the Models tab observes it.
    """
    from huggingface_hub import snapshot_download

    is_cached = is_model_cached(_CONTENTVEC_REPO)
    with model_load_progress("contentvec", is_cached):
        snapshot_download(_CONTENTVEC_REPO)


def unload_contentvec() -> None:
    """Drop the cached ContentVec model and free its device memory."""
    global _contentvec_model, _contentvec_key
    with _contentvec_lock:
        device = _contentvec_key[0] if _contentvec_key is not None else None
        _contentvec_model = None
        _contentvec_key = None
    if device:
        empty_device_cache(device)


def extract_contentvec_features(
    audio0: np.ndarray,
    version: str,
    *,
    device: Optional[str] = None,
    is_half: bool = False,
) -> torch.Tensor:
    """Extract ContentVec content features for one 16 kHz audio segment.

    Port of upstream ``Pipeline.vc`` lines 201-220: the raw waveform is fed to
    the encoder and the layer output the checkpoint's version expects is
    returned — layer 9 through ``final_proj`` (256 dims) for v1, layer 12 raw
    (768 dims) for v2. The index blend, 2x interpolation, protect mixing and
    synthesizer call that follow in upstream ``vc`` live in ``pipeline.py``.

    Args:
        audio0: Mono 16 kHz float audio for one (already padded) segment.
        version: RVC checkpoint version, ``"v1"`` or ``"v2"``.
        device: Torch device; defaults to ``get_torch_device(allow_mps=True)``.
        is_half: fp16 inference (CUDA only; keep False on CPU/MPS).

    Returns:
        Feature tensor of shape ``(1, T, 256|768)`` on the model's device, at
        the ContentVec ~50 Hz frame rate (the pipeline interpolates it 2x to the
        160-sample hop the synthesizer expects).
    """
    if version not in ("v1", "v2"):
        raise ValueError(f"Unsupported RVC version for features: {version!r}")

    model = _get_contentvec(device=device, is_half=is_half)
    dev = model.device

    feats = torch.from_numpy(audio0)  # pipeline.py:201
    feats = feats.half() if is_half else feats.float()  # pipeline.py:202-205
    if feats.dim() == 2:  # stereo -> mono (pipeline.py:206-207)
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()  # pipeline.py:208
    feats = feats.view(1, -1)  # pipeline.py:209

    # pipeline.py:215 — output_layer 9 for v1, 12 for v2; a False padding mask
    # (no padding) is the transformers default, so no attention_mask is passed.
    output_layer = 9 if version == "v1" else 12
    with torch.no_grad():
        outputs = model(feats.to(dev), output_hidden_states=True)
        feats = outputs.hidden_states[output_layer]  # pipeline.py:219 (logits[0])
        if version == "v1":  # pipeline.py:220
            feats = model.final_proj(feats)
    return feats
