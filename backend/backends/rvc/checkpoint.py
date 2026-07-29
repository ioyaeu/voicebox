"""Security and structure validation for user-uploaded RVC artifacts.

Uploaded ``.pth`` checkpoints are untrusted, so they are deserialized with
``torch.load(..., weights_only=True)`` (blocks pickle code execution) behind a
size cap, then their structure is validated before any of the tensors are used.
The ``ValueError`` messages here surface directly to the upload endpoint
(step 03), so they must read as user guidance.
"""

from dataclasses import dataclass
from pathlib import Path

# Community RVC checkpoints top out well under 500 MB (a 48k v2 model is ~55 MB).
# Index files from large training sets legitimately reach several hundred MB;
# the cap bounds what faiss.read_index parses in-process (untrusted input), not
# a functional ceiling. Loading keeps the index AND its reconstructed vector
# bank resident, so steady-state RAM is ~2x the file size.
MAX_CHECKPOINT_BYTES = 500 * 1024 * 1024
MAX_INDEX_BYTES = 1024 * 1024 * 1024

# ContentVec/HuBERT feature width feeding the RVC synthesizer: 256 for v1
# checkpoints, 768 for v2. This fixes the embedding dimension the FAISS index
# must also match.
_EMBEDDER_DIM = {"v1": 256, "v2": 768}

# The synthesizer classes take exactly 18 positional hyperparameters
# (spec_channels..sr, see synthesizer.SynthesizerTrnMs256NSFsid.__init__), which
# ``build_synthesizer`` splats as ``cls(*config, ...)`` and indexes at
# ``config[-3]``. A shorter list crashes conversion with a raw TypeError/IndexError.
_CONFIG_ARITY = 18


@dataclass
class RVCCheckpointInfo:
    """Validated hyperparameters extracted from an RVC checkpoint."""

    version: str
    sample_rate: int
    if_f0: int
    embedder_dim: int


def load_rvc_checkpoint(path: str) -> dict:
    """Safely deserialize an uploaded RVC ``.pth`` file into a dict.

    Enforces a size cap before reading and uses ``weights_only=True`` so a
    malicious pickle cannot execute code during load. Raises ``ValueError`` if
    the file is missing, too large, or does not deserialize to a dict.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"RVC checkpoint not found: {path}")

    size = p.stat().st_size
    if size > MAX_CHECKPOINT_BYTES:
        raise ValueError(
            f"RVC checkpoint is too large ({size / (1024 * 1024):.0f} MB); "
            f"the limit is {MAX_CHECKPOINT_BYTES // (1024 * 1024)} MB."
        )

    import torch

    # weights_only=True is passed explicitly: the repo's torch floor (2.2) does
    # not default to it, and it is the whole point of this loader.
    ckpt = torch.load(path, map_location="cpu", weights_only=True)

    if not isinstance(ckpt, dict):
        raise ValueError("RVC checkpoint did not deserialize to a dictionary.")
    return ckpt


def validate_rvc_checkpoint(ckpt: dict) -> RVCCheckpointInfo:
    """Validate the community-RVC checkpoint shape and extract its metadata.

    Raises ``ValueError`` with a user-readable message on any mismatch.
    """
    if not isinstance(ckpt, dict):
        raise ValueError("Not a valid RVC checkpoint: expected a dictionary.")

    weight = ckpt.get("weight")
    if not isinstance(weight, dict) or not weight:
        raise ValueError("Not a valid RVC checkpoint: missing the 'weight' state dict.")

    # build_synthesizer dereferences weight["emb_g.weight"] to size the speaker
    # embedding; without it, conversion dies with a raw KeyError (500).
    if "emb_g.weight" not in weight:
        raise ValueError(
            "Not a valid RVC checkpoint: the 'weight' state dict is missing "
            "'emb_g.weight' (the speaker embedding)."
        )

    config = ckpt.get("config")
    if not isinstance(config, (list, tuple)) or not config:
        raise ValueError("Not a valid RVC checkpoint: missing the 'config' hyperparameter list.")

    if len(config) != _CONFIG_ARITY:
        raise ValueError(
            f"Not a valid RVC checkpoint: 'config' must list {_CONFIG_ARITY} "
            f"hyperparameters, got {len(config)}."
        )

    f0 = ckpt.get("f0")
    # bool is an int subclass; True/False collapse to 1/0, which is acceptable.
    if not isinstance(f0, int) or int(f0) not in (0, 1):
        raise ValueError("Not a valid RVC checkpoint: 'f0' must be 0 or 1.")

    version = ckpt.get("version")
    if version not in _EMBEDDER_DIM:
        raise ValueError(
            f"Unsupported RVC checkpoint version {version!r}: expected 'v1' or 'v2'."
        )

    sample_rate = _extract_sample_rate(ckpt, config)

    return RVCCheckpointInfo(
        version=version,
        sample_rate=sample_rate,
        if_f0=int(f0),
        embedder_dim=_EMBEDDER_DIM[version],
    )


def validate_faiss_index(path: str, embedder_dim: int) -> None:
    """Validate an uploaded FAISS retrieval index against the model.

    Caps the file size, loads it, and checks the index dimension matches the
    checkpoint's embedding width. Raises ``ValueError`` on any mismatch.
    """
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"FAISS index not found: {path}")

    size = p.stat().st_size
    if size > MAX_INDEX_BYTES:
        raise ValueError(
            f"FAISS index is too large ({size / (1024 * 1024):.0f} MB); "
            f"the limit is {MAX_INDEX_BYTES // (1024 * 1024)} MB."
        )

    # Imported lazily: faiss-cpu may not be installed yet, and this module must
    # import without it (checkpoint validation does not need faiss).
    import faiss

    index = faiss.read_index(str(p))
    if index.d != embedder_dim:
        raise ValueError(
            f"FAISS index dimension ({index.d}) does not match the model's "
            f"embedding dimension ({embedder_dim})."
        )


def _extract_sample_rate(ckpt: dict, config) -> int:
    """Resolve the output sample rate from the checkpoint's 'sr' or config tail."""
    parsed = _parse_sample_rate(ckpt.get("sr"))
    if parsed is not None:
        return parsed

    # RVC stores the sampling rate as the final positional entry of config.
    tail = config[-1]
    if isinstance(tail, int) and not isinstance(tail, bool) and tail >= 8000:
        return tail

    raise ValueError("Not a valid RVC checkpoint: could not determine the sample rate.")


def _parse_sample_rate(value) -> int | None:
    """Parse an RVC sample-rate value (int like 40000 or string like '40k')."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        s = value.strip().lower()
        if not s:
            return None
        try:
            if s.endswith("k"):
                return int(float(s[:-1]) * 1000)
            return int(s)
        except ValueError:
            return None
    return None
