# RVC engine test fixtures

`test_rvc_engine.py` runs a real offline conversion only when a community RVC
checkpoint is present in this directory; otherwise it skips (so the suite stays
green without a GPU, network, or a large model download).

To enable the real conversion, drop an **extracted inference** checkpoint here —
one that passes `validate_rvc_checkpoint` (carries `weight`/`config`/`f0`/
`version`/`sr`). Raw training generators like `f0G40k.pth` do **not** qualify.
A small 40k v2 model (~55 MB) is ideal:

```
python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download(
    "trojblue/rvc-kanade-voice",
    "_weights_unsorted/keruanv2.pth",
    local_dir="backend/tests/fixtures/rvc",
)
PY
```

The first `*.pth` (sorted) is used; an optional matching `*.index` FAISS file in
this directory is picked up to exercise the retrieval blend. Do not commit large
checkpoint binaries — keep them local (add `*.pth` / `*.index` to a local
`.gitignore` if needed).
