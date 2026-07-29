"""Step 09 Wave 0 — worklet assets must be emitted into every packaged dist.

The realtime RVC AudioWorklet modules are referenced via
``new URL('../worklets/*.js', import.meta.url)`` and both ``app/vite.config.ts``
and ``web/vite.config.ts`` carve them out of ``assetsInlineLimit`` so Vite emits
them as physical hashed assets rather than inlined ``data:`` URIs.
``AudioWorklet.addModule()`` must fetch a real script URL — an inlined
base64 ``data:`` worklet is unreliable on the Tauri macOS WebKit view and, worse,
is invisible as a build artifact, which is exactly how the "worklets missing from
the packaged build" regression was born.

This pins the regression class: for every built dist, the two worklet processors
exist as ``rvc-*-processor-*.js`` hashed assets, the app bundle references them
by that hashed path, and no built JS chunk inlines a worklet as a
``data:...javascript`` URI. Skips a dist that has not been built yet (run
``cd app && npm run build`` and ``cd web && npm run build`` first).
"""

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DISTS = {
    "app": _REPO_ROOT / "app" / "dist",
    "web": _REPO_ROOT / "web" / "dist",
}
_WORKLET_STEMS = ("rvc-capture-processor", "rvc-playback-processor")
_DATA_URI_JS = re.compile(r"data:(?:text|application)/javascript")


def _any_dist_built() -> bool:
    return any(d.is_dir() for d in _DISTS.values())


@pytest.mark.skipif(
    not _any_dist_built(),
    reason="no frontend dist built; run `cd app && npm run build` (and web) first",
)
@pytest.mark.parametrize("dist_name", sorted(_DISTS))
def test_worklets_emitted_as_hashed_assets(dist_name):
    dist = _DISTS[dist_name]
    if not dist.is_dir():
        pytest.skip(f"{dist_name}/dist not built")

    assets = dist / "assets"
    assert assets.is_dir(), f"{dist_name}/dist/assets missing"

    # 1. Both worklet processors emitted as physical hashed assets.
    for stem in _WORKLET_STEMS:
        matches = list(assets.glob(f"{stem}-*.js"))
        assert matches, f"{dist_name}: expected a hashed {stem}-*.js asset in {assets}"

    # 2. The capture worklet is referenced from a bundle by its hashed path
    #    (proving it resolves to an asset URL, not an inlined data: URI).
    js_chunks = list(assets.glob("*.js"))
    referenced = any(
        re.search(r"rvc-capture-processor-[A-Za-z0-9_-]+\.js", chunk.read_text(errors="ignore"))
        for chunk in js_chunks
    )
    assert referenced, f"{dist_name}: no bundle references the hashed capture worklet asset"

    # 3. No built JS chunk inlines a worklet as a data: URI.
    for chunk in js_chunks:
        text = chunk.read_text(errors="ignore")
        assert not _DATA_URI_JS.search(text), (
            f"{dist_name}: {chunk.name} inlines JS as a data: URI (worklet must be a hashed asset)"
        )
