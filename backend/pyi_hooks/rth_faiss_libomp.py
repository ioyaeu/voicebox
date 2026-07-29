"""Runtime hook: make faiss-cpu resolve to torch's single OpenMP image (macOS).

torch and faiss-cpu each ship their own ``libomp.dylib``; loading two distinct
OpenMP runtimes into one process aborts on macOS (the classic dual-OpenMP
segfault). The two resolution paths inside an extracted bundle are (verified
with ``otool -L``):

  * torch : ``libtorch_cpu.dylib`` -> ``@rpath/libomp.dylib`` (rpath ``@loader_path``)
            => ``torch/lib/libomp.dylib``   (torch's canonical real image)
  * faiss : ``_swigfaiss*.so``     -> ``@loader_path/.dylibs/libomp.dylib``
            => ``faiss/.dylibs/libomp.dylib``

Step 01 fixes the dev venv by symlinking faiss's copy onto torch's. PyInstaller's
``--onefile`` CArchive, however, has no symlink type code, so that symlink is
DEREFERENCED into a full copy when the bundle is packed — at extraction both
``torch/lib/libomp.dylib`` and ``faiss/.dylibs/libomp.dylib`` are real, distinct
files (two OpenMP images -> the crash). This hook re-establishes the invariant
before torch or faiss is imported: it points ``faiss/.dylibs/libomp.dylib`` (and,
if present, the optional top-level ``libomp.dylib`` alias) at torch's single real
copy. It also materialises the faiss alias if a spec-level filter excluded it
(``voicebox-server.spec``), since faiss's ``@loader_path`` lookup MUST find a file
there. When the aliases already coincide with torch's copy it is a no-op.

Deliberately scoped to the faiss/top-level aliases: other packages that ship
their own libomp (e.g. scikit-learn, which links it via its own private
``.dylibs`` and is NOT loaded alongside faiss here) are pre-existing, separate
images and outside RVC packaging. KMP_DUPLICATE_LIB_OK is intentionally NOT used
— it silences the abort while leaving two runtimes racing, which corrupts results.
"""

import logging
import os
import sys

_logger = logging.getLogger("voicebox.rvc.libomp")


def _point_at_torch(canonical: str, alias: str, *, create: bool = False) -> None:
    """Make ``alias`` a symlink to torch's single libomp image (``canonical``).

    No-op when ``alias`` already resolves to ``canonical``. Replaces a distinct
    file/symlink. When ``alias`` is missing and ``create`` is set, creates it
    (materialising the parent directory if a spec-level exclusion dropped it) —
    faiss's ``@loader_path/.dylibs/libomp.dylib`` lookup must find a file there.
    """
    exists = os.path.lexists(alias)
    if not exists and not create:
        return
    if exists and os.path.realpath(alias) == os.path.realpath(canonical):
        return  # already the single torch image
    parent = os.path.dirname(alias)
    try:
        if create and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)
        if exists:
            os.remove(alias)
        os.symlink(os.path.relpath(canonical, parent), alias)
    except OSError as exc:
        # Never crash startup over the dedupe; worst case is the pre-existing
        # (dual-image) layout. Surface it as a warning instead of swallowing it
        # so a broken bundle is diagnosable rather than a silent segfault later.
        _logger.warning("faiss/torch libomp dedupe failed for %s -> %s: %s", alias, canonical, exc)


def _dedupe_faiss_libomp() -> None:
    if sys.platform != "darwin":
        return

    meipass = getattr(sys, "_MEIPASS", None)
    if not meipass:
        return

    canonical = os.path.join(meipass, "torch", "lib", "libomp.dylib")
    if not os.path.isfile(canonical):  # need torch's real copy to point faiss at
        return

    _point_at_torch(canonical, os.path.join(meipass, "libomp.dylib"))
    _point_at_torch(canonical, os.path.join(meipass, "faiss", ".dylibs", "libomp.dylib"), create=True)


_dedupe_faiss_libomp()
