"""Collect faiss-cpu's native libraries and swig submodules.

faiss/loader.py imports the arch-specific ``swigfaiss_*`` wrappers dynamically
(a try/except per CPU instruction set) and finally falls back to
``faiss.swigfaiss`` (-> the ``faiss._swigfaiss`` extension). ``collect_submodules``
pulls in whichever variants exist so the frozen loader resolves the same module
it does in the venv; ``collect_dynamic_libs`` bundles ``libfaiss.dylib`` and the
``_swigfaiss`` extension.

The ``libomp.dylib`` that faiss ships in ``faiss/.dylibs`` (symlinked to torch's
copy in the dev venv, step 01) is deduplicated at runtime by
``pyi_hooks/rth_faiss_libomp.py`` so exactly one OpenMP image is ever loaded.
"""

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

hiddenimports = collect_submodules("faiss")
binaries = collect_dynamic_libs("faiss")
