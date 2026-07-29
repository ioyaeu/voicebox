#!/usr/bin/env bash
# Idempotent, macOS-only post-install fixup for the faiss/torch dual-OpenMP crash.
#
# faiss-cpu wheels bundle their own libomp.dylib and torch bundles another. Two
# OpenMP runtimes in one process SIGSEGV under concurrent load. dyld dedupes loaded
# images by RESOLVED path, so a file copy still yields two images — only a symlink
# collapses faiss's libomp onto torch's, leaving a single loaded runtime.
# The OpenMP duplicate-lib override env var is unsupported (silent audio corruption
# risk) and is deliberately not used here.
#
# Safe to run repeatedly and on non-macOS (no-op).
set -euo pipefail

if [ "$(uname)" != "Darwin" ]; then
    echo "fix-macos-openmp: not macOS (uname=$(uname)); nothing to do."
    exit 0
fi

# Resolve the interpreter that actually has faiss+torch installed, rather than
# hardcoding the dev venv path. Preference order:
#   1. $VOICEBOX_PYTHON override (explicit),
#   2. the repo's dev venv (<repo>/backend/venv/bin/python),
#   3. python3 / python on PATH — CI (e.g. the release workflow) installs the
#      deps into the setup-python interpreter, not a venv.
# A total absence of any interpreter is a HARD error: previously this reported
# success while skipping, so a broken CI setup silently shipped the dual-OpenMP
# crash instead of failing the build.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

VENV_PY=""
if [ -n "${VOICEBOX_PYTHON:-}" ] && [ -x "${VOICEBOX_PYTHON}" ]; then
    VENV_PY="${VOICEBOX_PYTHON}"
elif [ -x "$REPO_ROOT/backend/venv/bin/python" ]; then
    VENV_PY="$REPO_ROOT/backend/venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    VENV_PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    VENV_PY="$(command -v python)"
fi

if [ -z "$VENV_PY" ]; then
    echo "fix-macos-openmp: ERROR — no Python interpreter found (checked \$VOICEBOX_PYTHON, $REPO_ROOT/backend/venv, and PATH)." >&2
    exit 1
fi

echo "fix-macos-openmp: using interpreter $VENV_PY"

# Ask the venv python where faiss's bundled libomp and torch's libomp live, plus
# the relative path from the former's directory to the latter (keeps the venv
# relocatable). If faiss or torch is not importable yet (setup may run before both
# are present), skip cleanly.
out="$("$VENV_PY" - <<'PY' || true
import os, sys
try:
    import faiss
    import torch
except Exception as exc:
    sys.stdout.write("SKIP\t%s\n" % exc)
    sys.exit(0)
faiss_omp = os.path.join(os.path.dirname(os.path.abspath(faiss.__file__)), ".dylibs", "libomp.dylib")
torch_omp = os.path.join(os.path.dirname(os.path.abspath(torch.__file__)), "lib", "libomp.dylib")
rel = os.path.relpath(torch_omp, os.path.dirname(faiss_omp))
sys.stdout.write("OK\t%s\t%s\t%s\n" % (faiss_omp, torch_omp, rel))
PY
)"

status=""; faiss_omp=""; torch_omp=""; rel_target=""
IFS=$'\t' read -r status faiss_omp torch_omp rel_target <<<"$out" || true

case "$status" in
    OK) ;;
    SKIP)
        echo "fix-macos-openmp: faiss/torch not both importable yet (${faiss_omp:-unknown}); skipping."
        exit 0
        ;;
    *)
        echo "fix-macos-openmp: could not determine libomp paths; skipping."
        exit 0
        ;;
esac

realpath_py() {
    "$VENV_PY" -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$1"
}

if [ ! -e "$torch_omp" ]; then
    echo "fix-macos-openmp: torch libomp not found at $torch_omp; skipping."
    exit 0
fi

# Already a symlink resolving to torch's libomp -> nothing to do (idempotent).
if [ -L "$faiss_omp" ] && [ -e "$faiss_omp" ] && [ "$(realpath_py "$faiss_omp")" = "$(realpath_py "$torch_omp")" ]; then
    echo "fix-macos-openmp: already fixed (faiss libomp is a symlink to torch libomp)."
    ls -la "$faiss_omp"
    exit 0
fi

mkdir -p "$(dirname "$faiss_omp")"

# Preserve the original bundled libomp exactly once; never clobber an existing .bak.
bak="${faiss_omp}.bak"
if [ -e "$bak" ]; then
    echo "fix-macos-openmp: existing backup preserved at $bak"
elif [ -f "$faiss_omp" ] && [ ! -L "$faiss_omp" ]; then
    cp -p "$faiss_omp" "$bak"
    echo "fix-macos-openmp: backed up original faiss libomp -> $bak"
fi

# Atomically swap in a relative symlink to torch's libomp (ln to a temp name in the
# same directory, then rename over the original).
tmp="${faiss_omp}.tmp.$$"
rm -f "$tmp"
ln -s "$rel_target" "$tmp"
mv -f "$tmp" "$faiss_omp"

# Verify the relative target actually resolves and points at torch's libomp.
if [ ! -e "$faiss_omp" ]; then
    echo "fix-macos-openmp: ERROR — new symlink does not resolve: $faiss_omp -> $rel_target" >&2
    exit 1
fi
if [ "$(realpath_py "$faiss_omp")" != "$(realpath_py "$torch_omp")" ]; then
    echo "fix-macos-openmp: ERROR — symlink does not resolve to torch libomp" >&2
    exit 1
fi

echo "fix-macos-openmp: linked faiss libomp -> $rel_target (torch's libomp)."
ls -la "$faiss_omp"
