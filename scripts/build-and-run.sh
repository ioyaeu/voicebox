#!/bin/bash
# Build and run the Voicebox DESKTOP APP (Tauri), not just the backend/API.
#
# Two modes:
#   (default)  Production: build the bundled desktop app and launch it.
#              Runs `tauri build` (Vite build + Rust release compile + bundle),
#              then opens Voicebox.app. The app spawns its own bundled backend
#              sidecar — nothing else to start.
#   --dev      Development: start the backend (uvicorn --reload) and open the
#              app window via `tauri dev` (Vite on :5173, hot reload). Fastest
#              way to iterate; runs against the live backend, so it always has
#              the latest code (no sidecar rebuild).
#
# Production builds rebuild the frozen backend sidecar automatically when it's
# missing or STALE (backend code changed since it was built), so the app never
# silently bundles old backend code. Pass --reuse-server to skip that.
#
# The updater's release signing key is NOT required for a local build: if
# TAURI_SIGNING_PRIVATE_KEY (or ~/.tauri/voicebox.key) is present the updater
# artifacts are signed, otherwise they are skipped for this build only (the .app
# still builds and runs). Signed release artifacts are produced by CI /
# scripts/prepare-release.sh, which is where the real key lives.
#
# Usage:
#   scripts/build-and-run.sh [--dev] [--rebuild-server | --reuse-server]
#                            [--host H] [--port P] [--data-dir DIR]
#                            [--no-open] [-h|--help]
#
# Options:
#   --dev             Dev mode (backend + tauri dev) instead of a production build
#   --rebuild-server  (production) Force-rebuild the frozen backend sidecar (SLOW)
#   --reuse-server    (production) Reuse the existing sidecar even if it is stale
#                     (fast, but the app may bundle old backend code)
#   --host H          (dev) backend bind address (default 127.0.0.1)
#   --port P          (dev) backend port (default 17493)
#   --data-dir DIR    (dev) data dir for the DB / profiles / generated audio
#   --no-open         (production) build the .app but don't launch it
#   -h, --help        Show this help

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Rust (cargo/rustc) is installed via rustup under ~/.cargo/bin but is often not
# on a non-login shell's PATH; Tauri needs it. bun is the project's JS runtime.
export PATH="$HOME/.cargo/bin:$PATH"

MODE="release"
REBUILD_SERVER=0
REUSE_SERVER=0
NO_OPEN=0
HOST="127.0.0.1"
PORT="17493"
DATA_DIR=""

while [ $# -gt 0 ]; do
    case "$1" in
        --dev) MODE="dev"; shift ;;
        --rebuild-server) REBUILD_SERVER=1; shift ;;
        --reuse-server) REUSE_SERVER=1; shift ;;
        --no-open) NO_OPEN=1; shift ;;
        --host) HOST="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        -h|--help) sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done

need() { command -v "$1" >/dev/null 2>&1 || { echo "Error: '$1' not found on PATH. $2" >&2; exit 1; }; }
need bun "Install bun (https://bun.sh) — the project's JS runtime."
need cargo "Install Rust via https://rustup.rs (cargo/rustc are needed to build the Tauri app)."

# Tauri's build.rs compiles the app icon with `actool`, which ships with FULL
# Xcode — not the Command Line Tools. If xcode-select points at the CLT, actool
# is absent; point this build at a full Xcode via DEVELOPER_DIR (no sudo needed).
if [ "$(uname)" = "Darwin" ] && ! xcrun -f actool >/dev/null 2>&1; then
    for xc in /Applications/Xcode.app /Applications/Xcode*.app; do
        if [ -x "$xc/Contents/Developer/usr/bin/actool" ]; then
            export DEVELOPER_DIR="$xc/Contents/Developer"
            echo "Using Xcode for actool: $DEVELOPER_DIR (xcode-select -> $(xcode-select -p 2>/dev/null))"
            break
        fi
    done
    if ! xcrun -f actool >/dev/null 2>&1; then
        echo "Error: 'actool' not found (needed to compile the app icon)." >&2
        echo "  Install full Xcode (App Store), then either let this script find it" >&2
        echo "  or set it globally:  sudo xcode-select -s /Applications/Xcode.app/Contents/Developer" >&2
        exit 1
    fi
fi

VENV_PY="$REPO_ROOT/backend/venv/bin/python"
[ -x "$VENV_PY" ] || { echo "Error: backend venv missing (backend/venv). Run: just setup-python" >&2; exit 1; }

# Install JS deps if the hoisted workspace node_modules is absent.
[ -d "$REPO_ROOT/node_modules" ] || { echo "Installing JS dependencies (bun) ..."; ( cd "$REPO_ROOT" && bun install ); }

if [ "$MODE" = "dev" ]; then
    # macOS: apply the single-libomp fix so the dev-venv backend's faiss+torch coexist.
    [ "$(uname)" = "Darwin" ] && [ -x scripts/fix-macos-openmp.sh ] && ./scripts/fix-macos-openmp.sh || true

    backend_pid=""
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo "Backend already running on http://127.0.0.1:$PORT"
    else
        echo "Starting backend on http://$HOST:$PORT ..."
        BACKEND_ARGS=(--host "$HOST" --port "$PORT")
        [ -n "$DATA_DIR" ] && BACKEND_ARGS+=(--data-dir "$DATA_DIR")
        "$VENV_PY" -m backend.main "${BACKEND_ARGS[@]}" &
        backend_pid=$!
        sleep 2
    fi
    trap '[ -n "$backend_pid" ] && kill "$backend_pid" 2>/dev/null; wait 2>/dev/null || true' EXIT

    echo "Launching Voicebox (tauri dev) ..."
    ( cd tauri && bun run tauri dev )
    exit 0
fi

# ── Production build + launch ────────────────────────────────────────────────
# The bundled app carries a frozen backend sidecar. A production build must ship
# the CURRENT backend, so rebuild the sidecar when it is missing OR STALE (any
# backend runtime source newer than it) — otherwise the app silently bundles old
# backend code (e.g. an endpoint that rejects a value the new UI sends). Use
# --reuse-server to skip the (slow) rebuild and accept a possibly-stale sidecar.
SIDECAR_GLOB=(tauri/src-tauri/binaries/voicebox-server-*)
sidecar_stale=0
if [ -e "${SIDECAR_GLOB[0]}" ] && \
   [ -n "$(find backend -path backend/tests -prune -o -name '*.py' -newer "${SIDECAR_GLOB[0]}" -print -quit 2>/dev/null)" ]; then
    sidecar_stale=1
fi
if [ "$REBUILD_SERVER" -eq 1 ] || [ ! -e "${SIDECAR_GLOB[0]}" ] || { [ "$sidecar_stale" -eq 1 ] && [ "$REUSE_SERVER" -eq 0 ]; }; then
    if [ "$sidecar_stale" -eq 1 ]; then
        echo "Backend sidecar is STALE (backend code changed since it was built) — rebuilding it."
    else
        echo "Building the backend sidecar ..."
    fi
    echo "(this is slow — minutes; use --dev for fast iteration, or --reuse-server to skip)"
    PATH="$REPO_ROOT/backend/venv/bin:$PATH" ./scripts/build-server.sh
elif [ "$sidecar_stale" -eq 1 ]; then
    echo "WARNING: reusing a STALE backend sidecar (--reuse-server) — the app bundles OLD backend code."
else
    echo "Using the current backend sidecar in tauri/src-tauri/binaries/."
fi

# `tauri build` signs updater artifacts, which needs TAURI_SIGNING_PRIVATE_KEY
# because tauri.conf.json sets `bundle.createUpdaterArtifacts` + `updater.pubkey`.
# Release CI and scripts/prepare-release.sh provide that key; a plain local build
# has none, so `tauri build` would fail at the signing step. Sign when a key is
# available, otherwise skip updater-artifact creation via a build-time config
# override (this NEVER edits tauri.conf.json, so signed CI releases are
# unaffected). The .app is still fully built and runnable — it just doesn't
# carry the .app.tar.gz + .sig update payload, which a local build doesn't need.
if [ -z "${TAURI_SIGNING_PRIVATE_KEY:-}" ] && [ -f "$HOME/.tauri/voicebox.key" ]; then
    export TAURI_SIGNING_PRIVATE_KEY="$(cat "$HOME/.tauri/voicebox.key")"
    export TAURI_SIGNING_PRIVATE_KEY_PASSWORD="${TAURI_SIGNING_PRIVATE_KEY_PASSWORD:-}"
fi

if [ -n "${TAURI_SIGNING_PRIVATE_KEY:-}" ]; then
    echo "Building the Voicebox desktop app (tauri build, signing updater artifacts) ..."
    ( cd tauri && bun run tauri build )
else
    echo "Building the Voicebox desktop app (tauri build, updater artifacts disabled) ..."
    echo "  (no signing key found — release signing is handled by CI / scripts/prepare-release.sh)"
    ( cd tauri && bun run tauri build --config '{"bundle":{"createUpdaterArtifacts":false}}' )
fi

APP="tauri/src-tauri/target/release/bundle/macos/Voicebox.app"
if [ "$NO_OPEN" -eq 1 ]; then
    echo "Built. App bundle: $APP  (not launched; --no-open)"
elif [ -d "$APP" ]; then
    # `open` on an already-running app just focuses the OLD instance instead of
    # launching the freshly built binary — quit it first so the new build runs.
    if pgrep -f "$APP/Contents/MacOS/voicebox" >/dev/null 2>&1; then
        echo "Quitting the running Voicebox instance so the new build launches ..."
        osascript -e 'quit app "Voicebox"' >/dev/null 2>&1 || true
        sleep 1
        pkill -f "$APP/Contents/MacOS/voicebox" 2>/dev/null || true
        sleep 1
    fi
    echo "Launching $APP ..."
    open -n "$APP"
else
    echo "Build finished, but $APP was not found — check the tauri build output above." >&2
    exit 1
fi
