"""
PyInstaller build script for creating standalone Python server binary.

Usage:
    python build_binary.py           # Build default (CPU) server binary
    python build_binary.py --cuda    # Build CUDA-enabled server binary
"""

import PyInstaller.__main__
import argparse
import contextlib
import logging
import os
import platform
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)


def is_apple_silicon():
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


@contextlib.contextmanager
def quiet_optional_dependency_probe_noise():
    """Keep PyInstaller analysis from printing optional-dependency warnings.

    qwen_tts imports the Python ``sox`` package during analysis. That package
    shells out to ``sox -h`` at import time, which prints `/bin/sh: sox: command
    not found` when the system executable is absent. Runtime Qwen inference has
    a numpy fallback in ``backend.utils.qwen_sox_shim``; this temporary PATH stub
    is only to keep the build log readable.
    """
    old_path = os.environ.get("PATH", "")
    old_pythonwarnings = os.environ.get("PYTHONWARNINGS")
    warning_filter = "ignore:pkg_resources is deprecated as an API:UserWarning"
    os.environ["PYTHONWARNINGS"] = (
        f"{old_pythonwarnings},{warning_filter}" if old_pythonwarnings else warning_filter
    )

    with tempfile.TemporaryDirectory(prefix="voicebox-pyi-tools-") as tmp:
        stub_path: Path | None = None
        if shutil.which("sox") is None:
            stub_path = Path(tmp) / ("sox.bat" if platform.system() == "Windows" else "sox")
            if platform.system() == "Windows":
                stub_path.write_text(
                    "@echo off\necho AUDIO FILE FORMATS: wav mp3 flac ogg m4a webm\n",
                    encoding="utf-8",
                )
            else:
                stub_path.write_text(
                    "#!/bin/sh\nprintf '%s\\n' 'AUDIO FILE FORMATS: wav mp3 flac ogg m4a webm'\n",
                    encoding="utf-8",
                )
                stub_path.chmod(0o755)
            os.environ["PATH"] = tmp + os.pathsep + old_path
            logger.info("Using temporary SoX analysis stub for PyInstaller")

        try:
            warnings_cm = warnings.catch_warnings()
            warnings_cm.__enter__()
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            yield
        finally:
            warnings_cm.__exit__(None, None, None)
            os.environ["PATH"] = old_path
            if old_pythonwarnings is None:
                os.environ.pop("PYTHONWARNINGS", None)
            else:
                os.environ["PYTHONWARNINGS"] = old_pythonwarnings


# ---------------------------------------------------------------------------
# macOS single-OpenMP (libomp) verification
# ---------------------------------------------------------------------------
# torch and faiss-cpu each vendor a libomp.dylib. Loading two distinct OpenMP
# runtimes into one process aborts on macOS (the classic dual-OpenMP segfault).
# The two runtime resolution paths inside an *extracted* bundle are:
#
#   * torch : libtorch_cpu.dylib -> @rpath/libomp.dylib (rpath=@loader_path)
#             => torch/lib/libomp.dylib
#   * faiss : _swigfaiss*.so     -> @loader_path/.dylibs/libomp.dylib
#             => faiss/.dylibs/libomp.dylib
#
# Both MUST resolve to a single real image. PyInstaller's --onefile CArchive has
# no symlink type code, so step 01's faiss->torch symlink is DEREFERENCED into a
# full copy when packed; the invariant is therefore only established at RUNTIME,
# after pyi_hooks/rth_faiss_libomp.py re-points faiss/.dylibs/libomp.dylib at
# torch's copy (before torch/faiss import). Consequently this check is meaningful
# against a tree where that hook has run: an extracted _MEIPASS of a launched
# onefile binary, or a --onedir COLLECT tree.
#
# NOTE ON sklearn: scikit-learn (pulled in via librosa) ships its OWN, separate
# libomp under sklearn/.dylibs/ (or scikit_learn.libs/). It is NOT loaded
# alongside torch/faiss here, so a naive "one libomp in the whole bundle" check
# is WRONG. We assert exactly one real libomp *on the torch+faiss resolution
# path* and merely report any others (sklearn's) as allowed.

# Relative paths (inside an extracted bundle) that participate in faiss/torch
# OpenMP resolution. The top-level alias is optional (present on some layouts).
_FAISS_TORCH_LIBOMP_RELPATHS = (
    os.path.join("torch", "lib", "libomp.dylib"),
    os.path.join("faiss", ".dylibs", "libomp.dylib"),
    "libomp.dylib",
)


def _iter_libomp_files(root: "Path"):
    """Yield (relpath, abspath) for every ``libomp*.dylib`` under ``root``."""
    for p in root.rglob("libomp*.dylib"):
        try:
            rel = str(p.relative_to(root))
        except ValueError:
            rel = str(p)
        yield rel, p


def verify_single_libomp(root) -> bool:
    """Assert the faiss+torch OpenMP resolution path collapses to ONE real image.

    ``root`` is an extracted bundle tree — a --onedir COLLECT dir or the runtime
    ``_MEIPASS`` of a launched --onefile binary (i.e. AFTER rth_faiss_libomp.py
    has run). Raises ``SystemExit`` (non-zero) if the invariant is violated so it
    can gate a build/CI step; returns ``True`` on success.

    Asserts:
      * ``torch/lib/libomp.dylib`` exists (torch's canonical real image);
      * every present faiss/torch resolution-path libomp resolves (realpath) to
        that same single file — i.e. faiss's is a symlink to torch's (or absent),
        and any top-level alias points there too.
    Reports (but allows) sklearn's separate libomp.
    """
    root = Path(root)
    if not root.exists():
        raise SystemExit(f"single-libomp check FAILED: bundle root does not exist: {root}")

    present = []
    for rel in _FAISS_TORCH_LIBOMP_RELPATHS:
        p = root / rel
        if p.is_symlink() or p.exists():
            present.append((rel, p))

    torch_omp = root / "torch" / "lib" / "libomp.dylib"
    if not torch_omp.exists():
        raise SystemExit(
            f"single-libomp check FAILED: torch/lib/libomp.dylib missing under {root}"
        )

    path_rels = {os.path.normpath(rel) for rel in _FAISS_TORCH_LIBOMP_RELPATHS}
    others = [
        (rel, p) for rel, p in _iter_libomp_files(root) if os.path.normpath(rel) not in path_rels
    ]

    real_targets = {os.path.realpath(p) for _, p in present}
    canonical = os.path.realpath(torch_omp)

    print("single-libomp check: bundle =", root)
    for rel, p in present:
        kind = "symlink->" + os.path.relpath(os.path.realpath(p), root) if p.is_symlink() else "real"
        print(f"  faiss/torch path : {rel:<34} [{kind}]")
    for rel, p in others:
        print(f"  other (allowed)  : {rel:<34} [sklearn/etc — separate image]")

    if len(real_targets) != 1 or canonical not in real_targets:
        raise SystemExit(
            "single-libomp check FAILED: faiss+torch resolution path has "
            f"{len(real_targets)} distinct real libomp image(s): "
            + ", ".join(sorted(real_targets))
            + f"; expected exactly 1 == {canonical}. "
            "faiss/.dylibs/libomp.dylib must be a symlink to torch/lib/libomp.dylib "
            "(rth_faiss_libomp.py enforces this at runtime)."
        )

    print(
        f"single-libomp check PASSED: 1 real libomp on faiss+torch path ({canonical}); "
        f"{len(others)} separate (allowed) copy/ies."
    )
    return True


def _report_onefile_libomp(binary_path: "Path") -> None:
    """Best-effort: list the libomp entries packed into a --onefile binary.

    Cannot assert single-image here — onefile dereferences step 01's symlink, so
    both torch's and faiss's libomp are packed as real copies and are only
    collapsed at runtime by rth_faiss_libomp.py. This is purely informational
    plus a hand-off pointer to the authoritative runtime check.
    """
    try:
        from PyInstaller.archive.readers import CArchiveReader

        reader = CArchiveReader(str(binary_path))
        toc = getattr(reader, "toc", None)
        if toc is None:
            toc, _opts = reader._parse_toc(reader.raw_pkg_data()[1])  # pragma: no cover
        names = [n for n in toc if "libomp" in n.lower()]
        print(f"single-libomp report: {binary_path.name} packs {len(names)} libomp entr(ies):")
        for n in sorted(names):
            print(f"    {n} [typecode {toc[n][-1]!r}]")
    except Exception as exc:  # noqa: BLE001 - report only, never fail the build
        print(f"single-libomp report: could not introspect {binary_path} ({exc!r})")
    print(
        "single-libomp: onefile packs BOTH torch's and faiss's copies (symlinks are "
        "dereferenced by the CArchive); rth_faiss_libomp.py collapses them to a single "
        "image at startup. Confirm the runtime invariant by launching the binary and "
        "running:\n"
        "    python build_binary.py --check-libomp <extracted _MEIPASS>"
    )


def build_server(cuda=False, rocm=False):
    """Build Python server as standalone binary.

    Args:
        cuda: If True, build with CUDA support and name the binary
              voicebox-server-cuda instead of voicebox-server.
        rocm: If True, build with ROCm support and name the binary
              voicebox-server-rocm instead of voicebox-server.
    """
    if cuda and rocm:
        raise ValueError("Cannot build with both CUDA and ROCm support")

    backend_dir = Path(__file__).parent

    if rocm:
        binary_name = "voicebox-server-rocm"
    elif cuda:
        binary_name = "voicebox-server-cuda"
    else:
        binary_name = "voicebox-server"

    # PyInstaller arguments
    # CUDA and ROCm builds use --onedir so we can split the output into two archives:
    #   1. Server core (~200-400MB) — versioned with the app
    #   2. GPU libs (~2GB) — versioned independently (only redownloaded on
    #      GPU toolkit / torch major version changes)
    # CPU builds remain --onefile for simplicity.
    pack_mode = "--onedir" if (cuda or rocm) else "--onefile"
    args = [
        "server.py",  # Use server.py as entry point instead of main.py
        pack_mode,
        "--name",
        binary_name,
    ]

    # Hide console window on Windows only. On macOS/Linux the sidecar needs
    # stdout/stderr for Tauri to capture logs.
    if platform.system() == "Windows":
        args.append("--noconsole")

    # numpy 2.x / torch ABI mismatch fix: install memmove fallback for
    # torch.from_numpy() before the app starts. Runtime hooks run after
    # FrozenImporter is registered so frozen torch/numpy are importable.
    # Paths are passed relative to backend_dir because os.chdir(backend_dir)
    # runs before PyInstaller. Absolute paths would get baked into the
    # generated .spec, breaking reproducible builds on other machines / CI.
    args.extend(
        [
            # macOS dual-OpenMP: torch and faiss-cpu each ship a libomp.dylib.
            # PyInstaller 6.20 preserves step 01's faiss->torch symlink, but this
            # hook guarantees faiss resolves to torch's single libomp image even
            # when built from an un-fixed venv. See pyi_hooks/rth_faiss_libomp.py.
            # No-op on non-macOS and when faiss already links to torch's copy.
            "--runtime-hook",
            "pyi_hooks/rth_faiss_libomp.py",
            "--runtime-hook",
            "pyi_rth_numpy_compat.py",
            # Stub torch.compiler.disable before transformers imports
            # flex_attention, which otherwise triggers torch._dynamo →
            # torch._numpy._ufuncs and crashes at module load under
            # PyInstaller. See pyi_rth_torch_compiler_disable.py.
            "--runtime-hook",
            "pyi_rth_torch_compiler_disable.py",
            # Per-module collection overrides (e.g. forcing scipy.stats._distn_infrastructure
            # to bundle .py source alongside .pyc so the runtime hook can source-patch it).
            "--additional-hooks-dir",
            "pyi_hooks",
        ]
    )

    # Add local qwen_tts path if specified (for editable installs)
    qwen_tts_path = os.getenv("QWEN_TTS_PATH")
    if qwen_tts_path and Path(qwen_tts_path).exists():
        args.extend(["--paths", str(qwen_tts_path)])
        logger.info("Using local qwen_tts source from: %s", qwen_tts_path)

    # Add common hidden imports
    args.extend(
        [
            "--hidden-import",
            "backend",
            "--hidden-import",
            "backend.main",
            "--hidden-import",
            "backend.config",
            "--hidden-import",
            "backend.database",
            "--hidden-import",
            "backend.models",
            "--hidden-import",
            "backend.services.profiles",
            "--hidden-import",
            "backend.services.history",
            "--hidden-import",
            "backend.services.tts",
            "--hidden-import",
            "backend.services.transcribe",
            "--hidden-import",
            "backend.utils.platform_detect",
            "--hidden-import",
            "backend.backends",
            "--hidden-import",
            "backend.backends.pytorch_backend",
            "--hidden-import",
            "backend.backends.qwen_custom_voice_backend",
            "--hidden-import",
            "backend.utils.audio",
            "--hidden-import",
            "backend.utils.cache",
            "--hidden-import",
            "backend.utils.progress",
            "--hidden-import",
            "backend.utils.hf_progress",
            "--hidden-import",
            "backend.utils.qwen_sox_shim",
            "--hidden-import",
            "backend.services.cuda",
            "--hidden-import",
            "backend.services.effects",
            "--hidden-import",
            "backend.utils.effects",
            "--hidden-import",
            "backend.services.versions",
            "--hidden-import",
            "pedalboard",
            "--hidden-import",
            "chatterbox",
            "--hidden-import",
            "chatterbox.tts_turbo",
            "--hidden-import",
            "chatterbox.mtl_tts",
            "--hidden-import",
            "backend.backends.chatterbox_backend",
            "--hidden-import",
            "backend.backends.chatterbox_turbo_backend",
            # chatterbox multilingual uses spacy_pkuseg for Chinese word
            # segmentation, which ships pickled dict files (dicts/default.pkl)
            # and native .so extensions that --hidden-import alone won't bundle.
            "--collect-all",
            "spacy_pkuseg",
            "--hidden-import",
            "backend.backends.luxtts_backend",
            "--hidden-import",
            "zipvoice",
            "--hidden-import",
            "zipvoice.luxvoice",
            "--collect-all",
            "zipvoice",
            "--collect-all",
            "linacodec",
            "--hidden-import",
            "torch",
            "--hidden-import",
            "transformers",
            "--hidden-import",
            "fastapi",
            "--hidden-import",
            "uvicorn",
            "--hidden-import",
            "sqlalchemy",
            # librosa uses lazy_loader which generates .pyi stub files at
            # install time and reads them at runtime to discover submodules.
            # --hidden-import alone doesn't bundle the stubs, causing
            # "Cannot load imports from non-existent stub" at runtime.
            "--collect-all",
            "lazy_loader",
            "--collect-all",
            "librosa",
            "--hidden-import",
            "soundfile",
            "--hidden-import",
            "qwen_tts",
            "--hidden-import",
            "qwen_tts.inference",
            "--hidden-import",
            "qwen_tts.inference.qwen3_tts_model",
            "--hidden-import",
            "qwen_tts.inference.qwen3_tts_tokenizer",
            "--hidden-import",
            "qwen_tts.core",
            "--hidden-import",
            "qwen_tts.cli",
            "--copy-metadata",
            "qwen-tts",
            "--copy-metadata",
            "requests",
            "--copy-metadata",
            "transformers",
            "--copy-metadata",
            "huggingface-hub",
            "--copy-metadata",
            "tokenizers",
            "--copy-metadata",
            "safetensors",
            "--copy-metadata",
            "tqdm",
            "--hidden-import",
            "requests",
            # qwen_tts uses inspect.getsource() at runtime to locate
            # modeling_qwen3_tts.py — needs physical .py source files bundled
            "--collect-all",
            "qwen_tts",
            # Fix for jaraco namespace packages
            "--collect-submodules",
            "jaraco",
            # inflect uses typeguard @typechecked which calls inspect.getsource()
            # at import time — needs .py source files, not just .pyc bytecode
            "--collect-all",
            "inflect",
            # perth ships pretrained watermark model files (hparams.yaml,
            # .pth.tar) in perth/perth_net/pretrained/ — needed by chatterbox
            # at runtime. Do not collect-all: perth.cli imports matplotlib,
            # which Voicebox does not need and does not install.
            "--hidden-import",
            "perth",
            "--collect-submodules",
            "perth.perth_net",
            "--collect-data",
            "perth",
            # piper_phonemize ships espeak-ng-data/ (phoneme tables, language dicts)
            # needed by LuxTTS for text-to-phoneme conversion
            "--collect-all",
            "piper_phonemize",
            # HumeAI TADA — speech-language model using Llama + flow matching
            "--hidden-import",
            "backend.backends.hume_backend",
            "--hidden-import",
            "tada",
            "--hidden-import",
            "tada.modules",
            "--hidden-import",
            "tada.modules.tada",
            "--hidden-import",
            "tada.modules.encoder",
            "--hidden-import",
            "tada.modules.decoder",
            "--hidden-import",
            "tada.modules.aligner",
            "--hidden-import",
            "tada.modules.acoustic_spkr_verf",
            "--hidden-import",
            "tada.nn",
            "--hidden-import",
            "tada.nn.vibevoice",
            "--hidden-import",
            "tada.utils",
            "--hidden-import",
            "tada.utils.gray_code",
            "--hidden-import",
            "tada.utils.text",
            # DAC shim — provides dac.nn.layers.Snake1d without the real
            # descript-audio-codec package (which pulls onnx/tensorboard via
            # descript-audiotools). The shim is in backend/utils/dac_shim.py.
            "--hidden-import",
            "backend.utils.dac_shim",
            "--hidden-import",
            "torchaudio",
            # Do not collect-submodules tada wholesale: PyInstaller imports
            # tada.modules in an isolated analyser before our DAC shim is
            # installed, which raises "No module named 'dac'". The runtime
            # Hume backend installs backend.utils.dac_shim before importing
            # the explicit TADA modules listed above.
            # Kokoro 82M — lightweight TTS engine using misaki G2P
            # collect-all is required because transformers introspects .py source
            # files at runtime (e.g. _can_set_attn_implementation opens the class
            # file); hidden-import alone only bundles bytecode.
            "--hidden-import",
            "backend.backends.kokoro_backend",
            "--collect-all",
            "kokoro",
            # misaki ships G2P data files (dictionaries, phoneme tables)
            # that must be bundled for espeak/en/ja/zh G2P to work
            "--collect-all",
            "misaki",
            # language_tags ships JSON data files (index.json etc.) loaded at
            # runtime via: misaki → phonemizer → segments → csvw → language_tags
            "--collect-all",
            "language_tags",
            # espeakng_loader ships the entire espeak-ng-data directory (369 files)
            # loaded at import time by misaki.espeak via get_data_path()
            "--collect-all",
            "espeakng_loader",
            # spacy en_core_web_sm model — misaki.en tries to spacy.cli.download()
            # at runtime if not found, which calls pip as a subprocess and crashes
            # the frozen binary. Bundle the model so spacy.util.is_package() passes.
            "--collect-all",
            "en_core_web_sm",
            "--copy-metadata",
            "en_core_web_sm",
            "--hidden-import",
            "en_core_web_sm",
            # unidic-lite ships the MeCab dictionary used by fugashi (pulled in
            # by misaki[ja]). The dict lives in unidic_lite/dicdir/ and is
            # discovered via the package's DICDIR constant, so the data files
            # must be collected or Japanese Kokoro voices crash at runtime.
            "--collect-all",
            "unidic_lite",
            "--hidden-import",
            "loguru",
            # MCP server — Streamable-HTTP endpoint and the 4 voicebox.* tools.
            # FastMCP pulls in a chain of deps (mcp, cyclopts, openapi-pydantic,
            # etc.) that don't auto-discover cleanly under PyInstaller, so we
            # collect them whole. Small compared to torch.
            "--hidden-import",
            "backend.mcp_server",
            "--hidden-import",
            "backend.mcp_server.server",
            "--hidden-import",
            "backend.mcp_server.tools",
            "--hidden-import",
            "backend.mcp_server.context",
            "--hidden-import",
            "backend.mcp_server.resolve",
            "--hidden-import",
            "backend.mcp_server.events",
            "--collect-all",
            "fastmcp",
            "--collect-all",
            "mcp",
            "--hidden-import",
            "sse_starlette",
            # RVC voice conversion (Phase A). The vendored engine is reached only
            # via function-local imports (services/convert.py, services/profiles.py,
            # routes/profiles.py), so list its modules explicitly rather than
            # relying on modulegraph to walk into the frozen backend package.
            "--hidden-import",
            "backend.services.convert",
            "--hidden-import",
            "backend.backends.rvc",
            "--hidden-import",
            "backend.backends.rvc.pipeline",
            "--hidden-import",
            "backend.backends.rvc.checkpoint",
            "--hidden-import",
            "backend.backends.rvc.features",
            "--hidden-import",
            "backend.backends.rvc.pitch",
            "--hidden-import",
            "backend.backends.rvc.synthesizer",
            # rvc.streaming: reached only via routes/convert.py's function-local
            # import on the first WS handshake, so modulegraph never sees it.
            "--hidden-import",
            "backend.backends.rvc.streaming",
            # faiss-cpu: native libfaiss.dylib + _swigfaiss extension and the
            # dynamically-imported swigfaiss_* variants are collected by
            # pyi_hooks/hook-faiss.py; libomp dedup by rth_faiss_libomp.py.
            "--hidden-import",
            "faiss",
            # pyworld: single compiled extension. Its __init__ reads its own dist
            # metadata via pkg_resources.get_distribution() at import time, so the
            # metadata must be bundled or `import pyworld` raises DistributionNotFound.
            "--hidden-import",
            "pyworld",
            "--hidden-import",
            "pyworld.pyworld",
            "--copy-metadata",
            "pyworld",
            # torchcrepe: the optional 'crepe' f0 estimator. Collect the package
            # CODE only via --collect-submodules; its assets/{full,tiny}.pth
            # (~87 MB) are deliberately NOT bundled (no --collect-data). Upstream
            # torchcrepe has no download fallback of its own, so instead crepe-full
            # is registered in the ModelConfig registry and fetched on demand
            # (commit-pinned GitHub source + sha256), the same way contentvec/rmvpe
            # are. Until it's downloaded, f0_method=crepe returns a clear 400 rather
            # than crashing; rmvpe (the default) needs none of this and is unaffected.
            "--hidden-import",
            "torchcrepe",
            "--collect-submodules",
            "torchcrepe",
        ]
    )

    if sys.version_info >= (3, 13):
        args.extend(["--hidden-import", "audioop"])

    # Add CUDA/ROCm-specific hidden imports
    if cuda or rocm:
        variant = "ROCm" if rocm else "CUDA"
        logger.info("Building with %s support", variant)
        gpu_hidden = [
            "--hidden-import",
            "torch.cuda",
        ]
        # cudnn is NVIDIA-specific; ROCm uses MIOpen under the abstraction layer
        if cuda:
            gpu_hidden.extend(
                [
                    "--hidden-import",
                    "torch.backends.cudnn",
                ]
            )
        args.extend(gpu_hidden)

    if rocm:
        # rocm_sdk imports its backend packages dynamically via
        # importlib.import_module(py_package_name), which PyInstaller's
        # static analyzer cannot see. We must collect them explicitly —
        # otherwise only the pure-python rocm_sdk wrapper ships and
        # rocm_sdk.find_libraries crashes with UnboundLocalError at boot.
        #
        # The backend packages also contain the HIP/MIOpen/hipBLAS DLLs
        # under bin/ (plus ~750 MB of tensile kernel files under
        # bin/rocblas/library and bin/hipblaslt/library) — collect-all
        # walks the tree recursively so both DLLs and kernel data are
        # bundled. See rocm_sdk/_dist_info.py for the package mapping.
        args.extend(
            [
                "--collect-all",
                "rocm_sdk",
                "--collect-all",
                "_rocm_sdk_core",
                "--collect-all",
                "_rocm_sdk_libraries_custom",
                "--collect-all",
                "rocm_sdk_core",
                "--collect-all",
                "rocm_sdk_libraries_custom",
                "--hidden-import",
                "_rocm_sdk_core",
                "--hidden-import",
                "_rocm_sdk_libraries_custom",
                "--hidden-import",
                "rocm_sdk_core",
                "--hidden-import",
                "rocm_sdk_libraries_custom",
                "--copy-metadata",
                "rocm",
                "--copy-metadata",
                "rocm-sdk-core",
                "--copy-metadata",
                "rocm-sdk-libraries-custom",
                # Repair rocm_sdk.find_libraries (masks UnboundLocalError
                # with a readable ModuleNotFoundError on missing backends).
                "--runtime-hook",
                "pyi_rth_rocm_sdk.py",
            ]
        )

    # Exclude NVIDIA CUDA packages from non-CUDA builds to keep binary small.
    # When building from a venv with CUDA torch installed, PyInstaller would
    # bundle ~3GB of NVIDIA shared libraries. We exclude both the Python
    # modules and the binary DLLs. This applies to CPU and ROCm builds.
    if not cuda:
        nvidia_packages = [
            "nvidia",
            "nvidia.cublas",
            "nvidia.cuda_cupti",
            "nvidia.cuda_nvrtc",
            "nvidia.cuda_runtime",
            "nvidia.cudnn",
            "nvidia.cufft",
            "nvidia.curand",
            "nvidia.cusolver",
            "nvidia.cusparse",
            "nvidia.nccl",
            "nvidia.nvjitlink",
            "nvidia.nvtx",
        ]
        for pkg in nvidia_packages:
            args.extend(["--exclude-module", pkg])

    # Add MLX-specific imports if building on Apple Silicon (never for GPU builds)
    if is_apple_silicon() and not cuda and not rocm:
        logger.info("Building for Apple Silicon - including MLX dependencies")
        args.extend(
            [
                "--hidden-import",
                "backend.backends.mlx_backend",
                "--hidden-import",
                "backend.backends.voxtral_backend",
                "--hidden-import",
                "backend.backends.mlx_tada_backend",
                "--hidden-import",
                "mlx",
                "--hidden-import",
                "mlx.core",
                "--hidden-import",
                "mlx.nn",
                "--hidden-import",
                "mlx_audio",
                "--hidden-import",
                "mlx_audio.tts",
                "--hidden-import",
                "mlx_audio.tts.models.voxtral_tts",
                "--hidden-import",
                "mlx_audio.stt",
                "--hidden-import",
                "mlx_tada",
                "--hidden-import",
                "mlx_tada.model",
                "--hidden-import",
                "mlx_tada.config",
                "--hidden-import",
                "mlx_tada.audio",
                "--hidden-import",
                "mlx_lm",
                "--hidden-import",
                "mistral_common.tokens.tokenizers.mistral",
                "--hidden-import",
                "mistral_common.protocol.speech.request",
                "--hidden-import",
                "sentencepiece",
                "--hidden-import",
                "sounddevice",
                "--hidden-import",
                "tiktoken",
                "--hidden-import",
                "backend.backends.qwen_llm_backend",
                "--collect-submodules",
                "mlx",
                "--collect-submodules",
                "mlx_audio",
                "--collect-submodules",
                "mlx_lm",
                # Use --collect-all so PyInstaller bundles both data files AND
                # native shared libraries (.dylib, .metallib) for MLX.
                # Previously only --collect-data was used, which caused MLX to
                # raise OSError at runtime inside the bundled binary because
                # the Metal shader libraries were missing.
                "--collect-all",
                "mlx",
                "--collect-all",
                "mlx_audio",
                "--collect-all",
                "mlx_tada",
                # mlx_lm ships chat_templates/ JSON files and loads tool_parsers
                # submodules dynamically via importlib at tokenizer load time,
                # which --hidden-import alone can't resolve.
                "--collect-all",
                "mlx_lm",
                # Voxtral TTS uses Mistral's Tekken tokenizer helpers.
                # mistral_common reads package metadata and ships tokenizer data;
                # sentencepiece/tiktoken/sounddevice carry native/data files.
                "--collect-all",
                "mistral_common",
                "--collect-all",
                "sentencepiece",
                "--collect-all",
                "sounddevice",
                "--collect-all",
                "tiktoken",
                "--copy-metadata",
                "mlx-tada",
                "--copy-metadata",
                "mistral-common",
                "--copy-metadata",
                "sentencepiece",
                "--copy-metadata",
                "sounddevice",
                "--copy-metadata",
                "tiktoken",
            ]
        )
    elif not cuda and not rocm:
        logger.info("Building for non-Apple Silicon platform - PyTorch only")

    dist_dir = str(backend_dir / "dist")
    build_dir = str(backend_dir / "build")

    args.extend(
        [
            "--distpath",
            dist_dir,
            "--workpath",
            build_dir,
            "--noconfirm",
            "--clean",
        ]
    )

    # Change to backend directory
    os.chdir(backend_dir)

    # For CPU builds on Windows, ensure we're using CPU-only torch.
    # If CUDA or ROCm torch is installed (local dev), swap to CPU torch before
    # building, then restore afterwards. This prevents PyInstaller from bundling
    # GPU libraries into the CPU binary.
    restore_torch = None
    try:
        if not cuda and not rocm and platform.system() == "Windows":
            import subprocess

            cuda_result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"], capture_output=True, text=True
            )
            rocm_result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.version.hip or '')"], capture_output=True, text=True
            )

            if cuda_result.stdout.strip():
                restore_torch = "cuda"
                logger.info("CUDA torch detected — installing CPU torch for CPU build...")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                        "--force-reinstall",
                        "--no-deps",
                        "-q",
                    ],
                    check=True,
                )
            elif rocm_result.stdout.strip():
                restore_torch = "rocm"
                logger.info("ROCm torch detected — installing CPU torch for CPU build...")
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "torch",
                        "torchvision",
                        "torchaudio",
                        "--index-url",
                        "https://download.pytorch.org/whl/cpu",
                        "--force-reinstall",
                        "--no-deps",
                        "-q",
                    ],
                    check=True,
                )

        # For ROCm builds on Windows, ensure ROCm torch is installed.
        if rocm and platform.system() == "Windows":
            import subprocess

            if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
                raise RuntimeError(
                    "ROCm wheels are cp312-cp312-specific; "
                    f"got {sys.implementation.name} {sys.version.split()[0]}. "
                    "Use CPython 3.12 to build the ROCm binary."
                )

            result = subprocess.run(
                [sys.executable, "-c", "import torch; print(torch.version.hip or '')"], capture_output=True, text=True
            )
            has_rocm_torch = bool(result.stdout.strip())
            if not has_rocm_torch:
                logger.info("ROCm torch not detected — installing ROCm torch for ROCm build...")

                # Determine what to restore BEFORE overwriting the environment
                cuda_result = subprocess.run(
                    [sys.executable, "-c", "import torch; print(torch.version.cuda or '')"],
                    capture_output=True,
                    text=True,
                )
                if cuda_result.stdout.strip():
                    restore_torch = "cuda"
                else:
                    restore_torch = "cpu"

                # Now overwrite the environment safely
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz",
                        "--no-deps",
                        "-q",
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pip",
                        "install",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                        "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                        "--force-reinstall",
                        "--no-deps",
                        "-q",
                    ],
                    check=True,
                )

        # Run PyInstaller
        with quiet_optional_dependency_probe_noise():
            PyInstaller.__main__.run(args)
    finally:
        # Restore torch if we swapped it out (even on build failure)
        if restore_torch == "cuda":
            logger.info("Restoring CUDA torch...")
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cu128",
                    "--force-reinstall",
                    "--no-deps",
                    "-q",
                ],
                check=True,
            )
        elif restore_torch == "rocm":
            logger.info("Restoring ROCm torch...")
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchaudio-2.9.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                    "https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1%2Brocm7.2.1-cp312-cp312-win_amd64.whl",
                    "--force-reinstall",
                    "--no-deps",
                    "-q",
                ],
                check=True,
            )
        elif restore_torch == "cpu":
            logger.info("Restoring CPU torch...")
            import subprocess

            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "torch",
                    "torchvision",
                    "torchaudio",
                    "--index-url",
                    "https://download.pytorch.org/whl/cpu",
                    "--force-reinstall",
                    "--no-deps",
                    "-q",
                ],
                check=True,
            )


    logger.info("Binary built in %s", backend_dir / "dist" / binary_name)

    # macOS single-OpenMP post-build check. libomp is macOS-only; a naive count
    # would trip over sklearn's separate copy, so we verify only the faiss+torch
    # resolution path (see verify_single_libomp / _report_onefile_libomp).
    if platform.system() == "Darwin":
        try:
            out = backend_dir / "dist" / binary_name
            if out.is_dir():
                # --onedir: the COLLECT tree is on disk; assert the invariant now.
                inner = out / "_internal" if (out / "_internal").is_dir() else out
                verify_single_libomp(inner)
            elif out.is_file():
                # --onefile: real invariant is established at runtime by the hook.
                _report_onefile_libomp(out)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 - never let the report crash a good build
            logger.warning("single-libomp post-build check skipped: %r", exc)


def build_shim():
    """Build the voicebox-mcp stdio shim as a tiny standalone binary.

    This is the bridge for MCP clients that only speak stdio — it proxies
    JSON-RPC to the main voicebox-server's /mcp endpoint. Keep it small: no
    torch, no ML deps, just httpx + asyncio.
    """
    backend_dir = Path(__file__).parent

    args = [
        "mcp_shim/__main__.py",
        "--onefile",
        "--name",
        "voicebox-mcp",
        # Stdio-only — no console hiding needed on Windows since the parent
        # MCP client is spawning this as a child process and wants stdio.
        "--hidden-import",
        "backend.mcp_shim",
        "--hidden-import",
        "backend.mcp_shim.__main__",
        "--hidden-import",
        "httpx",
        "--hidden-import",
        "httpx._transports.default",
        "--hidden-import",
        "anyio",
        # Exclude everything heavy that httpx/asyncio don't actually need so
        # the binary stays tiny (~15 MB instead of ~400 MB).
        "--exclude-module",
        "torch",
        "--exclude-module",
        "transformers",
        "--exclude-module",
        "mlx",
        "--exclude-module",
        "mlx_audio",
        "--exclude-module",
        "mlx_lm",
        "--exclude-module",
        "qwen_tts",
        "--exclude-module",
        "chatterbox",
        "--exclude-module",
        "zipvoice",
        "--exclude-module",
        "tada",
        "--exclude-module",
        "kokoro",
        "--exclude-module",
        "misaki",
        "--exclude-module",
        "spacy",
        "--exclude-module",
        "librosa",
        "--exclude-module",
        "numba",
        "--exclude-module",
        "numpy",
        "--exclude-module",
        "pedalboard",
        "--exclude-module",
        "fastapi",
        "--exclude-module",
        "uvicorn",
        "--exclude-module",
        "sqlalchemy",
        "--exclude-module",
        "fastmcp",
        "--exclude-module",
        "mcp",
    ]

    dist_dir = str(backend_dir / "dist")
    build_dir = str(backend_dir / "build")
    args.extend(
        [
            "--distpath",
            dist_dir,
            "--workpath",
            build_dir,
            "--noconfirm",
            "--clean",
        ]
    )

    os.chdir(backend_dir)
    PyInstaller.__main__.run(args)
    logger.info("Shim built: %s", backend_dir / "dist" / "voicebox-mcp")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build voicebox binaries")
    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Build CUDA-enabled binary (voicebox-server-cuda)",
    )
    parser.add_argument(
        "--rocm",
        action="store_true",
        help="Build ROCm-enabled binary (voicebox-server-rocm) for AMD GPUs",
    )
    parser.add_argument(
        "--shim",
        action="store_true",
        help="Build the voicebox-mcp stdio shim binary instead of the server",
    )
    parser.add_argument(
        "--check-libomp",
        metavar="BUNDLE_DIR",
        default=None,
        help=(
            "Do not build. Assert the faiss+torch OpenMP resolution path in an "
            "extracted bundle tree (a --onedir dir, or the runtime _MEIPASS of a "
            "launched --onefile binary) collapses to a single real libomp image. "
            "sklearn's separate copy is reported but allowed."
        ),
    )
    cli_args = parser.parse_args()
    if cli_args.check_libomp:
        verify_single_libomp(cli_args.check_libomp)
    elif cli_args.shim:
        build_shim()
    else:
        build_server(cuda=cli_args.cuda, rocm=cli_args.rocm)
