# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all
from PyInstaller.utils.hooks import copy_metadata

datas = []
binaries = []
hiddenimports = ['backend', 'backend.main', 'backend.config', 'backend.database', 'backend.models', 'backend.services.profiles', 'backend.services.history', 'backend.services.tts', 'backend.services.transcribe', 'backend.utils.platform_detect', 'backend.backends', 'backend.backends.pytorch_backend', 'backend.backends.qwen_custom_voice_backend', 'backend.utils.audio', 'backend.utils.cache', 'backend.utils.progress', 'backend.utils.hf_progress', 'backend.utils.qwen_sox_shim', 'backend.services.cuda', 'backend.services.effects', 'backend.utils.effects', 'backend.services.versions', 'pedalboard', 'chatterbox', 'chatterbox.tts_turbo', 'chatterbox.mtl_tts', 'backend.backends.chatterbox_backend', 'backend.backends.chatterbox_turbo_backend', 'backend.backends.luxtts_backend', 'zipvoice', 'zipvoice.luxvoice', 'torch', 'transformers', 'fastapi', 'uvicorn', 'sqlalchemy', 'soundfile', 'qwen_tts', 'qwen_tts.inference', 'qwen_tts.inference.qwen3_tts_model', 'qwen_tts.inference.qwen3_tts_tokenizer', 'qwen_tts.core', 'qwen_tts.cli', 'requests', 'perth', 'backend.backends.hume_backend', 'tada', 'tada.modules', 'tada.modules.tada', 'tada.modules.encoder', 'tada.modules.decoder', 'tada.modules.aligner', 'tada.modules.acoustic_spkr_verf', 'tada.nn', 'tada.nn.vibevoice', 'tada.utils', 'tada.utils.gray_code', 'tada.utils.text', 'backend.utils.dac_shim', 'torchaudio', 'backend.backends.kokoro_backend', 'en_core_web_sm', 'loguru', 'backend.mcp_server', 'backend.mcp_server.server', 'backend.mcp_server.tools', 'backend.mcp_server.context', 'backend.mcp_server.resolve', 'backend.mcp_server.events', 'sse_starlette', 'backend.services.convert', 'backend.backends.rvc', 'backend.backends.rvc.pipeline', 'backend.backends.rvc.checkpoint', 'backend.backends.rvc.features', 'backend.backends.rvc.pitch', 'backend.backends.rvc.synthesizer', 'backend.backends.rvc.streaming', 'faiss', 'pyworld', 'pyworld.pyworld', 'torchcrepe', 'backend.backends.mlx_backend', 'backend.backends.voxtral_backend', 'mlx', 'mlx.core', 'mlx.nn', 'mlx_audio', 'mlx_audio.tts', 'mlx_audio.tts.models.voxtral_tts', 'mlx_audio.stt', 'mlx_lm', 'mistral_common.tokens.tokenizers.mistral', 'mistral_common.protocol.speech.request', 'sentencepiece', 'sounddevice', 'tiktoken', 'backend.backends.qwen_llm_backend']
datas += collect_data_files('perth')
datas += copy_metadata('qwen-tts')
datas += copy_metadata('requests')
datas += copy_metadata('transformers')
datas += copy_metadata('huggingface-hub')
datas += copy_metadata('tokenizers')
datas += copy_metadata('safetensors')
datas += copy_metadata('tqdm')
datas += copy_metadata('en_core_web_sm')
datas += copy_metadata('pyworld')
datas += copy_metadata('mistral-common')
datas += copy_metadata('sentencepiece')
datas += copy_metadata('sounddevice')
datas += copy_metadata('tiktoken')
hiddenimports += collect_submodules('jaraco')
hiddenimports += collect_submodules('perth.perth_net')
hiddenimports += collect_submodules('torchcrepe')
hiddenimports += collect_submodules('mlx')
hiddenimports += collect_submodules('mlx_audio')
hiddenimports += collect_submodules('mlx_lm')
tmp_ret = collect_all('spacy_pkuseg')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('zipvoice')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('linacodec')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('lazy_loader')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('librosa')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('qwen_tts')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('inflect')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('piper_phonemize')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('kokoro')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('misaki')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('language_tags')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('espeakng_loader')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('en_core_web_sm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('unidic_lite')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('fastmcp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mcp')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mlx')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mlx_audio')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mlx_lm')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('mistral_common')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sentencepiece')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('sounddevice')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('tiktoken')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['server.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=['pyi_hooks'],
    hooksconfig={},
    runtime_hooks=['pyi_hooks/rth_faiss_libomp.py', 'pyi_rth_numpy_compat.py', 'pyi_rth_torch_compiler_disable.py'],
    excludes=['nvidia', 'nvidia.cublas', 'nvidia.cuda_cupti', 'nvidia.cuda_nvrtc', 'nvidia.cuda_runtime', 'nvidia.cudnn', 'nvidia.cufft', 'nvidia.curand', 'nvidia.cusolver', 'nvidia.cusparse', 'nvidia.nccl', 'nvidia.nvjitlink', 'nvidia.nvtx'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='voicebox-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
