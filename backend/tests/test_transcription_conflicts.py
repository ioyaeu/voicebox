import io

import numpy as np
import pytest
from fastapi import HTTPException
from starlette.datastructures import UploadFile


REALTIME_ERROR = (
    "MLX Whisper transcription is blocked while Voice Changer real-time "
    "streaming is active. Stop the live stream and try again."
)


@pytest.mark.asyncio
async def test_transcribe_realtime_conflict_returns_409(monkeypatch):
    from backend.routes import transcription
    from backend.utils import audio as audio_utils

    class WhisperStub:
        model_size = "base"

        def is_loaded(self):
            return True

        def _is_model_cached(self, model_size):
            return True

        async def transcribe(self, audio_path, language, model_size):
            raise RuntimeError(REALTIME_ERROR)

    monkeypatch.setattr(
        audio_utils,
        "load_audio",
        lambda path: (np.zeros(16000, dtype=np.float32), 16000),
    )
    monkeypatch.setattr(
        transcription.transcribe,
        "get_whisper_model",
        lambda: WhisperStub(),
    )

    upload = UploadFile(file=io.BytesIO(b"not-real-audio"), filename="sample.webm")

    with pytest.raises(HTTPException) as excinfo:
        await transcription.transcribe_audio(file=upload, language=None, model=None)

    assert excinfo.value.status_code == 409
    assert "real-time streaming is active" in excinfo.value.detail


@pytest.mark.asyncio
async def test_create_capture_realtime_conflict_returns_409(monkeypatch):
    from backend.routes import captures

    class CaptureSettingsStub:
        stt_model = "base"
        language = "auto"
        auto_refine = False
        allow_auto_paste = False

    async def create_capture_stub(**kwargs):
        raise RuntimeError(REALTIME_ERROR)

    monkeypatch.setattr(
        captures.settings_service,
        "get_capture_settings",
        lambda db: CaptureSettingsStub(),
    )
    monkeypatch.setattr(
        captures.captures_service,
        "create_capture",
        create_capture_stub,
    )

    upload = UploadFile(file=io.BytesIO(b"non-empty"), filename="capture.webm")

    with pytest.raises(HTTPException) as excinfo:
        await captures.create_capture_endpoint(file=upload, db=object())

    assert excinfo.value.status_code == 409
    assert "real-time streaming is active" in excinfo.value.detail
