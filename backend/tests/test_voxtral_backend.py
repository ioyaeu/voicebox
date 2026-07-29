from backend.backends import TTS_ENGINES, get_model_config
from backend.backends.voxtral_backend import (
    VOXTRAL_HF_REPO,
    VOXTRAL_VOICE_IDS,
    VOXTRAL_VOICES,
)
from backend.models import GenerationRequest, MCPClientBindingUpsert, SpeakRequest
from backend.services.profiles import _get_preset_voice_ids, _preset_voice_language


def test_voxtral_model_config_is_registered():
    cfg = get_model_config("voxtral-4b-tts-4bit")

    assert TTS_ENGINES["voxtral"] == "Voxtral 4B TTS"
    assert cfg is not None
    assert cfg.engine == "voxtral"
    assert cfg.hf_repo_id == VOXTRAL_HF_REPO
    assert cfg.languages == ["en", "fr", "es", "de", "it", "pt", "nl", "ar", "hi"]


def test_voxtral_preset_voice_table_is_exposed_to_profile_validation():
    assert len(VOXTRAL_VOICES) == 20
    assert _get_preset_voice_ids("voxtral") == VOXTRAL_VOICE_IDS
    assert _preset_voice_language("voxtral", "fr_female") == "fr"
    assert _preset_voice_language("voxtral", "hi_male") == "hi"


def test_voxtral_is_accepted_by_public_engine_validators():
    generation = GenerationRequest(
        profile_id="profile-id",
        text="Bonjour",
        language="fr",
        engine="voxtral",
    )
    speak = SpeakRequest(text="Bonjour", engine="voxtral")
    binding = MCPClientBindingUpsert(client_id="client-id", default_engine="voxtral")

    assert generation.engine == "voxtral"
    assert speak.engine == "voxtral"
    assert binding.default_engine == "voxtral"
