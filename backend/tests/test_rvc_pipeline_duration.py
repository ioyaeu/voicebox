"""RVC duration-preservation unit tests.

These tests avoid the real neural backbones and target the frame-length contract
inside ``RVCPipeline._vc``: the synthesizer frame count must come from the audio
hop grid, not from ContentVec's exact framing.
"""

import numpy as np
import pytest
import torch

from backend.backends.rvc import pipeline as pipeline_mod
from backend.backends.rvc.pipeline import _WINDOW, RVCPipeline


class _FakeSynth:
    def __init__(self):
        self.frame_count = None
        self.feature_frames = None
        self.pitch_frames = None

    def infer(self, feats, phone_lengths, pitch, pitchf, sid):
        self.frame_count = int(phone_lengths.item())
        self.feature_frames = int(feats.shape[1])
        self.pitch_frames = int(pitch.shape[1])
        return (torch.zeros(1, 1, self.frame_count * 480),)


@pytest.mark.parametrize("raw_feature_frames", [32, 70])
def test_vc_matches_content_features_to_audio_frame_count(
    monkeypatch, raw_feature_frames
):
    expected_frames = 100
    fake_synth = _FakeSynth()

    def fake_extract_contentvec_features(audio0, version, *, device=None, is_half=False):
        assert audio0.shape[0] == expected_frames * _WINDOW
        return torch.zeros(1, raw_feature_frames, 768)

    monkeypatch.setattr(
        pipeline_mod, "extract_contentvec_features", fake_extract_contentvec_features
    )

    pipe = RVCPipeline()
    pipe._device = "cpu"
    pipe._version = "v2"
    pipe._is_half = False
    pipe._net_g = fake_synth

    audio = np.zeros(expected_frames * _WINDOW, dtype=np.float32)
    pitch = torch.ones(1, expected_frames, dtype=torch.long)
    pitchf = torch.ones(1, expected_frames, dtype=torch.float32)
    sid = torch.tensor([0], dtype=torch.long)

    out = pipe._vc(
        sid,
        audio,
        pitch,
        pitchf,
        index=None,
        big_npy=None,
        index_rate=0.0,
        protect=0.33,
    )

    assert fake_synth.frame_count == expected_frames
    assert fake_synth.feature_frames == expected_frames
    assert fake_synth.pitch_frames == expected_frames
    assert out.shape[0] == expected_frames * 480
