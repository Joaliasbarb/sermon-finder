import wave

import pytest


@pytest.fixture
def sample_wav(tmp_path):
    """Minimal valid 16kHz mono WAV file (0.1 s of silence)."""
    wav_path = tmp_path / "sample.wav"
    with wave.open(str(wav_path), "w") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(16000)
        f.writeframes(b"\x00\x00" * 1600)
    return str(wav_path)
