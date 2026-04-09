import os

import pytest

from sermon_finder.audio import get_duration_seconds, prepare_audio, validate_audio_file


def test_validate_missing_file():
    with pytest.raises(FileNotFoundError):
        validate_audio_file("/nonexistent/path/audio.mp3")


def test_validate_unsupported_extension(tmp_path):
    f = tmp_path / "audio.xyz"
    f.write_bytes(b"fake")
    with pytest.raises(ValueError, match="Unsupported format"):
        validate_audio_file(str(f))


def test_validate_supported_wav(sample_wav):
    result = validate_audio_file(sample_wav)
    assert result.suffix == ".wav"


def test_prepare_audio_yields_wav(sample_wav):
    with prepare_audio(sample_wav) as wav_path:
        assert wav_path.endswith(".wav")
        assert os.path.exists(wav_path)


def test_prepare_audio_cleans_up(sample_wav):
    with prepare_audio(sample_wav) as wav_path:
        tmp_dir = os.path.dirname(wav_path)
    assert not os.path.exists(tmp_dir)


def test_get_duration_seconds(sample_wav):
    duration = get_duration_seconds(sample_wav)
    assert duration == pytest.approx(0.1, abs=0.02)
