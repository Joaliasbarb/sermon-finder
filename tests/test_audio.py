import os

import pytest
from pydub import AudioSegment

from sermon_finder.audio import (
    extract_window,
    get_duration_seconds,
    prepare_audio,
    split_wav,
    validate_audio_file,
)


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


# --- split_wav (T1 / V4) ---
# sample_wav is 0.1 s (100 ms). With segment=0.06 s, overlap=0.02 s → step=0.04 s:
#   chunk 0: [0, 60ms],  offset=0.0,  keep_until_s=0.04
#   chunk 1: [40, 100ms], offset=0.04, keep_until_s=0.08
#   chunk 2: [80, 100ms], offset=0.08, keep_until_s=None

def test_split_wav_single_segment(sample_wav):
    """Audio shorter than segment_s → one chunk with keep_until_s=None."""
    with split_wav(sample_wav, segment_s=0.5, overlap_s=0.05) as chunks:
        assert len(chunks) == 1
        _, offset_s, keep_until_s = chunks[0]
        assert offset_s == pytest.approx(0.0)
        assert keep_until_s is None


def test_split_wav_multiple_segments_count(sample_wav):
    """Correct number of chunks for 0.1 s audio with step=0.04 s."""
    with split_wav(sample_wav, segment_s=0.06, overlap_s=0.02) as chunks:
        assert len(chunks) == 3


def test_split_wav_offsets(sample_wav):
    """Chunk offsets advance by step_s each time."""
    segment_s, overlap_s = 0.06, 0.02
    step_s = int((segment_s - overlap_s) * 1000) / 1000.0  # mirrors implementation
    with split_wav(sample_wav, segment_s=segment_s, overlap_s=overlap_s) as chunks:
        offsets = [c[1] for c in chunks]
        assert offsets == pytest.approx([0.0, step_s, 2 * step_s])


def test_split_wav_keep_until_s_non_last(sample_wav):
    """Non-last chunks carry keep_until_s = start of next chunk (V4)."""
    segment_s, overlap_s = 0.06, 0.02
    step_s = int((segment_s - overlap_s) * 1000) / 1000.0
    with split_wav(sample_wav, segment_s=segment_s, overlap_s=overlap_s) as chunks:
        assert chunks[0][2] == pytest.approx(step_s)
        assert chunks[1][2] == pytest.approx(2 * step_s)


def test_split_wav_last_chunk_keep_until_s_none(sample_wav):
    """Last chunk has keep_until_s=None — no overlap tail to drop (V4)."""
    with split_wav(sample_wav, segment_s=0.06, overlap_s=0.02) as chunks:
        assert chunks[-1][2] is None


def test_split_wav_chunks_exist_inside_context(sample_wav):
    """Chunk WAV files exist while inside the context manager."""
    with split_wav(sample_wav, segment_s=0.06, overlap_s=0.02) as chunks:
        for path, _, _ in chunks:
            assert os.path.exists(path)


def test_split_wav_cleans_up_after_context(sample_wav):
    """Temp directory is deleted after the context manager exits (V10)."""
    with split_wav(sample_wav, segment_s=0.06, overlap_s=0.02) as chunks:
        tmp_dir = os.path.dirname(chunks[0][0])
    assert not os.path.exists(tmp_dir)


# --- extract_window (T2 / V5) ---

def test_extract_window_normal(sample_wav):
    """Window within bounds: actual_start_s equals requested start."""
    with extract_window(sample_wav, 0.02, 0.08) as (win_path, actual_start_s):
        assert actual_start_s == pytest.approx(0.02)
        assert os.path.exists(win_path)


def test_extract_window_clamps_negative_start(sample_wav):
    """Negative start clamped to 0; actual_start_s = 0.0 (V5)."""
    with extract_window(sample_wav, -0.05, 0.05) as (_, actual_start_s):
        assert actual_start_s == 0.0


def test_extract_window_clamps_end_beyond_duration(sample_wav):
    """End beyond audio duration clamped to audio length (V5)."""
    duration_s = len(AudioSegment.from_file(sample_wav)) / 1000.0
    with extract_window(sample_wav, 0.0, duration_s + 10.0) as (win_path, _):
        win_duration_s = len(AudioSegment.from_file(win_path)) / 1000.0
        assert win_duration_s == pytest.approx(duration_s, abs=0.01)


def test_extract_window_both_clamped(sample_wav):
    """start < 0 and end > duration both clamped correctly (V5)."""
    duration_s = len(AudioSegment.from_file(sample_wav)) / 1000.0
    with extract_window(sample_wav, -5.0, duration_s + 5.0) as (win_path, actual_start_s):
        assert actual_start_s == 0.0
        win_duration_s = len(AudioSegment.from_file(win_path)) / 1000.0
        assert win_duration_s == pytest.approx(duration_s, abs=0.01)


def test_extract_window_cleans_up(sample_wav):
    """Temp directory deleted after context manager exits (V10)."""
    with extract_window(sample_wav, 0.0, 0.05) as (win_path, _):
        tmp_dir = os.path.dirname(win_path)
    assert not os.path.exists(tmp_dir)
