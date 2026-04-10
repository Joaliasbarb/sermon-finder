from unittest.mock import MagicMock, patch

from sermon_finder.diarizer import get_speaker_transitions, run_diarization


def _seg(speaker: str, start: float, end: float) -> dict:
    return {"speaker": speaker, "start": start, "end": end}


# --- get_speaker_transitions ---

def test_get_speaker_transitions_empty():
    assert get_speaker_transitions([]) == []


def test_get_speaker_transitions_no_change():
    segs = [_seg("A", 0.0, 5.0), _seg("A", 5.0, 10.0), _seg("A", 10.0, 15.0)]
    assert get_speaker_transitions(segs) == []


def test_get_speaker_transitions_one_change():
    segs = [_seg("A", 0.0, 5.0), _seg("B", 5.0, 10.0)]
    assert get_speaker_transitions(segs) == [5.0]


def test_get_speaker_transitions_multiple():
    segs = [
        _seg("A", 0.0, 5.0),
        _seg("B", 5.0, 10.0),
        _seg("A", 10.0, 15.0),
        _seg("A", 15.0, 20.0),
        _seg("C", 20.0, 25.0),
    ]
    assert get_speaker_transitions(segs) == [5.0, 10.0, 20.0]


# --- run_diarization ---

def _make_mock_diarize_result(segments):
    """Build a mock DiarizeResult from (speaker, start, end) tuples."""
    mock_segs = []
    for speaker, start, end in segments:
        s = MagicMock()
        s.speaker = speaker
        s.start = start
        s.end = end
        mock_segs.append(s)
    result = MagicMock()
    result.segments = mock_segs
    return result


def test_run_diarization_applies_offset():
    raw = [("SPEAKER_0", 1.0, 3.0), ("SPEAKER_1", 3.5, 6.0)]
    with patch("sermon_finder.diarizer._diarize", return_value=_make_mock_diarize_result(raw)) as mock_fn:
        result = run_diarization("chunk.wav", offset_s=10.0)

    assert result == [
        {"speaker": "SPEAKER_0", "start": 11.0, "end": 13.0},
        {"speaker": "SPEAKER_1", "start": 13.5, "end": 16.0},
    ]


def test_run_diarization_zero_offset():
    raw = [("SPEAKER_0", 0.0, 5.0)]
    with patch("sermon_finder.diarizer._diarize", return_value=_make_mock_diarize_result(raw)):
        result = run_diarization("chunk.wav")

    assert result == [{"speaker": "SPEAKER_0", "start": 0.0, "end": 5.0}]


def test_run_diarization_no_forced_speaker_count():
    """_diarize must be called without min_speakers to allow auto-detection."""
    raw = [("SPEAKER_0", 0.0, 5.0)]
    with patch("sermon_finder.diarizer._diarize", return_value=_make_mock_diarize_result(raw)) as mock_fn:
        run_diarization("chunk.wav")

    mock_fn.assert_called_once_with("chunk.wav")
