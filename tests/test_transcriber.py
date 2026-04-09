from unittest.mock import MagicMock, patch

from sermon_finder.transcriber import transcribe


def _make_segment(start: float, end: float, text: str) -> MagicMock:
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = f" {text}"  # faster-whisper often adds a leading space
    return seg


def _make_info(duration: float = 10.0) -> MagicMock:
    info = MagicMock()
    info.duration = duration
    return info


def test_transcribe_returns_segments():
    mock_segments = [
        _make_segment(0.0, 5.0, "Bonjour à tous"),
        _make_segment(5.0, 10.0, "Nous allons commencer"),
    ]
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(mock_segments), _make_info(10.0))
        result = transcribe("audio.wav")

    assert result == [
        {"start": 0.0, "end": 5.0, "text": "Bonjour à tous"},
        {"start": 5.0, "end": 10.0, "text": "Nous allons commencer"},
    ]


def test_transcribe_forces_french():
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([]), _make_info())
        transcribe("audio.wav")
        _, kwargs = MockModel.return_value.transcribe.call_args
    assert kwargs.get("language") == "fr"


def test_transcribe_enables_vad_filter():
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([]), _make_info())
        transcribe("audio.wav")
        _, kwargs = MockModel.return_value.transcribe.call_args
    assert kwargs.get("vad_filter") is True


def test_transcribe_passes_model_size():
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([]), _make_info())
        transcribe("audio.wav", model_size="small")
    MockModel.assert_called_once_with("small")
