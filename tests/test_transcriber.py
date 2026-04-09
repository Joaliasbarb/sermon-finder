import threading
from unittest.mock import MagicMock, patch

from sermon_finder.transcriber import _transcribe_segment, transcribe


def _make_segment(start: float, end: float, text: str) -> MagicMock:
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = f" {text}"  # faster-whisper often adds a leading space
    return seg


def _make_split_wav_ctx(chunks):
    """Return a mock context manager for split_wav that yields chunks."""
    ctx = MagicMock()
    ctx.__enter__.return_value = chunks
    ctx.__exit__.return_value = False
    return ctx


# --- Core transcription tests (num_workers=1) ---

def test_transcribe_returns_segments():
    mock_segments = [
        _make_segment(0.0, 5.0, "Bonjour à tous"),
        _make_segment(5.0, 10.0, "Nous allons commencer"),
    ]
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel, \
         patch("sermon_finder.transcriber.split_wav") as mock_split:
        mock_split.return_value = _make_split_wav_ctx([("/tmp/c.wav", 0.0, None)])
        MockModel.return_value.transcribe.return_value = (iter(mock_segments), MagicMock())
        result = transcribe("audio.wav")

    assert result == [
        {"start": 0.0, "end": 5.0, "text": "Bonjour à tous"},
        {"start": 5.0, "end": 10.0, "text": "Nous allons commencer"},
    ]


def test_transcribe_forces_french():
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel, \
         patch("sermon_finder.transcriber.split_wav") as mock_split:
        mock_split.return_value = _make_split_wav_ctx([("/tmp/c.wav", 0.0, None)])
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        transcribe("audio.wav")
        _, kwargs = MockModel.return_value.transcribe.call_args
    assert kwargs.get("language") == "fr"


def test_transcribe_enables_vad_filter():
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel, \
         patch("sermon_finder.transcriber.split_wav") as mock_split:
        mock_split.return_value = _make_split_wav_ctx([("/tmp/c.wav", 0.0, None)])
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        transcribe("audio.wav")
        _, kwargs = MockModel.return_value.transcribe.call_args
    assert kwargs.get("vad_filter") is True


def test_transcribe_passes_model_size():
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel, \
         patch("sermon_finder.transcriber.split_wav") as mock_split:
        mock_split.return_value = _make_split_wav_ctx([("/tmp/c.wav", 0.0, None)])
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        transcribe("audio.wav", model_size="small")
    MockModel.assert_called_once_with("small")


# --- _transcribe_segment unit tests ---

def test_transcribe_segment_applies_offset():
    """Segment timestamps are shifted by offset_s."""
    seg = _make_segment(1.0, 2.0, "bonjour")
    tl = threading.local()
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([seg]), MagicMock())
        result = _transcribe_segment("/tmp/chunk.wav", 100.0, None, "small", tl)
    assert result == [{"start": 101.0, "end": 102.0, "text": "bonjour"}]


def test_transcribe_segment_drops_overlap_tail():
    """Segments at or after keep_until_s are dropped (overlap zone)."""
    segs = [_make_segment(5.0, 6.0, "keep"), _make_segment(65.0, 66.0, "drop")]
    tl = threading.local()
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(segs), MagicMock())
        result = _transcribe_segment("/tmp/chunk.wav", 0.0, 60.0, "small", tl)
    assert len(result) == 1
    assert result[0]["start"] == 5.0


def test_transcribe_segment_last_chunk_keeps_all():
    """keep_until_s=None (last segment) keeps all segments regardless of position."""
    segs = [_make_segment(5.0, 6.0, "a"), _make_segment(65.0, 66.0, "b")]
    tl = threading.local()
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(segs), MagicMock())
        result = _transcribe_segment("/tmp/chunk.wav", 0.0, None, "small", tl)
    assert len(result) == 2


def test_transcribe_segment_reuses_thread_local_model():
    """WhisperModel is instantiated only once per thread_local instance."""
    tl = threading.local()
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        _transcribe_segment("/tmp/c.wav", 0.0, None, "small", tl)
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        _transcribe_segment("/tmp/c.wav", 0.0, None, "small", tl)
    MockModel.assert_called_once_with("small")


# --- Parallel / merge tests ---

def test_transcribe_parallel_merges_sorted():
    """With num_workers=2, results from all chunks are merged in start-time order."""
    chunks = [("/tmp/c0.wav", 0.0, 90.0), ("/tmp/c1.wav", 90.0, None)]

    def fake_segment(path, offset, keep_until, model_size, tl):
        if path == "/tmp/c0.wav":
            return [{"start": 65.0, "end": 70.0, "text": "second"}]
        return [{"start": 5.0, "end": 10.0, "text": "first"}]

    with patch("sermon_finder.transcriber.split_wav") as mock_split, \
         patch("sermon_finder.transcriber._transcribe_segment", side_effect=fake_segment):
        mock_split.return_value = _make_split_wav_ctx(chunks)
        result = transcribe("audio.wav", num_workers=2)

    assert result[0] == {"start": 5.0, "end": 10.0, "text": "first"}
    assert result[1] == {"start": 65.0, "end": 70.0, "text": "second"}


def test_transcribe_single_segment_audio():
    """Audio shorter than segment_s produces one chunk and returns its segments."""
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel, \
         patch("sermon_finder.transcriber.split_wav") as mock_split:
        mock_split.return_value = _make_split_wav_ctx([("/tmp/c0.wav", 0.0, None)])
        MockModel.return_value.transcribe.return_value = (
            iter([_make_segment(1.0, 2.0, "hello")]), MagicMock()
        )
        result = transcribe("audio.wav")
    assert result == [{"start": 1.0, "end": 2.0, "text": "hello"}]
