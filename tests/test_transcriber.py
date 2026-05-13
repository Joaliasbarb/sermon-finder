from unittest.mock import MagicMock, patch

from sermon_finder.transcriber import transcribe_segment


def _make_segment(start: float, end: float, text: str) -> MagicMock:
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = f" {text}"  # faster-whisper often adds a leading space
    return seg


def testtranscribe_segment_applies_offset():
    """Segment timestamps are shifted by offset_s."""
    seg = _make_segment(1.0, 2.0, "bonjour")
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([seg]), MagicMock())
        result = transcribe_segment("/tmp/chunk.wav", 100.0, None, "small", {})
    assert result == [{"start": 101.0, "end": 102.0, "text": "bonjour"}]


def testtranscribe_segment_drops_overlap_tail():
    """Segments at or after keep_until_s are dropped (overlap zone)."""
    segs = [_make_segment(5.0, 6.0, "keep"), _make_segment(65.0, 66.0, "drop")]
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(segs), MagicMock())
        result = transcribe_segment("/tmp/chunk.wav", 0.0, 60.0, "small", {})
    assert len(result) == 1
    assert result[0]["start"] == 5.0


def testtranscribe_segment_last_chunk_keeps_all():
    """keep_until_s=None (last segment) keeps all segments regardless of position."""
    segs = [_make_segment(5.0, 6.0, "a"), _make_segment(65.0, 66.0, "b")]
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(segs), MagicMock())
        result = transcribe_segment("/tmp/chunk.wav", 0.0, None, "small", {})
    assert len(result) == 2


def testtranscribe_segment_reuses_cached_model():
    """WhisperModel is instantiated only once per model_cache dict (V6)."""
    cache: dict = {}
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        transcribe_segment("/tmp/c.wav", 0.0, None, "small", cache)
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        transcribe_segment("/tmp/c.wav", 0.0, None, "small", cache)
    MockModel.assert_called_once_with("small")


def testtranscribe_segment_separate_cache_per_model_size():
    """Different model sizes use separate cache entries; both get instantiated."""
    cache: dict = {}
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        transcribe_segment("/tmp/c.wav", 0.0, None, "small", cache)
        transcribe_segment("/tmp/c.wav", 0.0, None, "medium", cache)
    assert MockModel.call_count == 2
