import queue
import threading
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from sermon_finder.transcriber import transcribe_segment, transcriber_worker


def _make_segment(start: float, end: float, text: str) -> MagicMock:
    seg = MagicMock()
    seg.start = start
    seg.end = end
    seg.text = f" {text}"  # faster-whisper often adds a leading space
    return seg


# --- transcribe_segment unit tests ---

def testtranscribe_segment_applies_offset():
    """Segment timestamps are shifted by offset_s."""
    seg = _make_segment(1.0, 2.0, "bonjour")
    tl = threading.local()
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([seg]), MagicMock())
        result = transcribe_segment("/tmp/chunk.wav", 100.0, None, "small", tl)
    assert result == [{"start": 101.0, "end": 102.0, "text": "bonjour"}]


def testtranscribe_segment_drops_overlap_tail():
    """Segments at or after keep_until_s are dropped (overlap zone)."""
    segs = [_make_segment(5.0, 6.0, "keep"), _make_segment(65.0, 66.0, "drop")]
    tl = threading.local()
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(segs), MagicMock())
        result = transcribe_segment("/tmp/chunk.wav", 0.0, 60.0, "small", tl)
    assert len(result) == 1
    assert result[0]["start"] == 5.0


def testtranscribe_segment_last_chunk_keeps_all():
    """keep_until_s=None (last segment) keeps all segments regardless of position."""
    segs = [_make_segment(5.0, 6.0, "a"), _make_segment(65.0, 66.0, "b")]
    tl = threading.local()
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter(segs), MagicMock())
        result = transcribe_segment("/tmp/chunk.wav", 0.0, None, "small", tl)
    assert len(result) == 2


def testtranscribe_segment_reuses_thread_local_model():
    """WhisperModel is instantiated only once per thread_local instance."""
    tl = threading.local()
    with patch("sermon_finder.transcriber.WhisperModel") as MockModel:
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        transcribe_segment("/tmp/c.wav", 0.0, None, "small", tl)
        MockModel.return_value.transcribe.return_value = (iter([]), MagicMock())
        transcribe_segment("/tmp/c.wav", 0.0, None, "small", tl)
    MockModel.assert_called_once_with("small")


# --- transcriber_worker tests ---

@contextmanager
def _mock_extract_window(win_path="/tmp/win.wav", win_start=0.0):
    yield win_path, win_start


def _run_transcriber_worker(trans_items, found=None):
    """Fill transition_queue, run worker with mocked audio+transcribe, return queue items."""
    trans_q = queue.Queue()
    transcription_q = queue.Queue()
    found = found or threading.Event()

    for item in trans_items:
        trans_q.put(item)
    trans_q.put(None)

    fake_segments = [{"start": 10.0, "end": 12.0, "text": "bonjour"}]

    with patch("sermon_finder.transcriber.audio.extract_window", return_value=_mock_extract_window()), \
         patch("sermon_finder.transcriber.transcribe_segment", return_value=fake_segments) as mock_transcribe:
        transcriber_worker(trans_q, transcription_q, found, "/tmp/audio.wav", "small", threading.local())

    items = []
    while not transcription_q.empty():
        items.append(transcription_q.get_nowait())
    return items, mock_transcribe


def test_transcriber_worker_forwards_sentinel_immediately():
    trans_q = queue.Queue()
    transcription_q = queue.Queue()
    trans_q.put(None)
    with patch("sermon_finder.transcriber.audio.extract_window"):
        transcriber_worker(trans_q, transcription_q, threading.Event(), "/tmp/a.wav", "small", threading.local())
    assert transcription_q.get_nowait() is None
    assert transcription_q.empty()


def test_transcriber_worker_pushes_transcription():
    item = (30.0, 1, 1, 2, 0.0, 240.0)
    items, _ = _run_transcriber_worker([item])
    result = items[0]
    assert result[0] == 30.0                                   # t
    assert result[1] == [{"start": 10.0, "end": 12.0, "text": "bonjour"}]  # segments
    assert result[2] == "small"                                # model_size
    assert result[3] == 1                                      # segment_idx
    assert result[4] == 1                                      # transition_idx
    assert result[5] == 2                                      # total_transitions
    assert items[-1] is None                                   # sentinel last


def test_transcriber_worker_uses_correct_window():
    item = (60.0, 1, 1, 1, 0.0, 240.0)
    with patch("sermon_finder.transcriber.audio.extract_window", return_value=_mock_extract_window()) as mock_win, \
         patch("sermon_finder.transcriber.transcribe_segment", return_value=[]):
        trans_q, transcription_q = queue.Queue(), queue.Queue()
        trans_q.put(item)
        trans_q.put(None)
        transcriber_worker(trans_q, transcription_q, threading.Event(), "/tmp/a.wav", "small", threading.local())
    mock_win.assert_called_once_with("/tmp/a.wav", 30.0, 90.0)  # t-30, t+30


def test_transcriber_worker_stops_on_found_event():
    found = threading.Event()
    found.set()
    trans_q, transcription_q = queue.Queue(), queue.Queue()
    trans_q.put((30.0, 1, 1, 1, 0.0, 240.0))
    trans_q.put(None)
    with patch("sermon_finder.transcriber.audio.extract_window") as mock_win:
        transcriber_worker(trans_q, transcription_q, found, "/tmp/a.wav", "small", threading.local())
    mock_win.assert_not_called()
    assert transcription_q.get_nowait() is None
