import queue
import threading
from unittest.mock import MagicMock, patch

from sermon_finder.diarizer import diarizer_worker, get_speaker_transitions, run_diarization


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


# --- diarizer_worker tests ---

def _run_worker(seg_items, transitions_per_segment, found=None):
    """Helper: fill segment_queue, run worker, return all transition_queue items."""
    seg_q = queue.Queue()
    trans_q = queue.Queue()
    found = found or threading.Event()

    for item in seg_items:
        seg_q.put(item)
    seg_q.put(None)  # sentinel

    side_effects = []
    for transitions in transitions_per_segment:
        mock_segs = [{"speaker": "A", "start": 0.0, "end": 1.0}]
        side_effects.append(mock_segs)

    with patch("sermon_finder.diarizer.run_diarization", side_effect=side_effects), \
         patch("sermon_finder.diarizer.get_speaker_transitions", side_effect=transitions_per_segment):
        diarizer_worker(seg_q, trans_q, found)

    items = []
    while not trans_q.empty():
        items.append(trans_q.get_nowait())
    return items


def test_diarizer_worker_forwards_sentinel_immediately():
    seg_q = queue.Queue()
    trans_q = queue.Queue()
    seg_q.put(None)
    diarizer_worker(seg_q, trans_q, threading.Event())
    assert trans_q.get_nowait() is None
    assert trans_q.empty()


def test_diarizer_worker_pushes_transitions():
    seg = ("/tmp/c.wav", 0.0, None, 1, 1)
    items = _run_worker([seg], [[10.0, 20.0]])
    transitions = [i for i in items if i is not None]
    assert len(transitions) == 2
    assert transitions[0][0] == 10.0   # t
    assert transitions[0][1] == 1      # segment_idx
    assert transitions[0][2] == 1      # transition_idx
    assert transitions[0][3] == 2      # total_transitions
    assert transitions[1][0] == 20.0
    assert transitions[1][2] == 2
    assert items[-1] is None           # sentinel last


def test_diarizer_worker_skips_empty_segment():
    seg = ("/tmp/c.wav", 0.0, None, 1, 1)
    items = _run_worker([seg], [[]])
    assert items == [None]             # only sentinel, no transitions


def test_diarizer_worker_stops_on_found_event():
    found = threading.Event()
    found.set()
    seg_q = queue.Queue()
    trans_q = queue.Queue()
    seg_q.put(("/tmp/c.wav", 0.0, None, 1, 1))
    seg_q.put(None)
    with patch("sermon_finder.diarizer.run_diarization") as mock_diarize:
        diarizer_worker(seg_q, trans_q, found)
    mock_diarize.assert_not_called()
    assert trans_q.get_nowait() is None


def test_diarizer_worker_seg_end_s_last_segment():
    """Last segment (keep_until_s=None) uses offset_s + 240 for seg_end_s."""
    seg = ("/tmp/c.wav", 60.0, None, 1, 1)
    items = _run_worker([seg], [[90.0]])
    t, seg_idx, t_idx, total, offset_s, seg_end_s = items[0]
    assert seg_end_s == 60.0 + 240.0


def test_diarizer_worker_seg_end_s_non_last_segment():
    """Non-last segment uses keep_until_s + 30 for seg_end_s."""
    seg = ("/tmp/c.wav", 0.0, 210.0, 1, 2)
    items = _run_worker([seg], [[90.0]])
    t, seg_idx, t_idx, total, offset_s, seg_end_s = items[0]
    assert seg_end_s == 210.0 + 30.0
