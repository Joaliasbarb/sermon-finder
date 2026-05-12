import queue
import threading
import time
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from sermon_finder.analyzer import (
    TransitionResult,
    WHISPER_MODELS,
    _next_whisper_model,
    validator_worker,
)


def _ok():
    return TransitionResult(is_sermon=True, uncertain=False, quality_ok=True)


def _no():
    return TransitionResult(is_sermon=False, uncertain=False, quality_ok=True)


def _poor_no():
    return TransitionResult(is_sermon=False, uncertain=False, quality_ok=False)


def _poor_yes():
    return TransitionResult(is_sermon=True, uncertain=False, quality_ok=False)


def _item(t=30.0, model_size="small", seg_idx=1, trans_idx=1, total=1):
    segments = [{"start": t, "end": t + 5, "text": "test"}]
    return (t, segments, model_size, seg_idx, trans_idx, total, 0.0, 240.0)


def _run_validator(items, llm_results, retranscribe_fn=None, retry_cap_idx=-1,
                   found=None, model_ready=None):
    trans_q = queue.Queue()
    found = found or threading.Event()
    model_ready = model_ready or threading.Event()
    model_ready.set()
    result_holder = []

    for item in items:
        trans_q.put(item)
    trans_q.put(None)

    with patch("sermon_finder.analyzer.is_sermon_transition", side_effect=llm_results):
        validator_worker(
            trans_q, found, model_ready,
            MagicMock(), retry_cap_idx, result_holder,
            retranscribe_fn=retranscribe_fn,
        )
    return result_holder, found


# --- _next_whisper_model ---

def test_next_whisper_model_returns_next():
    assert _next_whisper_model("small", WHISPER_MODELS.index("medium")) == "medium"


def test_next_whisper_model_no_retry_cap():
    assert _next_whisper_model("small", -1) is None


def test_next_whisper_model_cap_exceeded():
    # "medium" is at index 3; cap at "small" (index 2) — cannot step up
    assert _next_whisper_model("medium", WHISPER_MODELS.index("small")) is None


def test_next_whisper_model_at_largest():
    assert _next_whisper_model("large-v3", WHISPER_MODELS.index("large-v3")) is None


# --- sentinel / early-exit ---

def test_validator_worker_forwards_sentinel_immediately():
    trans_q = queue.Queue()
    trans_q.put(None)
    found = threading.Event()
    model_ready = threading.Event()
    model_ready.set()
    result_holder = []
    with patch("sermon_finder.analyzer.is_sermon_transition") as mock_llm:
        validator_worker(trans_q, found, model_ready, MagicMock(), -1, result_holder)
    mock_llm.assert_not_called()
    assert not found.is_set()
    assert result_holder == []


def test_validator_worker_stops_on_found_event():
    found = threading.Event()
    found.set()
    trans_q = queue.Queue()
    trans_q.put(_item())
    trans_q.put(None)
    model_ready = threading.Event()
    model_ready.set()
    with patch("sermon_finder.analyzer.is_sermon_transition") as mock_llm:
        validator_worker(trans_q, found, model_ready, MagicMock(), -1, [])
    mock_llm.assert_not_called()


# --- V13: blocks on model_ready ---

def test_validator_worker_blocks_until_model_ready():
    """LLM should not be called before model_ready is set."""
    trans_q = queue.Queue()
    trans_q.put(_item())
    trans_q.put(None)
    found = threading.Event()
    model_ready = threading.Event()  # NOT yet set
    result_holder = []
    llm_called = threading.Event()

    def fake_llm(*args, **kwargs):
        llm_called.set()
        return _ok()

    t = threading.Thread(
        target=validator_worker,
        args=(trans_q, found, model_ready, MagicMock(), -1, result_holder),
        kwargs={"retranscribe_fn": None},
    )
    with patch("sermon_finder.analyzer.is_sermon_transition", side_effect=fake_llm):
        t.start()
        time.sleep(0.05)
        assert not llm_called.is_set(), "LLM called before model_ready was set"
        model_ready.set()
        t.join(timeout=2.0)
    assert llm_called.is_set()


# --- V8: YES accepted ---

def test_validator_worker_yes_sets_found():
    result_holder, found = _run_validator([_item(t=60.0)], [_ok()])
    assert found.is_set()
    assert result_holder == [(1, 0)]  # 60s = 1'00


def test_validator_worker_poor_yes_accepted_without_retry():
    """V8: YES accepted regardless of quality — no retry even when POOR."""
    retranscribe = MagicMock()
    result_holder, found = _run_validator(
        [_item()], [_poor_yes()],
        retranscribe_fn=retranscribe,
        retry_cap_idx=WHISPER_MODELS.index("medium"),
    )
    assert found.is_set()
    retranscribe.assert_not_called()


# --- V8: POOR + NO retry ---

def test_validator_worker_poor_no_retries_with_larger_model():
    retranscribe = MagicMock(return_value=[{"start": 30.0, "end": 35.0, "text": "retry"}])
    result_holder, found = _run_validator(
        [_item(model_size="small")],
        [_poor_no(), _ok()],
        retranscribe_fn=retranscribe,
        retry_cap_idx=WHISPER_MODELS.index("medium"),
    )
    assert found.is_set()
    retranscribe.assert_called_once_with(30.0, "medium")


def test_validator_worker_poor_no_no_retry_without_cap():
    retranscribe = MagicMock()
    result_holder, found = _run_validator(
        [_item()], [_poor_no()],
        retranscribe_fn=retranscribe,
        retry_cap_idx=-1,
    )
    assert not found.is_set()
    retranscribe.assert_not_called()


def test_validator_worker_poor_no_retries_respect_cap():
    """Retry stops at cap even if result is still POOR."""
    retranscribe = MagicMock(return_value=[])
    result_holder, found = _run_validator(
        [_item(model_size="small")],
        [_poor_no(), _poor_no()],  # small → POOR, medium → POOR, cap=medium so no further retry
        retranscribe_fn=retranscribe,
        retry_cap_idx=WHISPER_MODELS.index("medium"),
    )
    assert not found.is_set()
    assert retranscribe.call_count == 1


# --- on_result callback ---

def test_validator_worker_on_result_called_for_yes():
    on_result = MagicMock()
    trans_q = queue.Queue()
    trans_q.put(_item(t=90.0, seg_idx=2, trans_idx=3, total=5))
    trans_q.put(None)
    found = threading.Event()
    model_ready = threading.Event()
    model_ready.set()
    with patch("sermon_finder.analyzer.is_sermon_transition", return_value=_ok()):
        validator_worker(trans_q, found, model_ready, MagicMock(), -1, [], on_result=on_result)
    on_result.assert_called_once()
    args = on_result.call_args[0]
    assert args[0] == 90.0    # t
    assert args[1].is_sermon  # result
    assert args[4] == 3        # trans_idx
    assert args[5] == 5        # total_trans
    assert args[6] == 2        # seg_idx


def test_validator_worker_on_result_called_for_no():
    on_result = MagicMock()
    trans_q = queue.Queue()
    trans_q.put(_item())
    trans_q.put(None)
    found = threading.Event()
    model_ready = threading.Event()
    model_ready.set()
    with patch("sermon_finder.analyzer.is_sermon_transition", return_value=_no()):
        validator_worker(trans_q, found, model_ready, MagicMock(), -1, [], on_result=on_result)
    on_result.assert_called_once()
    assert not on_result.call_args[0][1].is_sermon
