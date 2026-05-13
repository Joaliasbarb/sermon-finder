from unittest.mock import MagicMock, patch

from sermon_finder.analyzer import (
    TransitionResult,
    WHISPER_MODELS,
    _next_whisper_model,
    validate_transition,
)


def _ok():
    return TransitionResult(is_sermon=True, uncertain=False, quality_ok=True)


def _no():
    return TransitionResult(is_sermon=False, uncertain=False, quality_ok=True)


def _poor_no():
    return TransitionResult(is_sermon=False, uncertain=False, quality_ok=False)


def _poor_yes():
    return TransitionResult(is_sermon=True, uncertain=False, quality_ok=False)


def _segments(t=30.0):
    return [{"start": t, "end": t + 5, "text": "test"}]


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


# --- validate_transition ---

def test_validate_transition_yes_returns_result():
    with patch("sermon_finder.analyzer.is_sermon_transition", return_value=_ok()):
        result, models = validate_transition(30.0, _segments(), "small", MagicMock(), -1)
    assert result.is_sermon
    assert models == ["small"]


def test_validate_transition_poor_yes_no_retry():
    """V8: YES accepted regardless of quality — no retry even when POOR."""
    retranscribe = MagicMock()
    with patch("sermon_finder.analyzer.is_sermon_transition", return_value=_poor_yes()):
        result, models = validate_transition(
            30.0, _segments(), "small", MagicMock(),
            WHISPER_MODELS.index("medium"),
            retranscribe_fn=retranscribe,
        )
    assert result.is_sermon
    retranscribe.assert_not_called()


def test_validate_transition_poor_no_retries_with_larger_model():
    """V8: POOR+NO triggers retry with next model."""
    retranscribe = MagicMock(return_value=_segments())
    with patch("sermon_finder.analyzer.is_sermon_transition", side_effect=[_poor_no(), _ok()]):
        result, models = validate_transition(
            30.0, _segments(), "small", MagicMock(),
            WHISPER_MODELS.index("medium"),
            retranscribe_fn=retranscribe,
        )
    assert result.is_sermon
    assert models == ["small", "medium"]
    retranscribe.assert_called_once_with(30.0, "medium")


def test_validate_transition_no_retry_without_cap():
    """No retry when retry_cap_idx=-1."""
    retranscribe = MagicMock()
    with patch("sermon_finder.analyzer.is_sermon_transition", return_value=_poor_no()):
        result, models = validate_transition(
            30.0, _segments(), "small", MagicMock(), -1,
            retranscribe_fn=retranscribe,
        )
    assert not result.is_sermon
    assert models == ["small"]
    retranscribe.assert_not_called()


def test_validate_transition_poor_no_stops_at_cap():
    """Retry stops at cap even if result is still POOR."""
    retranscribe = MagicMock(return_value=_segments())
    with patch("sermon_finder.analyzer.is_sermon_transition", side_effect=[_poor_no(), _poor_no()]):
        result, models = validate_transition(
            30.0, _segments(), "small", MagicMock(),
            WHISPER_MODELS.index("medium"),
            retranscribe_fn=retranscribe,
        )
    assert not result.is_sermon
    assert models == ["small", "medium"]
    assert retranscribe.call_count == 1


def test_validate_transition_no_returns_models_tried():
    """NO result still returns models_tried list."""
    with patch("sermon_finder.analyzer.is_sermon_transition", return_value=_no()):
        result, models = validate_transition(30.0, _segments(), "small", MagicMock(), -1)
    assert not result.is_sermon
    assert models == ["small"]
