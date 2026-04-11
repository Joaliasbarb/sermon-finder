import pytest
from unittest.mock import MagicMock

from sermon_finder.analyzer import (
    ClaudeProvider,
    TransitionResult,
    _format_chunk,
    _format_timestamp,
    is_sermon_transition,
)


def seg(start: float, end: float, text: str = "text") -> dict:
    return {"start": start, "end": end, "text": text}


# --- Helper function tests ---

def test_format_timestamp_zero():
    assert _format_timestamp(0) == "00:00"


def test_format_timestamp_with_minutes():
    assert _format_timestamp(65) == "01:05"


def test_format_timestamp_large():
    assert _format_timestamp(3661) == "61:01"


def test_format_chunk():
    segments = [seg(65.0, 70.0, "Bonjour"), seg(70.0, 75.0, "Bienvenue")]
    result = _format_chunk(segments)
    assert result == "[01:05] Bonjour\n[01:10] Bienvenue"


def test_format_chunk_with_transition_marker():
    segments = [seg(60.0, 65.0, "Avant"), seg(75.0, 80.0, "Après")]
    result = _format_chunk(segments, transition_t=70.0)
    assert result == "[01:00] Avant\n--- transition at 01:10 ---\n[01:15] Après"


def test_format_chunk_marker_at_exact_segment_start():
    segments = [seg(60.0, 65.0, "Avant"), seg(70.0, 75.0, "Après")]
    result = _format_chunk(segments, transition_t=70.0)
    assert result == "[01:00] Avant\n--- transition at 01:10 ---\n[01:10] Après"


# --- ClaudeProvider tests ---

def test_claude_provider_api_call():
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [MagicMock(text=" DECISION: YES\nQUALITY: GOOD ")]
    provider = ClaudeProvider(client=mock_client)
    result = provider.complete("system msg", "user msg")
    assert result == "DECISION: YES\nQUALITY: GOOD"
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["max_tokens"] == 50
    assert kwargs["system"] == "system msg"
    assert kwargs["messages"] == [{"role": "user", "content": "user msg"}]


# --- is_sermon_transition tests ---

def test_is_sermon_transition_yes():
    provider = MagicMock()
    provider.complete.return_value = "DECISION: YES\nQUALITY: GOOD"
    result = is_sermon_transition([seg(0, 5)], provider=provider)
    assert result.is_sermon is True
    assert result.uncertain is False
    assert result.quality_ok is True


def test_is_sermon_transition_no():
    provider = MagicMock()
    provider.complete.return_value = "DECISION: NO\nQUALITY: GOOD"
    result = is_sermon_transition([seg(0, 5)], provider=provider)
    assert result.is_sermon is False
    assert result.uncertain is False
    assert result.quality_ok is True


def test_is_sermon_transition_yes_case_insensitive():
    provider = MagicMock()
    provider.complete.return_value = "decision: yes\nquality: good"
    result = is_sermon_transition([seg(0, 5)], provider=provider)
    assert result.is_sermon is True


def test_is_sermon_transition_unsure():
    provider = MagicMock()
    provider.complete.return_value = "DECISION: UNSURE\nQUALITY: GOOD"
    result = is_sermon_transition([seg(0, 5)], provider=provider)
    assert result.is_sermon is False
    assert result.uncertain is True
    assert result.quality_ok is True


def test_is_sermon_transition_poor_quality():
    provider = MagicMock()
    provider.complete.return_value = "DECISION: NO\nQUALITY: POOR"
    result = is_sermon_transition([seg(0, 5)], provider=provider)
    assert result.is_sermon is False
    assert result.quality_ok is False


def test_is_sermon_transition_uses_correct_prompts():
    provider = MagicMock()
    provider.complete.return_value = "DECISION: NO\nQUALITY: GOOD"
    segments = [seg(55, 62, "Bonjour frères"), seg(62, 70, "Chers amis")]
    is_sermon_transition(segments, transition_t=62.0, provider=provider)
    system, user = provider.complete.call_args[0]
    assert "YES" in system and "NO" in system and "UNSURE" in system
    assert "Bonjour frères" in user
    assert "transition at 01:02" in user
