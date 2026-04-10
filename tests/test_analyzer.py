import pytest
from unittest.mock import MagicMock

from sermon_finder.analyzer import (
    ClaudeProvider,
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


# --- ClaudeProvider tests ---

def test_claude_provider_api_call():
    mock_client = MagicMock()
    mock_client.messages.create.return_value.content = [MagicMock(text=" [10:00] ")]
    provider = ClaudeProvider(client=mock_client)
    result = provider.complete("system msg", "user msg")
    assert result == "[10:00]"
    kwargs = mock_client.messages.create.call_args[1]
    assert kwargs["max_tokens"] == 20
    assert kwargs["system"] == "system msg"
    assert kwargs["messages"] == [{"role": "user", "content": "user msg"}]


# --- is_sermon_transition tests ---

def test_is_sermon_transition_yes():
    provider = MagicMock()
    provider.complete.return_value = "YES"
    assert is_sermon_transition([seg(0, 5)], provider=provider) is True


def test_is_sermon_transition_no():
    provider = MagicMock()
    provider.complete.return_value = "NO"
    assert is_sermon_transition([seg(0, 5)], provider=provider) is False


def test_is_sermon_transition_yes_case_insensitive():
    provider = MagicMock()
    provider.complete.return_value = "yes, this is the transition"
    assert is_sermon_transition([seg(0, 5)], provider=provider) is True


def test_is_sermon_transition_uses_correct_prompts():
    provider = MagicMock()
    provider.complete.return_value = "NO"
    is_sermon_transition([seg(60, 65, "Bonjour frères")], provider=provider)
    system, user = provider.complete.call_args[0]
    assert "YES" in system and "NO" in system
    assert "Bonjour frères" in user
