import pytest
from unittest.mock import MagicMock, patch

import httpx

from sermon_finder.analyzer import (
    ClaudeProvider,
    OllamaProvider,
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


# --- OllamaProvider tests ---

def test_ollama_provider_sends_correct_request():
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": " DECISION: YES\nQUALITY: GOOD "}}
    mock_response.raise_for_status = MagicMock()

    with patch("sermon_finder.analyzer.httpx.post", return_value=mock_response) as mock_post:
        provider = OllamaProvider(model="mistral", base_url="http://localhost:11434")
        result = provider.complete("sys prompt", "user prompt")

    assert result == "DECISION: YES\nQUALITY: GOOD"
    call_kwargs = mock_post.call_args
    assert call_kwargs[0][0] == "http://localhost:11434/api/chat"
    body = call_kwargs[1]["json"]
    assert body["model"] == "mistral"
    assert body["stream"] is False
    assert body["keep_alive"] == -1
    assert body["messages"][0] == {"role": "system", "content": "sys prompt"}
    assert body["messages"][1] == {"role": "user", "content": "user prompt"}


def test_ollama_provider_warm_up_sends_keep_alive():
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("sermon_finder.analyzer.httpx.post", return_value=mock_response) as mock_post:
        provider = OllamaProvider(model="mistral", base_url="http://localhost:11434")
        provider.warm_up()

    body = mock_post.call_args[1]["json"]
    assert body["keep_alive"] == -1
    assert body["stream"] is False
    assert body["model"] == "mistral"
    assert len(body["messages"]) == 1


def test_ollama_provider_parses_response():
    mock_response = MagicMock()
    mock_response.json.return_value = {"message": {"content": "DECISION: NO\nQUALITY: POOR"}}
    mock_response.raise_for_status = MagicMock()

    with patch("sermon_finder.analyzer.httpx.post", return_value=mock_response):
        provider = OllamaProvider()
        result = provider.complete("s", "u")

    assert result == "DECISION: NO\nQUALITY: POOR"


def test_ollama_provider_teardown_sends_keep_alive_zero():
    mock_response = MagicMock()
    with patch("sermon_finder.analyzer.httpx.post", return_value=mock_response) as mock_post:
        OllamaProvider().teardown()
    body = mock_post.call_args[1]["json"]
    assert body["keep_alive"] == 0


def test_ollama_provider_teardown_swallows_errors():
    with patch("sermon_finder.analyzer.httpx.post", side_effect=httpx.ConnectError("refused")):
        OllamaProvider().teardown()  # must not raise


def test_ollama_provider_connection_error_raises():
    with patch("sermon_finder.analyzer.httpx.post", side_effect=httpx.ConnectError("refused")):
        provider = OllamaProvider()
        with pytest.raises(httpx.ConnectError):
            provider.complete("s", "u")
