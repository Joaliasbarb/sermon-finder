import pytest
from unittest.mock import MagicMock

from sermon_finder.analyzer import (
    ClaudeProvider,
    _format_chunk,
    _format_timestamp,
    _make_chunks,
    _parse_response,
    find_sermon_start,
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


def test_make_chunks_empty():
    assert _make_chunks([]) == []


def test_make_chunks_splits_correctly():
    # 25 minutes of segments, one per minute — expect 3 chunks (at 0, 9, 18 min)
    segments = [seg(i * 60, i * 60 + 50) for i in range(25)]
    chunks = _make_chunks(segments)
    assert len(chunks) == 3


def test_make_chunks_overlap():
    # Segments spanning 12 minutes — first and second chunks share 1 minute
    segments = [seg(i * 60, i * 60 + 50) for i in range(12)]
    chunks = _make_chunks(segments)
    # Chunk 1: [0, 10), Chunk 2: [9, 19)
    chunk1_starts = {s["start"] for s in chunks[0]}
    chunk2_starts = {s["start"] for s in chunks[1]}
    assert 9 * 60 in chunk1_starts  # 9-min segment is in chunk 1
    assert 9 * 60 in chunk2_starts  # and also in chunk 2 (overlap)


def test_parse_response_timestamp():
    assert _parse_response("[35:42]") == (35, 42)


def test_parse_response_timestamp_in_sentence():
    assert _parse_response("The sermon starts at [35:42]") == (35, 42)


def test_parse_response_not_found():
    assert _parse_response("not found") is None


def test_parse_response_not_found_uppercase():
    assert _parse_response("NOT FOUND") is None


def test_parse_response_unrecognised():
    assert _parse_response("I cannot determine this") is None


# --- find_sermon_start tests ---

def test_find_sermon_start_found_immediately():
    """Single chunk — sermon found on the only API call."""
    provider = MagicMock()
    provider.complete.return_value = "[05:00]"
    segments = [seg(i * 60, i * 60 + 50) for i in range(8)]  # 8 min → 1 chunk
    result = find_sermon_start(segments, provider=provider)
    assert result == (5, 0)
    assert provider.complete.call_count == 1


def test_find_sermon_start_fallback_to_second_half():
    """First-half chunks return nothing; second-half chunk has the answer."""
    # 25 min → 3 chunks (mid=2); second half = chunk starting at 18 min
    segments = [seg(i * 60, i * 60 + 50) for i in range(25)]

    def respond(system, user):
        return "[22:15]" if "[18:00]" in user else "not found"

    provider = MagicMock()
    provider.complete.side_effect = respond
    result = find_sermon_start(segments, provider=provider)
    assert result == (22, 15)


def test_find_sermon_start_earliest_timestamp_wins():
    """Two chunks in the same batch both match — earliest timestamp returned."""
    # 22 min → 3 chunks (mid=2); both first-half chunks respond positively
    segments = [seg(i * 60, i * 60 + 50) for i in range(22)]

    def respond(system, user):
        if "[00:00]" in user:   # chunk 1 starts at 0:00
            return "[05:30]"
        if "[09:00]" in user:   # chunk 2 starts at 9:00
            return "[12:00]"
        return "not found"

    provider = MagicMock()
    provider.complete.side_effect = respond
    result = find_sermon_start(segments, provider=provider)
    assert result == (5, 30)   # 5:30 < 12:00


def test_find_sermon_start_second_half_not_queried_when_first_half_matches():
    """When first half finds the sermon, second half chunks are never queried."""
    # 40 min → 5 chunks (mid=3); first half has 3 chunks, second half has 2
    segments = [seg(i * 60, i * 60 + 50) for i in range(40)]
    provider = MagicMock()
    provider.complete.return_value = "[35:42]"
    result = find_sermon_start(segments, provider=provider)
    assert result == (35, 42)
    assert provider.complete.call_count == 3   # only the 3 first-half chunks


def test_find_sermon_start_not_found_raises():
    provider = MagicMock()
    provider.complete.return_value = "not found"
    segments = [seg(i * 60, i * 60 + 50) for i in range(10)]
    with pytest.raises(ValueError, match="Could not find"):
        find_sermon_start(segments, provider=provider)


def test_find_sermon_start_empty_raises():
    provider = MagicMock()
    with pytest.raises(ValueError):
        find_sermon_start([], provider=provider)


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
