import re
from typing import Protocol, runtime_checkable

import anthropic

CLAUDE_MODEL = "claude-sonnet-4-5"
CHUNK_SECONDS = 10 * 60   # 10-minute windows
OVERLAP_SECONDS = 1 * 60  # 1-minute overlap between windows

SYSTEM_PROMPT = """\
You are an assistant analyzing French Protestant church worship service transcripts.
Identify when the sermon (prédication) begins. The transition is:
the service president finishes introducing the preacher → the preacher begins with a
thematic opening statement. This is a content/speaker transition, not an acoustic one.
If this chunk contains the sermon start, respond with ONLY the timestamp: [mm:ss]
If the sermon has not started yet in this chunk, respond with: not found\
"""


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ClaudeProvider:
    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client or anthropic.Anthropic()

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=20,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()


def _format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _format_chunk(segments: list[dict]) -> str:
    return "\n".join(
        f"[{_format_timestamp(seg['start'])}] {seg['text']}"
        for seg in segments
    )


def _make_chunks(segments: list[dict]) -> list[list[dict]]:
    """Split segments into overlapping fixed-duration windows."""
    if not segments:
        return []

    chunks = []
    window_start = segments[0]["start"]
    total_end = segments[-1]["end"]

    while window_start < total_end:
        window_end = window_start + CHUNK_SECONDS
        chunk = [s for s in segments if window_start <= s["start"] < window_end]
        if chunk:
            chunks.append(chunk)
        window_start += CHUNK_SECONDS - OVERLAP_SECONDS

    return chunks


def _parse_response(response: str) -> tuple[int, int] | None:
    """Return (minutes, seconds) if a timestamp is found, None otherwise."""
    if "not found" in response.lower():
        return None
    match = re.search(r'\[(\d+):(\d+)\]', response)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def find_sermon_start(
    segments: list[dict],
    provider: LLMProvider | None = None,
) -> tuple[int, int]:
    """Find the timestamp when the sermon begins.

    Analyzes the transcript in overlapping 10-minute chunks, querying the
    LLM for each chunk until the sermon start is identified.

    Returns (minutes, seconds).
    Raises ValueError if the sermon start cannot be found.
    """
    if provider is None:
        provider = ClaudeProvider()

    chunks = _make_chunks(segments)
    if not chunks:
        raise ValueError("No transcript segments provided.")

    for chunk in chunks:
        transcript = _format_chunk(chunk)
        user_msg = (
            "Here is the timestamped transcript. "
            "Identify where the sermon begins:\n\n" + transcript
        )
        response = provider.complete(SYSTEM_PROMPT, user_msg)
        result = _parse_response(response)
        if result is not None:
            return result

    raise ValueError("Could not find the sermon start in the transcript.")
