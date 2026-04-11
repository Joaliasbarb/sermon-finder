from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import anthropic

CLAUDE_MODEL = "claude-sonnet-4-5"

TRANSITION_SYSTEM_PROMPT = """\
You are analyzing a French Protestant church service transcript around a detected speaker change.
Your task: determine whether this transition marks the start of the sermon (prédication).

Structure of a French Protestant service immediately before the sermon:
- The service president leads the liturgy. Just before the sermon they may:
  • Read one or more Bible passages, then step aside
  • Lead a prayer (for the sunday school children, for the preaching, or other),
    often closing with "Amen", then step aside
  • The congregation may sing a hymn right before the sermon, in which case the
    transcript may show no president speech immediately before the preacher starts
- The hand-over is not always explicit; it can be silent or abrupt.
- Very occasionally the president and the preacher are the same person.

How the preacher typically begins (any combination is possible):
- Greets the assembly ("Frères et sœurs…", "Bonjour…", etc.)
- Relays salutations from another church or community
- Opens with a prayer of their own
- Announces a Bible passage ("Ouvrez votre Bible en…", "Tournez-vous en…")
- Moves directly into the sermon theme or content

Key signal: after this transition the new speaker holds the floor in a sustained,
substantive way as the preacher — not merely for a short reading or a liturgical element.

Respond with exactly two lines:
DECISION: YES | NO | UNSURE
QUALITY: GOOD | POOR

DECISION values:
  YES    — this transition is the sermon start
  NO     — this is a different speaker change within the liturgy
  UNSURE — the transcript does not give enough context to decide

QUALITY values:
  GOOD — the transcript is legible enough to make a reliable determination
  POOR — the transcript is too garbled, fragmented, or incomplete to be reliable\
"""


@dataclass
class TransitionResult:
    is_sermon: bool   # True for YES; False for NO or UNSURE
    uncertain: bool   # True when DECISION is UNSURE
    quality_ok: bool  # True for GOOD; False for POOR


@runtime_checkable
class LLMProvider(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class ClaudeProvider:
    def __init__(self, client: anthropic.Anthropic | None = None):
        self._client = client or anthropic.Anthropic()

    def complete(self, system: str, user: str) -> str:
        response = self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=50,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text.strip()


def is_sermon_transition(
    segments: list[dict],
    transition_t: float | None = None,
    provider: LLMProvider | None = None,
) -> TransitionResult:
    """Return a TransitionResult for the transcript around a speaker transition."""
    if provider is None:
        provider = ClaudeProvider()
    transcript = _format_chunk(segments, transition_t)
    user_msg = (
        "Here is the transcript around a detected speaker transition. "
        "Is this the start of the sermon (prédication)?\n\n"
        + transcript
    )
    response = provider.complete(TRANSITION_SYSTEM_PROMPT, user_msg)
    return _parse_result(response)


def _parse_result(response: str) -> TransitionResult:
    decision = "NO"
    quality = "GOOD"
    for line in response.strip().splitlines():
        upper = line.strip().upper()
        if upper.startswith("DECISION:"):
            decision = upper.split(":", 1)[1].strip()
        elif upper.startswith("QUALITY:"):
            quality = upper.split(":", 1)[1].strip()
    return TransitionResult(
        is_sermon=(decision == "YES"),
        uncertain=(decision == "UNSURE"),
        quality_ok=(quality != "POOR"),
    )


def _format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _format_chunk(segments: list[dict], transition_t: float | None = None) -> str:
    lines = []
    marker_done = False
    for seg in segments:
        if transition_t is not None and not marker_done and seg["start"] >= transition_t:
            lines.append(f"--- transition at {_format_timestamp(transition_t)} ---")
            marker_done = True
        lines.append(f"[{_format_timestamp(seg['start'])}] {seg['text']}")
    return "\n".join(lines)
