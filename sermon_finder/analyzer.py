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

Answer with exactly YES if this transition is the start of the sermon,
or NO if it is any other speaker change within the liturgy.\
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


def is_sermon_transition(
    segments: list[dict],
    provider: LLMProvider | None = None,
) -> bool:
    """Return True if the transcript around a speaker transition is the sermon start."""
    if provider is None:
        provider = ClaudeProvider()
    transcript = _format_chunk(segments)
    user_msg = (
        "Here is the transcript around a detected speaker transition. "
        "Is this the start of the sermon (prédication)?\n\n"
        + transcript
    )
    response = provider.complete(TRANSITION_SYSTEM_PROMPT, user_msg)
    return response.strip().upper().startswith("YES")


def _format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


def _format_chunk(segments: list[dict]) -> str:
    return "\n".join(
        f"[{_format_timestamp(seg['start'])}] {seg['text']}"
        for seg in segments
    )
