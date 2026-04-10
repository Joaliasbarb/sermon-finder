from diarize import diarize as _diarize


def run_diarization(wav_path: str, offset_s: float = 0.0) -> list[dict]:
    """Diarize one audio chunk. Returns segments with absolute timestamps.

    offset_s is added to every segment timestamp so results are in absolute
    recording time, consistent with how transcribe_segment handles offsets.
    """
    result = _diarize(wav_path)
    return [
        {"speaker": seg.speaker, "start": seg.start + offset_s, "end": seg.end + offset_s}
        for seg in result.segments
    ]


def get_speaker_transitions(segments: list[dict]) -> list[float]:
    """Return absolute timestamps (seconds) where the speaker changes."""
    return [
        segments[i]["start"]
        for i in range(1, len(segments))
        if segments[i]["speaker"] != segments[i - 1]["speaker"]
    ]
