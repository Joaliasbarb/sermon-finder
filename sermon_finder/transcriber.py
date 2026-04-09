from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str = "medium") -> list[dict]:
    """Transcribe audio and return segments with timestamps.

    Returns a list of dicts: [{"start": float, "end": float, "text": str}, ...]
    """
    model = WhisperModel(model_size)
    segments, _ = model.transcribe(
        audio_path,
        language="fr",
        vad_filter=True,
    )
    return [
        {"start": seg.start, "end": seg.end, "text": seg.text.strip()}
        for seg in segments
    ]
