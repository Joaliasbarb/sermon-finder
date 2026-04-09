import sys

from faster_whisper import WhisperModel
from tqdm import tqdm


def transcribe(
    audio_path: str,
    model_size: str = "medium",
    verbose: bool = False,
) -> list[dict]:
    """Transcribe audio and return segments with timestamps.

    Returns a list of dicts: [{"start": float, "end": float, "text": str}, ...]
    Shows a progress bar based on audio position by default.
    """
    print(f"Loading Whisper model ({model_size})...", file=sys.stderr)
    model = WhisperModel(model_size)

    segments_gen, info = model.transcribe(
        audio_path,
        language="fr",
        vad_filter=True,
    )

    segments = []
    prev_end = 0.0

    with tqdm(
        total=info.duration,
        unit="s",
        unit_scale=True,
        desc="Transcribing",
        file=sys.stderr,
        bar_format="{l_bar}{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]",
    ) as pbar:
        for seg in segments_gen:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            pbar.update(seg.end - prev_end)
            prev_end = seg.end
            if verbose:
                tqdm.write(f"  [{seg.start:.1f}s] {seg.text.strip()}", file=sys.stderr)

    return segments
