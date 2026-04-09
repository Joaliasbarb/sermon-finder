import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

from pydub import AudioSegment

SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}


def validate_audio_file(path: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported format: {p.suffix}. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return p


def get_duration_seconds(path: str) -> float:
    audio = AudioSegment.from_file(path)
    return len(audio) / 1000.0


@contextmanager
def prepare_audio(path: str):
    """Validate and convert audio to 16kHz mono WAV. Cleans up on exit."""
    validate_audio_file(path)
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1).set_frame_rate(16000)
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "audio.wav")
        audio.export(out, format="wav")
        yield out


@contextmanager
def split_wav(wav_path: str, segment_s: float = 120.0, overlap_s: float = 30.0):
    """Split a prepared WAV into fixed-duration overlapping segments.

    Yields a list of (chunk_wav_path, offset_s, keep_until_s) tuples.
    keep_until_s is the absolute timestamp boundary for overlap deduplication;
    None for the last segment (keep everything). All temp files are cleaned up on exit.
    """
    audio = AudioSegment.from_file(wav_path)
    duration_ms = len(audio)
    segment_ms = int(segment_s * 1000)
    step_ms = int((segment_s - overlap_s) * 1000)

    with tempfile.TemporaryDirectory() as tmpdir:
        chunks = []
        start_ms = 0
        while start_ms < duration_ms:
            end_ms = min(start_ms + segment_ms, duration_ms)
            next_start_ms = start_ms + step_ms
            keep_until_s = next_start_ms / 1000.0 if next_start_ms < duration_ms else None
            path = os.path.join(tmpdir, f"chunk_{len(chunks):03d}.wav")
            audio[start_ms:end_ms].export(path, format="wav")
            chunks.append((path, start_ms / 1000.0, keep_until_s))
            start_ms = next_start_ms
        yield chunks
