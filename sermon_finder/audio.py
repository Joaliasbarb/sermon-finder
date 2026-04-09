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
