import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()

from sermon_finder import audio, transcriber, analyzer
from sermon_finder.analyzer import ClaudeProvider


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("audio_file", metavar="AUDIO_FILE")
@click.option(
    "--model",
    default="small",
    show_default=True,
    metavar="SIZE",
    help=(
        "Whisper model to use for transcription. "
        "Larger models are more accurate but slower. "
        "Choices: tiny, base, small, medium, large-v3."
    ),
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    metavar="N",
    help=(
        "Number of parallel transcription workers. "
        "Each worker loads its own copy of the Whisper model "
        "(e.g. 2 workers × 2 GB = 4 GB for 'small'). "
        "Default: 1 (sequential)."
    ),
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Print each transcribed segment to stderr as it is produced.",
)
@click.version_option(version="0.1.0")
def main(audio_file: str, model: str, workers: int, verbose: bool) -> None:
    """Find the timestamp when the sermon begins in a church service recording.

    AUDIO_FILE is the path to the audio file (MP3, WAV, M4A, …).

    \b
    The tool works in three steps:
      1. Convert the audio to a format Whisper can read
      2. Transcribe the full recording locally using faster-whisper
      3. Send the transcript in chunks to Claude to locate the sermon start

    \b
    Output:
      A single timestamp on stdout in mm'ss format, e.g.:  35'42
      All progress messages go to stderr, so the output is pipe-friendly.

    \b
    Requirements:
      - ffmpeg must be installed (sudo apt install ffmpeg)
      - ANTHROPIC_API_KEY must be set (or present in a .env file)

    \b
    Examples:
      sermon-finder service.mp3
      sermon-finder service.mp3 --model small
      sermon-finder service.mp3 --verbose
      sermon-finder service.mp3 | xargs echo "Sermon starts at:"
    """
    if not os.environ.get("ANTHROPIC_API_KEY"):
        click.echo("Error: ANTHROPIC_API_KEY environment variable is not set.", err=True)
        sys.exit(1)

    try:
        audio.validate_audio_file(audio_file)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    try:
        duration = audio.get_duration_seconds(audio_file)
        if duration > 3600:
            est_min = int(duration / 120)
            click.echo(
                f"Warning: Long recording (~{int(duration / 60)} min). "
                f"Transcription may take ~{est_min} min on CPU.",
                err=True,
            )
    except Exception:
        pass

    try:
        click.echo("Preparing audio...", err=True)

        with audio.prepare_audio(audio_file) as wav_path:
            segments = transcriber.transcribe(wav_path, model_size=model, verbose=verbose, num_workers=workers)
            click.echo(f"Transcribed {len(segments)} segments.", err=True)

            provider = ClaudeProvider()
            minutes, seconds = analyzer.find_sermon_start(segments, provider=provider)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"{minutes}'{seconds:02d}")
