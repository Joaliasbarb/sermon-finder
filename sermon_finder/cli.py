import os
import sys

import click
from dotenv import load_dotenv

load_dotenv()

from sermon_finder import audio, transcriber, analyzer
from sermon_finder.analyzer import ClaudeProvider


@click.command()
@click.argument("audio_file")
@click.option(
    "--model",
    default="medium",
    show_default=True,
    help="Whisper model size [tiny|base|small|medium|large-v3]",
)
@click.option("--verbose", is_flag=True, help="Print progress to stderr.")
@click.version_option(version="0.1.0")
def main(audio_file: str, model: str, verbose: bool) -> None:
    """Detect the timestamp when the sermon begins in a church service recording.

    Outputs the timestamp in mm'ss format (e.g. 35'42).
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
        if verbose:
            click.echo("Preparing audio...", err=True)

        with audio.prepare_audio(audio_file) as wav_path:
            if verbose:
                click.echo(f"Transcribing with Whisper ({model} model)...", err=True)

            segments = transcriber.transcribe(wav_path, model_size=model)

            if verbose:
                click.echo(f"Transcribed {len(segments)} segments. Analysing...", err=True)

            provider = ClaudeProvider()
            minutes, seconds = analyzer.find_sermon_start(segments, provider=provider)

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"{minutes}'{seconds:02d}")
