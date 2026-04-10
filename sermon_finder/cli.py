import os
import sys
import threading
import time
from dataclasses import dataclass, field

import click
from dotenv import load_dotenv
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text

load_dotenv()

from sermon_finder import audio, transcriber, analyzer, diarizer as _diarizer
from sermon_finder.analyzer import ClaudeProvider

_console = Console(stderr=True)


def _ts(s: float) -> str:
    """Format a timestamp in seconds as m:ss (e.g. 63 → '1:03')."""
    return f"{int(s) // 60}:{int(s) % 60:02d}"


@dataclass
class _StatusBar:
    """Live status bar rendered at the bottom of the terminal."""

    total: int = 1
    segment: int = 1
    seg_start_s: float = 0.0
    seg_end_s: float = 0.0
    phase: str = "diarizing"      # "diarizing" | "transcribing" | "validating"
    transition: int = 0
    total_transitions: int = 0
    _spinner: Spinner = field(
        default_factory=lambda: Spinner("dots", style="bold green"),
        repr=False,
        compare=False,
    )

    def __rich_console__(self, console, options):
        row = Text()
        row.append_text(self._spinner.render(time.time()))
        row.append(f"  Segment {self.segment}/{self.total}", style="bold white")
        row.append(f"  [{_ts(self.seg_start_s)} – {_ts(self.seg_end_s)}]", style="dim")
        row.append("    ")

        if self.phase == "diarizing":
            row.append("diarizing", style="bold yellow")
        elif self.phase == "transcribing":
            row.append("transcribing", style="bold cyan")
            row.append(
                f"  —  transition {self.transition}/{self.total_transitions}",
                style="dim cyan",
            )
        elif self.phase == "validating":
            row.append("validating with LLM", style="bold magenta")
            row.append(
                f"  —  transition {self.transition}/{self.total_transitions}",
                style="dim magenta",
            )

        yield row


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
    "--verbose", "-v",
    is_flag=True,
    help="Print each transcribed segment as it is produced (useful for debugging).",
)
@click.version_option(version="0.1.0")
def main(audio_file: str, model: str, verbose: bool) -> None:
    """Find the timestamp when the sermon begins in a church service recording.

    AUDIO_FILE is the path to the audio file (MP3, WAV, M4A, …).

    \b
    The tool works in three steps:
      1. Convert the audio to a format Whisper can read
      2. Diarize 4-minute segments to detect speaker transitions
      3. Transcribe a ±30s window around each transition and ask Claude yes/no

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
        _console.print("Preparing audio...", style="dim")

        with audio.prepare_audio(audio_file) as wav_path:
            thread_local = threading.local()
            found: tuple[int, int] | None = None
            provider = ClaudeProvider()

            with audio.split_wav(wav_path, segment_s=240.0, overlap_s=30.0) as chunks:
                status = _StatusBar(total=len(chunks))

                with Live(status, console=_console, refresh_per_second=8):
                    for i, (chunk_path, offset_s, keep_until_s) in enumerate(chunks, 1):
                        seg_end_s = (
                            keep_until_s + 30.0
                            if keep_until_s is not None
                            else offset_s + 240.0
                        )
                        status.segment = i
                        status.seg_start_s = offset_s
                        status.seg_end_s = seg_end_s
                        status.phase = "diarizing"

                        speaker_segs = _diarizer.run_diarization(chunk_path, offset_s)
                        transitions = _diarizer.get_speaker_transitions(speaker_segs)

                        if not transitions:
                            _console.print(
                                f"  [dim]Segment {i}/{len(chunks)}"
                                f" [{_ts(offset_s)} – {_ts(seg_end_s)}]"
                                f"  no transitions — skipped[/dim]"
                            )
                            continue

                        for j, t in enumerate(transitions, 1):
                            status.phase = "transcribing"
                            status.transition = j
                            status.total_transitions = len(transitions)

                            with audio.extract_window(wav_path, t - 30.0, t + 60.0) as (win_path, win_start):
                                segs = transcriber.transcribe_segment(
                                    win_path, win_start, keep_until_s=None,
                                    model_size=model, thread_local=thread_local,
                                )

                            if verbose:
                                for seg in segs:
                                    _console.print(
                                        f"    [dim][{_ts(seg['start'])}] {seg['text']}[/dim]"
                                    )

                            status.phase = "validating"

                            if analyzer.is_sermon_transition(segs, provider=provider):
                                found = (int(t) // 60, int(t) % 60)
                                _console.print(
                                    f"[bold green]Sermon start found at {_ts(t)}[/bold green]"
                                )
                                break
                            else:
                                _console.print(
                                    f"  [yellow]Segment {i} — transition {j}/{len(transitions)}"
                                    f" at {_ts(t)} discarded by LLM[/yellow]"
                                )

                        if found:
                            break

                        _console.print(
                            f"  [dim]Segment {i}/{len(chunks)}"
                            f" [{_ts(offset_s)} – {_ts(seg_end_s)}]"
                            f"  {len(transitions)} transition(s) — no sermon start[/dim]"
                        )

            if found is None:
                raise ValueError("Could not find the sermon start in any detected transition.")

            minutes, seconds = found

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    click.echo(f"{minutes}'{seconds:02d}")
