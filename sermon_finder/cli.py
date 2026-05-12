import os
import queue
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
from sermon_finder.analyzer import ClaudeProvider, OllamaProvider, WHISPER_MODELS

_console = Console(stderr=True)


def _ts(s: float) -> str:
    """Format a timestamp in seconds as m:ss (e.g. 63 → '1:03')."""
    return f"{int(s) // 60}:{int(s) % 60:02d}"


@dataclass
class _StatusBar:
    """Live status bar rendered at the bottom of the terminal."""

    total: int = 0
    segment: int = 1
    seg_start_s: float = 0.0
    seg_end_s: float = 0.0
    phase: str = "diarizing"      # "diarizing" | "transcribing" | "validating"
    transition: int = 0
    total_transitions: int = 0
    current_model: str = ""
    llm_ready: bool = True
    found_at_s: float | None = None
    _spinner: Spinner = field(
        default_factory=lambda: Spinner("dots", style="bold green"),
        repr=False,
        compare=False,
    )

    def __rich_console__(self, console, options):
        row = Text()
        row.append_text(self._spinner.render(time.time()))

        if self.found_at_s is not None:
            row.append(f"  Sermon confirmed at {_ts(self.found_at_s)}", style="bold green")
            yield row
            return

        if self.total == 0:
            row.append("  initializing...", style="dim")
        else:
            row.append(f"  Segment {self.segment}/{self.total}", style="bold white")
            row.append(f"  [{_ts(self.seg_start_s)} – {_ts(self.seg_end_s)}]", style="dim")
            row.append("    ")

            if self.phase == "diarizing":
                row.append("diarizing", style="bold yellow")
            elif self.phase == "transcribing":
                row.append("transcribing", style="bold cyan")
                row.append(
                    f"  —  transition {self.transition}/{self.total_transitions}"
                    + (f"  [{self.current_model}]" if self.current_model else ""),
                    style="dim cyan",
                )
            elif self.phase == "validating":
                row.append("validating with LLM", style="bold magenta")
                row.append(
                    f"  —  transition {self.transition}/{self.total_transitions}",
                    style="dim magenta",
                )

        row.append("    ")
        if self.llm_ready:
            row.append("LLM: ready", style="dim green")
        else:
            row.append("LLM: loading...", style="dim yellow")

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
    "--retry-model",
    default=None,
    metavar="SIZE",
    help=(
        "Enable quality-triggered retries up to this Whisper model size. "
        "When the LLM reports a poor-quality transcript and returns NO, "
        "the tool re-transcribes with successively larger models until this cap. "
        "Choices: tiny, base, small, medium, large-v3."
    ),
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Print each transcribed segment as it is produced (useful for debugging).",
)
@click.option(
    "--ollama",
    is_flag=True,
    default=False,
    help="Use a local ollama model instead of the Claude API.",
)
@click.option(
    "--ollama-model",
    default="mistral",
    show_default=True,
    metavar="NAME",
    help="Ollama model name to use when --ollama is set.",
)
@click.version_option(version="0.1.0")
def main(
    audio_file: str,
    model: str,
    retry_model: str | None,
    verbose: bool,
    ollama: bool,
    ollama_model: str,
) -> None:
    """Find the timestamp when the sermon begins in a church service recording.

    AUDIO_FILE is the path to the audio file (MP3, WAV, M4A, …).

    \b
    The tool works in three steps:
      1. Convert the audio to a format Whisper can read
      2. Diarize 4-minute segments to detect speaker transitions
      3. Transcribe a [t−30s, t+30s] window around each transition and ask Claude yes/no

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
    if not ollama and not os.environ.get("ANTHROPIC_API_KEY"):
        click.echo("Error: ANTHROPIC_API_KEY environment variable is not set.", err=True)
        sys.exit(1)

    if retry_model is not None and retry_model not in WHISPER_MODELS:
        click.echo(
            f"Error: invalid --retry-model '{retry_model}'. "
            f"Choices: {', '.join(WHISPER_MODELS)}",
            err=True,
        )
        sys.exit(1)
    retry_cap_idx = WHISPER_MODELS.index(retry_model) if retry_model is not None else -1

    try:
        audio.validate_audio_file(audio_file)
    except (FileNotFoundError, ValueError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    provider = OllamaProvider(model=ollama_model) if ollama else ClaudeProvider()
    try:
        _console.print("Preparing audio...", style="dim")

        with audio.prepare_audio(audio_file) as wav_path:
            thread_local = threading.local()
            found = threading.Event()
            result_holder: list[tuple[int, int]] = []
            model_ready = threading.Event()
            thread_errors: list[Exception] = []

            status = _StatusBar(llm_ready=not ollama)

            def _load_model_thread():
                try:
                    provider.warm_up()
                    model_ready.set()
                    status.llm_ready = True
                except Exception as e:
                    thread_errors.append(e)
                    found.set()
                    model_ready.set()  # unblock validator

            with Live(status, console=_console, refresh_per_second=8):
                if ollama:
                    threading.Thread(
                        target=_load_model_thread, daemon=True, name="model-loader"
                    ).start()
                else:
                    model_ready.set()

                with audio.split_wav(wav_path, segment_s=240.0, overlap_s=30.0) as chunks:
                    status.total = len(chunks)

                    seg_q: queue.Queue = queue.Queue()
                    trans_q: queue.Queue = queue.Queue()
                    transcription_q: queue.Queue = queue.Queue()

                    for i, (chunk_path, offset_s, keep_until_s) in enumerate(chunks, 1):
                        seg_q.put((chunk_path, offset_s, keep_until_s, i, len(chunks)))
                    seg_q.put(None)

                    # log_q: worker threads enqueue pre-formatted rich markup strings;
                    # the main thread drains and prints them. This avoids calling
                    # _console.print() from background threads while Live holds its
                    # internal lock, which can deadlock.
                    log_q: queue.Queue = queue.Queue()

                    # --- status + log callbacks ---

                    def on_segment_start(seg_idx, total_segs, off_s, end_s):
                        status.segment = seg_idx
                        status.seg_start_s = off_s
                        status.seg_end_s = end_s
                        status.phase = "diarizing"

                    def on_no_transitions(seg_idx, total_segs, off_s, end_s):
                        log_q.put(
                            f"  [dim]Segment {seg_idx}/{total_segs}"
                            f" [{_ts(off_s)} – {_ts(end_s)}]"
                            f"  no transitions — skipped[/dim]"
                        )

                    def on_transcribe_start(t, trans_idx, total_trans, seg_idx, m_size):
                        status.phase = "transcribing"
                        status.transition = trans_idx
                        status.total_transitions = total_trans
                        status.current_model = m_size

                    def on_validate_start(t, trans_idx, total_trans, seg_idx):
                        status.phase = "validating"
                        status.transition = trans_idx
                        status.total_transitions = total_trans

                    def on_result(t, result, models_tried, segments, trans_idx, total_trans, seg_idx):
                        if verbose:
                            for seg in segments:
                                log_q.put(
                                    f"    [dim][{_ts(seg['start'])}] {seg['text']}[/dim]"
                                )
                        model_chain = " → ".join(models_tried)
                        entry = (
                            f"Segment {seg_idx} — transition {trans_idx}/{total_trans}"
                            f" at {_ts(t)}  {model_chain}"
                        )
                        if result.is_sermon:
                            status.found_at_s = t
                            log_q.put(f"  {entry} → [bold green]confirmed[/bold green]")
                            log_q.put(
                                f"[bold green]Sermon start found at {_ts(t)}[/bold green]"
                            )
                        else:
                            if result.uncertain:
                                outcome = "rejected [dim](uncertain)[/dim]"
                            elif not result.quality_ok:
                                outcome = "rejected [dim](poor quality)[/dim]"
                            else:
                                outcome = "rejected"
                            log_q.put(f"  [yellow]{entry} → {outcome}[/yellow]")

                    def retranscribe(t, m_size):
                        with audio.extract_window(wav_path, t - 30.0, t + 30.0) as (wp, ws):
                            return transcriber.transcribe_segment(
                                wp, ws, keep_until_s=None,
                                model_size=m_size, thread_local=thread_local,
                            )

                    # --- thread launcher with exception capture ---

                    def run_worker(name, target, *args, **kwargs):
                        def _wrap():
                            try:
                                target(*args, **kwargs)
                            except Exception as e:
                                thread_errors.append(e)
                                found.set()
                        t = threading.Thread(target=_wrap, name=name, daemon=True)
                        t.start()
                        return t

                    t_diarizer = run_worker(
                        "diarizer", _diarizer.diarizer_worker,
                        seg_q, trans_q, found,
                        on_segment_start=on_segment_start,
                        on_no_transitions=on_no_transitions,
                    )
                    t_transcriber = run_worker(
                        "transcriber", transcriber.transcriber_worker,
                        trans_q, transcription_q, found, wav_path, model, thread_local,
                        on_transcribe_start=on_transcribe_start,
                    )
                    t_validator = run_worker(
                        "validator", analyzer.validator_worker,
                        transcription_q, found, model_ready, provider,
                        retry_cap_idx, result_holder,
                        retranscribe_fn=retranscribe,
                        on_validate_start=on_validate_start,
                        on_result=on_result,
                    )

                    def _drain_log():
                        while True:
                            try:
                                _console.print(log_q.get_nowait())
                            except queue.Empty:
                                break

                    threads = [t_diarizer, t_transcriber, t_validator]
                    try:
                        while any(t.is_alive() for t in threads):
                            _drain_log()
                            time.sleep(0.1)
                        _drain_log()
                    except KeyboardInterrupt:
                        found.set()
                        raise

            if thread_errors:
                raise thread_errors[0]

            if not result_holder:
                raise ValueError("Could not find the sermon start in any detected transition.")

            minutes, seconds = result_holder[0]

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        if ollama:
            provider.teardown()

    click.echo(f"{minutes}'{seconds:02d}")
