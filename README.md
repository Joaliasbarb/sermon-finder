# sermon-finder

A CLI tool that automatically finds the timestamp when the sermon begins in a French Protestant church service audio recording.

It uses speaker diarization to detect speaker transitions, transcribes a short window around each candidate transition with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), and asks the Claude API to confirm whether that transition is the sermon start. Because it only transcribes short clips rather than the full recording, it is fast even on CPU.

## Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/)
- [ffmpeg](https://ffmpeg.org/) (for MP3 and other non-WAV formats)
- An [Anthropic API key](https://console.anthropic.com/)

Install ffmpeg on Ubuntu/Debian:

```bash
sudo apt install ffmpeg
```

## Installation

```bash
git clone https://github.com/Joaliasbarb/sermon-finder.git
cd sermon-finder
poetry install
```

## Configuration

Create a `.env` file with your API key:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

The tool loads `.env` automatically — no need to `export` the variable.

## Usage

```
Usage: sermon-finder [OPTIONS] AUDIO_FILE

  Find the timestamp when the sermon begins in a church service recording.

Options:
  --model SIZE         Whisper model for transcription. [default: small]
                       Choices: tiny, base, small, medium, large-v3.
  --retry-model SIZE   Enable quality-triggered retries up to this model size.
                       Choices: tiny, base, small, medium, large-v3.
  -v, --verbose        Print each transcribed segment (useful for debugging).
  --version            Show the version and exit.
  -h, --help           Show this message and exit.
```

### Basic usage

```bash
poetry run sermon-finder service.mp3
```

Output (stdout only):

```
35'42
```

All progress is printed to stderr so the result can be piped:

```bash
poetry run sermon-finder service.mp3 | xargs echo "Sermon starts at:"
```

### Progress display

The tool shows a live status bar during processing:

```
  Segment 1/12 [0:00 – 4:00]  no transitions — skipped
  Segment 2/12 [3:30 – 7:30]  no transitions — skipped
  Segment 3/12 [7:00 – 11:00]  transition 1/1 at 9:15 discarded by LLM
Sermon start found at 12:43
⣾  Segment 4/12 [10:30 – 14:30]    validating with LLM  —  transition 1/1
```

The bottom line updates live; log messages scroll above it.

### Whisper model

The `--model` option controls the Whisper model used to transcribe the short windows around candidate transitions. Because only brief clips are transcribed (not the full recording), even `small` is fast and accurate enough for most recordings.

| Model    | RAM    | French quality |
|----------|--------|----------------|
| tiny     | 1 GB   | Poor           |
| base     | 1 GB   | Acceptable     |
| small    | 2 GB   | Good (default) |
| medium   | 5 GB   | Very good      |
| large-v3 | 10 GB  | Best           |

> **Note:** On first run the chosen model is downloaded from HuggingFace (~500 MB for `small`). Subsequent runs use the cached model.

### Quality-triggered retries

When Claude determines that the transcript quality is too poor to make a reliable decision, the tool can automatically re-transcribe using a larger Whisper model. Use `--retry-model` to enable this and set the ceiling:

```bash
# Retry with medium if small produces a poor-quality transcript
poetry run sermon-finder service.mp3 --retry-model medium

# Step through small → medium → large-v3 until quality is acceptable
poetry run sermon-finder service.mp3 --retry-model large-v3
```

Without `--retry-model`, poor-quality transcripts are accepted as-is and the transition is discarded if Claude returns NO. Retries only happen when quality is poor **and** Claude's answer is NO — a YES is always accepted regardless of quality.

### Verbose mode

Print each transcribed segment for debugging:

```bash
poetry run sermon-finder service.mp3 --verbose
```

## How it works

1. **Audio preparation** — The file is converted to a 16 kHz mono WAV. A temporary file is used and cleaned up automatically.

2. **Diarization** — The audio is split into 4-minute segments with 30-second overlap. Each segment is diarized locally with the [diarize](https://github.com/FoxNoseTech/diarize) library (CPU-only, no API key required) to detect speaker transitions. Segments with no transitions are skipped immediately.

3. **Transcription** — For each detected speaker transition at time *t*, a short window around *t* is extracted and transcribed with `faster-whisper`. Only these short clips are transcribed — not the full recording.

4. **LLM validation** — The transcript is sent to Claude (`claude-sonnet-4-5`) with a prompt describing the typical structure of a French Protestant service. Claude returns a decision (YES / NO / UNSURE) and a quality assessment (GOOD / POOR). If quality is POOR and the decision is NO, the tool can re-transcribe with a larger model (see `--retry-model`).

5. **Early exit** — The tool processes segments in chronological order and stops as soon as a transition is confirmed, keeping both compute and API costs low.

## Development

```bash
# Run tests
poetry run pytest -v tests/

# Lint
poetry run ruff check .
```

## Supported formats

MP3, WAV, M4A, AAC, OGG, FLAC — anything ffmpeg can read.
