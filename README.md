# sermon-finder

A CLI tool that automatically finds the timestamp when the sermon begins in a French church worship service audio recording.

It transcribes the recording locally with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), then uses the Claude API to locate the transition from the service president's introduction to the preacher's sermon.

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
git clone https://github.com/Joaliasbarb/sermon_publisher.git
cd sermon_publisher
poetry install
```

## Configuration

Copy the example and fill in your API key:

```bash
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

The tool loads `.env` automatically — no need to `export` the variable.

## Usage

```
Usage: sermon-finder [OPTIONS] AUDIO_FILE

  Find the timestamp when the sermon begins in a church service recording.

Options:
  --model SIZE    Whisper model to use for transcription. [default: medium]
                  Choices: tiny, base, small, medium, large-v3.
  -v, --verbose   Print each transcribed segment to stderr as it is produced.
  --version       Show the version and exit.
  -h, --help      Show this message and exit.
```

### Basic usage

```bash
poetry run sermon-finder service.mp3
```

Output (stdout only):

```
35'42
```

Progress is printed to stderr so the result can be piped:

```bash
poetry run sermon-finder service.mp3 | xargs echo "Sermon starts at:"
```

### Speed vs. accuracy

The default model (`medium`) gives good accuracy for French speech on CPU.
Use a smaller model to trade accuracy for speed:

```bash
poetry run sermon-finder service.mp3 --model small
```

| Model    | RAM    | Speed on CPU | French quality |
|----------|--------|--------------|----------------|
| tiny     | 1 GB   | ~32× real-time | Poor         |
| base     | 1 GB   | ~16× real-time | Acceptable   |
| small    | 2 GB   | ~6× real-time  | Good         |
| medium   | 5 GB   | ~2× real-time  | Very good    |
| large-v3 | 10 GB  | ~1× real-time  | Best         |

> **Note:** On first run the chosen model is downloaded from HuggingFace (~500 MB for `small`, ~1.5 GB for `medium`). Subsequent runs use the cached model.

### Verbose mode

Print each transcribed segment to stderr as it is produced:

```bash
poetry run sermon-finder service.mp3 --verbose
```

## How it works

1. **Audio preparation** — The file is converted to a 16 kHz mono WAV (what Whisper expects). A temporary file is used and cleaned up automatically.

2. **Transcription** — The full recording is transcribed locally with `faster-whisper`. A progress bar shows the current position in the audio. Voice activity detection (`vad_filter`) skips silence and music automatically.

3. **Sermon detection** — The transcript is split into 10-minute overlapping chunks. Each chunk is sent to Claude (`claude-sonnet-4-5`) with a prompt that describes the typical French Protestant service structure. Claude responds with the timestamp or "not found". The tool stops at the first chunk where the sermon is detected, keeping API costs low.

## Development

```bash
# Run tests
poetry run pytest -v tests/

# Lint
poetry run ruff check .
```

## Supported formats

MP3, WAV, M4A, AAC, OGG, FLAC — anything ffmpeg can read.
