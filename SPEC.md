## §G

Find sermon-start timestamp in French Protestant church service audio. Output `mm'ss` to stdout.

---

## §C

- Python ≥3.10; Poetry
- CPU-only inference (no GPU)
- ffmpeg required for audio conversion
- `ANTHROPIC_API_KEY` in env or `.env` file (not required when `--ollama` used)
- ollama running at `localhost:11434` + at least one model pulled (required when `--ollama` used)
- No full-recording transcription — short windows around transitions only
- Pipe-friendly: stdout = result only, stderr = progress
- Early-exit: stop at first confirmed transition

---

## §I

- CLI: `sermon-finder AUDIO_FILE [--model SIZE] [--retry-model SIZE] [-v/--verbose] [--ollama] [--ollama-model NAME]`
- Config: `ANTHROPIC_API_KEY` env var or `.env` (loaded via python-dotenv)
- Claude API: `claude-sonnet-4-5` via `anthropic` SDK; `max_tokens=50`
- Ollama API: `POST /api/chat` at `localhost:11434`; `keep_alive=-1`; timeout 120s
- Diarize: `diarize` lib (CPU-local, no API key)
- Transcribe: `faster-whisper` with `language="fr"`, `vad_filter=True`
- Audio I/O: `pydub` + ffmpeg; accepted: `.mp3 .wav .m4a .aac .ogg .flac`
- Pipeline: `threading.Thread` workers + `queue.Queue` channels + `threading.Event` signals

---

## §V

| id  | invariant |
|-----|-----------|
| V1  | stdout = exactly `mm'ss` on success; nothing else written to stdout |
| V2  | all progress/log output → stderr only |
| V3  | exit code 1 on: missing API key \| bad/missing file \| unsupported format \| no sermon found |
| V4  | overlap dedup: discard transitions where offset-corrected start ≥ `keep_until_s`; last segment keeps all (`keep_until_s=None`) |
| V5  | transcription window = `[t−30, t+30]` seconds, clamped to `[0, audio_duration]` |
| V6  | `WhisperModel` instantiated lazily per `threading.local`; reused for all subsequent calls on same thread |
| V7  | `_diarize()` called without `min_speakers` — auto-detect speaker count |
| V8  | quality-triggered retry fires only when `POOR + NO`; `YES` accepted regardless of quality |
| V9  | audio converted to 16 kHz mono WAV before any ML processing |
| V10 | all temp files wrapped in context managers; cleanup on normal exit AND exception |
| V11 | `model_ready` Event is always used regardless of provider; for Claude it is set immediately after audio validation; for ollama it is set when warm-up response is received |
| V12 | single transcriber thread consumes `transition_queue` in order → `transcription_queue` items arrive in chronological order |
| V13 | validator thread blocks on `model_ready` Event before issuing first LLM call |
| V14 | all worker threads check `found` Event between items and stop immediately when set |
| V15 | sentinel `None` propagates through each queue to signal end-of-stream; each worker forwards it to the next queue before exiting |

---

## §T

| id  | status | tag        | task | cites |
|-----|--------|------------|------|-------|
| T1  | .      |            | Add unit tests for `split_wav` overlap/dedup logic | V4 |
| T2  | .      |            | Add unit tests for `extract_window` clamping | V5 |
| T3  | .      |            | Interactive TUI: allow user to review and override rejected transitions | I.CLI |
| T4  | .      |            | Track UNSURE transitions as potential-start candidates (log, don't discard) | V8 |
| T5  | .      |            | Merge diarization transitions < N seconds apart before LLM validation | V4 |
| T6  | x      |            | Add `OllamaProvider` implementing `LLMProvider` protocol; call local ollama REST API (`localhost:11434`) as drop-in replacement for `ClaudeProvider` | I.Claude |
| T7  | x      | pipeline   | Add `model_ready` Event in `cli.py`; for Claude set it immediately after audio validation; for ollama spawn loader thread calling `OllamaProvider.warm_up()` then set event; `complete()` sends `keep_alive=-1`; status bar shows "loading" or "ready" LLM state | V11,V13 |
| T8  | .      | pipeline   | Refactor diarizer into worker thread: consume `segment_queue`, push transitions to `transition_queue`, forward sentinel on done | V14,V15 |
| T9  | .      | pipeline   | Add transcriber worker thread: consume `transition_queue` in order, push `(t, segments)` to `transcription_queue`, forward sentinel on done | V12,V14,V15 |
| T10 | .      | pipeline   | Add validator worker thread: wait on `model_ready`, consume `transcription_queue` in order, set `found` Event on YES, forward sentinel on done | V8,V13,V14,V15 |
| T11 | .      | pipeline   | Update `_StatusBar` to reflect pipeline state: model loading, active phase per worker, early-exit confirmation | V2,V11 |
| T12 | .      | pipeline   | Wire T7–T11 into `cli.py`: replace sequential loop with pipeline; join all threads; propagate exceptions across threads to main | V3,V14 |

---

## §B

| id | date | cause | fix |
|----|------|-------|-----|
