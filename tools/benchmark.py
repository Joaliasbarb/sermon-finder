#!/usr/bin/env python3
"""
Benchmark sequential vs. parallel chunk analysis.

Transcribes a real recording once, then times both approaches using a mock
provider that sleeps SIMULATED_LATENCY seconds per call to mimic Claude API
response time — no real API calls made.

Usage:
    poetry run python tools/benchmark.py tests/data/<file>.mp3
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from unittest.mock import MagicMock

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from sermon_finder import audio, transcriber
from sermon_finder.analyzer import (
    SYSTEM_PROMPT,
    _format_chunk,
    _format_timestamp,
    _make_chunks,
    _parse_response,
)

SIMULATED_LATENCY = 2.0  # seconds — realistic Claude API round-trip


def _mock_provider(latency: float) -> MagicMock:
    """Return a provider whose complete() always sleeps then returns 'not found'.

    'not found' is the conservative worst-case: forces every batch to be
    exhausted, making the timing comparison purely about parallelism.
    """
    provider = MagicMock()

    def _complete(system, user):
        time.sleep(latency)
        return "not found"

    provider.complete.side_effect = _complete
    return provider


def run_sequential(chunks: list, latency: float) -> tuple[float, int]:
    """Process chunks one by one; stop at first hit. Returns (wall_time, calls)."""
    provider = _mock_provider(latency)
    t0 = time.perf_counter()
    total = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        transcript = _format_chunk(chunk)
        user_msg = (
            "Here is the timestamped transcript. "
            "Identify where the sermon begins:\n\n" + transcript
        )
        response = provider.complete(SYSTEM_PROMPT, user_msg)
        if _parse_response(response) is not None:
            return time.perf_counter() - t0, i
    return time.perf_counter() - t0, total


def run_parallel_batched(chunks: list, latency: float) -> tuple[float, int]:
    """Process first half in parallel, then second half if needed.
    Returns (wall_time, total_calls)."""
    provider = _mock_provider(latency)
    total = len(chunks)
    mid = (total + 1) // 2
    calls = 0
    t0 = time.perf_counter()

    for batch_start, batch_end in [(0, mid), (mid, total)]:
        batch = chunks[batch_start:batch_end]
        if not batch:
            continue
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    provider.complete,
                    SYSTEM_PROMPT,
                    "Here is the timestamped transcript. "
                    "Identify where the sermon begins:\n\n" + _format_chunk(chunk),
                ): chunk
                for chunk in batch
            }
            hits = []
            for future in as_completed(futures):
                calls += 1
                result = _parse_response(future.result())
                if result is not None:
                    hits.append(result)
        if hits:
            return time.perf_counter() - t0, calls

    return time.perf_counter() - t0, calls


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: poetry run python tools/benchmark.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    print(f"File        : {Path(audio_file).name}")
    print(f"API latency : {SIMULATED_LATENCY:.1f}s per call (simulated)\n")

    print("Step 1/2 — Transcribing audio with faster-whisper …", flush=True)
    t_start = time.perf_counter()
    with audio.prepare_audio(audio_file) as wav_path:
        segments = transcriber.transcribe(wav_path, model_size="medium", verbose=False)
    t_transcribe = time.perf_counter() - t_start
    print(f"  → {len(segments)} segments  ({t_transcribe:.1f}s)\n")

    chunks = _make_chunks(segments)
    total = len(chunks)
    mid = (total + 1) // 2
    print(f"Step 2/2 — Chunk analysis")
    print(f"  Total chunks   : {total}")
    print(f"  First half     : {mid}   (queried in parallel)")
    print(f"  Second half    : {total - mid}   (queried in parallel, only if needed)\n")

    print("Running sequential …", end=" ", flush=True)
    t_seq, calls_seq = run_sequential(chunks, SIMULATED_LATENCY)
    print(f"{t_seq:.2f}s  ({calls_seq} calls)")

    print("Running parallel   …", end=" ", flush=True)
    t_par, calls_par = run_parallel_batched(chunks, SIMULATED_LATENCY)
    print(f"{t_par:.2f}s  ({calls_par} calls)")

    speedup = t_seq / t_par if t_par > 0 else float("inf")
    saved = t_seq - t_par
    print(f"\nSpeedup : {speedup:.1f}×  ({saved:.1f}s saved on the analysis step)")
    print(
        f"Note: transcription ({t_transcribe:.0f}s) is the same for both approaches "
        "and dominates total runtime."
    )


if __name__ == "__main__":
    main()
