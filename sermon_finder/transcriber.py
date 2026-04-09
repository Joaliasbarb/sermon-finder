import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from faster_whisper import WhisperModel
from tqdm import tqdm

from sermon_finder.audio import split_wav

_MODEL_RAM_GB = {"tiny": 1, "base": 1, "small": 2, "medium": 5, "large-v3": 10}


def _transcribe_segment(
    wav_path: str,
    offset_s: float,
    keep_until_s: float | None,
    model_size: str,
    thread_local: threading.local,
) -> list[dict]:
    """Transcribe one audio segment using the thread-local WhisperModel.

    The model is created lazily on first call per thread and reused for all
    subsequent segments processed by that thread.

    offset_s: seconds to add to every segment timestamp
    keep_until_s: drop segments whose offset-corrected start >= this value
                  (trims the overlap tail); None means last segment, keep all
    """
    if not hasattr(thread_local, "model"):
        thread_local.model = WhisperModel(model_size)
    model = thread_local.model
    segments_gen, _ = model.transcribe(wav_path, language="fr", vad_filter=True)
    segments = []
    for seg in segments_gen:
        true_start = seg.start + offset_s
        true_end = seg.end + offset_s
        if keep_until_s is not None and true_start >= keep_until_s:
            continue
        segments.append({"start": true_start, "end": true_end, "text": seg.text.strip()})
    return segments


def transcribe(
    audio_path: str,
    model_size: str = "small",
    verbose: bool = False,
    num_workers: int = 1,
) -> list[dict]:
    """Transcribe audio and return segments with timestamps.

    Returns a list of dicts: [{"start": float, "end": float, "text": str}, ...]

    Audio is split into 2-minute segments with 30s overlap, processed in
    chronological order. With num_workers=1 (default): segments are transcribed
    sequentially. With num_workers>1: segments are processed in parallel using a
    thread pool, each thread holding its own WhisperModel instance.
    """
    if num_workers > 1:
        ram_gb = _MODEL_RAM_GB.get(model_size, 5) * num_workers
        print(
            f"Warning: {num_workers} threads × '{model_size}' ≈ {ram_gb} GB RAM required.",
            file=sys.stderr,
        )
    else:
        print(f"Loading Whisper model ({model_size})...", file=sys.stderr)

    thread_local = threading.local()
    with split_wav(audio_path, segment_s=120.0, overlap_s=30.0) as chunks:
        print(
            f"Splitting audio into {len(chunks)} segments (~2 min each)...",
            file=sys.stderr,
        )
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(
                    _transcribe_segment, path, offset, keep_until, model_size, thread_local
                )
                for path, offset, keep_until in chunks
            ]
            results: list[list[dict]] = [None] * len(futures)  # type: ignore[list-item]
            future_to_idx = {f: i for i, f in enumerate(futures)}
            with tqdm(
                total=len(futures), desc="Transcribing", unit="seg", file=sys.stderr
            ) as pbar:
                for future in as_completed(futures):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    pbar.update(1)
                    if verbose:
                        for seg in results[idx]:
                            tqdm.write(
                                f"  [{seg['start']:.1f}s] {seg['text']}", file=sys.stderr
                            )

    all_segments = [seg for segs in results for seg in segs]
    return sorted(all_segments, key=lambda s: s["start"])
