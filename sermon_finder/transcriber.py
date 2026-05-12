import logging
import queue
import threading

import ctranslate2
from faster_whisper import WhisperModel

from sermon_finder import audio

# faster-whisper / ctranslate2 emits a warning when float16 weights are
# silently converted to float32 on CPU. Suppress it — it is expected and harmless.
ctranslate2.set_log_level(logging.ERROR)


def transcriber_worker(
    transition_queue: queue.Queue,
    transcription_queue: queue.Queue,
    found: threading.Event,
    wav_path: str,
    model_size: str,
    thread_local: threading.local,
    on_transcribe_start=None,
) -> None:
    """Consume transition_queue, transcribe each window, push results to transcription_queue.

    transition_queue items: (t, segment_idx, transition_idx, total_transitions, offset_s, seg_end_s)
    transcription_queue items: (t, segments, model_size, segment_idx, transition_idx, total_transitions, offset_s, seg_end_s)
    Sentinel None signals end-of-stream.
    on_transcribe_start(t, transition_idx, total_transitions, segment_idx, model_size)
    """
    while True:
        if found.is_set():
            transcription_queue.put(None)
            return

        item = transition_queue.get()
        if item is None:
            transcription_queue.put(None)
            return

        t, segment_idx, transition_idx, total_transitions, offset_s, seg_end_s = item

        if found.is_set():
            transcription_queue.put(None)
            return

        if on_transcribe_start:
            on_transcribe_start(t, transition_idx, total_transitions, segment_idx, model_size)

        with audio.extract_window(wav_path, t - 30.0, t + 30.0) as (win_path, win_start):
            segments = transcribe_segment(
                win_path, win_start, keep_until_s=None,
                model_size=model_size, thread_local=thread_local,
            )

        transcription_queue.put((t, segments, model_size, segment_idx, transition_idx, total_transitions, offset_s, seg_end_s))


def transcribe_segment(
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
