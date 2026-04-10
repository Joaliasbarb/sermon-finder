import logging
import threading

import ctranslate2
from faster_whisper import WhisperModel

# faster-whisper / ctranslate2 emits a warning when float16 weights are
# silently converted to float32 on CPU. Suppress it — it is expected and harmless.
ctranslate2.set_log_level(logging.ERROR)


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
