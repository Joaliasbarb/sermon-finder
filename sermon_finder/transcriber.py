import logging

import ctranslate2
from faster_whisper import WhisperModel

from sermon_finder import audio

# faster-whisper / ctranslate2 emits a warning when float16 weights are
# silently converted to float32 on CPU. Suppress it — it is expected and harmless.
ctranslate2.set_log_level(logging.ERROR)

_model_cache: dict[str, "WhisperModel"] = {}


def transcribe_segment(
    wav_path: str,
    offset_s: float,
    keep_until_s: float | None,
    model_size: str,
    model_cache: dict | None = None,
) -> list[dict]:
    """Transcribe one audio segment using a cached WhisperModel (V6).

    model_cache: caller-owned dict keyed by model_size; pass {} per run for isolation.
                 None falls back to the module-level cache.
    offset_s: seconds to add to every segment timestamp
    keep_until_s: drop segments whose offset-corrected start >= this value; None keeps all
    """
    cache = model_cache if model_cache is not None else _model_cache
    if model_size not in cache:
        cache[model_size] = WhisperModel(model_size)
    model = cache[model_size]
    segments_gen, _ = model.transcribe(wav_path, language="fr", vad_filter=True)
    segments = []
    for seg in segments_gen:
        true_start = seg.start + offset_s
        true_end = seg.end + offset_s
        if keep_until_s is not None and true_start >= keep_until_s:
            continue
        segments.append({"start": true_start, "end": true_end, "text": seg.text.strip()})
    return segments
