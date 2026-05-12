import queue
import threading

from diarize import diarize as _diarize


def run_diarization(wav_path: str, offset_s: float = 0.0) -> list[dict]:
    """Diarize one audio chunk. Returns segments with absolute timestamps.

    offset_s is added to every segment timestamp so results are in absolute
    recording time, consistent with how transcribe_segment handles offsets.
    """
    result = _diarize(wav_path)
    return [
        {"speaker": seg.speaker, "start": seg.start + offset_s, "end": seg.end + offset_s}
        for seg in result.segments
    ]


def diarizer_worker(
    segment_queue: queue.Queue,
    transition_queue: queue.Queue,
    found: threading.Event,
    on_segment_start=None,
    on_no_transitions=None,
) -> None:
    """Consume segment_queue, diarize each chunk, push transitions to transition_queue.

    segment_queue items: (chunk_path, offset_s, keep_until_s, segment_idx, total_segments)
    transition_queue items: (t, segment_idx, transition_idx, total_transitions, offset_s, seg_end_s)
    Sentinel None in either queue signals end-of-stream.
    on_segment_start(segment_idx, total_segments, offset_s, seg_end_s)
    on_no_transitions(segment_idx, total_segments, offset_s, seg_end_s)
    """
    while True:
        if found.is_set():
            transition_queue.put(None)
            return

        try:
            item = segment_queue.get(timeout=0.5)
        except queue.Empty:
            continue
        if item is None:
            transition_queue.put(None)
            return

        chunk_path, offset_s, keep_until_s, segment_idx, total_segments = item
        seg_end_s = keep_until_s + 30.0 if keep_until_s is not None else offset_s + 240.0

        if on_segment_start:
            on_segment_start(segment_idx, total_segments, offset_s, seg_end_s)

        speaker_segs = run_diarization(chunk_path, offset_s)
        transitions = get_speaker_transitions(speaker_segs)

        if not transitions and on_no_transitions:
            on_no_transitions(segment_idx, total_segments, offset_s, seg_end_s)

        for j, t in enumerate(transitions, 1):
            if found.is_set():
                transition_queue.put(None)
                return
            transition_queue.put((t, segment_idx, j, len(transitions), offset_s, seg_end_s))


def get_speaker_transitions(segments: list[dict]) -> list[float]:
    """Return absolute timestamps (seconds) where the speaker changes."""
    return [
        segments[i]["start"]
        for i in range(1, len(segments))
        if segments[i]["speaker"] != segments[i - 1]["speaker"]
    ]
