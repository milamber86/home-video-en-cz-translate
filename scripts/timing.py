"""Speech-timing helpers shared by TTS fitting and the translation reflow pass.

The pipeline already has an isolated vocal stem (`vocals.wav`) per video. Its
energy profile tells us where the original speaker actually talks inside each
subtitle cue, which is what the dubbed line should match: start when the
original starts, and only stretch or spill into real silence.
"""

from __future__ import annotations

from typing import NamedTuple


class CueTiming(NamedTuple):
    """Where a dubbed cue should start and how long it may run."""

    start_at: float
    slot: float
    onset: float
    offset: float


def frame_rms(audio, sr: int, frame_sec: float = 0.02):
    """Per-frame RMS energy of a mono signal."""
    import numpy as np

    frame = max(int(frame_sec * sr), 1)
    n = len(audio) // frame
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    blocks = audio[: n * frame].reshape(n, frame)
    return np.sqrt(np.mean(blocks.astype("float64") ** 2, axis=1)).astype(np.float32)


def _mask_regions(mask, frame_sec: float, min_dur: float, merge_gap: float):
    """Collapse a boolean frame mask into (start, end) seconds."""
    regions: list[tuple[float, float]] = []
    start = None
    for i, hit in enumerate(mask):
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            regions.append((start * frame_sec, (i) * frame_sec))
            start = None
    if start is not None:
        regions.append((start * frame_sec, len(mask) * frame_sec))

    merged: list[tuple[float, float]] = []
    for region in regions:
        if merged and region[0] - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], region[1])
        else:
            merged.append(region)
    return [r for r in merged if r[1] - r[0] >= min_dur]


def speech_regions(
    audio,
    sr: int,
    frame_sec: float = 0.02,
    thr_ratio: float = 0.08,
    min_dur: float = 0.15,
    merge_gap: float = 0.25,
) -> list[tuple[float, float]]:
    """Voiced spans (absolute seconds) of a vocal stem."""
    import numpy as np

    rms = frame_rms(audio, sr, frame_sec)
    if rms.size == 0:
        return []
    peak = float(np.percentile(rms, 98)) if rms.size > 8 else float(np.max(rms))
    thr = max(peak * thr_ratio, 1e-4)
    return _mask_regions(rms > thr, frame_sec, min_dur, merge_gap)


def load_speech_regions(path, **kwargs) -> list[tuple[float, float]]:
    """Speech regions of a WAV file (mono mixdown)."""
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(str(path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return speech_regions(np.asarray(audio, dtype="float32"), int(sr), **kwargs)


def voiced_end(audio, sr: int, frame_sec: float = 0.02, thr_ratio: float = 0.08) -> float:
    """Seconds at which the last voiced frame ends (len/sr when voiced throughout)."""
    import numpy as np

    rms = frame_rms(audio, sr, frame_sec)
    if rms.size == 0:
        return len(audio) / float(sr)
    thr = max(float(np.max(rms)) * thr_ratio, 1e-4)
    voiced = rms > thr
    if not voiced.any():
        return 0.0
    last = int(np.max(np.nonzero(voiced)[0]))
    return min((last + 1) * frame_sec, len(audio) / float(sr))


def voiced_start(audio, sr: int, frame_sec: float = 0.02, thr_ratio: float = 0.08) -> float:
    """Seconds at which the first voiced frame starts (0.0 when voiced throughout)."""
    import numpy as np

    rms = frame_rms(audio, sr, frame_sec)
    if rms.size == 0:
        return 0.0
    thr = max(float(np.max(rms)) * thr_ratio, 1e-4)
    voiced = rms > thr
    if not voiced.any():
        return len(audio) / float(sr)
    first = int(np.min(np.nonzero(voiced)[0]))
    return first * frame_sec


def cue_slot(cue, next_cue) -> float:
    """Fallback slot without vocal analysis: cue length plus bounded gap spill."""
    from srtutil import cue_seconds

    slot = max(cue_seconds(cue), 0.08)
    if next_cue is None:
        return slot
    gap = (next_cue.start - cue.end).total_seconds()
    if gap < 0.15:
        return slot
    return slot + min(gap * 0.7, gap - 0.1, 2.0)


def _span(regions: list[tuple[float, float]], start: float, end: float):
    hits = [r for r in regions if r[1] > start and r[0] < end]
    if not hits:
        return None
    return max(start, hits[0][0]), min(end, hits[-1][1])


def cue_timings(
    cues,
    regions: list[tuple[float, float]],
    pad: float = 0.05,
    keep_gap: float = 0.1,
    spill_ratio: float = 0.7,
    max_spill: float = 2.0,
) -> list[CueTiming]:
    """Per-cue start/slot derived from where the original voice speaks.

    `start_at` is shifted to the original speech onset so the dub lines up with
    the source speaker instead of always starting at the subtitle edge. `slot`
    covers the original speech span plus a bounded share of the silence that
    follows it, never reaching into the next cue's speech.
    """
    bounds = [
        (c.start.total_seconds(), c.end.total_seconds()) for c in cues
    ]
    spans = [_span(regions, start, end) for start, end in bounds]
    onsets = [s[0] if s else start for s, (start, _end) in zip(spans, bounds)]

    timings: list[CueTiming] = []
    for i, ((start, end), span) in enumerate(zip(bounds, spans)):
        onset, offset = span if span else (start, end)
        start_at = max(start, onset - pad)
        next_onset = onsets[i + 1] if i + 1 < len(onsets) else None
        allowed_end = end if next_onset is None else min(end + max_spill, next_onset - keep_gap)
        room = max(0.08, allowed_end - start_at)
        target = max(offset - start_at, 0.08)
        slot = min(target + spill_ratio * max(0.0, room - target), room)
        timings.append(CueTiming(start_at, slot, onset, offset))
    return timings


def est_duration(text: str, chars_per_sec: float = 14.0) -> float:
    """Rough spoken duration of Czech text (calibrated for the local engines)."""
    return len((text or "").strip()) / max(chars_per_sec, 1.0)


def char_budget(slot_sec: float, max_speed: float, chars_per_sec: float = 14.0) -> int:
    """Characters a cue may have and still fit `slot_sec` at `max_speed`."""
    return int(slot_sec * max(max_speed, 1.0) * chars_per_sec)
