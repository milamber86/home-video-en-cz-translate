"""Tests for timing: speech regions, cue onsets/slots, and duration budgets."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
import srt

from timing import (
    char_budget,
    cue_slot,
    cue_timings,
    est_duration,
    frame_rms,
    speech_regions,
    voiced_end,
)


def _cue(start: float, end: float, text: str = "x") -> srt.Subtitle:
    return srt.Subtitle(
        index=1,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=text,
    )


def _track(sr: int = 1000, spans=()) -> np.ndarray:
    audio = np.zeros(sr * 4, dtype=np.float32)
    for start, end in spans:
        audio[int(start * sr) : int(end * sr)] = 0.5
    return audio


def test_frame_rms_shape_and_energy():
    sr = 1000
    audio = _track(sr, [(1.0, 2.0)])
    rms = frame_rms(audio, sr, 0.02)
    assert rms.size == 200
    assert rms[75] > 0.4
    assert rms[0] == 0.0


def test_speech_regions_finds_voiced_span():
    sr = 1000
    regions = speech_regions(_track(sr, [(1.0, 2.0)]), sr)
    assert len(regions) == 1
    start, end = regions[0]
    assert start == pytest.approx(1.0, abs=0.05)
    assert end == pytest.approx(2.0, abs=0.05)


def test_speech_regions_merges_short_gaps():
    sr = 1000
    regions = speech_regions(_track(sr, [(1.0, 1.5), (1.6, 2.5)]), sr, merge_gap=0.25)
    assert len(regions) == 1
    assert regions[0][0] == pytest.approx(1.0, abs=0.05)
    assert regions[0][1] == pytest.approx(2.5, abs=0.05)


def test_speech_regions_drops_blips():
    sr = 1000
    regions = speech_regions(_track(sr, [(1.0, 1.05)]), sr, min_dur=0.15)
    assert regions == []


def test_voiced_end_stops_at_last_voiced_frame():
    sr = 1000
    audio = _track(sr, [(0.5, 1.5)])
    assert voiced_end(audio, sr) == pytest.approx(1.5, abs=0.05)


def test_voiced_end_full_when_voiced_throughout():
    sr = 1000
    audio = np.full(sr * 2, 0.4, dtype=np.float32)
    assert voiced_end(audio, sr) == pytest.approx(2.0, abs=0.05)


def test_cue_slot_back_to_back_and_gap():
    assert cue_slot(_cue(0.0, 2.0), _cue(2.0, 4.0)) == pytest.approx(2.0)
    slot = cue_slot(_cue(0.0, 2.0), _cue(4.0, 6.0))
    assert slot > 2.0
    assert slot <= 4.0 - 0.1


def test_cue_timings_starts_at_original_onset():
    cues = [_cue(0.0, 4.0), _cue(4.0, 8.0)]
    regions = [(1.0, 3.0), (5.0, 7.0)]
    timings = cue_timings(cues, regions)
    assert timings[0].start_at == pytest.approx(0.95, abs=0.01)
    assert timings[0].onset == pytest.approx(1.0)
    assert timings[0].offset == pytest.approx(3.0)
    assert timings[1].start_at == pytest.approx(4.95, abs=0.01)


def test_cue_timings_slot_never_reaches_next_onset():
    cues = [_cue(0.0, 4.0), _cue(4.0, 8.0)]
    regions = [(1.0, 3.0), (5.0, 7.0)]
    timings = cue_timings(cues, regions)
    first = timings[0]
    assert first.start_at + first.slot <= 5.0 - 0.1 + 1e-6
    assert first.slot > first.offset - first.start_at


def test_cue_timings_falls_back_without_speech():
    cues = [_cue(0.0, 4.0)]
    timings = cue_timings(cues, [])
    assert timings[0].start_at == pytest.approx(0.0)
    assert timings[0].slot == pytest.approx(4.0, abs=0.01)


def test_est_duration_and_char_budget():
    assert est_duration("a" * 140, 14.0) == pytest.approx(10.0)
    assert char_budget(2.0, 1.25, 14.0) == 35
    assert char_budget(2.0, 1.0, 14.0) == 28
