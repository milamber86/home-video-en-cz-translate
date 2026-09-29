"""Tests for the reflow (translation compression) planning helpers."""

from __future__ import annotations

import argparse
from datetime import timedelta

import srt

import reflow_czech as rf


def _cue(start: float, end: float, text: str) -> srt.Subtitle:
    return srt.Subtitle(
        index=1,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=text,
    )


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "chars_per_sec": 14.0,
        "max_speed": 1.25,
        "base_speed": 1.15,
        "batch": 6,
        "timeout": 60.0,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_spoken_form_expands_numbers_and_keeps_period():
    assert rf.spoken_form("v roce 1976", None) == "v roce devatenáct set sedmdesát šest."


def test_spoken_form_preserves_existing_punctuation():
    assert rf.spoken_form("Ahoj!", None) == "Ahoj!"


def test_plan_flags_overlong_cue_only():
    cues = [
        _cue(0.0, 2.0, "slovo " * 16),
        _cue(2.0, 4.0, "Krátká věta."),
    ]
    plan = rf.plan_cues(cues, None, None, None, _args())
    assert plan[0]["action"] == "compress"
    assert plan[1]["action"] == "keep"
    assert plan[0]["needed_speed"] > 1.25
    assert plan[1]["needed_speed"] < 1.0


def test_plan_budget_scales_to_srt_chars():
    cues = [_cue(0.0, 2.0, "v roce 1976")]
    plan = rf.plan_cues(cues, None, None, None, _args())
    entry = plan[0]
    assert entry["budget_chars"] > 0
    assert entry["budget_chars"] <= int(2.0 * 1.25 * 14.0)


def test_plan_uses_vocal_timings_when_given():
    cues = [_cue(0.0, 4.0, "slovo " * 16)]
    from timing import cue_timings

    timings = cue_timings(cues, [(2.0, 3.0)])
    plan = rf.plan_cues(cues, None, timings, None, _args())
    assert plan[0]["slot"] == round(timings[0].slot, 3)
    assert plan[0]["slot"] < 4.0


def test_validate_rejects_empty_and_overlong():
    assert rf.validate("původní věta", "", 30) is None
    assert rf.validate("původní věta", "x" * 200, 30) is None


def test_validate_accepts_shorter_line():
    out = rf.validate("To je velmi dlouhá původní věta.", "Krátká věta.", 30)
    assert out == "Krátká věta."


def test_validate_rejects_no_op_when_original_fits():
    assert rf.validate("Krátká.", "Krátká.", 30) is None


def test_validate_strips_instruction_leak():
    out = rf.validate("dlouhá původní věta", "Remember: kratší věta.", 40)
    assert out == "Kratší věta."
