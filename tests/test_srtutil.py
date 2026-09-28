"""Tests for srtutil: caption cleanup, rolling captions, sentence cues, echoes."""

from __future__ import annotations

from datetime import timedelta

import srt

from srtutil import (
    clean_caption,
    cues_to_sentences,
    drop_echo_cues,
    incremental_text,
    is_echo_cue,
    is_rolling_captions,
    load_srt,
    save_srt,
    segments_to_cues,
    unroll_rolling_cues,
)


def cue(start: float, end: float, text: str, index: int = 1) -> srt.Subtitle:
    return srt.Subtitle(
        index=index,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=text,
    )


def test_clean_caption_strips_tags_soundfx_and_whitespace():
    assert clean_caption("<i>Hello</i>  world\xa0again") == "Hello world again"
    assert clean_caption("[Music] real words [Applause]") == "real words"
    assert clean_caption("a\n\nb") == "a b"


def test_clean_caption_empty():
    assert clean_caption("") == ""
    assert clean_caption("[Laughter]") == ""


def test_is_rolling_captions_detects_sliding_captions():
    words = [f"w{i}" for i in range(40)]
    cues = [
        cue(i * 2.0, i * 2.0 + 2.0, " ".join(words[i : i + 5]))
        for i in range(20)
    ]
    assert is_rolling_captions(cues) is True


def test_is_rolling_captions_false_for_distinct_sentences():
    sentences = [
        "The city was quiet.",
        "Rain fell all night long.",
        "She never wrote back.",
        "Trains stop running at midnight.",
        "He kept the letter hidden.",
        "Nobody remembered the promise.",
        "Winter came early that year.",
        "The market closed on Monday.",
        "Her voice carried through the hall.",
        "A dog barked near the gate.",
        "They left without a word.",
        "The lights went out again.",
        "He counted the days twice.",
        "The river froze in February.",
        "Nobody answered the telephone.",
        "The old bridge was closed.",
        "Snow piled against the door.",
        "She sold the family farm.",
        "The letter arrived on Friday.",
        "He never came back home.",
        "The choir sang one more hymn.",
    ]
    cues = [
        cue(i * 2.0, i * 2.0 + 2.0, sentence)
        for i, sentence in enumerate(sentences)
    ]
    assert is_rolling_captions(cues) is False


def test_is_rolling_captions_false_when_few_cues():
    cues = [cue(0, 2, "a b"), cue(2, 4, "a b c")]
    assert is_rolling_captions(cues) is False


def test_incremental_text_variants():
    assert incremental_text("", "hello") == "hello"
    assert incremental_text("a b", "a b c") == "c"
    assert incremental_text("a b c", "a b c") == ""
    assert incremental_text("a b c", "a b c d") == "d"
    assert incremental_text("a b c d", "b c d e") == "e"
    assert incremental_text("x y", "z w") == "z w"


def test_unroll_rolling_cues_packs_into_sentences():
    cues = [
        cue(0.0, 1.0, "Hello there."),
        cue(1.0, 2.0, "Hello there. General"),
        cue(2.0, 3.0, "Hello there. General Kenobi."),
    ]
    packed = unroll_rolling_cues(cues)
    assert len(packed) == 2
    assert packed[0].content == "Hello there."
    assert packed[1].content == "General Kenobi."
    assert packed[1].start == timedelta(seconds=1)
    assert packed[1].end == timedelta(seconds=3)


def test_unroll_rolling_cues_hard_packs_long_runs():
    text = "word1 word2"
    cues = [cue(i * 3.0, i * 3.0 + 3.0, f"{text} part{i}") for i in range(8)]
    packed = unroll_rolling_cues(cues)
    assert packed
    for item in packed:
        assert (item.end - item.start).total_seconds() <= 15.0


def test_cues_to_sentences_interpolates_times():
    cues = [cue(0.0, 4.0, "One. Two."), cue(4.0, 8.0, "Three.")]
    out = cues_to_sentences(cues)
    assert [c.content for c in out] == ["One.", "Two.", "Three."]
    assert out[0].start == timedelta(0)
    assert out[-1].end == timedelta(seconds=8)
    mid = out[1]
    assert timedelta(seconds=2) < mid.start < timedelta(seconds=3)
    assert mid.end == timedelta(seconds=4)


def test_cues_to_sentences_empty_returns_empty():
    assert cues_to_sentences([]) == []


def test_is_echo_cue_matches_repeated_tail():
    assert is_echo_cue("They were a theologian.", "theologian.")
    assert is_echo_cue("Every Sunday I go.", "I go.")
    assert not is_echo_cue("Short.", "A completely different sentence here.")


def test_is_echo_cue_exempts_period_token():
    assert not is_echo_cue("He paused. Period.", "period")
    assert not is_echo_cue("Tečka.", "tečka")


def test_drop_echo_cues_extends_previous_end():
    cues = [
        cue(0.0, 4.0, "They were a theologian."),
        cue(4.0, 5.0, "theologian."),
        cue(5.0, 7.0, "Next sentence."),
    ]
    out = drop_echo_cues(cues)
    assert len(out) == 2
    assert out[0].end == timedelta(seconds=5)
    assert out[0].content == "They were a theologian."
    assert out[1].content == "Next sentence."


def test_save_and_load_roundtrip(tmp_path):
    cues = [
        cue(0.0, 2.0, "První věta.", 1),
        cue(2.0, 4.0, "Druhá věta.", 2),
    ]
    path = tmp_path / "round.srt"
    save_srt(path, cues)
    loaded = load_srt(path, sentences=False)
    assert len(loaded) == 2
    assert loaded[0].content == "První věta."
    assert loaded[0].start == timedelta(seconds=0)
    assert loaded[1].start == timedelta(seconds=2)
    assert loaded[1].end == timedelta(seconds=4)


def test_load_srt_raises_when_no_usable_cues(tmp_path):
    path = tmp_path / "empty.srt"
    path.write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n[Music]\n",
        encoding="utf-8",
    )
    try:
        load_srt(path, sentences=False)
    except ValueError:
        pass
    else:
        raise AssertionError("load_srt should reject caption-only files")


def test_segments_to_cues_splits_sentences():
    segments = [
        {"text": "Hello there.", "start": 0.0, "end": 2.0},
        {"text": "Bye now.", "start": 2.0, "end": 4.0},
    ]
    cues = segments_to_cues(segments)
    assert [c.content for c in cues] == ["Hello there.", "Bye now."]
    assert cues[0].start == timedelta(0)
    assert cues[-1].end == timedelta(seconds=4)


def test_segments_to_cues_rejects_empty():
    try:
        segments_to_cues([])
    except ValueError:
        pass
    else:
        raise AssertionError("segments_to_cues should reject empty input")
