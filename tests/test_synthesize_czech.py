"""Tests for synthesize_czech pure helpers (no model loads)."""

from __future__ import annotations

import argparse
import shutil
from datetime import timedelta

import numpy as np
import pytest
import soundfile as sf
import srt

import synthesize_czech as sc


def test_atempo_chain_single():
    assert sc.atempo_chain(1.0) == "atempo=1.000000"
    assert sc.atempo_chain(1.15) == "atempo=1.150000"
    assert sc.atempo_chain(2.0) == "atempo=2.000000"


def test_atempo_chain_splits_above_two():
    assert sc.atempo_chain(4.0) == "atempo=2.0,atempo=2.000000"


def test_atempo_chain_splits_below_half():
    assert sc.atempo_chain(0.25) == "atempo=0.5,atempo=0.500000"


def test_atempo_chain_rejects_zero():
    try:
        sc.atempo_chain(0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("ratio 0 must raise")


def test_voice_cache_id_and_stock_engine():
    assert sc.voice_cache_id("piper") == "piper_jirka"
    assert sc.voice_cache_id("xtts") == "xtts_m2"
    assert sc.voice_cache_id("zzz") == "zzz"
    assert sc.stock_engine("female") == "vits"
    assert sc.stock_engine("male") == "piper"


def test_fit_cache_tag():
    assert sc.fit_cache_tag("xtts_m2", 1.15, 1.25) == "xtts_m2_b115m125v2"


def test_spoken_cache_tag_deterministic():
    assert sc.spoken_cache_tag("ahoj") == sc.spoken_cache_tag("ahoj")
    assert sc.spoken_cache_tag("ahoj") != sc.spoken_cache_tag("nazdar")


def test_normalize_czech_appends_period():
    assert sc.normalize_czech("ahoj", "piper", True) == "ahoj."


def test_normalize_czech_f5_base_grapheme_fix():
    assert sc.normalize_czech("kůň", "f5", False) == "kúň."
    assert sc.normalize_czech("kůň", "f5", True) == "kůň."


def test_overlay_sums_and_normalizes():
    sr = 100
    ones = np.ones(100, dtype=np.float32)
    pieces = [(0.0, ones), (1.0, ones)]
    canvas = sc.overlay(pieces, 2.0, sr)
    assert canvas.shape == (200,)
    assert canvas.max() == pytest.approx(0.99)
    assert canvas[50] == pytest.approx(0.99)
    assert canvas[150] == pytest.approx(0.99)


def test_overlay_drops_piece_beyond_duration():
    sr = 100
    ones = np.ones(100, dtype=np.float32)
    canvas = sc.overlay([(3.0, ones)], 1.0, sr)
    assert canvas.shape == (100,)
    assert canvas.max() == 0.0


def _tone(seconds: float, sr: int, amp: float = 0.4) -> np.ndarray:
    t = np.arange(int(seconds * sr), dtype=np.float32) / sr
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def test_trim_tts_tail_cuts_trailing_silence():
    sr = 48000
    audio = np.concatenate([_tone(1.0, sr), np.full(int(0.6 * sr), 1e-6, np.float32)])
    out = sc.trim_tts_tail(audio, sr)
    dur = len(out) / sr
    assert 0.9 <= dur <= 1.15


def test_trim_tts_tail_keeps_voiced_audio():
    sr = 48000
    audio = _tone(1.0, sr)
    out = sc.trim_tts_tail(audio, sr)
    assert len(out) == len(audio)


def test_trim_tts_tail_returns_short_audio_with_fade():
    sr = 48000
    audio = _tone(0.05, sr)
    out = sc.trim_tts_tail(audio, sr)
    assert len(out) == len(audio)


def test_usable_segment(tmp_path):
    path = tmp_path / "seg.wav"
    assert sc.usable_segment(path) is False
    sf.write(str(path), _tone(0.2, 16000), 16000)
    assert sc.usable_segment(path) is True


def test_pick_reference_clip_prefers_voiced_window(tmp_path):
    pytest.importorskip("librosa")
    sr = 16000
    voiced = _tone(2.0, sr, amp=0.4)
    quiet = np.full(int(1.0 * sr), 1e-5, np.float32)
    vocals = tmp_path / "vocals.wav"
    sf.write(str(vocals), np.concatenate([quiet, voiced]), sr)
    dest = tmp_path / "ref.wav"
    out = sc.pick_reference_clip(vocals, dest, target_sec=0.5)
    assert out == dest
    audio, out_sr = sc.load_mono(dest)
    assert out_sr == sr
    assert sc.rms(audio) > 1e-3


def test_fade_out():
    sr = 100
    audio = np.ones(1000, dtype=np.float32)
    out = sc.fade_out(audio, sr, 0.1)
    assert out[0] == pytest.approx(1.0)
    assert out[-1] == pytest.approx(0.0)


def test_read_ref_text_from_file_and_inline(tmp_path):
    ref = tmp_path / "ref.txt"
    ref.write_text("referenční text\n", encoding="utf-8")
    assert sc.read_ref_text(str(ref)) == "referenční text"
    assert sc.read_ref_text("inline text") == "inline text"
    assert sc.read_ref_text(None) == ""


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_fit_to_slot_stretches_to_slot(tmp_path):
    sr = 16000
    src = tmp_path / "raw.wav"
    sf.write(str(src), _tone(1.0, sr), sr)
    dest = tmp_path / "fitted.wav"
    audio = sc.fit_to_slot(
        src,
        dest,
        target_sec=0.8,
        max_speed=1.25,
        tmp_dir=tmp_path,
        base_speed=1.15,
    )
    dur = len(audio) / sr
    assert dest.is_file()
    assert 0.78 <= dur <= 0.84


def _srt_cue(start: float, end: float, text: str = "x") -> srt.Subtitle:
    return srt.Subtitle(
        index=1,
        start=timedelta(seconds=start),
        end=timedelta(seconds=end),
        content=text,
    )


def test_cue_slot_no_spill_when_back_to_back():
    cue = _srt_cue(0.0, 2.0)
    nxt = _srt_cue(2.0, 4.0)
    assert sc.cue_slot(cue, nxt) == pytest.approx(2.0)


def test_cue_slot_spills_into_gap():
    cue = _srt_cue(0.0, 2.0)
    nxt = _srt_cue(4.0, 6.0)
    slot = sc.cue_slot(cue, nxt)
    assert 2.0 < slot <= 2.0 + 0.7 * 2.0 + 1e-6
    assert slot <= 4.0 - 0.1


def test_cue_slot_no_next_cue():
    cue = _srt_cue(0.0, 2.0)
    assert sc.cue_slot(cue, None) == pytest.approx(2.0)


def test_cue_slot_caps_huge_gap():
    cue = _srt_cue(0.0, 2.0)
    nxt = _srt_cue(30.0, 32.0)
    assert sc.cue_slot(cue, nxt) == pytest.approx(4.0)


def test_voice_cache_id_pocket():
    assert sc.voice_cache_id("pocket") == "pocket_cs"
    assert sc.voice_cache_id("xtts") == "xtts_m2"


def test_normalize_for_cer():
    assert sc.normalize_for_cer("Dobrý den, světe!") == "dobrý den světe"
    assert sc.normalize_for_cer("") == ""


def test_char_error_rate():
    assert sc.char_error_rate("kocka", "kocka") == 0.0
    assert sc.char_error_rate("kocka", "koka") == pytest.approx(1 / 5)
    assert sc.char_error_rate("", "") == 0.0
    assert sc.char_error_rate("", "x") == 1.0
    assert sc.char_error_rate("Dobrý den", "dobrý den") == 0.0


def test_tail_omissions_detects_missing_ending():
    ref = ["dobrý", "den", "toto", "je", "zkouška"]
    assert sc.tail_omissions(ref, ["dobrý", "den", "toto", "je"]) == ["zkouška"]
    assert sc.tail_omissions(ref, ["dobrý", "den", "toto", "je", "zkouška"]) == []
    assert sc.tail_omissions(ref, []) == ref


def test_missing_words_finds_omissions_anywhere():
    ref = ["ahoj", "světe", "jak", "se", "máš"]
    assert sc.missing_words(ref, ["ahoj", "jak", "se", "máš"]) == ["světe"]
    assert sc.missing_words(ref, ref) == []


def test_assess_cue_accepts_matching_transcript():
    v = sc.assess_cue("Dobrý den, toto je zkouška.", "Dobrý den, toto je zkouška.", 2.5, 0.15, 22.0)
    assert v["ok"]
    assert v["tail"] == []
    assert not v["too_fast"]


def test_assess_cue_flags_truncated_tail():
    v = sc.assess_cue("Dobrý den, toto je zkouška řeči.", "Dobrý den, toto je zkouška", 2.0, 0.15, 22.0)
    assert not v["ok"]
    assert v["tail"] == ["řeči"]


def test_assess_cue_flags_empty_and_too_fast():
    v = sc.assess_cue("Dobrý den, toto je zkouška řeči.", "", 0.0, 0.15, 22.0)
    assert not v["ok"] and v["empty"]
    v = sc.assess_cue("Dobrý den, toto je zkouška řeči.", "Dobrý den toto je zkouška řeči", 0.2, 0.15, 22.0)
    assert not v["ok"] and v["too_fast"]


def test_assess_cue_aligns_digits_with_spoken_form():
    v = sc.assess_cue(
        "V roce devatenáct set sedmdesát šest to funguje.",
        "V roce 1976 to funguje.",
        4.0,
        0.15,
        22.0,
    )
    assert v["ok"]


def test_regen_temperature_schedule():
    assert sc.regen_temperature("pocket", 1) == 0.45
    assert sc.regen_temperature("pocket", 2) == 0.2
    assert sc.regen_temperature("pocket", 5) == 0.45
    assert sc.regen_temperature("xtts", 1) == 0.45
    assert sc.regen_temperature("piper", 1) is None
    assert sc.regen_temperature("vits", 1) is None


def test_verdict_score_prefers_ok_take():
    good = sc.assess_cue("ahoj světe", "ahoj světe.", 1.0, 0.15, 22.0)
    bad = sc.assess_cue("ahoj světe", "ahoj", 1.0, 0.15, 22.0)
    assert sc._verdict_score(good) < sc._verdict_score(bad)


def test_trim_by_alignment_keeps_voiced_tail():
    sr = 100
    audio = np.ones(1000, dtype=np.float32)
    out, changed = sc.trim_by_alignment(audio, sr, [{"end": 3.0}])
    assert not changed
    assert len(out) == 1000


def test_trim_by_alignment_cuts_trailing_silence():
    sr = 100
    audio = np.concatenate(
        [np.ones(300, dtype=np.float32), np.zeros(700, dtype=np.float32)]
    )
    out, changed = sc.trim_by_alignment(audio, sr, [{"end": 2.5}])
    assert changed
    assert len(out) == 306
    same, changed = sc.trim_by_alignment(audio, sr, [])
    assert not changed
    assert len(same) == 1000


def test_select_engine_prefers_pocket(tmp_path):
    fake_python = tmp_path / "python"
    fake_python.write_text("#!/bin/sh\n")
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF")
    args = argparse.Namespace(
        engine="auto",
        voice_gender="male",
        voice_mode="clone",
        vocals=None,
        ref_audio=str(ref),
        pocket_python=str(fake_python),
        pocket_config=sc.POCKET_CONFIG,
        xtts_python="",
        ckpt="",
        vocab="",
    )
    assert sc.select_engine(args) == "pocket"


def test_select_engine_falls_back_without_pocket(tmp_path):
    args = argparse.Namespace(
        engine="auto",
        voice_gender="male",
        voice_mode="clone",
        vocals=None,
        ref_audio=None,
        pocket_python="",
        pocket_config=sc.POCKET_CONFIG,
        xtts_python="",
        ckpt="",
        vocab="",
    )
    assert sc.select_engine(args) == "piper"


def test_select_engine_rejects_explicit_pocket_without_venv(tmp_path):
    args = argparse.Namespace(
        engine="pocket",
        voice_gender="male",
        voice_mode="clone",
        vocals=None,
        ref_audio=None,
        pocket_python=str(tmp_path / "missing"),
        pocket_config=sc.POCKET_CONFIG,
        xtts_python="",
        ckpt="",
        vocab="",
    )
    try:
        sc.select_engine(args)
    except SystemExit:
        pass
    else:
        raise AssertionError("pocket without .venv-pocket must raise")
