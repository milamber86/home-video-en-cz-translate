"""Tests for synthesize_czech pure helpers (no model loads)."""

from __future__ import annotations

import shutil

import numpy as np
import pytest
import soundfile as sf

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
    assert sc.fit_cache_tag("xtts_m2", 1.15, 1.25) == "xtts_m2_b115m125"


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
    assert 0.9 <= dur <= 1.1


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
