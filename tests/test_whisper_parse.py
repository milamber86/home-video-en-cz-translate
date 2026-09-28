"""Tests for whisper_translate parsing, alignment, and cleaning helpers."""

from __future__ import annotations

import argparse
from pathlib import Path

import whisper_translate as wt


def test_parse_json_array_plain():
    assert wt.parse_json_array('["Ahoj.", "Nazdar."]') == ["Ahoj.", "Nazdar."]


def test_parse_json_array_fenced():
    text = "```json\n[\"Ahoj.\"]\n```"
    assert wt.parse_json_array(text) == ["Ahoj."]


def test_parse_json_array_object_with_cues_key():
    assert wt.parse_json_array('{"cues": ["A.", "B."]}') == ["A.", "B."]


def test_parse_json_array_dict_with_cs():
    assert wt.parse_json_array('{"cs": "Ahoj."}') == ["Ahoj."]


def test_parse_json_array_embedded_in_prose():
    text = 'Here you go: ["První.", "Druhá."] hope that helps'
    assert wt.parse_json_array(text) == ["První.", "Druhá."]


def test_parse_json_array_single_item_list_keys():
    assert wt.parse_json_array('{"translations": ["x"]} ') == ["x"]


def test_parse_json_array_rejects_garbage():
    for bad in ("", "no json here at all"):
        try:
            wt.parse_json_array(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_as_str_list_nested_list():
    assert wt._as_str_list({"result": {"items": ["a", "b"]}}) == ["a", "b"]


def test_as_str_list_str():
    assert wt._as_str_list("hello") == ["hello"]
    assert wt._as_str_list("  ") == []


def test_align_translations_exact():
    assert wt.align_translations(["a", "b"], ["x", "y"], "") == ["a", "b"]


def test_align_translations_drops_prev_when_present():
    assert wt.align_translations(["prev", "a", "b"], ["x", "y"], "prev") == ["a", "b"]


def test_align_translations_drops_last_without_prev():
    assert wt.align_translations(["a", "b"], ["x"], "") == ["a"]


def test_align_translations_truncates_extra():
    assert wt.align_translations(["a", "b", "c"], ["x", "y"], "") == ["a", "b"]


def test_align_translations_rejects_missing():
    try:
        wt.align_translations(["a"], ["x", "y"], "")
    except ValueError:
        pass
    else:
        raise AssertionError("short output must raise")


def test_strip_instruction_leak_removes_prefixes():
    assert wt.strip_instruction_leak("Remember: Stay here.") == "Stay here."
    assert wt.strip_instruction_leak("Ale: nic.") == "Nic."
    assert wt.strip_instruction_leak("Pamatujte si: jedno.") == "Jedno."


def test_strip_instruction_leak_drops_leak_only_lines():
    assert wt.strip_instruction_leak("Mluvčí je žena.") == ""
    assert wt.strip_instruction_leak("The speaker is a woman.") == ""
    assert wt.strip_instruction_leak("Ženský rod.") == ""


def test_strip_instruction_leak_capitalizes_and_unlabels():
    assert wt.strip_instruction_leak("ale vzpomínám.") == "Ale vzpomínám."
    assert (
        wt.strip_instruction_leak("NARRATOR: this is spoken text.")
        == "This is spoken text."
    )
    assert (
        wt.strip_instruction_leak("SPEAKER: Proper case.")
        == "SPEAKER: Proper case."
    )


def test_capitalize_echoes():
    out = wt.capitalize_echoes(
        ["every Sunday", "Sunday."],
        ["každou neděli", "každou neděli."],
    )
    assert out == ["každou neděli", "Každou neděli."]


def test_capitalize_echoes_ignores_long_cues():
    out = wt.capitalize_echoes(["a b c d e f", "a b c d e f"], ["lower", "lower"])
    assert out == ["lower", "lower"]


def test_local_cue_translation_period():
    assert wt.local_cue_translation("Period.", "", "") == "Tečka."


def test_local_cue_translation_echo_word():
    assert (
        wt.local_cue_translation("Sunday.", "every Sunday.", "Každou neděli.")
        == "Neděli."
    )


def test_local_cue_translation_returns_none_for_normal_text():
    assert wt.local_cue_translation("A new thought.", "every Sunday.", "Každou neděli.") is None


def test_clean_translategemma_strips_quotes_and_prefixes():
    assert wt.clean_translategemma('"Ahoj světe."') == "Ahoj světe."
    assert wt.clean_translategemma("Czech: Ahoj.") == "Ahoj."
    assert wt.clean_translategemma("Translation: Nazdar.") == "Nazdar."


def test_clean_translategemma_removes_context_banned_lines():
    text = "Každou neděli.\nSprávně jsme to měli."
    out = wt.clean_translategemma(text, ["Každou neděli."])
    assert out == "Správně jsme to měli."


def test_clean_translategemma_merges_near_duplicates():
    out = wt.clean_translategemma("Každou neděli.\nKaždou neděli")
    assert out == "Každou neděli."


def test_pick_ollama_model_prefers_exact():
    tags = {"models": [{"name": "qwen2.5:14b"}, {"name": "translategemma:12b"}]}
    assert wt.pick_ollama_model(tags, "translategemma:12b") == "translategemma:12b"


def test_pick_ollama_model_falls_back_to_family():
    tags = {"models": [{"name": "translategemma:4b"}]}
    assert wt.pick_ollama_model(tags, "translategemma:12b") == "translategemma:4b"


def test_pick_ollama_model_raises_without_models():
    try:
        wt.pick_ollama_model({"models": []}, "translategemma:12b")
    except RuntimeError:
        pass
    else:
        raise AssertionError("empty model list must raise")


def test_pick_chat_model_skips_translategemma():
    tags = {"models": [{"name": "translategemma:12b"}, {"name": "qwen2.5:14b"}]}
    assert wt.pick_chat_model(tags, "qwen2.5:14b") == "qwen2.5:14b"


def test_pick_chat_model_returns_none_when_only_translategemma():
    tags = {"models": [{"name": "translategemma:12b"}]}
    assert wt.pick_chat_model(tags, "translategemma:12b") is None


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    defaults = {
        "audio": None,
        "en_srt": None,
        "out_en": str(tmp_path / "en.srt"),
        "force_asr": False,
        "whisper_model": "test-model",
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def _write_srt(path: Path, lines: list[str]) -> Path:
    from datetime import timedelta

    import srt as srt_mod

    cues = [
        srt_mod.Subtitle(
            index=i,
            start=timedelta(seconds=i * 2),
            end=timedelta(seconds=i * 2 + 2),
            content=text,
        )
        for i, text in enumerate(lines, start=1)
    ]
    path.write_text(srt_mod.compose(cues), encoding="utf-8")
    return path


def test_english_cues_falls_back_to_youtube_srt(tmp_path):
    en_srt = _write_srt(tmp_path / "yt.en.srt", ["One sentence.", "Another one."])
    cues = wt.english_cues(_args(tmp_path, en_srt=str(en_srt)))
    assert [c.content for c in cues] == ["One sentence.", "Another one."]


def test_english_cues_prefers_whisper_cache(tmp_path):
    out_en = tmp_path / "en.srt"
    _write_srt(tmp_path / "whisper.en.srt", ["Cached whisper line."])
    _write_srt(tmp_path / "yt.en.srt", ["YouTube line."])
    cues = wt.english_cues(_args(tmp_path, out_en=str(out_en)))
    assert [c.content for c in cues] == ["Cached whisper line."]


def test_english_cues_requires_audio_or_srt(tmp_path):
    try:
        wt.english_cues(_args(tmp_path))
    except SystemExit:
        pass
    else:
        raise AssertionError("no sources must raise SystemExit")
