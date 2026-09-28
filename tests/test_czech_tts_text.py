"""Tests for czech_tts_text: numbers, abbreviations, names, TTS expansion."""

from __future__ import annotations

import czech_tts_text as ctt


def test_expand_numbers_year():
    assert ctt.expand_numbers("v roce 1976") == "v roce devatenáct set sedmdesát šest"


def test_expand_numbers_decades():
    assert ctt.expand_numbers("v 50. letech") == "v padesátých letech"
    assert ctt.expand_numbers("za 30. let") == "za třicátých let"


def test_expand_numbers_percent():
    assert ctt.expand_numbers("3,5 %") == "tři celá pět procent"
    assert ctt.expand_numbers("10 %") == "deset procent"


def test_expand_numbers_multiple_and_decimal():
    assert ctt.expand_numbers("3 krát") == "tři krát"
    assert ctt.expand_numbers("2,5") == "dva celá pět"


def test_expand_numbers_plain_integer():
    assert ctt.expand_numbers("mám 42") == "mám čtyřicet dva"


def test_expand_abbreviations_known():
    assert ctt.expand_abbreviations("v USA") == "v ú es á"
    assert ctt.expand_abbreviations("NATO schůzka") == "Náto schůzka"
    assert ctt.expand_abbreviations("na TV") == "na té vé"


def test_expand_abbreviations_unknown_allcaps_spelled():
    assert ctt.expand_abbreviations("FEMA řekla") == "ef é em á řekla"


def test_spell_letters():
    assert ctt.spell_letters("W") == "dvojité vé"
    assert ctt.spell_letters("X") == "iks"


def test_expand_names_from_glossary():
    glossary = {
        "terms": [
            {"en": "Satoshi", "cs": "Satoshi", "kind": "person", "tts": "Satoši"}
        ]
    }
    names = ctt.load_name_pronunciations(glossary)
    assert names == [("Satoshi", "Satoši")]
    assert ctt.expand_names("Satoshi to vymyslel", names) == "Satoši to vymyslel"


def test_load_name_pronunciations_skips_identical_and_missing(tmp_path):
    glossary = {
        "terms": [
            {"en": "Havel", "cs": "Havel", "tts": ""},
            {"en": "Nobody", "cs": "Nikdo", "tts": "Nikdo"},
            "junk",
        ]
    }
    assert ctt.load_name_pronunciations(glossary) == []


def test_load_name_pronunciations_from_missing_path(tmp_path):
    assert ctt.load_name_pronunciations(tmp_path / "nope.json") == []


def test_expand_for_tts_combines_all_stages():
    out = ctt.expand_for_tts("V 50. letech v USA měli 3,5 %." )
    assert out == "V padesátých letech v ú es á měli tři celá pět procent."


def test_expand_for_tts_keeps_display_text_without_numbers():
    assert ctt.expand_for_tts("") == ""
    assert ctt.expand_for_tts("Jen obyčejná věta.") == "Jen obyčejná věta."
