"""Tests for xtts_synth text chunking and helpers."""

from __future__ import annotations

import xtts_synth as xs
from text_split import _split_words, split_sentences


def test_is_xtts_model():
    assert xs.is_xtts_model("tts_models/multilingual/multi-dataset/xtts_v2")
    assert not xs.is_xtts_model("tts_models/cs/cv/vits")
    assert not xs.is_xtts_model("")


def test_split_cs_chunks_short():
    assert xs.split_cs_chunks("Krátká věta.", 180) == ["Krátká věta."]


def test_split_cs_chunks_empty():
    assert xs.split_cs_chunks("   ", 180) == []


def test_split_cs_chunks_splits_on_punctuation():
    text = "První. " + "Druhá druhá druhá druhá."
    chunks = xs.split_cs_chunks(text, 10)
    assert chunks == ["První.", "Druhá", "druhá", "druhá", "druhá."]
    assert all(len(chunk) <= 10 for chunk in chunks)


def test_split_cs_chunks_word_fallback():
    text = " ".join(["slovo"] * 50)
    chunks = xs.split_cs_chunks(text, 10)
    assert all(len(chunk) <= 10 for chunk in chunks)
    assert len(chunks) == 50


def test_split_cs_chunks_long_single_word_kept():
    word = "A" * 200
    assert xs.split_cs_chunks(word, 180) == [word]


def test_split_words():
    assert _split_words("aaa bbb ccc", 7) == ["aaa bbb", "ccc"]
    assert _split_words("ab cd", 10) == ["ab cd"]


def test_split_sentences():
    assert split_sentences("První. Druhá! Třetí?") == ["První.", "Druhá!", "Třetí?"]
    assert split_sentences("Bez tečky") == ["Bez tečky"]
    assert split_sentences("") == []
