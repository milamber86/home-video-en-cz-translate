#!/usr/bin/env python3
"""Expand Czech subtitle text so TTS can pronounce numbers, abbrevs, and names."""

from __future__ import annotations

import json
import re
from pathlib import Path

# Spoken letter names (Czech). Used for leftover Latin abbreviations.
_LETTER = {
    "A": "á",
    "B": "bé",
    "C": "cé",
    "D": "dé",
    "E": "é",
    "F": "ef",
    "G": "gé",
    "H": "há",
    "I": "í",
    "J": "jé",
    "K": "ká",
    "L": "el",
    "M": "em",
    "N": "en",
    "O": "ó",
    "P": "pé",
    "Q": "kvé",
    "R": "er",
    "S": "es",
    "T": "té",
    "U": "ú",
    "V": "vé",
    "W": "dvojité vé",
    "X": "iks",
    "Y": "ypsilon",
    "Z": "zet",
}

# Written form → spoken Czech. Longer keys first via sort at apply time.
ABBREVIATIONS = {
    "USA": "ú es á",
    "US": "ú es",
    "UK": "u ká",
    "EU": "é ú",
    "OSN": "O es en",
    "UN": "O es en",
    "NATO": "Náto",
    "FBI": "ef bé í",
    "CIA": "sí á",
    "CEO": "generální ředitel",
    "CFO": "finanční ředitel",
    "AI": "umělá inteligence",
    "HDP": "há dé pé",
    "GDP": "há dé pé",
    "MMF": "em em ef",
    "IMF": "em em ef",
    "WHO": "vé há ó",
    "BBC": "bé bé cé",
    "CNN": "cé en en",
    "NBC": "en bé cé",
    "ETF": "é té ef",
    "OECD": "ó é cé dé",
    "WTO": "vé té ó",
    "ECB": "é cé bé",
    "FED": "Fed",
    "IPO": "í pé ó",
    "PDF": "pé dé ef",
    "URL": "ú er el",
    "USB": "ú es bé",
    "TV": "té vé",
    "PC": "pé cé",
    "OK": "okej",
}

_DECADE = {
    10: "desátých",
    20: "dvacátých",
    30: "třicátých",
    40: "čtyřicátých",
    50: "padesátých",
    60: "šedesátých",
    70: "sedmdesátých",
    80: "osmdesátých",
    90: "devadesátých",
}

_NUMBER_RE = re.compile(
    r"(?P<decade>\b(?P<dnum>\d{1,2})\.\s*(?P<dlet>letech|leta|let)\b)"
    r"|(?P<year>\b(?:19|20)\d{2}\b)"
    r"|(?P<mult>\b(?P<mnum>\d+(?:[.,]\d+)?)\s*(?:krát|x|×))"
    r"|(?P<pct>\b(?P<pnum>\d+(?:[.,]\d+)?)\s*%)"
    r"|(?P<dec>\b\d+[.,]\d+\b)"
    r"|(?P<int>\b\d{1,6}\b)",
    re.IGNORECASE,
)

_ABBREV_RE: re.Pattern[str] | None = None
_ALLCAPS_RE = re.compile(r"\b[A-Z]{2,6}\b")


def _num2words_cs(n: int, ordinal: bool = False) -> str:
    import num2words

    for lang in ("cs", "cz"):
        try:
            return num2words.num2words(n, lang=lang, ordinal=ordinal)
        except NotImplementedError:
            continue
        except Exception:
            continue
    return str(n)


def _parse_decimal(raw: str) -> tuple[int, str | None]:
    text = raw.replace(" ", "").replace(",", ".")
    if "." in text:
        whole, frac = text.split(".", 1)
        frac = frac.rstrip("0") or "0"
        return int(whole or "0"), frac
    return int(text or "0"), None


def cardinal(raw: str) -> str:
    whole, frac = _parse_decimal(raw)
    spoken = _num2words_cs(whole)
    if frac is None:
        return spoken
    if len(frac) == 1:
        return f"{spoken} celá {_num2words_cs(int(frac))}"
    return f"{spoken} celá {_num2words_cs(int(frac))}"


def year_words(year: int) -> str:
    if 1100 <= year <= 1999 and year % 100 != 0:
        century = year // 100
        rest = year % 100
        return f"{_num2words_cs(century)} set {_num2words_cs(rest)}"
    return _num2words_cs(year)


def decade_words(n: int, let: str) -> str:
    stem = _DECADE.get(n)
    if stem is None:
        ordinal = _num2words_cs(n, ordinal=True)
        stem = re.sub(r"[ýí]$", "ých", ordinal)
    if let.startswith("letech"):
        return f"{stem} letech"
    return f"{stem} let"


def _replace_number(m: re.Match[str]) -> str:
    if m.group("decade"):
        return decade_words(int(m.group("dnum")), m.group("dlet").lower())
    if m.group("year"):
        return year_words(int(m.group("year")))
    if m.group("mult"):
        return f"{cardinal(m.group('mnum'))} krát"
    if m.group("pct"):
        return f"{cardinal(m.group('pnum'))} procent"
    if m.group("dec"):
        return cardinal(m.group("dec"))
    return cardinal(m.group("int"))


def expand_numbers(text: str) -> str:
    return _NUMBER_RE.sub(_replace_number, text)


def _abbrev_pattern() -> re.Pattern[str]:
    global _ABBREV_RE
    if _ABBREV_RE is None:
        keys = sorted(ABBREVIATIONS, key=len, reverse=True)
        _ABBREV_RE = re.compile(r"\b(" + "|".join(re.escape(k) for k in keys) + r")\b")
    return _ABBREV_RE


def spell_letters(token: str) -> str:
    parts = [_LETTER.get(ch.upper(), ch) for ch in token if ch.isalpha()]
    return " ".join(parts) if parts else token


def expand_abbreviations(text: str) -> str:
    def known(m: re.Match[str]) -> str:
        return ABBREVIATIONS[m.group(1).upper()]

    text = _abbrev_pattern().sub(known, text)

    def unknown(m: re.Match[str]) -> str:
        token = m.group(0)
        if token in ABBREVIATIONS:
            return ABBREVIATIONS[token]
        return spell_letters(token)

    return _ALLCAPS_RE.sub(unknown, text)


def load_name_pronunciations(glossary: dict | Path | None) -> list[tuple[str, str]]:
    if glossary is None:
        return []
    if isinstance(glossary, Path):
        if not glossary.is_file():
            return []
        glossary = json.loads(glossary.read_text(encoding="utf-8"))
    terms = glossary.get("terms") if isinstance(glossary, dict) else None
    if not isinstance(terms, list):
        return []
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for item in terms:
        if not isinstance(item, dict):
            continue
        spoken = (item.get("tts") or "").strip()
        written = (item.get("cs") or item.get("en") or "").strip()
        if not spoken or not written or spoken == written:
            continue
        key = written.casefold()
        if key in seen:
            continue
        seen.add(key)
        pairs.append((written, spoken))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def expand_names(text: str, names: list[tuple[str, str]]) -> str:
    for written, spoken in names:
        text = re.sub(r"\b" + re.escape(written) + r"\b", spoken, text, flags=re.IGNORECASE)
    return text


def expand_for_tts(text: str, glossary: dict | Path | None = None) -> str:
    """Spoken Czech for TTS. Display subtitles should keep the unexpanded form."""
    text = (text or "").strip()
    if not text:
        return text
    names = load_name_pronunciations(glossary)
    text = expand_names(text, names)
    text = expand_abbreviations(text)
    text = expand_numbers(text)
    return re.sub(r"\s+", " ", text).strip()
