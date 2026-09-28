"""Sentence-aware text chunking shared by the TTS engines."""

from __future__ import annotations

import re

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")

DEFAULT_CHUNK_LIMIT = 180


def _split_words(text: str, limit: int) -> list[str]:
    out: list[str] = []
    buf = ""
    for word in text.split():
        if not buf:
            buf = word
        elif len(buf) + 1 + len(word) <= limit:
            buf = f"{buf} {word}"
        else:
            out.append(buf)
            buf = word
    if buf:
        out.append(buf)
    return out


def split_sentences(text: str) -> list[str]:
    """Split into individual sentences; Pocket TTS emits EOS after one."""
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    parts = [p.strip() for p in _SENT_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def split_cs_chunks(text: str, limit: int = DEFAULT_CHUNK_LIMIT) -> list[str]:
    """Split Czech TTS text into chunks no longer than `limit` chars.

    Prefers sentence boundaries; falls back to word boundaries so a chunk
    never exceeds the limit (except a single over-long word).
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    parts = _SENT_SPLIT_RE.split(text)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        pieces = [part] if len(part) <= limit else _split_words(part, limit)
        for piece in pieces:
            if buf and len(buf) + 1 + len(piece) > limit:
                chunks.append(buf)
                buf = piece
            elif buf:
                buf = f"{buf} {piece}"
            else:
                buf = piece
    if buf:
        chunks.append(buf)
    return chunks
