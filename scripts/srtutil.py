"""SRT load/save and caption cleanup."""

from __future__ import annotations

import re
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path

import srt

_TAG_RE = re.compile(r"<[^>]+>")
_SFX_RE = re.compile(
    r"\[(?:music|applause|laughter|cheers|singing|inaudible|noise)\]",
    re.IGNORECASE,
)
_WS_RE = re.compile(r"\s+")
_MIN_CUE_SECONDS = 0.05
_MAX_PACK_SECONDS = 7.0
_MAX_PACK_WORDS = 22


def clean_caption(text: str) -> str:
    text = _TAG_RE.sub("", text or "")
    text = _SFX_RE.sub(" ", text)
    text = text.replace("\xa0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def _renumber(cues: list[srt.Subtitle]) -> list[srt.Subtitle]:
    return [
        srt.Subtitle(index=i, start=c.start, end=c.end, content=c.content)
        for i, c in enumerate(cues, start=1)
    ]


def _overlap_word_count(prev: str, nxt: str) -> int:
    pw, nw = prev.split(), nxt.split()
    if not pw or not nw:
        return 0
    max_n = min(len(pw), len(nw))
    for n in range(max_n, 2, -1):
        if pw[-n:] == nw[:n]:
            return n
    return 0


def _is_rolling_pair(prev: str, nxt: str) -> bool:
    if not prev or not nxt:
        return False
    if prev == nxt or prev in nxt or nxt in prev:
        return True
    if _overlap_word_count(prev, nxt) >= 3:
        return True
    return SequenceMatcher(None, prev, nxt).ratio() >= 0.4


def is_rolling_captions(cues: list[srt.Subtitle]) -> bool:
    """True for karaoke-style auto-captions that slide a few words at a time."""
    if len(cues) < 12:
        return False
    n = min(len(cues) - 1, 80)
    hits = sum(
        1
        for a, b in zip(cues[:n], cues[1 : n + 1])
        if _is_rolling_pair(a.content, b.content)
    )
    tiny = sum(1 for c in cues if cue_seconds(c) <= 0.2)
    return hits / n >= 0.30 or tiny / len(cues) >= 0.25


def incremental_text(prev: str, nxt: str) -> str:
    """New words added by a rolling YouTube caption relative to the previous cue."""
    if not nxt:
        return ""
    if not prev:
        return nxt
    if nxt == prev or prev.startswith(nxt) or nxt in prev:
        return ""
    if nxt.startswith(prev):
        return nxt[len(prev) :].strip()
    overlap = _overlap_word_count(prev, nxt)
    if overlap:
        return " ".join(nxt.split()[overlap:])
    return nxt


def unroll_rolling_cues(cues: list[srt.Subtitle]) -> list[srt.Subtitle]:
    """Turn sliding auto-captions into non-overlapping, sentence-sized cues."""
    if not cues:
        return []

    pieces: list[tuple] = []
    prev_text = ""
    for cue in cues:
        piece = incremental_text(prev_text, cue.content)
        prev_text = cue.content
        if not piece:
            if pieces:
                start, _, text = pieces[-1]
                pieces[-1] = (start, cue.end, text)
            continue
        pieces.append((cue.start, cue.end, piece))

    packed: list[srt.Subtitle] = []
    buf_start = None
    buf_end = None
    buf_words: list[str] = []

    def flush() -> None:
        nonlocal buf_start, buf_end, buf_words
        if buf_words and buf_start is not None and buf_end is not None:
            packed.append(
                srt.Subtitle(
                    index=len(packed) + 1,
                    start=buf_start,
                    end=buf_end,
                    content=" ".join(buf_words),
                )
            )
        buf_start = None
        buf_end = None
        buf_words = []

    for start, end, piece in pieces:
        words = piece.split()
        if not words:
            continue
        if buf_start is None:
            buf_start, buf_end, buf_words = start, end, list(words)
        else:
            buf_end = end
            buf_words.extend(words)
        text = " ".join(buf_words)
        dur = (buf_end - buf_start).total_seconds()
        ended = text.endswith((".", "?", "!"))
        if ended and (dur >= 1.5 or len(buf_words) >= 6):
            flush()
        elif dur >= _MAX_PACK_SECONDS or len(buf_words) >= _MAX_PACK_WORDS:
            flush()
    flush()
    return packed


def load_srt(path: str | Path) -> list[srt.Subtitle]:
    raw = Path(path).read_text(encoding="utf-8")
    cues = list(srt.parse(raw))
    cleaned: list[srt.Subtitle] = []
    for cue in cues:
        content = clean_caption(cue.content.replace("\n", " "))
        if not content:
            continue
        dur = (cue.end - cue.start).total_seconds()
        if dur < _MIN_CUE_SECONDS:
            continue
        cleaned.append(
            srt.Subtitle(
                index=len(cleaned) + 1,
                start=cue.start,
                end=cue.end,
                content=content,
            )
        )
    if is_rolling_captions(cleaned):
        cleaned = unroll_rolling_cues(cleaned)
    if not cleaned:
        raise ValueError(f"No usable cues in {path}")
    return _renumber(cleaned)


def save_srt(path: str | Path, cues: list[srt.Subtitle]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(srt.compose(_renumber(cues)), encoding="utf-8")


def cue_seconds(cue: srt.Subtitle) -> float:
    dur = (cue.end - cue.start).total_seconds()
    return max(dur, 0.05)


def segments_to_cues(segments: list[dict]) -> list[srt.Subtitle]:
    cues: list[srt.Subtitle] = []
    for seg in segments:
        text = clean_caption(str(seg.get("text") or ""))
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if end <= start:
            end = start + 0.05
        cues.append(
            srt.Subtitle(
                index=len(cues) + 1,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=text,
            )
        )
    if not cues:
        raise ValueError("Whisper returned no subtitle cues")
    return _renumber(cues)
