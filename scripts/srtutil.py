"""SRT load/save and caption cleanup."""

from __future__ import annotations

import re
from datetime import timedelta
from pathlib import Path

import srt

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def clean_caption(text: str) -> str:
    text = _TAG_RE.sub("", text or "")
    text = text.replace("\xa0", " ")
    text = _WS_RE.sub(" ", text).strip()
    return text


def load_srt(path: str | Path) -> list[srt.Subtitle]:
    raw = Path(path).read_text(encoding="utf-8")
    cues = list(srt.parse(raw))
    cleaned: list[srt.Subtitle] = []
    for i, cue in enumerate(cues, start=1):
        content = clean_caption(cue.content.replace("\n", " "))
        if not content:
            continue
        cleaned.append(
            srt.Subtitle(
                index=i,
                start=cue.start,
                end=cue.end,
                content=content,
            )
        )
    if not cleaned:
        raise ValueError(f"No usable cues in {path}")
    return cleaned


def save_srt(path: str | Path, cues: list[srt.Subtitle]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    numbered = []
    for i, cue in enumerate(cues, start=1):
        numbered.append(
            srt.Subtitle(index=i, start=cue.start, end=cue.end, content=cue.content)
        )
    out.write_text(srt.compose(numbered), encoding="utf-8")


def cue_seconds(cue: srt.Subtitle) -> float:
    dur = (cue.end - cue.start).total_seconds()
    return max(dur, 0.05)


def segments_to_cues(segments: list[dict]) -> list[srt.Subtitle]:
    cues: list[srt.Subtitle] = []
    for i, seg in enumerate(segments, start=1):
        text = clean_caption(str(seg.get("text") or ""))
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or start)
        if end <= start:
            end = start + 0.05
        cues.append(
            srt.Subtitle(
                index=i,
                start=timedelta(seconds=start),
                end=timedelta(seconds=end),
                content=text,
            )
        )
    if not cues:
        raise ValueError("Whisper returned no subtitle cues")
    return cues
