#!/usr/bin/env python3
"""Synthesize Czech speech with Coqui TTS in the isolated .venv-xtts.

XTTS clone (needs a speaker WAV):
  xtts_synth.py --model tts_models/multilingual/multi-dataset/xtts_v2 \\
    --speaker ref.wav --text '...' --out out.wav

Czech VITS stock voice (no speaker):
  xtts_synth.py --model tts_models/cs/cv/vits --text '...' --out out.wav

Batch (loads the model once):
  xtts_synth.py --model ... --jobs jobs.json
  jobs.json is a list of {"text": "...", "out": "path.wav"}
Existing out files are skipped so a crash can be resumed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("COQUI_TOS_AGREED", "1")

# XTTS tokenizer warns/truncates Czech above this many characters.
XTTS_CS_CHAR_LIMIT = 180
VITS_CHAR_LIMIT = 400


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", default="", help="One Czech sentence")
    p.add_argument("--speaker", default="", help="Reference speaker WAV (XTTS only)")
    p.add_argument("--out", default="", help="Output WAV for --text")
    p.add_argument("--jobs", default="", help="JSON list of {text, out}")
    p.add_argument("--language", default="cs")
    p.add_argument(
        "--model",
        default="tts_models/multilingual/multi-dataset/xtts_v2",
    )
    return p.parse_args()


def is_xtts_model(model_name: str) -> bool:
    return "xtts" in (model_name or "").lower()


def patch_czech_ordinals() -> None:
    """num2words has no Czech ordinals; XTTS treats '1976.' as an ordinal."""
    import num2words
    import TTS.tts.layers.xtts.tokenizer as tok

    def _expand_ordinal(m, lang="en"):
        n = int(m.group(1))
        try:
            return num2words.num2words(n, ordinal=True, lang=lang)
        except NotImplementedError:
            return num2words.num2words(n, lang=lang)

    tok._expand_ordinal = _expand_ordinal


def split_cs_chunks(text: str, limit: int) -> list[str]:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return []
    if len(text) <= limit:
        return [text]
    parts = [p.strip() for p in re.split(r"(?<=[;:,.!?…])\s+", text) if p.strip()]
    chunks: list[str] = []
    buf = ""
    for part in parts:
        pieces = [part] if len(part) <= limit else _split_words(part, limit)
        for piece in pieces:
            if buf and len(buf) + 1 + len(piece) > limit:
                chunks.append(buf)
                buf = piece
            else:
                buf = f"{buf} {piece}".strip()
    if buf:
        chunks.append(buf)
    return chunks


def _split_words(text: str, limit: int) -> list[str]:
    words = text.split()
    out: list[str] = []
    buf = ""
    for word in words:
        if buf and len(buf) + 1 + len(word) > limit:
            out.append(buf)
            buf = word
        else:
            buf = f"{buf} {word}".strip()
    if buf:
        out.append(buf)
    return out


def load_tts(model_name: str):
    from TTS.api import TTS

    if is_xtts_model(model_name):
        patch_czech_ordinals()
    print(f"Loading Coqui model={model_name} device=cpu", file=sys.stderr, flush=True)
    try:
        return TTS(model_name, gpu=False)
    except TypeError:
        return TTS(model_name)


def _concat_wavs(paths: list[Path], dest: Path) -> None:
    import numpy as np
    import soundfile as sf

    pieces = []
    sr = None
    for path in paths:
        audio, this_sr = sf.read(str(path), always_2d=False)
        if sr is None:
            sr = this_sr
        elif this_sr != sr:
            raise RuntimeError(f"Sample-rate mismatch {this_sr} vs {sr}")
        pieces.append(np.asarray(audio, dtype=np.float32))
    if not pieces:
        raise RuntimeError("No Coqui chunks to concatenate")
    dest.parent.mkdir(parents=True, exist_ok=True)
    audio = np.concatenate(pieces)
    sf.write(str(dest), audio, int(sr), format="WAV")


def tts_to_file(tts, text: str, dest: Path, speaker: str, language: str, xtts: bool) -> None:
    kwargs = {
        "text": text,
        "file_path": str(dest),
        "split_sentences": not xtts,
    }
    if xtts:
        kwargs["speaker_wav"] = speaker
        kwargs["language"] = language
        kwargs["split_sentences"] = False
    tts.tts_to_file(**kwargs)


def synth_one(tts, text: str, speaker: str, out: str, language: str, xtts: bool) -> None:
    text = (text or "").strip()
    if not text:
        raise SystemExit("Empty text")
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    limit = XTTS_CS_CHAR_LIMIT if xtts else VITS_CHAR_LIMIT
    chunks = split_cs_chunks(text, limit)
    if not chunks:
        raise SystemExit("Empty text after split")
    # Keep a .wav suffix so SoundFile/Coqui can infer the container.
    partial = dest.with_name(dest.stem + ".partial.wav")
    if len(chunks) == 1:
        tts_to_file(tts, chunks[0], partial, speaker, language, xtts)
        partial.replace(dest)
        return
    with tempfile.TemporaryDirectory(prefix="coqui_chunks_") as tmp:
        tmp_dir = Path(tmp)
        parts: list[Path] = []
        for i, chunk in enumerate(chunks, start=1):
            part = tmp_dir / f"{i:02d}.wav"
            print(f"  chunk {i}/{len(chunks)} ({len(chunk)} chars)", file=sys.stderr, flush=True)
            tts_to_file(tts, chunk, part, speaker, language, xtts)
            parts.append(part)
        _concat_wavs(parts, partial)
    partial.replace(dest)


def usable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    try:
        import soundfile as sf

        return float(sf.info(str(path)).duration) > 0.05
    except Exception:
        return False


def main() -> int:
    args = parse_args()
    xtts = is_xtts_model(args.model)
    speaker = Path(args.speaker) if args.speaker else None
    if xtts:
        if speaker is None or not speaker.is_file():
            raise SystemExit("XTTS requires --speaker pointing at a reference WAV")
    speaker_arg = str(speaker) if speaker else ""

    jobs: list[dict]
    if args.jobs:
        jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
        if not isinstance(jobs, list) or not jobs:
            raise SystemExit("--jobs must be a non-empty JSON list")
    elif args.text and args.out:
        jobs = [{"text": args.text, "out": args.out}]
    else:
        raise SystemExit("Provide --text and --out, or --jobs")

    pending: list[tuple[int, str, str]] = []
    for i, job in enumerate(jobs, start=1):
        text = str(job.get("text") or "")
        out = str(job.get("out") or "")
        if not out:
            raise SystemExit(f"Job {i} missing out path")
        dest = Path(out)
        if usable(dest):
            print(f"Coqui {i}/{len(jobs)} resume {dest}", file=sys.stderr, flush=True)
            continue
        pending.append((i, text, out))
    if not pending:
        return 0

    tts = load_tts(args.model)
    for i, text, out in pending:
        print(f"Coqui {i}/{len(jobs)} -> {out}", file=sys.stderr, flush=True)
        synth_one(tts, text, speaker_arg, out, args.language, xtts)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
