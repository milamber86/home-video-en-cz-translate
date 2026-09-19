#!/usr/bin/env python3
"""Synthesize Czech speech with XTTS-v2 on CPU in the isolated .venv-xtts.

Single clip:
  xtts_synth.py --text '...' --speaker ref.wav --out out.wav

Batch (loads the model once):
  xtts_synth.py --speaker ref.wav --jobs jobs.json
  jobs.json is a list of {"text": "...", "out": "path.wav"}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("COQUI_TOS_AGREED", "1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", default="", help="One Czech sentence")
    p.add_argument("--speaker", required=True, help="Reference speaker WAV")
    p.add_argument("--out", default="", help="Output WAV for --text")
    p.add_argument("--jobs", default="", help="JSON list of {text, out}")
    p.add_argument("--language", default="cs")
    p.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    return p.parse_args()


def load_tts(model_name: str):
    from TTS.api import TTS

    print(f"Loading XTTS model={model_name} device=cpu", file=sys.stderr, flush=True)
    try:
        return TTS(model_name, gpu=False)
    except TypeError:
        return TTS(model_name)


def synth_one(tts, text: str, speaker: str, out: str, language: str) -> None:
    text = (text or "").strip()
    if not text:
        raise SystemExit("Empty text")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker,
        language=language,
        file_path=out,
    )


def main() -> int:
    args = parse_args()
    speaker = Path(args.speaker)
    if not speaker.is_file():
        raise SystemExit(f"Speaker WAV missing: {speaker}")

    jobs: list[dict]
    if args.jobs:
        jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
        if not isinstance(jobs, list) or not jobs:
            raise SystemExit("--jobs must be a non-empty JSON list")
    elif args.text and args.out:
        jobs = [{"text": args.text, "out": args.out}]
    else:
        raise SystemExit("Provide --text and --out, or --jobs")

    tts = load_tts(args.model)
    for i, job in enumerate(jobs, start=1):
        text = str(job.get("text") or "")
        out = str(job.get("out") or "")
        if not out:
            raise SystemExit(f"Job {i} missing out path")
        print(f"XTTS {i}/{len(jobs)} -> {out}", file=sys.stderr, flush=True)
        synth_one(tts, text, str(speaker), out, args.language)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
