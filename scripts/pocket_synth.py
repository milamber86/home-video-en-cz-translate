#!/usr/bin/env python3
"""Batch Czech synthesis driver for Kyutai Pocket TTS.

Runs inside .venv-pocket (pip install pocket-tts soundfile). Reads a JSON job
list and writes one WAV per job:

    [{"text": "...", "out": "0001.raw.wav", "speaker": "voice_ref.wav"}, ...]

`speaker` is optional; when omitted the default speaker passed via --speaker
(or the model's built-in voice) is used. Voice states are cached in-process,
and speakers shared by several jobs are exported to `<wav>.voice.safetensors`
next to the WAV so later runs skip the slow audio-prompt encoding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pocket_tts import TTSModel, export_model_state
from pocket_tts.default_parameters import get_default_voice_for_language

from text_split import split_cs_chunks, split_sentences

POCKET_CHUNK_LIMIT = 120

DEFAULT_CONFIG = (
    "hf://vvolhejn/pocket-tts-czech/czech.yaml"
    "@7c1fbd0acba765617749dd17f3dbddc2be791cc7"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=DEFAULT_CONFIG, help="Pocket TTS model config")
    p.add_argument("--jobs", help="JSON file with the synthesis jobs")
    p.add_argument("--text", help="Single-line mode: text to synthesize")
    p.add_argument("--out", help="Single-line output WAV")
    p.add_argument("--speaker", help="Default reference WAV or .voice.safetensors")
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--sampler-decode-steps", type=int, default=1)
    p.add_argument("--prefetch", action="store_true", help="Load model weights and exit")
    p.add_argument("--prefetch-marker", default="")
    return p.parse_args()


def load_jobs(args: argparse.Namespace) -> list[dict]:
    if args.jobs:
        jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
        if not isinstance(jobs, list) or not jobs:
            raise SystemExit("--jobs must be a non-empty JSON list")
        return jobs
    if args.text and args.out:
        return [{"text": args.text, "out": args.out}]
    raise SystemExit("Provide --text and --out, or --jobs")


def voice_state_path(speaker: Path) -> Path:
    return speaker.with_suffix(".voice.safetensors")


def is_remote(speaker: str) -> bool:
    return speaker.startswith(("http://", "https://", "hf://"))


def speaker_state(
    model: TTSModel, speaker: str, shared: set[str], states: dict[str, dict]
) -> dict:
    if speaker in states:
        return states[speaker]
    if is_remote(speaker):
        states[speaker] = model.get_state_for_audio_prompt(speaker)
        return states[speaker]
    path = Path(speaker)
    if path.suffix == ".safetensors":
        states[speaker] = model.get_state_for_audio_prompt(str(path))
        return states[speaker]
    exported = path.with_suffix(".voice.safetensors")
    if exported.is_file():
        states[speaker] = model.get_state_for_audio_prompt(str(exported))
        return states[speaker]
    state = model.get_state_for_audio_prompt(str(path))
    if path.name in shared:
        export_model_state(state, str(exported))
    states[speaker] = state
    return states[speaker]


def main() -> int:
    args = parse_args()
    model = TTSModel.load_model(
        config=args.config,
        temp=args.temperature,
        sampler_decode_steps=args.sampler_decode_steps,
    )
    model.to("cpu")
    if args.prefetch:
        if args.prefetch_marker:
            marker = Path(args.prefetch_marker)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("ok\n", encoding="utf-8")
        print("Pocket TTS model ready", file=sys.stderr, flush=True)
        return 0

    jobs = load_jobs(args)
    if args.speaker:
        jobs = [{"speaker": args.speaker, **job} for job in jobs]
    shared = {
        str(job["speaker"])
        for job in jobs
        if job.get("speaker") and sum(1 for j in jobs if j.get("speaker") == job["speaker"]) > 1
    }
    states: dict[str, dict] = {}
    sr = int(model.sample_rate)
    for i, job in enumerate(jobs, start=1):
        text = str(job.get("text") or "").strip()
        out = Path(str(job.get("out") or ""))
        if not text or not str(out):
            raise SystemExit(f"Job {i} needs text and out")
        speaker = str(job["speaker"]) if job.get("speaker") else None
        if speaker is None:
            default_voice = get_default_voice_for_language(None, args.config)
            speaker = str(default_voice)
        state = speaker_state(model, speaker, shared, states)
        print(
            f"pocket cue {i}/{len(jobs)} ({len(text)} chars)",
            file=sys.stderr,
            flush=True,
        )
        pieces: list = []
        sentences = split_sentences(text)
        for sentence in sentences:
            for chunk in split_cs_chunks(sentence, POCKET_CHUNK_LIMIT):
                audio = model.generate_audio(state, chunk)
                pieces.append(audio.detach().cpu().numpy())
        out.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np
        import soundfile as sf

        samples = np.concatenate([p if p.ndim == 1 else p[0] for p in pieces])
        sf.write(str(out), samples, sr, subtype="FLOAT")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
