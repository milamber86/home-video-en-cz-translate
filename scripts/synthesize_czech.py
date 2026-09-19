#!/usr/bin/env python3
"""Generate timed Czech speech from an SRT with F5-TTS on Apple Silicon MPS."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import device as apple_device  # noqa: E402

apple_device.bootstrap_mps_fallback()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from srtutil import cue_seconds, load_srt  # noqa: E402

BASE_GRAPHEME_FIX = str.maketrans({"ů": "ú", "Ů": "Ú", "ď": "d", "Ď": "D"})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--srt", help="Czech SRT (required unless --smoke-test)")
    p.add_argument("--out", required=True, help="Output czech_vocals.wav")
    p.add_argument("--duration", type=float, help="Video duration seconds (required unless --smoke-test)")
    p.add_argument("--device", default="mps")
    p.add_argument("--voice-mode", choices=("clone", "bundled"), default="clone")
    p.add_argument("--vocals", help="Original vocals.wav for clone mode")
    p.add_argument("--ref-audio", help="Bundled or explicit reference WAV")
    p.add_argument("--ref-out", help="Where to write the extracted clone reference WAV")
    p.add_argument("--ref-text", help="Transcript of --ref-audio, or path to a .txt file")
    p.add_argument("--ckpt", default="", help="Fine-tuned F5 checkpoint")
    p.add_argument("--vocab", default="", help="Fine-tuned vocab.txt")
    p.add_argument("--f5-model", default="F5TTS_v1_Base")
    p.add_argument("--nfe-step", type=int, default=32)
    p.add_argument("--max-speed", type=float, default=1.35)
    p.add_argument("--segments-dir", help="Optional directory for per-cue WAVs")
    p.add_argument("--whisper-model", default="mlx-community/whisper-large-v3-mlx")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--smoke-text", default="Za svítání šel dům přes louku.")
    return p.parse_args()


def run_ffmpeg(argv: list[str]) -> None:
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed ({proc.returncode}): {' '.join(argv)}\n{proc.stderr}"
        )


def atempo_chain(ratio: float) -> str:
    filters: list[str] = []
    r = float(ratio)
    if r <= 0:
        raise ValueError(f"Invalid atempo ratio {ratio}")
    while r > 2.0:
        filters.append("atempo=2.0")
        r /= 2.0
    while r < 0.5:
        filters.append("atempo=0.5")
        r /= 0.5
    filters.append(f"atempo={r:.6f}")
    return ",".join(filters)


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    data, sr = sf.read(str(path), always_2d=True)
    mono = data.mean(axis=1).astype(np.float32)
    return mono, int(sr)


def pick_reference_clip(vocals: Path, dest: Path, target_sec: float = 10.0) -> Path:
    mono, sr = load_mono(vocals)
    window = int(target_sec * sr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if len(mono) <= window:
        sf.write(str(dest), mono, sr)
        return dest
    hop = max(int(0.25 * sr), 1)
    best_i, best_e = 0, -1.0
    for i in range(0, len(mono) - window + 1, hop):
        chunk = mono[i : i + window]
        energy = float(np.mean(chunk * chunk))
        if energy > best_e:
            best_e, best_i = energy, i
    sf.write(str(dest), mono[best_i : best_i + window], sr)
    return dest


def transcribe_ref(path: Path, model: str) -> str:
    import mlx_whisper

    apple_device.log(f"Transcribing voice reference with mlx-whisper")
    result = mlx_whisper.transcribe(
        str(path),
        path_or_hf_repo=model,
        word_timestamps=False,
        condition_on_previous_text=False,
    )
    text = (result.get("text") or "").strip()
    if not text:
        raise RuntimeError(f"Empty transcript for reference audio {path}")
    release_mlx()
    return text


def read_ref_text(value: str | None) -> str:
    if not value:
        return ""
    p = Path(value)
    if p.is_file():
        return p.read_text(encoding="utf-8").strip()
    return value.strip()


def normalize_czech(text: str, using_finetune: bool) -> str:
    text = (text or "").strip()
    if not using_finetune:
        text = text.translate(BASE_GRAPHEME_FIX)
    if text and text[-1] not in ".!?…":
        text += "."
    return text


class _DoneFuture:
    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value


class _SerialExecutor:
    """F5-TTS infer_batch_process uses ThreadPoolExecutor for text chunks.

    Two Metal command encoders on one MPS buffer abort the process:
    `A command encoder is already encoding to this command buffer`.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, fn, *args, **kwargs):
        return _DoneFuture(fn(*args, **kwargs))


def patch_f5_serial_mps(device: str) -> None:
    if device != "mps":
        return
    import f5_tts.infer.utils_infer as infer_utils

    infer_utils.ThreadPoolExecutor = _SerialExecutor
    apple_device.log("F5-TTS: serial MPS inference (avoid Metal encoder clash)")


def sync_mps(device: str) -> None:
    if device != "mps":
        return
    import torch

    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def release_mlx() -> None:
    try:
        import mlx.core as mx

        mx.metal.clear_cache()
    except Exception:
        pass


def usable_segment(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1000:
        return False
    try:
        return float(sf.info(str(path)).duration) > 0.05
    except Exception:
        return False


def load_f5(args: argparse.Namespace):
    from f5_tts.api import F5TTS

    device = apple_device.device_str(args.device)
    patch_f5_serial_mps(device)
    apple_device.log(f"Loading F5-TTS on device={device}")
    ckpt = args.ckpt if args.ckpt and Path(args.ckpt).is_file() else ""
    vocab = args.vocab if args.vocab and Path(args.vocab).is_file() else ""
    kwargs = {"model": args.f5_model, "device": device}
    if ckpt:
        kwargs["ckpt_file"] = ckpt
        apple_device.log(f"Using fine-tuned checkpoint {ckpt}")
    if vocab:
        kwargs["vocab_file"] = vocab
    return F5TTS(**kwargs), bool(ckpt and vocab), device


def fit_to_slot(
    src: Path, dest: Path, target_sec: float, max_speed: float, tmp_dir: Path
) -> np.ndarray:
    info = sf.info(str(src))
    gen_sec = float(info.duration)
    sr = int(info.samplerate)
    if gen_sec <= 0:
        raise RuntimeError(f"Generated silent/empty WAV: {src}")

    work = src
    stem = dest.stem
    if gen_sec > target_sec + 0.02:
        ratio = gen_sec / max(target_sec, 0.05)
        speed = min(ratio, max_speed)
        sped = tmp_dir / f"{stem}.sped.wav"
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(work),
                "-af",
                atempo_chain(speed),
                str(sped),
            ]
        )
        work = sped
        gen_sec = float(sf.info(str(work)).duration)

    if gen_sec > target_sec + 0.02:
        trimmed = tmp_dir / f"{stem}.trim.wav"
        fade = min(0.04, target_sec / 4)
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(work),
                "-af",
                f"atrim=0:{target_sec:.6f},afade=t=out:st={max(target_sec - fade, 0):.6f}:d={fade:.6f}",
                str(trimmed),
            ]
        )
        work = trimmed
        gen_sec = float(sf.info(str(work)).duration)

    if gen_sec < target_sec - 0.02:
        pad = target_sec - gen_sec
        padded = tmp_dir / f"{stem}.pad.wav"
        run_ffmpeg(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(work),
                "-af",
                f"apad=pad_dur={pad:.6f}",
                str(padded),
            ]
        )
        work = padded

    shutil.copyfile(work, dest)
    audio, out_sr = load_mono(dest)
    if out_sr != sr:
        apple_device.log(f"Warning: sample rate changed {sr} -> {out_sr}")
    return audio


def resample_mono(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio.astype(np.float32)
    import torch
    import torchaudio

    wav = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
    out = torchaudio.functional.resample(wav, src_sr, dst_sr)
    return out.squeeze(0).cpu().numpy().astype(np.float32)


def overlay(pieces: list[tuple[float, np.ndarray]], duration: float, sr: int) -> np.ndarray:
    n = max(int(math.ceil(duration * sr)), 1)
    canvas = np.zeros(n, dtype=np.float32)
    for start_s, samples in pieces:
        start = int(round(start_s * sr))
        if start >= n:
            continue
        end = min(start + len(samples), n)
        sl = end - start
        if sl <= 0:
            continue
        canvas[start:end] += samples[:sl]
    peak = float(np.max(np.abs(canvas))) if canvas.size else 0.0
    if peak > 0.99:
        canvas *= 0.99 / peak
    return canvas


def resolve_reference(args: argparse.Namespace, job_ref: Path) -> tuple[Path, str]:
    if args.voice_mode == "bundled":
        if not args.ref_audio or not Path(args.ref_audio).is_file():
            raise SystemExit(
                "voice_mode=bundled requires files/voices/czech_default_ref.wav "
                "(see files/voices/README.md)"
            )
        text = read_ref_text(args.ref_text)
        if not text:
            raise SystemExit("Bundled voice requires a non-empty reference transcript")
        return Path(args.ref_audio), text

    if args.ref_audio and Path(args.ref_audio).is_file():
        text = read_ref_text(args.ref_text) or transcribe_ref(
            Path(args.ref_audio), args.whisper_model
        )
        return Path(args.ref_audio), text

    if not args.vocals or not Path(args.vocals).is_file():
        raise SystemExit("voice_mode=clone requires --vocals pointing at vocals.wav")
    ref = pick_reference_clip(Path(args.vocals), job_ref)
    text = read_ref_text(args.ref_text) or transcribe_ref(ref, args.whisper_model)
    return ref, text


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def main() -> int:
    args = parse_args()
    if not args.smoke_test and (not args.srt or args.duration is None):
        raise SystemExit("--srt and --duration are required unless --smoke-test")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    segments_dir = Path(args.segments_dir) if args.segments_dir else out_path.parent / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    default_ref = Path(args.ref_out) if args.ref_out else out_path.parent / "voice_ref.wav"
    ref_audio, ref_text = resolve_reference(args, default_ref)
    tts, using_finetune, tts_device = load_f5(args)
    apple_device.log(f"Reference audio={ref_audio} text={ref_text[:80]!r}")

    if args.smoke_test:
        text = normalize_czech(args.smoke_text, using_finetune)
        wav, sr, _ = tts.infer(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=text,
            nfe_step=args.nfe_step,
            speed=1.0,
        )
        sync_mps(tts_device)
        if wav is None:
            raise SystemExit("F5-TTS smoke test returned no audio")
        audio = np.asarray(wav, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=-1)
        dur = len(audio) / float(sr)
        energy = rms(audio)
        sf.write(str(out_path), audio, int(sr))
        apple_device.log(f"Smoke test duration={dur:.2f}s rms={energy:.5f}")
        if dur < 0.3 or energy < 1e-4:
            raise SystemExit("F5-TTS smoke test produced silent or too-short audio")
        return 0

    cues = load_srt(args.srt)
    pieces: list[tuple[float, np.ndarray]] = []
    gen_sr = 24000

    with tempfile.TemporaryDirectory(prefix="f5cue_") as tmp:
        tmp_dir = Path(tmp)
        for i, cue in enumerate(cues, start=1):
            text = normalize_czech(cue.content, using_finetune)
            if not text:
                continue
            fitted = segments_dir / f"{i:04d}.wav"
            start = cue.start.total_seconds()
            if usable_segment(fitted):
                samples, seg_sr = load_mono(fitted)
                gen_sr = int(seg_sr)
                apple_device.log(f"TTS cue {i}/{len(cues)} resume {fitted.name}")
                pieces.append((start, samples))
                continue
            raw = tmp_dir / f"{i:04d}_raw.wav"
            apple_device.log(f"TTS cue {i}/{len(cues)} ({cue_seconds(cue):.2f}s): {text[:80]}")
            wav, sr, _ = tts.infer(
                ref_file=str(ref_audio),
                ref_text=ref_text,
                gen_text=text,
                nfe_step=args.nfe_step,
                speed=1.0,
                file_wave=str(raw),
            )
            sync_mps(tts_device)
            if wav is None:
                raise RuntimeError(f"F5-TTS returned no audio for cue {i}")
            gen_sr = int(sr)
            slot = max(cue_seconds(cue), 0.08)
            samples = fit_to_slot(raw, fitted, slot, args.max_speed, tmp_dir)
            pieces.append((start, samples))

    if not pieces:
        raise SystemExit("No Czech speech segments were generated")

    duration = max(args.duration, pieces[-1][0] + 0.1)
    canvas = overlay(pieces, duration, gen_sr)
    canvas_48k = resample_mono(canvas, gen_sr, args.sample_rate)
    stereo = np.stack([canvas_48k, canvas_48k], axis=1)
    sf.write(str(out_path), stereo, args.sample_rate, subtype="PCM_16")
    apple_device.log(f"Wrote {out_path} ({duration:.2f}s @ {args.sample_rate} Hz stereo)")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
