#!/usr/bin/env python3
"""Generate timed Czech speech from an SRT (Piper, Coqui VITS, or Czech F5)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import device as apple_device  # noqa: E402

apple_device.bootstrap_mps_fallback()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from srtutil import cue_seconds, load_srt  # noqa: E402
from czech_tts_text import expand_for_tts  # noqa: E402

BASE_GRAPHEME_FIX = str.maketrans({"ů": "ú", "Ů": "Ú", "ď": "d", "Ď": "D"})
VITS_MODEL = "tts_models/cs/cv/vits"
XTTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--srt", help="Czech SRT (required unless --smoke-test)")
    p.add_argument("--out", required=True, help="Output czech_vocals.wav")
    p.add_argument("--duration", type=float, help="Video duration seconds (required unless --smoke-test)")
    p.add_argument("--device", default="mps")
    p.add_argument("--engine", default="auto", choices=("auto", "f5", "xtts", "piper", "vits"))
    p.add_argument("--voice-mode", choices=("clone", "bundled"), default="clone")
    p.add_argument("--voice-gender", choices=("male", "female"), default="male")
    p.add_argument("--vocals", help="Original vocals.wav for clone mode")
    p.add_argument("--ref-audio", help="Bundled or explicit reference WAV")
    p.add_argument("--ref-out", help="Where to write the extracted clone reference WAV")
    p.add_argument("--ref-text", help="Transcript of --ref-audio, or path to a .txt file")
    p.add_argument("--ckpt", default="", help="Fine-tuned F5 checkpoint")
    p.add_argument("--vocab", default="", help="Fine-tuned vocab.txt")
    p.add_argument("--f5-model", default="F5TTS_v1_Base")
    p.add_argument("--nfe-step", type=int, default=32)
    p.add_argument(
        "--base-speed",
        type=float,
        default=1.15,
        help="Minimum atempo applied to every cue so Czech can keep up",
    )
    p.add_argument(
        "--max-speed",
        type=float,
        default=1.25,
        help="Cap when a cue is still longer than its English slot",
    )
    p.add_argument("--segments-dir", help="Optional directory for per-cue WAVs")
    p.add_argument("--whisper-model", default="mlx-community/whisper-large-v3-mlx")
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--piper-model", default="", help="Piper ONNX voice (cs_CZ-jirka-medium.onnx)")
    p.add_argument("--xtts-python", default="", help="Interpreter for isolated .venv-xtts (VITS/XTTS)")
    p.add_argument("--vits-model", default=VITS_MODEL, help="Coqui Czech VITS model name")
    p.add_argument(
        "--glossary",
        default="",
        help="glossary.json with name pronunciations (default: next to --srt)",
    )
    p.add_argument("--fresh", action="store_true", help="Ignore existing per-cue WAVs")
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


def _resample_16k(mono: np.ndarray, sr: int) -> tuple[np.ndarray, int]:
    if sr == 16000:
        return mono.astype(np.float32), 16000
    import librosa

    return librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=16000), 16000


def pick_reference_clip(vocals: Path, dest: Path, target_sec: float = 8.0) -> Path:
    """Loudest *voiced* window so the clone ref is speech, not a music sting."""
    mono, sr = load_mono(vocals)
    dest.parent.mkdir(parents=True, exist_ok=True)
    search = mono
    search_sr = sr
    max_search = int(90.0 * sr)
    if len(mono) > max_search:
        search = mono[:max_search]
    window = int(target_sec * search_sr)
    if len(search) <= window:
        sf.write(str(dest), search, search_sr)
        return dest
    skip = int(2.0 * search_sr)
    hop = max(int(0.5 * search_sr), 1)
    best_i, best_score = skip, -1.0
    y16, sr16 = _resample_16k(search, search_sr)
    import librosa

    f0, _, _ = librosa.pyin(y16, fmin=75, fmax=300, sr=sr16)
    for i in range(skip, len(search) - window + 1, hop):
        chunk = search[i : i + window]
        energy = float(np.mean(chunk * chunk))
        a = int(i * sr16 / search_sr / 512)
        b = int((i + window) * sr16 / search_sr / 512)
        voiced = f0[a:b] if b > a else f0[:1]
        voiced = voiced[np.isfinite(voiced)] if voiced.size else voiced
        frac = float(voiced.size / max(b - a, 1))
        score = energy * (0.25 + 0.75 * frac)
        if score > best_score:
            best_score, best_i = score, i
    sf.write(str(dest), search[best_i : best_i + window], search_sr)
    return dest


def fade_out(audio: np.ndarray, sr: int, sec: float = 0.035) -> np.ndarray:
    n = min(len(audio), max(int(sec * sr), 1))
    if n <= 1:
        return audio
    out = audio.copy()
    out[-n:] *= np.linspace(1.0, 0.0, n, dtype=np.float32)
    return out


def trim_tts_tail(audio: np.ndarray, sr: int) -> np.ndarray:
    """Cut XTTS trailing silence and the short echo burst it often appends."""
    if audio.size < int(0.12 * sr):
        return fade_out(audio, sr, 0.02)
    frame = max(int(0.02 * sr), 1)
    rms = np.array(
        [
            float(np.sqrt(np.mean(audio[i : i + frame] ** 2)))
            for i in range(0, len(audio) - frame + 1, frame)
        ],
        dtype=np.float32,
    )
    if rms.size == 0:
        return fade_out(audio, sr, 0.02)
    thr = max(float(np.max(rms)) * 0.08, 1e-4)
    speech = rms > thr
    regions: list[tuple[int, int]] = []
    start = None
    for i, hit in enumerate(speech):
        if hit and start is None:
            start = i
        elif not hit and start is not None:
            regions.append((start, i))
            start = None
    if start is not None:
        regions.append((start, len(speech)))
    if len(regions) >= 2:
        last_s, last_e = regions[-1]
        prev_s, prev_e = regions[-2]
        last_dur = (last_e - last_s) * frame / sr
        gap = (last_s - prev_e) * frame / sr
        if last_dur <= 0.55 and gap >= 0.12 and last_s / max(len(rms), 1) >= 0.55:
            audio = audio[: max(prev_e * frame, frame)]
            rms = rms[:prev_e]
            speech = speech[:prev_e]
    if speech.any():
        last = int(np.max(np.nonzero(speech)[0]))
        cut = min(len(audio), (last + 2) * frame)
        audio = audio[:cut]
    # Drop the last unstable XTTS samples.
    keep = max(int(0.08 * sr), len(audio) - int(0.045 * sr))
    audio = audio[:keep]
    return fade_out(audio.astype(np.float32), sr, 0.04)


def transcribe_ref(path: Path, model: str) -> str:
    import mlx_whisper

    apple_device.log("Transcribing voice reference with mlx-whisper")
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


def czech_f5_ready(args: argparse.Namespace) -> bool:
    return bool(
        args.ckpt
        and Path(args.ckpt).is_file()
        and args.vocab
        and Path(args.vocab).is_file()
    )


def xtts_cache_dir(model: str = XTTS_MODEL) -> Path:
    slug = (model or XTTS_MODEL).replace("/", "--")
    return Path.home() / "Library/Application Support/tts" / slug


def xtts_pretrained_ready(args: argparse.Namespace) -> bool:
    """True when .venv-xtts exists and Coqui's pretrained XTTS-v2 (incl. Czech) is cached."""
    if not _path_ready(args.xtts_python):
        return False
    return (xtts_cache_dir(XTTS_MODEL) / "model.pth").is_file()


def stock_engine(gender: str) -> str:
    return "vits" if gender == "female" else "piper"


def _path_ready(value: str | None) -> bool:
    return bool(value and Path(value).is_file())


def select_engine(args: argparse.Namespace) -> str:
    gender = (args.voice_gender or "male").strip().lower()
    if gender not in ("male", "female"):
        raise SystemExit(f"Unsupported --voice-gender {args.voice_gender}")
    requested = (args.engine or "auto").strip().lower()
    if requested == "auto":
        speaker_ok = _path_ready(args.vocals) or _path_ready(args.ref_audio)
        if xtts_pretrained_ready(args) and speaker_ok:
            apple_device.log("TTS auto: pretrained XTTS-v2 (Czech)")
            return "xtts"
        if czech_f5_ready(args):
            apple_device.log(f"TTS auto: Czech F5 checkpoint {args.ckpt}")
            return "f5"
        if args.voice_mode == "clone" and not xtts_pretrained_ready(args):
            apple_device.log(
                "voice_mode=clone needs pretrained XTTS-v2 "
                "(run setup, or tts_engine=f5); falling back to a stock "
                f"{gender} voice"
            )
        return stock_engine(gender)
    if requested == "f5" and not czech_f5_ready(args):
        raise SystemExit(
            "engine=f5 requires a Czech fine-tune (--ckpt and --vocab). "
            "Official F5TTS_v1_Base is ZH+EN and is not used for Czech."
        )
    return requested


def resolve_voice_mode(args: argparse.Namespace, engine: str) -> str:
    """Use vocals.wav for F5 when the bundled reference WAV is absent."""
    mode = (args.voice_mode or "bundled").strip().lower()
    if engine != "f5":
        return mode
    if mode == "clone":
        return mode
    if _path_ready(args.ref_audio):
        return "bundled"
    if _path_ready(args.vocals):
        apple_device.log(
            "voice_mode=bundled has no reference WAV; cloning from vocals.wav"
        )
        return "clone"
    return mode


def voice_cache_id(engine: str) -> str:
    return {
        "piper": "piper_jirka",
        "vits": "vits_cv",
        "f5": "f5_clone",
        "xtts": "xtts_m2",
    }.get(engine, engine)


def spoken_cache_tag(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:10]


def normalize_czech(text: str, engine: str, using_finetune: bool) -> str:
    text = (text or "").strip()
    if engine == "f5" and not using_finetune:
        text = text.translate(BASE_GRAPHEME_FIX)
    if text and text[-1] not in ".!?…":
        text += "."
    return text


def spoken_czech(text: str, engine: str, using_finetune: bool, glossary_path: Path | None) -> str:
    text = normalize_czech(text, engine, using_finetune)
    return expand_for_tts(text, glossary_path)


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
    kwargs = {"model": args.f5_model, "device": device, "ckpt_file": args.ckpt}
    if args.vocab:
        kwargs["vocab_file"] = args.vocab
    apple_device.log(f"Using Czech F5 checkpoint {args.ckpt}")
    return F5TTS(**kwargs), device


def piper_to_wav(text: str, model: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from piper import PiperVoice
    except ImportError:
        piper_bin = shutil.which("piper")
        if not piper_bin:
            raise SystemExit("piper-tts is not installed and no piper CLI is on PATH")
        proc = subprocess.run(
            [piper_bin, "--model", str(model), "--output_file", str(dest)],
            input=text,
            text=True,
            capture_output=True,
        )
        if proc.returncode != 0 or not dest.is_file():
            raise RuntimeError(f"piper CLI failed: {proc.stderr}")
        return

    try:
        voice = PiperVoice.load(str(model), use_cuda=False)
    except TypeError:
        voice = PiperVoice.load(str(model))
    with wave.open(str(dest), "wb") as wf:
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, wf)
            return
        chunks = list(voice.synthesize(text))
        if not chunks:
            raise RuntimeError("Piper returned no audio")
        first = chunks[0]
        sample_rate = getattr(first, "sample_rate", 22050)
        sample_width = getattr(first, "sample_width", 2)
        sample_channels = getattr(first, "sample_channels", 1)
        wf.setnchannels(sample_channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        for chunk in chunks:
            audio_bytes = getattr(chunk, "audio_int16_bytes", None)
            if audio_bytes is None:
                audio_int16 = getattr(chunk, "audio_int16", None)
                if audio_int16 is None:
                    raise RuntimeError("Unexpected Piper chunk type")
                audio_bytes = np.asarray(audio_int16, dtype=np.int16).tobytes()
            wf.writeframes(audio_bytes)


def coqui_batch(
    coqui_python: Path,
    jobs: list[dict],
    model: str,
    speaker: Path | None = None,
) -> None:
    if not coqui_python.is_file():
        raise SystemExit(
            f"Coqui interpreter missing: {coqui_python}. "
            "Re-run the setup role to create .venv-xtts."
        )
    if not jobs:
        return
    with tempfile.NamedTemporaryFile(
        prefix="coqui_jobs_", suffix=".json", delete=False, mode="w", encoding="utf-8"
    ) as fh:
        json.dump(jobs, fh, ensure_ascii=False)
        jobs_path = Path(fh.name)
    try:
        cmd = [
            str(coqui_python),
            str(SCRIPTS_DIR / "xtts_synth.py"),
            "--model",
            model,
            "--language",
            "cs",
            "--jobs",
            str(jobs_path),
        ]
        if speaker is not None:
            cmd.extend(["--speaker", str(speaker)])
        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(f"xtts_synth.py failed ({proc.returncode})")
    finally:
        jobs_path.unlink(missing_ok=True)


def fit_to_slot(
    src: Path,
    dest: Path,
    target_sec: float,
    max_speed: float,
    tmp_dir: Path,
    base_speed: float = 1.0,
) -> np.ndarray:
    info = sf.info(str(src))
    gen_sec = float(info.duration)
    sr = int(info.samplerate)
    if gen_sec <= 0:
        raise RuntimeError(f"Generated silent/empty WAV: {src}")

    work = src
    stem = dest.stem
    base = max(1.0, float(base_speed))
    cap = max(base, float(max_speed))
    needed = gen_sec / max(target_sec, 0.05)
    # Every cue gets at least base_speed so overflow stretch is a small bump.
    speed = min(max(base, needed), cap) if needed > base + 0.02 else base
    if speed > 1.001:
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
        apple_device.log(
            f"TTS slot overflow {gen_sec:.2f}s > {target_sec:.2f}s after "
            f"{speed:.2f}x (cap {cap:.2f}x); keeping the rest (may overlap)"
        )

    shutil.copyfile(work, dest)
    audio, out_sr = load_mono(dest)
    if out_sr != sr:
        apple_device.log(f"Warning: sample rate changed {sr} -> {out_sr}")
    return audio


def fit_cue_audio(
    src: Path,
    dest: Path,
    target_sec: float,
    max_speed: float,
    tmp_dir: Path,
    base_speed: float = 1.0,
) -> np.ndarray:
    audio, sr = load_mono(src)
    audio = trim_tts_tail(audio, sr)
    cleaned = tmp_dir / f"{dest.stem}.trim.wav"
    sf.write(str(cleaned), audio, sr)
    return fit_to_slot(
        cleaned, dest, target_sec, max_speed, tmp_dir, base_speed=base_speed
    )


def fit_cache_tag(cache_id: str, base_speed: float, max_speed: float) -> str:
    return (
        f"{cache_id}_b{int(round(base_speed * 100)):03d}"
        f"m{int(round(max_speed * 100)):03d}"
    )


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


def _cue_window(
    mono: np.ndarray, sr: int, start: float, end: float, min_sec: float = 3.0
) -> np.ndarray:
    start = max(0.0, float(start))
    end = max(start + 0.05, float(end))
    if end - start < min_sec:
        extra = min_sec - (end - start)
        start = max(0.0, start - extra * 0.3)
        end = start + min(min_sec, len(mono) / sr)
    a = int(start * sr)
    b = min(len(mono), max(a + 1, int(end * sr)))
    return mono[a:b]


def _clip_f0(clip: np.ndarray, sr: int) -> float:
    if clip.size < int(0.2 * sr):
        return float("nan")
    import librosa

    y16, sr16 = _resample_16k(clip, sr)
    f0 = librosa.yin(y16, fmin=80, fmax=280, sr=sr16)
    if f0.size < 8:
        return float("nan")
    return float(np.median(f0))


def _is_voiced_clip(clip: np.ndarray, sr: int) -> bool:
    if clip.size < int(0.4 * sr):
        return False
    return float(np.sqrt(np.mean(np.square(clip)))) > 8e-4


def _write_ref_from_clips(
    clips: list[np.ndarray], sr: int, dest: Path, target_sec: float = 8.0
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not clips:
        raise RuntimeError(f"No audio to build speaker ref {dest}")
    need = int(target_sec * sr)
    chunks: list[np.ndarray] = []
    total = 0
    for clip in sorted(clips, key=len, reverse=True):
        if total >= need:
            break
        take = clip[: max(0, need - total)]
        if take.size:
            chunks.append(take)
            total += len(take)
    sf.write(str(dest), np.concatenate(chunks) if chunks else clips[0], sr)
    return dest


_PASSIVE_SAID = {
    "is",
    "was",
    "were",
    "been",
    "be",
    "are",
    "am",
    "it's",
    "its",
}
_QUOTE_START_RE = re.compile(
    r"^(you know[, ]+)?(i |i've |i’m |i'm |he said|she said|we )",
    re.IGNORECASE,
)


def _has_attr(text: str) -> bool:
    """True when a line attributes speech to someone else."""
    if not text:
        return False
    if re.search(r"\b(?:he|she)\s+said\b", text, re.IGNORECASE):
        return True
    if re.search(
        r"(?:televangelist|pastor|preacher)s?\s+\S.{0,50}?\b(?:says|said|told)\b",
        text,
        re.IGNORECASE,
    ):
        return True
    for match in re.finditer(
        r"\b([A-Za-z][\w'.-]+)(?:\s+([A-Za-z][\w'.-]+))?\s+(says|said|told)\b",
        text,
    ):
        if match.group(1).lower() in _PASSIVE_SAID:
            continue
        if (match.group(2) or "").lower() in _PASSIVE_SAID:
            continue
        return True
    return False


def _is_guest_f0(f0: float, narrator_f0: float) -> bool | None:
    if not np.isfinite(f0) or not np.isfinite(narrator_f0):
        return None
    if narrator_f0 >= 155:
        return f0 <= 142
    if narrator_f0 <= 145:
        return f0 >= 175
    return abs(f0 - narrator_f0) >= 38


def classify_cue_speakers(vocals: Path, cues: list) -> list[dict]:
    """Label each cue narrator/guest from that line's vocals (no neighbor bleed)."""
    mono, sr = load_mono(vocals)
    rows: list[dict] = []
    for i, cue in enumerate(cues, start=1):
        start = cue.start.total_seconds()
        end = cue.end.total_seconds()
        clip = _cue_window(mono, sr, start, end, min_sec=0.05)
        f0 = _clip_f0(clip, sr)
        rows.append(
            {
                "i": i,
                "start": start,
                "dur": max(end - start, 0.05),
                "f0": f0,
                "clip": clip,
                "text": cue.content or "",
            }
        )

    early = [
        r["f0"]
        for r in rows
        if r["start"] < 45.0 and r["dur"] >= 1.0 and np.isfinite(r["f0"])
    ]
    if not early:
        early = [r["f0"] for r in rows if r["dur"] >= 1.2 and np.isfinite(r["f0"])]
    narrator_f0 = float(np.median(early)) if early else float("nan")

    labels: list[int] = []
    prev = 0
    for r in rows:
        guest = _is_guest_f0(r["f0"], narrator_f0)
        if r["dur"] < 1.2 and guest is None:
            labels.append(prev)
        else:
            labels.append(1 if guest else 0)
        prev = labels[-1]
    for i, r in enumerate(rows):
        if labels[i] == 1:
            continue
        prev = rows[i - 1]["text"] if i else ""
        nxt = rows[i + 1]["text"] if i + 1 < len(rows) else ""
        quoted = bool(_QUOTE_START_RE.match(r["text"]))
        if quoted and (_has_attr(prev) or _has_attr(nxt)):
            labels[i] = 1
    for i, r in enumerate(rows):
        if r["dur"] >= 1.2:
            continue
        guest = _is_guest_f0(r["f0"], narrator_f0)
        if guest is True:
            labels[i] = 1
        elif i:
            labels[i] = labels[i - 1]
    for i, r in enumerate(rows):
        if labels[i] == 1 or i == 0 or i + 1 >= len(rows):
            continue
        if labels[i - 1] != 1 or labels[i + 1] != 1:
            continue
        if (
            np.isfinite(r["f0"])
            and np.isfinite(narrator_f0)
            and narrator_f0 >= 155
            and r["f0"] >= 155
        ):
            continue
        labels[i] = 1

    out: list[dict] = []
    for r, lab in zip(rows, labels):
        item = dict(r)
        item["label"] = lab
        item["narrator_f0"] = narrator_f0
        out.append(item)
    return out


def speaker_label_cues(cs_cues: list, srt_path: Path) -> list:
    """Use English cue text for quote/attribution; Czech SRT has no 'he said'."""
    for name in ("en.srt", "whisper.en.srt"):
        alt = srt_path.with_name(name)
        if not alt.is_file():
            continue
        try:
            en_cues = load_srt(alt, sentences=False)
        except ValueError:
            continue
        if len(en_cues) != len(cs_cues):
            continue
        if any(a.start != b.start or a.end != b.end for a, b in zip(en_cues, cs_cues)):
            continue
        apple_device.log(f"TTS speaker labels from {alt.name}")
        return en_cues
    return cs_cues


def plan_clone_speakers(
    vocals: Path,
    cues: list,
    narrator_ref: Path,
    dest_dir: Path,
) -> list[Path]:
    """Map each cue to narrator vs guest using that line's vocals only.

    Windows are not expanded into neighboring cues (that mixed voices).
    Narrator lines share one clean ref. Guest lines use that line's clip
    when it is long enough, else a concatenated guest ref.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows = classify_cue_speakers(vocals, cues)
    labels = [r["label"] for r in rows]
    narrator_f0 = rows[0]["narrator_f0"] if rows else float("nan")
    sr = 48000
    if rows:
        _, sr = load_mono(vocals)

    guest_clips = [
        r["clip"]
        for r, lab in zip(rows, labels)
        if lab == 1 and r["dur"] >= 0.8 and _is_voiced_clip(r["clip"], sr)
    ]
    guest_ref = narrator_ref
    if guest_clips:
        guest_ref = _write_ref_from_clips(guest_clips, sr, dest_dir / "other.wav")
        apple_device.log(
            f"TTS speakers: narrator f0~{narrator_f0:.0f} Hz, "
            f"{sum(labels)} guest cues, other ref={guest_ref.name}"
        )
    else:
        apple_device.log(
            f"TTS speakers: narrator only (f0~{narrator_f0:.0f} Hz)"
            if np.isfinite(narrator_f0)
            else "TTS speakers: narrator only"
        )

    paths: list[Path] = []
    meta: list[dict] = []
    for r, lab in zip(rows, labels):
        if lab == 1 and r["dur"] >= 1.5 and _is_voiced_clip(r["clip"], sr):
            dest = dest_dir / f"cue_{r['i']:04d}.wav"
            sf.write(str(dest), r["clip"], sr)
            use = dest
            source = "guest-cue"
        elif lab == 1:
            use = guest_ref
            source = "guest"
        else:
            use = narrator_ref
            source = "narrator"
        paths.append(use)
        meta.append(
            {
                "cue": r["i"],
                "speaker": "guest" if lab == 1 else "narrator",
                "ref": use.name,
                "source": source,
                "f0": None if not np.isfinite(r["f0"]) else round(float(r["f0"]), 1),
            }
        )
    dest_dir.joinpath("speakers.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return paths


def resolve_speaker(args: argparse.Namespace, job_ref: Path, engine: str) -> tuple[Path | None, str]:
    if engine in ("piper", "vits"):
        return None, ""
    if args.voice_mode == "bundled":
        if not args.ref_audio or not Path(args.ref_audio).is_file():
            raise SystemExit(
                "voice_mode=bundled requires files/voices/czech_default_ref.wav "
                "(see files/voices/README.md)"
            )
        text = read_ref_text(args.ref_text)
        if engine == "f5" and not text:
            raise SystemExit("Bundled F5 voice requires a non-empty reference transcript")
        return Path(args.ref_audio), text

    if not args.vocals or not Path(args.vocals).is_file():
        raise SystemExit("voice_mode=clone requires --vocals pointing at vocals.wav")
    ref = pick_reference_clip(Path(args.vocals), job_ref)
    if engine != "f5":
        return ref, ""
    # F5 must pair the extracted clip with its own transcript. The bundled
    # czech_default_ref.txt is a dummy Czech line and does not describe vocals.wav.
    if args.ref_text:
        apple_device.log("Ignoring --ref-text in clone mode; transcribing the source clip")
    text = transcribe_ref(ref, args.whisper_model)
    job_ref.with_suffix(".txt").write_text(text + "\n", encoding="utf-8")
    return ref, text


def rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio))))


def f5_infer(tts, ref_audio: Path, ref_text: str, text: str, args, raw: Path, device: str):
    wav, sr, _ = tts.infer(
        ref_file=str(ref_audio),
        ref_text=ref_text,
        gen_text=text,
        nfe_step=args.nfe_step,
        speed=1.0,
        file_wave=str(raw),
    )
    sync_mps(device)
    if wav is None:
        raise RuntimeError("F5-TTS returned no audio")
    return int(sr)


def main() -> int:
    args = parse_args()
    if not args.smoke_test and (not args.srt or args.duration is None):
        raise SystemExit("--srt and --duration are required unless --smoke-test")
    engine = select_engine(args)
    args.voice_mode = resolve_voice_mode(args, engine)
    cache_id = voice_cache_id(engine)
    base_speed = max(1.0, float(args.base_speed))
    max_speed = max(base_speed, float(args.max_speed))
    fit_tag = fit_cache_tag(cache_id, base_speed, max_speed)
    apple_device.log(
        f"TTS engine={engine} voice_mode={args.voice_mode} "
        f"gender={args.voice_gender} cache={cache_id} "
        f"pace={base_speed:.2f}–{max_speed:.2f}x"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    segments_dir = Path(args.segments_dir) if args.segments_dir else out_path.parent / "segments"
    if args.fresh and segments_dir.is_dir():
        shutil.rmtree(segments_dir)
    segments_dir.mkdir(parents=True, exist_ok=True)

    default_ref = Path(args.ref_out) if args.ref_out else out_path.parent / "voice_ref.wav"
    speaker, ref_text = resolve_speaker(args, default_ref, engine)
    tts = None
    tts_device = "cpu"
    using_finetune = engine == "f5"
    if engine == "f5":
        tts, tts_device = load_f5(args)
        apple_device.log(f"Reference audio={speaker} text={ref_text[:80]!r}")
    elif engine == "xtts":
        if speaker is None:
            raise SystemExit("XTTS clone requires a speaker WAV")
        apple_device.log(f"XTTS speaker={speaker}")
    elif engine == "vits":
        apple_device.log(f"Coqui VITS model={args.vits_model}")
    else:
        model = Path(args.piper_model) if args.piper_model else Path()
        if not model.is_file():
            raise SystemExit(
                "Piper voice ONNX missing. Pass --piper-model "
                "models/piper/cs_CZ-jirka-medium.onnx"
            )
        apple_device.log(f"Piper voice={model}")

    if args.smoke_test:
        text = spoken_czech(args.smoke_text, engine, using_finetune, None)
        with tempfile.TemporaryDirectory(prefix="ttssmoke_") as tmp:
            raw = Path(tmp) / "smoke.wav"
            if engine == "f5":
                f5_infer(tts, speaker, ref_text, text, args, raw, tts_device)
            elif engine in ("xtts", "vits"):
                coqui_python = Path(args.xtts_python) if args.xtts_python else Path()
                model = XTTS_MODEL if engine == "xtts" else (args.vits_model or VITS_MODEL)
                coqui_batch(
                    coqui_python,
                    [{"text": text, "out": str(raw)}],
                    model,
                    speaker if engine == "xtts" else None,
                )
            else:
                piper_to_wav(text, Path(args.piper_model), raw)
            audio, sr = load_mono(raw)
        dur = len(audio) / float(sr)
        energy = rms(audio)
        sf.write(str(out_path), audio, int(sr))
        apple_device.log(f"Smoke test duration={dur:.2f}s rms={energy:.5f}")
        if dur < 0.3 or energy < 1e-4:
            raise SystemExit("TTS smoke test produced silent or too-short audio")
        return 0

    cues = load_srt(args.srt, sentences=False)
    speaker_for_cue: list[Path] | None = None
    if (
        engine == "xtts"
        and args.voice_mode == "clone"
        and speaker is not None
        and _path_ready(args.vocals)
    ):
        speaker_for_cue = plan_clone_speakers(
            Path(args.vocals),
            speaker_label_cues(cues, Path(args.srt)),
            speaker,
            segments_dir / "speakers",
        )
    glossary_file = Path(args.glossary) if args.glossary else Path(args.srt).with_name("glossary.json")
    glossary_path: Path | None = glossary_file if glossary_file.is_file() else None
    if glossary_path is not None:
        apple_device.log(f"TTS glossary={glossary_path}")
    pieces: list[tuple[float, np.ndarray]] = []
    gen_sr = 24000
    pending_coqui: list[dict] = []
    pending_meta: list[tuple[int, Path, Path, float, float]] = []

    with tempfile.TemporaryDirectory(prefix="ttscue_") as tmp:
        tmp_dir = Path(tmp)
        for i, cue in enumerate(cues, start=1):
            text = spoken_czech(cue.content, engine, using_finetune, glossary_path)
            if not text:
                continue
            text_tag = spoken_cache_tag(text)
            fitted = segments_dir / f"{i:04d}.{fit_tag}.{text_tag}.wav"
            start = cue.start.total_seconds()
            slot = max(cue_seconds(cue), 0.08)
            if not args.fresh and usable_segment(fitted):
                samples, seg_sr = load_mono(fitted)
                gen_sr = int(seg_sr)
                apple_device.log(f"TTS cue {i}/{len(cues)} resume {fitted.name}")
                pieces.append((start, samples))
                continue
            raw = segments_dir / f"{i:04d}.{cache_id}.{text_tag}_raw.wav"
            apple_device.log(f"TTS cue {i}/{len(cues)} ({slot:.2f}s): {text[:80]}")
            if engine in ("xtts", "vits"):
                if usable_segment(raw):
                    apple_device.log(f"TTS cue {i}/{len(cues)} reuse {raw.name}")
                else:
                    job = {"text": text, "out": str(raw)}
                    if speaker_for_cue is not None:
                        job["speaker"] = str(speaker_for_cue[i - 1])
                    pending_coqui.append(job)
                pending_meta.append((i, raw, fitted, start, slot))
                continue
            if engine == "f5":
                gen_sr = f5_infer(tts, speaker, ref_text, text, args, raw, tts_device)
            else:
                piper_to_wav(text, Path(args.piper_model), raw)
            samples = fit_cue_audio(
                raw, fitted, slot, max_speed, tmp_dir, base_speed=base_speed
            )
            gen_sr = int(sf.info(str(fitted)).samplerate)
            pieces.append((start, samples))

        if pending_coqui:
            coqui_python = Path(args.xtts_python) if args.xtts_python else Path()
            model = XTTS_MODEL if engine == "xtts" else (args.vits_model or VITS_MODEL)
            coqui_batch(
                coqui_python,
                pending_coqui,
                model,
                speaker if engine == "xtts" else None,
            )
        for _i, raw, fitted, start, slot in pending_meta:
            if not raw.is_file():
                raise RuntimeError(f"Coqui TTS did not write {raw}")
            samples = fit_cue_audio(
                raw, fitted, slot, max_speed, tmp_dir, base_speed=base_speed
            )
            gen_sr = int(sf.info(str(fitted)).samplerate)
            pieces.append((start, samples))

    if not pieces:
        raise SystemExit("No Czech speech segments were generated")

    last = max(
        (start + (len(samples) / float(gen_sr) if gen_sr else 0.0))
        for start, samples in pieces
    )
    duration = max(args.duration, last + 0.1)
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
