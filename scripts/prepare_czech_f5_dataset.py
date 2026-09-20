#!/usr/bin/env python3
"""Build an F5-TTS csv+wavs dataset from a Hugging Face speech corpus (or a local tree).

Default source is facebook/voxpopuli (Czech). Mozilla Common Voice left Hugging Face
in October 2025; pass --local-dir if you downloaded a CV tarball yourself.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import device as apple_device  # noqa: E402

apple_device.bootstrap_mps_fallback()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


PROGRESS_JSON = "progress.json"
PROGRESS_LOG = "progress.log"
HEARTBEAT_SEC = 15.0
KEEP_HEARTBEAT = 50


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True, help="Writes wavs/ and metadata.csv")
    p.add_argument("--hf-dataset", default="facebook/voxpopuli")
    p.add_argument("--hf-config", default="cs")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--text-column", default="raw_text")
    p.add_argument("--audio-column", default="audio")
    p.add_argument("--local-dir", default="", help="Existing dataset with metadata.csv")
    p.add_argument("--max-hours", type=float, default=20.0)
    p.add_argument("--min-seconds", type=float, default=1.0)
    p.add_argument("--max-seconds", type=float, default=12.0)
    p.add_argument("--sample-rate", type=int, default=24000)
    p.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 4),
        help="Threads for CPU resample + WAV write",
    )
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing wavs/metadata/progress and start over",
    )
    p.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Only for legacy HF loading scripts. Current datasets builds reject this.",
    )
    return p.parse_args()


def resample_mono(array: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    audio = np.asarray(array, dtype=np.float32)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=-1)
    audio = np.ascontiguousarray(audio)
    if src_sr == dst_sr:
        return audio
    import librosa

    return np.ascontiguousarray(
        librosa.resample(audio, orig_sr=src_sr, target_sr=dst_sr).astype(np.float32)
    )


def write_wav(path: Path, audio: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak * 0.99
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return len(audio) / float(sr)


def process_clip(
    array: np.ndarray,
    src_sr: int,
    dest: Path,
    sample_rate: int,
    min_seconds: float,
    max_seconds: float,
) -> float | None:
    audio = resample_mono(array, src_sr, sample_rate)
    dur = len(audio) / float(sample_rate)
    if dur < min_seconds or dur > max_seconds:
        return None
    return write_wav(dest, audio, sample_rate)


class Progress:
    def __init__(self, out_dir: Path, max_hours: float):
        self.out_dir = out_dir
        self.max_hours = max_hours
        self.log_path = out_dir / PROGRESS_LOG
        self.json_path = out_dir / PROGRESS_JSON
        self.meta_path = out_dir / "metadata.csv"
        self.scanned = 0
        self.kept = 0
        self.hours = 0.0
        self.last_audio_id = ""
        self.t0 = time.monotonic()
        self.last_beat = 0.0

    def emit(self, msg: str) -> None:
        apple_device.log(msg)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
            fh.flush()

    def heartbeat(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and (now - self.last_beat) < HEARTBEAT_SEC:
            return
        self.last_beat = now
        elapsed = max(now - self.t0, 1e-6)
        rate = self.kept / elapsed
        self.emit(
            f"prep scanned={self.scanned} kept={self.kept} "
            f"hours={self.hours:.2f}/{self.max_hours:.1f} "
            f"clips/s={rate:.2f} last_id={self.last_audio_id or '-'}"
        )
        self.save_json()

    def save_json(self) -> None:
        payload = {
            "kept": self.kept,
            "hours": self.hours,
            "scanned": self.scanned,
            "last_audio_id": self.last_audio_id,
        }
        tmp = self.json_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.json_path)

    def load_json(self) -> None:
        if not self.json_path.is_file():
            return
        try:
            data = json.loads(self.json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        self.kept = int(data.get("kept") or 0)
        self.hours = float(data.get("hours") or 0.0)
        self.scanned = int(data.get("scanned") or 0)
        self.last_audio_id = str(data.get("last_audio_id") or "")

    def ensure_metadata_header(self) -> None:
        if self.meta_path.is_file() and self.meta_path.stat().st_size > 0:
            return
        with self.meta_path.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh, delimiter="|", lineterminator="\n").writerow(
                ["audio_file", "text"]
            )

    def append_metadata(self, audio_file: str, text: str) -> None:
        with self.meta_path.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh, delimiter="|", lineterminator="\n").writerow(
                [audio_file, text]
            )


def hours_from_wavs(wavs: Path) -> tuple[int, float]:
    files = sorted(wavs.glob("*.wav"))
    hours = 0.0
    for path in files:
        try:
            hours += float(sf.info(str(path)).duration) / 3600.0
        except Exception:
            continue
    return len(files), hours


def metadata_rows(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    n = 0
    with path.open(encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            if line.strip():
                n += 1
    return n


def next_wav_index(wavs: Path) -> int:
    nums = []
    for path in wavs.glob("*.wav"):
        try:
            nums.append(int(path.stem))
        except ValueError:
            continue
    return (max(nums) + 1) if nums else 1


def write_metadata(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="|", lineterminator="\n")
        writer.writerow(["audio_file", "text"])
        for audio_file, text in rows:
            writer.writerow([audio_file, text])


def from_local(local_dir: Path, out_dir: Path, args: argparse.Namespace) -> int:
    meta = local_dir / "metadata.csv"
    if not meta.is_file():
        raise SystemExit(f"Local dataset missing {meta}")
    rows: list[tuple[str, str]] = []
    hours = 0.0
    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    with meta.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter="|")
        header = next(reader, None)
        for i, row in enumerate(reader, start=1):
            if len(row) < 2:
                continue
            src = Path(row[0].strip())
            text = row[1].strip()
            if not src.is_absolute():
                src = (local_dir / src).resolve()
                if not src.is_file():
                    src = (local_dir / "wavs" / Path(row[0].strip()).name).resolve()
            if not src.is_file() or not text:
                continue
            audio, sr = sf.read(str(src), always_2d=False)
            audio = resample_mono(audio, int(sr), args.sample_rate)
            dur = len(audio) / float(args.sample_rate)
            if dur < args.min_seconds or dur > args.max_seconds:
                continue
            dest = wavs / f"{i:08d}.wav"
            actual = write_wav(dest, audio, args.sample_rate)
            rows.append((str(dest.resolve()), text))
            hours += actual / 3600.0
            if hours >= args.max_hours:
                break
    if not rows:
        raise SystemExit("Local dataset produced no usable clips")
    write_metadata(out_dir / "metadata.csv", rows)
    apple_device.log(f"Wrote {len(rows)} clips ({hours:.2f} h) to {out_dir}")
    return 0


_TEXT_COLUMNS = (
    "sentence",
    "raw_text",
    "normalized_text",
    "transcription",
    "text",
)


def _common_voice_removed(name: str) -> bool:
    return "common_voice" in (name or "").lower() and "mozilla-foundation" in (name or "").lower()


def _row_text(row: dict, preferred: str) -> str:
    keys = [preferred, *_TEXT_COLUMNS]
    seen: set[str] = set()
    for key in keys:
        if not key or key in seen:
            continue
        seen.add(key)
        val = row.get(key)
        if val:
            return str(val).strip()
    return ""


def _decode_audio(audio_obj, default_sr: int) -> tuple[np.ndarray | None, int]:
    if audio_obj is None:
        return None, default_sr
    if isinstance(audio_obj, dict):
        array = audio_obj.get("array")
        sr = int(audio_obj.get("sampling_rate") or default_sr)
        if array is None:
            path = audio_obj.get("path")
            if not path:
                return None, default_sr
            array, sr = sf.read(path, always_2d=False)
        return np.asarray(array), int(sr)
    getter = getattr(audio_obj, "get_all_samples", None)
    if callable(getter):
        samples = getter()
        data = samples.data
        if hasattr(data, "cpu"):
            data = data.cpu().numpy()
        data = np.asarray(data, dtype=np.float32)
        if data.ndim > 1:
            data = np.mean(data, axis=tuple(range(data.ndim - 1)))
        sr = int(getattr(samples, "sample_rate", default_sr) or default_sr)
        return data, sr
    try:
        array = audio_obj["array"]
        sr = int(audio_obj["sampling_rate"] or default_sr)
        return np.asarray(array), sr
    except Exception:
        return None, default_sr


def drain_pending(
    pending: list[tuple[Future, Path, str, str]],
    progress: Progress,
    wait_all: bool,
    max_inflight: int,
) -> None:
    while pending and (wait_all or len(pending) >= max_inflight):
        fut, dest, text, audio_id = pending.pop(0)
        actual = fut.result()
        if actual is None:
            dest.unlink(missing_ok=True)
            continue
        progress.kept += 1
        progress.hours += actual / 3600.0
        progress.last_audio_id = audio_id
        progress.append_metadata(str(dest.resolve()), text)
        progress.heartbeat(force=progress.kept % KEEP_HEARTBEAT == 0)


def from_hf(args: argparse.Namespace, out_dir: Path) -> int:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required to download a Hugging Face speech corpus") from exc

    if _common_voice_removed(args.hf_dataset):
        raise SystemExit(
            "Mozilla Common Voice is no longer on Hugging Face (moved to "
            "Mozilla Data Collective in October 2025). Default training data is "
            "facebook/voxpopuli config=cs. To use Common Voice, download the Czech "
            "tarball, convert it to metadata.csv + wavs/, and pass --local-dir."
        )

    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    progress = Progress(out_dir, args.max_hours)
    progress.load_json()
    n_meta = metadata_rows(progress.meta_path)
    n_wavs, wav_hours = hours_from_wavs(wavs)
    if n_meta > 0 and max(progress.hours, wav_hours) >= args.max_hours:
        progress.emit(
            f"prep already complete: {max(progress.kept, n_meta)} clips "
            f"({max(progress.hours, wav_hours):.2f} h)"
        )
        return 0

    resume_id = progress.last_audio_id
    skipping_id = bool(resume_id) and n_meta > 0 and n_wavs <= n_meta
    if skipping_id:
        progress.ensure_metadata_header()
        next_index = next_wav_index(wavs)
        if progress.kept <= 0:
            progress.kept = n_meta
            progress.hours = wav_hours
    else:
        progress.meta_path.write_text("audio_file|text\n", encoding="utf-8")
        progress.kept = 0
        progress.hours = 0.0
        progress.last_audio_id = ""
        next_index = 1
        if n_wavs:
            progress.emit(
                f"prep reusing {n_wavs} existing wavs ({wav_hours:.2f} h); "
                "rebuilding metadata from the HF stream"
            )

    workers = max(1, args.workers)
    progress.emit(
        f"Loading {args.hf_dataset} config={args.hf_config} split={args.hf_split} "
        f"workers={workers} resume_id={(resume_id if skipping_id else '-') or '-'}"
    )
    load_kwargs = {
        "path": args.hf_dataset,
        "name": args.hf_config or None,
        "split": args.hf_split,
        "streaming": True,
    }
    if args.trust_remote_code:
        load_kwargs["trust_remote_code"] = True
    try:
        ds = load_dataset(**{k: v for k, v in load_kwargs.items() if v is not None})
    except TypeError:
        load_kwargs.pop("trust_remote_code", None)
        ds = load_dataset(**{k: v for k, v in load_kwargs.items() if v is not None})
    except Exception as exc:
        raise SystemExit(
            f"Failed to load {args.hf_dataset} ({exc}). "
            "If the repo is gated, run `huggingface-cli login`. "
            "Or pass --local-dir with a prepared metadata.csv."
        ) from exc

    pending: list[tuple[Future, Path, str, str]] = []
    max_inflight = workers * 2
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for row in ds:
            progress.scanned += 1
            progress.heartbeat()
            audio_id = str(row.get("audio_id") or "")
            if skipping_id:
                if audio_id == resume_id:
                    skipping_id = False
                continue
            text = _row_text(row, args.text_column)
            if not text:
                continue
            if int(row.get("down_votes") or 0) > 0:
                continue
            array, sr = _decode_audio(row.get(args.audio_column), 16000)
            if array is None:
                continue
            native_dur = len(array) / float(sr or 1)
            if native_dur < args.min_seconds or native_dur > args.max_seconds:
                continue
            dest = wavs / f"{next_index:08d}.wav"
            next_index += 1
            if dest.is_file():
                try:
                    actual = float(sf.info(str(dest)).duration)
                except Exception:
                    actual = 0.0
                if args.min_seconds <= actual <= args.max_seconds:
                    progress.kept += 1
                    progress.hours += actual / 3600.0
                    progress.last_audio_id = audio_id
                    progress.append_metadata(str(dest.resolve()), text)
                    progress.heartbeat(force=progress.kept % KEEP_HEARTBEAT == 0)
                    if progress.hours >= args.max_hours:
                        break
                    continue
                dest.unlink(missing_ok=True)
            fut = pool.submit(
                process_clip,
                np.array(array, dtype=np.float32, copy=True),
                int(sr),
                dest,
                args.sample_rate,
                args.min_seconds,
                args.max_seconds,
            )
            pending.append((fut, dest, text, audio_id))
            drain_pending(pending, progress, False, max_inflight)
            if progress.hours >= args.max_hours:
                break
        drain_pending(pending, progress, True, 0)

    if skipping_id:
        progress.emit(
            f"prep last_audio_id={resume_id} was not found in the stream; "
            "existing clips were kept"
        )
    if progress.kept <= 0:
        raise SystemExit("Hugging Face dataset produced no usable clips")
    progress.heartbeat(force=True)
    progress.emit(f"Wrote {progress.kept} clips ({progress.hours:.2f} h) to {out_dir}")
    return 0


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fresh:
        for name in ("wavs", PROGRESS_JSON, PROGRESS_LOG, "metadata.csv"):
            path = out_dir / name
            if path.is_dir():
                shutil.rmtree(path)
            elif path.is_file():
                path.unlink()
        (out_dir / "wavs").mkdir(parents=True, exist_ok=True)
    if args.local_dir:
        return from_local(Path(args.local_dir), out_dir, args)
    return from_hf(args, out_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
