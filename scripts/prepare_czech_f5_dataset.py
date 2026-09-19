#!/usr/bin/env python3
"""Build an F5-TTS csv+wavs dataset from a Hugging Face speech corpus (or a local tree).

Default source is facebook/voxpopuli (Czech). Mozilla Common Voice left Hugging Face
in October 2025; pass --local-dir if you downloaded a CV tarball yourself.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import device as apple_device  # noqa: E402

apple_device.bootstrap_mps_fallback()

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402


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
    if src_sr == dst_sr:
        return np.ascontiguousarray(audio)
    import torch
    import torchaudio

    wav = torch.from_numpy(np.ascontiguousarray(audio)).float().unsqueeze(0)
    out = torchaudio.functional.resample(wav, src_sr, dst_sr)
    return out.squeeze(0).cpu().numpy().astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sr: int) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak * 0.99
    sf.write(str(path), audio, sr, subtype="PCM_16")
    return len(audio) / float(sr)


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
    try:
        array = audio_obj["array"]
        sr = int(audio_obj["sampling_rate"] or default_sr)
        return np.asarray(array), sr
    except Exception:
        pass
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
    return None, default_sr


def from_hf(args: argparse.Namespace, out_dir: Path) -> int:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        raise SystemExit("datasets is required to download a Hugging Face speech corpus") from exc

    if _common_voice_removed(args.hf_dataset):
        raise SystemExit(
            "Mozilla Common Voice is no longer on Hugging Face (moved to "
            "Mozilla Data Collective in October 2025). Default training data is "
            "facebook/voxpopuli config=cs. To use Common Voice, download the Czech "
            "tarball, convert it to metadata.csv + wavs/, and pass --local-dir."
        )

    apple_device.log(
        f"Loading {args.hf_dataset} config={args.hf_config} split={args.hf_split}"
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

    try:
        ds = ds.cast_column(args.audio_column, Audio(sampling_rate=args.sample_rate))
    except Exception:
        pass

    wavs = out_dir / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str]] = []
    hours = 0.0
    kept = 0
    for i, row in enumerate(ds, start=1):
        text = _row_text(row, args.text_column)
        if not text:
            continue
        if int(row.get("down_votes") or 0) > 0:
            continue
        audio_obj = row.get(args.audio_column)
        array, sr = _decode_audio(audio_obj, args.sample_rate)
        if array is None:
            continue
        audio = resample_mono(array, sr, args.sample_rate)
        dur = len(audio) / float(args.sample_rate)
        if dur < args.min_seconds or dur > args.max_seconds:
            continue
        dest = wavs / f"{kept + 1:08d}.wav"
        actual = write_wav(dest, audio, args.sample_rate)
        rows.append((str(dest.resolve()), text))
        hours += actual / 3600.0
        kept += 1
        if kept % 100 == 0:
            apple_device.log(f"Kept {kept} clips ({hours:.2f} h)")
        if hours >= args.max_hours:
            break

    if not rows:
        raise SystemExit("Hugging Face dataset produced no usable clips")
    write_metadata(out_dir / "metadata.csv", rows)
    apple_device.log(f"Wrote {len(rows)} clips ({hours:.2f} h) to {out_dir}")
    return 0


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.local_dir:
        return from_local(Path(args.local_dir), out_dir, args)
    return from_hf(args, out_dir)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
