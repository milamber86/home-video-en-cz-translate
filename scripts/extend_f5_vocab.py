#!/usr/bin/env python3
"""Extend F5TTS_v1_Base vocab + text embeddings for missing Czech characters."""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import device as apple_device  # noqa: E402

apple_device.bootstrap_mps_fallback()

ALWAYS_ADD = list("ďůĎŮ")
EMBED_KEY = "ema_model.transformer.text_embed.text_embed.weight"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--out-vocab", required=True)
    p.add_argument("--out-ckpt", required=True)
    p.add_argument("--src-vocab", default="")
    p.add_argument("--src-ckpt", default="")
    p.add_argument("--model", default="F5TTS_v1_Base")
    return p.parse_args()


def load_vocab_lines(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8-sig")
    # Preserve a trailing empty entry the way F5 vocab files are stored.
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    return lines


def default_src_vocab() -> Path:
    try:
        from importlib.resources import files

        candidate = Path(str(files("f5_tts").joinpath("infer/examples/vocab.txt")))
        if candidate.is_file():
            return candidate
    except Exception:
        pass
    from cached_path import cached_path

    return Path(str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/vocab.txt")))


def default_src_ckpt(model: str) -> Path:
    from cached_path import cached_path

    if model == "F5TTS_v1_Base":
        uri = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    elif model == "F5TTS_Base":
        uri = "hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"
    else:
        uri = "hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
    return Path(str(cached_path(uri)))


def corpus_chars(metadata_csv: Path) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    with metadata_csv.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.reader(fh, delimiter="|")
        header = next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            for ch in row[1]:
                if ch not in seen:
                    seen.add(ch)
                    found.append(ch)
    return found


def expand_checkpoint(src: Path, dest: Path, n_new: int) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    dest.parent.mkdir(parents=True, exist_ok=True)
    if n_new <= 0:
        shutil.copy2(src, dest)
        return

    if src.suffix == ".safetensors":
        state = load_file(str(src), device="cpu")
    else:
        blob = torch.load(str(src), map_location="cpu", weights_only=True)
        state = blob.get("ema_model_state_dict") or blob

    if EMBED_KEY not in state:
        alt = [k for k in state if k.endswith("text_embed.weight")]
        raise SystemExit(
            f"Could not find {EMBED_KEY} in {src}. Candidates: {alt[:8]}"
        )

    old = state[EMBED_KEY]
    vocab_old, dim = old.shape
    extra = old.mean(dim=0, keepdim=True).repeat(n_new, 1)
    state[EMBED_KEY] = torch.cat([old, extra], dim=0)
    apple_device.log(
        f"Expanded text embeddings {vocab_old} -> {vocab_old + n_new} (dim={dim})"
    )

    if dest.suffix == ".safetensors" or src.suffix == ".safetensors":
        if dest.suffix != ".safetensors":
            dest = dest.with_suffix(".safetensors")
        save_file(state, str(dest))
    else:
        torch.save({"ema_model_state_dict": state}, str(dest))


def main() -> int:
    args = parse_args()
    metadata = Path(args.metadata_csv)
    if not metadata.is_file():
        raise SystemExit(f"metadata.csv not found: {metadata}")

    src_vocab = Path(args.src_vocab) if args.src_vocab else default_src_vocab()
    src_ckpt = Path(args.src_ckpt) if args.src_ckpt else default_src_ckpt(args.model)
    if not src_vocab.is_file():
        raise SystemExit(f"Source vocab not found: {src_vocab}")
    if not src_ckpt.is_file():
        raise SystemExit(f"Source checkpoint not found: {src_ckpt}")

    vocab = load_vocab_lines(src_vocab)
    existing = set(vocab)
    missing: list[str] = []
    for ch in ALWAYS_ADD + corpus_chars(metadata):
        if ch not in existing:
            existing.add(ch)
            missing.append(ch)

    out_vocab = Path(args.out_vocab)
    out_vocab.parent.mkdir(parents=True, exist_ok=True)
    out_lines = list(vocab) + missing
    out_vocab.write_text("\n".join(out_lines) + "\n", encoding="utf-8")

    out_ckpt = Path(args.out_ckpt)
    expand_checkpoint(src_ckpt, out_ckpt, len(missing))
    apple_device.log(
        f"Vocab {len(vocab)} -> {len(out_lines)} "
        f"(added {len(missing)}: {''.join(missing) or 'none'})"
    )
    apple_device.log(f"Wrote {out_vocab} and {out_ckpt}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
