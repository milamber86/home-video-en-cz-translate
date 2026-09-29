#!/usr/bin/env python3
"""Secondary pass: compress Czech cues that cannot fit their original slot.

Runs after translation and before TTS. For every cue it measures the time the
dubbed line may actually use (from the isolated vocal stem when available, so
leading/trailing silence inside the cue counts), estimates how long the Czech
text will take to speak, and asks the local chat model to shorten only the
lines that would otherwise be sped up past `--max-speed` or overlap the next
cue.

The pristine translation is kept as `cs.orig.srt`; the pass is idempotent and
always reflows from that file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import device as apple_device  # noqa: E402
from czech_tts_text import expand_for_tts  # noqa: E402
from srtutil import load_srt, save_srt  # noqa: E402
from timing import (  # noqa: E402
    char_budget,
    cue_slot,
    cue_timings,
    est_duration,
    load_speech_regions,
)
from whisper_translate import (  # noqa: E402
    align_translations,
    clean_translategemma,
    ollama_json,
    ollama_tags,
    parse_json_array,
    pick_chat_model,
    strip_instruction_leak,
)

SYSTEM_PROMPT = (
    "You compress Czech dubbing lines to a hard character budget.\n"
    "Rules:\n"
    '- Reply with one JSON object: {"lines": ["...", "..."]}, one string per '
    "input line, in the same order.\n"
    "- Each line MUST be at most its budget in characters, including spaces "
    "and the final period. Count them.\n"
    "- Keep the core meaning, gender agreement and proper nouns.\n"
    "- Drop filler, subordinate clauses and explanations first; keep the facts.\n"
    "- Natural spoken Czech, sentence case.\n"
    "- Never translate, never explain, never add notes."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cs-srt", required=True, help="Czech SRT to rewrite in place")
    p.add_argument("--en-srt", default="", help="English SRT for context (same cue count)")
    p.add_argument("--vocals", default="", help="Isolated vocals.wav for timing analysis")
    p.add_argument("--glossary", default="", help="glossary.json with name pronunciations")
    p.add_argument("--url", default="http://127.0.0.1:11434", help="Ollama base URL")
    p.add_argument("--model", default="qwen2.5:14b", help="Preferred chat model")
    p.add_argument("--base-speed", type=float, default=1.15)
    p.add_argument("--max-speed", type=float, default=1.25)
    p.add_argument("--chars-per-sec", type=float, default=14.0)
    p.add_argument("--batch", type=int, default=6, help="Cues per model request")
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--report", default="", help="Write a JSON report here")
    p.add_argument("--dry-run", action="store_true", help="Report only; leave cs.srt alone")
    p.add_argument("--force", action="store_true", help="Reflow even when cs.srt is already processed")
    return p.parse_args()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def spoken_form(text: str, glossary: Path | None) -> str:
    """What the TTS will actually say (numbers/abbreviations expanded)."""
    text = (text or "").strip()
    if text and text[-1] not in ".!?…":
        text += "."
    return expand_for_tts(text, glossary)


def plan_cues(cues, en_cues, timings, glossary: Path | None, args):
    """Flag cues whose spoken Czech cannot fit the available slot."""
    plan: list[dict] = []
    for i, cue in enumerate(cues):
        if timings is not None:
            slot = timings[i].slot
        else:
            slot = cue_slot(cue, cues[i + 1] if i + 1 < len(cues) else None)
        text = cue.content.strip()
        if not text:
            continue
        spoken = spoken_form(text, glossary)
        est = est_duration(spoken, args.chars_per_sec)
        needed = est / max(slot, 0.05)
        entry = {
            "index": i + 1,
            "start": round(cue.start.total_seconds(), 3),
            "slot": round(slot, 3),
            "chars": len(text),
            "spoken_chars": len(spoken),
            "est_sec": round(est, 3),
            "needed_speed": round(needed, 3),
            "budget_chars": char_budget(slot, args.max_speed, args.chars_per_sec),
            "en": en_cues[i].content.strip() if en_cues and i < len(en_cues) else "",
            "action": "keep",
        }
        # Budgets are spoken characters; scale back to SRT characters so a line
        # full of numbers is not allowed more text than it can speak.
        ratio = len(text) / max(len(spoken), 1)
        entry["budget_chars"] = max(4, int(entry["budget_chars"] * ratio))
        if needed > args.max_speed + 0.02:
            entry["action"] = "compress"
        plan.append(entry)
    return plan


def compress_batch(
    model: str, url: str, batch: list[dict], args, strict: bool = False
) -> list[str]:
    lines = []
    for item in batch:
        context = f" (source: {item['en']})" if item["en"] else ""
        lines.append(
            f"{item['index']}. at most {item['budget_chars']} chars "
            f"(now {item['chars']}){context}: \"{item['text']}\""
        )
    header = "Compress to the budget. Budgets are hard limits."
    if strict:
        header = (
            "STRICT RETRY: the previous answers were too long. "
            "Rewrite each line in at most its budget, counting characters."
        )
    user = (
        f"{header}\n"
        + "\n".join(lines)
        + f'\nReturn {{"lines": [...]}} with {len(batch)} strings.'
    )
    raw = ollama_json(url, model, SYSTEM_PROMPT, user, args.timeout)
    if isinstance(raw, str):
        return parse_json_array(raw)
    if isinstance(raw, list):
        return [str(x) for x in raw]
    return parse_json_array(json.dumps(raw, ensure_ascii=False))


def validate(original: str, candidate: str, budget: int) -> str | None:
    """Return the compressed line, or None when it is unusable."""
    text = clean_translategemma(strip_instruction_leak(candidate or ""))
    if not text:
        return None
    if len(text) > budget * 1.25:
        return None
    if len(text) >= len(original) and len(original) <= budget:
        return None
    return text


def apply_batch(
    chat_model: str,
    url: str,
    batch: list[dict],
    args,
    glossary: Path | None,
    strict: bool = False,
) -> None:
    """Ask the model to compress a batch and record accepted lines in place."""
    try:
        candidates = compress_batch(chat_model, url, batch, args, strict=strict)
    except Exception as exc:
        apple_device.log(f"Reflow batch failed ({exc}); keeping originals")
        return
    try:
        aligned = align_translations(
            candidates, [item["text"] for item in batch], batch[0]["text"]
        )
    except ValueError as exc:
        apple_device.log(f"Reflow batch misaligned ({exc}); keeping originals")
        return
    for item, candidate in zip(batch, aligned):
        compressed = validate(item["text"], candidate, item["budget_chars"])
        if compressed is None:
            apple_device.log(
                f"Reflow cue {item['index']}: kept original "
                f"({len(candidate or '')} chars vs budget {item['budget_chars']})"
            )
            continue
        est_after = est_duration(spoken_form(compressed, glossary), args.chars_per_sec)
        needed_after = est_after / max(item["slot"], 0.05)
        if needed_after > args.max_speed * 1.1:
            apple_device.log(
                f"Reflow cue {item['index']}: still {needed_after:.2f}x after "
                f"compression; retrying"
            )
            continue
        item["compressed"] = compressed
        item["chars_after"] = len(compressed)
        item["est_sec_after"] = round(est_after, 3)
        item["needed_speed_after"] = round(needed_after, 3)
        apple_device.log(
            f"Reflow cue {item['index']}: {item['chars']} -> {len(compressed)} chars "
            f"({item['needed_speed']:.2f}x -> {needed_after:.2f}x)"
        )


def main() -> int:
    args = parse_args()
    cs_path = Path(args.cs_srt)
    if not cs_path.is_file():
        raise SystemExit(f"Czech SRT missing: {cs_path}")
    original = cs_path.with_name("cs.orig.srt")
    done = cs_path.with_name("cs.reflow.done")
    if (
        not args.dry_run
        and not args.force
        and done.is_file()
        and original.is_file()
        and done.read_text(encoding="utf-8").strip() == _sha(cs_path)
    ):
        apple_device.log("Reflow: cs.srt already processed; nothing to do")
        return 0
    if not original.is_file() or args.force:
        shutil.copyfile(cs_path, original)
        apple_device.log(f"Reflow: pristine translation saved to {original.name}")
    cues = load_srt(original, sentences=False)

    en_cues = None
    if args.en_srt and Path(args.en_srt).is_file():
        try:
            en_loaded = load_srt(Path(args.en_srt), sentences=False)
        except ValueError:
            en_loaded = []
        if len(en_loaded) == len(cues):
            en_cues = en_loaded

    timings = None
    if args.vocals and Path(args.vocals).is_file():
        try:
            regions = load_speech_regions(args.vocals)
        except Exception as exc:
            apple_device.log(f"Reflow: vocal timing unavailable ({exc})")
            regions = []
        if regions:
            timings = cue_timings(cues, regions)
            apple_device.log(f"Reflow: timing from {len(regions)} voiced spans")

    glossary: Path | None = None
    if args.glossary and Path(args.glossary).is_file():
        glossary = Path(args.glossary)
    elif cs_path.with_name("glossary.json").is_file():
        glossary = cs_path.with_name("glossary.json")

    plan = plan_cues(cues, en_cues, timings, glossary, args)
    for item, cue in zip(plan, cues):
        item["text"] = cue.content.strip()

    flagged = [item for item in plan if item["action"] == "compress"]
    apple_device.log(
        f"Reflow: {len(flagged)}/{len(plan)} cues exceed {args.max_speed:.2f}x "
        f"({args.chars_per_sec:.1f} chars/s)"
    )

    tags = ollama_tags(args.url)
    chat_model: str | None = pick_chat_model(tags, args.model) if tags else None
    if flagged and chat_model is None:
        apple_device.log(
            "Reflow: no chat model reachable; keeping the translation as-is "
            f"(url={args.url})"
        )
        flagged = []

    batch_size = max(1, args.batch)
    if chat_model is not None:
        for start in range(0, len(flagged), batch_size):
            batch = flagged[start : start + batch_size]
            apple_device.log(
                f"Reflow batch {start // batch_size + 1}: "
                f"cues {batch[0]['index']}–{batch[-1]['index']}"
            )
            apply_batch(chat_model, args.url, batch, args, glossary)
            leftovers = [item for item in batch if "compressed" not in item]
            if leftovers:
                apple_device.log(
                    f"Reflow batch retry: {len(leftovers)} cue(s) still over budget"
                )
                apply_batch(
                    chat_model, args.url, leftovers, args, glossary, strict=True
                )

    applied = [item for item in flagged if item.get("compressed")]
    if applied and not args.dry_run:
        by_index = {item["index"]: item["compressed"] for item in applied}
        for i, cue in enumerate(cues, start=1):
            if i in by_index:
                cue.content = by_index[i]
        save_srt(cs_path, cues)
        done.write_text(_sha(cs_path) + "\n", encoding="utf-8")
        apple_device.log(f"Reflow: wrote {cs_path} ({len(applied)} cues compressed)")
    elif applied:
        apple_device.log(f"Reflow dry run: {len(applied)} cues would change")

    if args.report:
        report = Path(args.report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps(
                {
                    "chars_per_sec": args.chars_per_sec,
                    "max_speed": args.max_speed,
                    "timing_source": "vocals" if timings is not None else "cue",
                    "dry_run": bool(args.dry_run),
                    "cues": plan,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        apple_device.log(f"Reflow report: {report}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130) from None
