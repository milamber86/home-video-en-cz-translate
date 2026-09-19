#!/usr/bin/env python3
"""Transcribe English audio (or reuse SRT) and translate cues to Czech.

ASR uses mlx-whisper on Metal. Translation prefers a local Ollama API and
falls back to Marian (Helsinki-NLP) on torch MPS.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import device as apple_device  # noqa: E402

apple_device.bootstrap_mps_fallback()

import srt  # noqa: E402
from srtutil import (  # noqa: E402
    load_srt,
    save_srt,
    segments_to_cues,
)

CONTEXT_SENTENCES = 4

SYSTEM_PROMPT = (
    "You are a professional audiovisual translator from English to Czech. "
    "Translate each English subtitle sentence into natural, spoken Czech. "
    "The items are consecutive sentences from the same talk, not isolated fragments. "
    "Preserve clause links and finish each thought; do not restart meaning at "
    "array boundaries. Keep roughly the same length as the source. "
    "Use correct Czech gender, number, and case agreement, and Czech word order. "
    "Do not calque English syntax or stack tautologies "
    "(bad: 'této názoru' → good: 'tohoto názoru'; "
    "bad: 'současnou ekonomickou systémem' → good: 'současným ekonomickým systémem'; "
    "bad: 'by měla stát přednost vytváření a udržování institucí nezbytných pro "
    "funkční fungování trhů' → good: 'by stát měl dávat přednost vytváření a "
    "udržování institucí nezbytných pro fungování trhů'). "
    "Do not add explanations, numbering, or timestamps. "
    "Previous sentences given as context are read-only: do not translate or "
    "repeat them in the output. "
    "Return ONLY a JSON array of Czech strings with exactly the same length "
    "and order as the input sentences array."
)

REVISE_PROMPT = (
    "You are a native Czech subtitle editor. Revise each Czech draft so it is "
    "grammatical spoken Czech. Fix gender, number, and case agreement. Use Czech "
    "word order, not English calques. Do not change meaning or add content. "
    "Keep roughly the same length. Examples: 'této názoru' → 'tohoto názoru'; "
    "'současnou ekonomickou systémem' → 'současným ekonomickým systémem'; "
    "'by měla stát přednost … funkční fungování trhů' → "
    "'by stát měl dávat přednost … fungování trhů'. "
    "Return ONLY a JSON array of Czech strings with exactly the same length "
    "and order as the input drafts."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_LIST_KEYS = ("cues", "translations", "cs", "output", "result", "items")
_ITEM_KEYS = ("cs", "cs_text", "translation", "text", "czech", "content")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", help="WAV to transcribe if no English SRT is usable")
    p.add_argument("--en-srt", dest="en_srt", help="Existing English SRT")
    p.add_argument("--out-en", required=True, help="Write English SRT here")
    p.add_argument("--out-cs", required=True, help="Write Czech SRT here")
    p.add_argument(
        "--whisper-model",
        default="mlx-community/whisper-large-v3-mlx",
    )
    p.add_argument("--device", default="mps", help="torch device for Marian (mps|cpu)")
    p.add_argument(
        "--translation-backend",
        default="ollama_first",
        choices=("ollama_first", "ollama_only", "marian"),
    )
    p.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    p.add_argument("--ollama-model", default="qwen2.5:14b")
    p.add_argument(
        "--marian-model",
        default="Helsinki-NLP/opus-mt-tc-big-en-ces_slk",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--ollama-timeout", type=float, default=180.0)
    return p.parse_args()


def transcribe_mlx(audio: str, model: str) -> list[srt.Subtitle]:
    try:
        import mlx_whisper
    except ImportError as exc:
        raise SystemExit(
            "mlx-whisper is required for ASR on Apple Silicon. "
            "Install it in the project venv."
        ) from exc

    apple_device.log(f"Transcribing with mlx-whisper model={model}")
    result = mlx_whisper.transcribe(
        audio,
        path_or_hf_repo=model,
        language="en",
        word_timestamps=True,
        condition_on_previous_text=True,
    )
    segments = result.get("segments") or []
    return segments_to_cues(segments)


def _as_str_list(data: object) -> list[str]:
    if isinstance(data, str):
        stripped = data.strip()
        return [stripped] if stripped else []
    if isinstance(data, dict):
        if "cs" in data and isinstance(data["cs"], str):
            return [data["cs"].strip()]
        for key in _LIST_KEYS:
            if key in data:
                return _as_str_list(data[key])
        raise ValueError("JSON object has no translation list")
    if not isinstance(data, list):
        raise ValueError("JSON value is not a list")
    out: list[str] = []
    for item in data:
        if isinstance(item, dict):
            val = ""
            for key in _ITEM_KEYS:
                if key in item and item[key] is not None:
                    val = str(item[key])
                    break
            if not val and item:
                val = str(next(iter(item.values())))
            out.append(val.strip())
        else:
            out.append(str(item).strip())
    return out


def parse_json_array(text: str) -> list[str]:
    text = _FENCE_RE.sub("", (text or "").strip())
    if not text:
        raise ValueError("Empty model output")
    try:
        return _as_str_list(json.loads(text))
    except json.JSONDecodeError:
        pass
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        try:
            return _as_str_list(json.loads(text[start : end + 1]))
        except json.JSONDecodeError:
            pass
    obj_start = text.find("{")
    obj_end = text.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        return _as_str_list(json.loads(text[obj_start : obj_end + 1]))
    raise ValueError("No JSON array in model output")


def align_translations(out: list[str], texts: list[str], prev: str) -> list[str]:
    """Drop the extra string models emit when they also translate previous_cue."""
    n = len(texts)
    if len(out) == n:
        return out
    if len(out) == n + 1:
        if prev:
            return out[1:]
        return out[:n]
    if len(out) > n:
        return out[:n]
    raise ValueError(f"Ollama returned {len(out)} strings for {n} cues")


def ollama_tags(url: str, timeout: float = 5.0) -> dict | None:
    try:
        import httpx
    except ImportError as exc:
        raise SystemExit("httpx is required for Ollama translation") from exc
    try:
        r = httpx.get(f"{url.rstrip('/')}/api/tags", timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as exc:
        apple_device.log(f"Ollama health check failed: {exc}")
        return None


def pick_ollama_model(tags: dict, preferred: str) -> str:
    names = [m.get("name") for m in (tags.get("models") or []) if m.get("name")]
    if preferred in names:
        return preferred
    for name in names:
        if name.split(":")[0] == preferred.split(":")[0]:
            return name
    if names:
        apple_device.log(f"Preferred Ollama model {preferred} missing; using {names[0]}")
        return names[0]
    raise RuntimeError("Ollama is running but has no models")


def ollama_chat(
    url: str,
    model: str,
    system: str,
    user: str,
    texts: list[str],
    timeout: float,
    prev_blob: str = "",
) -> list[str]:
    import httpx

    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    last_exc: Exception | None = None
    for use_json_format in (True, False):
        body_payload = dict(payload)
        if use_json_format:
            body_payload["format"] = "json"
        try:
            r = httpx.post(
                f"{url.rstrip('/')}/api/chat",
                json=body_payload,
                timeout=timeout,
            )
            r.raise_for_status()
            body = r.json()
            content = (body.get("message") or {}).get("content") or ""
            raw = parse_json_array(content)
            out = align_translations(raw, texts, prev_blob)
            if any(not item for item in out):
                raise ValueError("Ollama returned an empty sentence")
            return out
        except Exception as exc:
            last_exc = exc
            continue
    raise last_exc or RuntimeError("Ollama chat failed")


def ollama_translate_batch(
    url: str,
    model: str,
    texts: list[str],
    context: list[tuple[str, str]],
    timeout: float,
) -> list[str]:
    if all(not t.strip() for t in texts):
        return [""] * len(texts)

    user_parts: list[str] = []
    if context:
        user_parts.append(
            "Previous sentences (read-only context, do not translate or output):\n"
            + json.dumps(
                [{"en": en, "cs": cs} for en, cs in context],
                ensure_ascii=False,
            )
        )
    user_parts.append(
        "Translate this JSON array of English sentences. Return a JSON array of "
        f"exactly {len(texts)} Czech strings:"
    )
    user_parts.append(json.dumps(texts, ensure_ascii=False))
    prev_blob = " ".join(cs for _, cs in context)
    return ollama_chat(
        url,
        model,
        SYSTEM_PROMPT,
        "\n\n".join(user_parts),
        texts,
        timeout,
        prev_blob=prev_blob,
    )


def ollama_revise_batch(
    url: str,
    model: str,
    english: list[str],
    drafts: list[str],
    timeout: float,
) -> list[str]:
    if all(not t.strip() for t in drafts):
        return list(drafts)
    pairs = [{"en": en, "cs": cs} for en, cs in zip(english, drafts)]
    user = (
        "Revise the Czech drafts. Each object has the English source (en) and "
        f"the Czech draft (cs). Return a JSON array of exactly {len(drafts)} "
        "revised Czech strings in the same order:\n"
        + json.dumps(pairs, ensure_ascii=False)
    )
    return ollama_chat(url, model, REVISE_PROMPT, user, drafts, timeout)


def ollama_translate_and_revise(
    url: str,
    model: str,
    texts: list[str],
    context: list[tuple[str, str]],
    timeout: float,
) -> list[str]:
    draft = ollama_translate_batch(url, model, texts, context, timeout)
    try:
        return ollama_revise_batch(url, model, texts, draft, timeout)
    except Exception as exc:
        apple_device.log(f"Ollama revise skipped ({exc}); keeping draft")
        return draft


def marian_translate(texts: list[str], model: str, device: str) -> list[str]:
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    apple_device.log(f"Marian MT model={model} device={device}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    mt_model = AutoModelForSeq2SeqLM.from_pretrained(model)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_device = torch.device(device)
    try:
        mt_model.to(torch_device)
    except Exception as exc:
        apple_device.log(f"Marian could not use {device} ({exc}); using CPU")
        torch_device = torch.device("cpu")
        mt_model.to(torch_device)
    mt_model.eval()

    out: list[str] = [""] * len(texts)
    work = [(i, t) for i, t in enumerate(texts) if t.strip()]
    batch = 8
    for start in range(0, len(work), batch):
        chunk = work[start : start + batch]
        encoded = tokenizer(
            [t for _, t in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        encoded = {k: v.to(torch_device) for k, v in encoded.items()}
        with torch.inference_mode():
            generated = mt_model.generate(**encoded, max_new_tokens=256)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(decoded) != len(chunk):
            raise RuntimeError("Marian batch size mismatch")
        for (idx, _), text in zip(chunk, decoded):
            out[idx] = text.strip()
    missing = [i for i, t in work if not out[i]]
    if missing:
        raise RuntimeError("Marian produced empty translations")
    return out


def translate_cues(
    texts: list[str],
    args: argparse.Namespace,
) -> list[str]:
    backend = args.translation_backend
    device = apple_device.device_str(args.device)

    def via_marian(subset: list[str] | None = None) -> list[str]:
        return marian_translate(subset if subset is not None else texts, args.marian_model, device)

    if backend == "marian":
        return via_marian()

    tags = ollama_tags(args.ollama_url)
    if tags is None:
        if backend == "ollama_only":
            raise SystemExit(f"Ollama is not reachable at {args.ollama_url}")
        apple_device.log("Ollama unavailable; falling back to Marian")
        return via_marian()

    model = pick_ollama_model(tags, args.ollama_model)
    apple_device.log(f"Translating with Ollama model={model} (grammar revise pass)")
    translated: list[str] = []
    bs = max(1, args.batch_size)
    ctx_n = CONTEXT_SENTENCES
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        ctx_en = texts[max(0, i - ctx_n) : i]
        ctx = list(zip(ctx_en, translated[-len(ctx_en) :] if ctx_en else []))
        try:
            translated.extend(
                ollama_translate_and_revise(
                    args.ollama_url, model, chunk, ctx, args.ollama_timeout
                )
            )
        except Exception as exc:
            apple_device.log(f"Ollama batch failed ({exc}); retrying per sentence")
            for j, cue_text in enumerate(chunk):
                one_en = texts[max(0, i + j - ctx_n) : i + j]
                one_ctx = list(
                    zip(one_en, translated[-len(one_en) :] if one_en else [])
                )
                try:
                    one = ollama_translate_and_revise(
                        args.ollama_url,
                        model,
                        [cue_text],
                        one_ctx,
                        args.ollama_timeout,
                    )
                    translated.append(one[0])
                except Exception as cue_exc:
                    if backend == "ollama_only":
                        raise SystemExit(
                            f"Ollama failed on sentence {i + j + 1}: {cue_exc}"
                        ) from cue_exc
                    apple_device.log(
                        f"Ollama failed on sentence {i + j + 1}; remaining use Marian"
                    )
                    rest = texts[len(translated) :]
                    translated.extend(via_marian(rest))
                    return translated
    if len(translated) != len(texts):
        raise RuntimeError("Translation count mismatch after Ollama")
    return translated


def english_cues(args: argparse.Namespace) -> list[srt.Subtitle]:
    if args.en_srt and Path(args.en_srt).is_file():
        try:
            cues = load_srt(args.en_srt, sentences=True)
            apple_device.log(f"Using English SRT as {len(cues)} sentences")
            return cues
        except ValueError as exc:
            apple_device.log(f"Ignoring unusable SRT {args.en_srt}: {exc}")
    if not args.audio or not Path(args.audio).is_file():
        raise SystemExit("Need --en-srt with cues or a readable --audio file")
    return transcribe_mlx(args.audio, args.whisper_model)


def main() -> int:
    args = parse_args()
    out_en = Path(args.out_en)
    out_cs = Path(args.out_cs)
    out_en.parent.mkdir(parents=True, exist_ok=True)
    out_cs.parent.mkdir(parents=True, exist_ok=True)

    cues = english_cues(args)
    save_srt(out_en, cues)
    texts = [c.content for c in cues]
    apple_device.log(f"Translating {len(texts)} sentences to Czech")
    czech = translate_cues(texts, args)
    if len(czech) != len(cues):
        raise SystemExit(
            f"Translation produced {len(czech)} cues, expected {len(cues)}"
        )

    cs_cues = [
        srt.Subtitle(index=c.index, start=c.start, end=c.end, content=t)
        for c, t in zip(cues, czech)
    ]
    save_srt(out_cs, cs_cues)
    apple_device.log(f"Wrote {out_en} and {out_cs}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
