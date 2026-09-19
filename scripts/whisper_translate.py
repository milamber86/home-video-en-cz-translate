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
from srtutil import load_srt, save_srt, segments_to_cues  # noqa: E402

SYSTEM_PROMPT = (
    "You are a professional audiovisual translator from English to Czech. "
    "Translate each subtitle cue into natural, spoken Czech. "
    "Keep roughly the same length as the source. "
    "Do not add explanations, numbering, or timestamps. "
    "Return ONLY a JSON array of strings, same length and order as the input array."
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


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


def parse_json_array(text: str) -> list[str]:
    text = _FENCE_RE.sub("", (text or "").strip())
    start = text.find("[")
    end = text.rfind("]")
    if start < 0 or end <= start:
        raise ValueError("No JSON array in model output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, list):
        raise ValueError("JSON value is not a list")
    return [str(item).strip() for item in data]


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


def ollama_translate_batch(
    url: str,
    model: str,
    texts: list[str],
    prev: str,
    timeout: float,
) -> list[str]:
    import httpx

    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {"previous_cue": prev, "cues": texts},
                    ensure_ascii=False,
                ),
            },
        ],
    }
    r = httpx.post(f"{url.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    body = r.json()
    content = (body.get("message") or {}).get("content") or ""
    out = parse_json_array(content)
    if len(out) != len(texts):
        raise ValueError(f"Ollama returned {len(out)} strings for {len(texts)} cues")
    if any(not item for item in out):
        raise ValueError("Ollama returned an empty cue")
    return out


def marian_translate(texts: list[str], model: str, device: str) -> list[str]:
    from transformers import pipeline

    apple_device.log(f"Marian MT model={model} device={device}")
    mt = pipeline("translation", model=model, device=device)
    out: list[str] = []
    batch = 16
    for i in range(0, len(texts), batch):
        chunk = texts[i : i + batch]
        results = mt(chunk, max_length=256, truncation=True)
        for item in results:
            out.append((item.get("translation_text") or "").strip())
    if len(out) != len(texts) or any(not t for t in out):
        raise RuntimeError("Marian produced empty or mismatched translations")
    return out


def translate_cues(
    texts: list[str],
    args: argparse.Namespace,
) -> list[str]:
    backend = args.translation_backend
    device = apple_device.device_str(args.device)

    def via_marian() -> list[str]:
        return marian_translate(texts, args.marian_model, device)

    if backend == "marian":
        return via_marian()

    tags = ollama_tags(args.ollama_url)
    if tags is None:
        if backend == "ollama_only":
            raise SystemExit(f"Ollama is not reachable at {args.ollama_url}")
        apple_device.log("Ollama unavailable; falling back to Marian")
        return via_marian()

    model = pick_ollama_model(tags, args.ollama_model)
    apple_device.log(f"Translating with Ollama model={model}")
    translated: list[str] = []
    bs = max(1, args.batch_size)
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        prev = translated[-1] if translated else (texts[i - 1] if i else "")
        try:
            translated.extend(
                ollama_translate_batch(
                    args.ollama_url, model, chunk, prev, args.ollama_timeout
                )
            )
        except Exception as exc:
            apple_device.log(f"Ollama batch failed ({exc}); retrying per cue")
            for j, cue_text in enumerate(chunk):
                cue_prev = translated[-1] if translated else prev
                try:
                    one = ollama_translate_batch(
                        args.ollama_url,
                        model,
                        [cue_text],
                        cue_prev,
                        args.ollama_timeout,
                    )
                    translated.append(one[0])
                except Exception as cue_exc:
                    if backend == "ollama_only":
                        raise SystemExit(
                            f"Ollama failed on cue {i + j + 1}: {cue_exc}"
                        ) from cue_exc
                    apple_device.log(
                        f"Ollama failed on cue {i + j + 1}; remaining cues use Marian"
                    )
                    rest = texts[len(translated) :]
                    translated.extend(
                        marian_translate(rest, args.marian_model, device)
                    )
                    return translated
    if len(translated) != len(texts):
        raise RuntimeError("Translation count mismatch after Ollama")
    return translated


def main() -> int:
    args = parse_args()
    out_en = Path(args.out_en)
    out_cs = Path(args.out_cs)
    out_en.parent.mkdir(parents=True, exist_ok=True)
    out_cs.parent.mkdir(parents=True, exist_ok=True)

    cues: list[srt.Subtitle] | None = None
    if args.en_srt and Path(args.en_srt).is_file():
        try:
            cues = load_srt(args.en_srt)
            apple_device.log(f"Using downloaded English SRT ({len(cues)} cues)")
        except ValueError as exc:
            apple_device.log(f"Ignoring unusable SRT {args.en_srt}: {exc}")

    if cues is None:
        if not args.audio or not Path(args.audio).is_file():
            raise SystemExit("Need --en-srt with cues or a readable --audio file")
        cues = transcribe_mlx(args.audio, args.whisper_model)

    save_srt(out_en, cues)
    texts = [c.content for c in cues]
    apple_device.log(f"Translating {len(texts)} cues to Czech")
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
