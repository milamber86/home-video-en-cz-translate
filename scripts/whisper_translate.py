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
import time
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

CONTEXT_SENTENCES = 6

SYSTEM_PROMPT = (
    "Překládej anglický dabing do přirozené mluvené češtiny. "
    "Ne doslovně: idiomy a slovní spojení nahraď tím, co by řekl rodilý mluvčí. "
    "Krátká replika, která jen opakuje konec předchozí věty, použij stejné české "
    "slova a začni velkým písmenem. "
    "Retorické 'Period.' je 'Tečka.' "
    "Rod vypravěče je v zadání — první osoba musí sedět v rodě. "
    "Neskloňuj špatně (video je střední). "
    "Nepřidávej vysvětlení. Předchozí věty v kontextu nepřekládej znovu. "
    "Vrať jen JSON pole českých vět, stejně dlouhé a ve stejném pořadí jako vstup."
)

TRANSLATEGEMMA_PREAMBLE = (
    "You are a professional English (en) to Czech (cs) translator. "
    "Your goal is to accurately convey the meaning and nuances of the original "
    "English text while adhering to Czech grammar, vocabulary, and cultural sensitivities. "
    "Write natural spoken Czech for a YouTube voice-over, not a word-for-word gloss. "
    "Short echo cues that repeat the previous line should reuse the same Czech words "
    "and start with a capital letter."
)

GENDER_PROMPT = (
    "Infer the first-person narrator's gender from this English voice-over. "
    "Use only a short quote of self-description ('I'm a woman', 'as a girl'), "
    "or he/she/him/her that clearly refers to the narrator. "
    "If there is no such quote, return unknown. Do not default to male. "
    "Do not guess from the topic or from a typical YouTuber. "
    "Return ONLY JSON: {\"speaker_gender\": \"male|female|unknown\", "
    "\"evidence\": \"quote or empty\"}"
)

GLOSSARY_PROMPT = (
    "List people mentioned in this English transcript. Do not translate "
    "organizations, laws, or common words into Czech. "
    "cs is the written name (correct obvious ASR typos, otherwise keep English). "
    "tts is a Czech phonetic spelling only when the English name would be misread. "
    "Return ONLY JSON: {\"speaker_gender\": \"male|female|unknown\", "
    "\"terms\": [{\"en\": \"\", \"cs\": \"\", \"kind\": \"person\", \"tts\": \"\"}]}"
)

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)
_LIST_KEYS = (
    "cues",
    "translations",
    "translated",
    "cs",
    "output",
    "result",
    "items",
)
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
    p.add_argument("--ollama-model", default="translategemma:12b")
    p.add_argument(
        "--marian-model",
        default="Helsinki-NLP/opus-mt-tc-big-en-ces_slk",
    )
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--ollama-timeout", type=float, default=240.0)
    p.add_argument(
        "--speaker-gender",
        choices=("auto", "male", "female", "unknown"),
        default="auto",
        help="Narrator gender for Czech agreement. auto infers from transcript, then vocals",
    )
    p.add_argument(
        "--glossary-out",
        default="",
        help="Write glossary JSON here (default: next to --out-cs)",
    )
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
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            return _as_str_list(lists[0])
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


def is_translategemma(model: str) -> bool:
    return "translategemma" in (model or "").lower()


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


def pick_chat_model(tags: dict, preferred: str) -> str | None:
    names = [m.get("name") for m in (tags.get("models") or []) if m.get("name")]
    ordered: list[str] = []
    if preferred and not is_translategemma(preferred):
        if preferred in names:
            ordered.append(preferred)
        for name in names:
            if name.split(":")[0] == preferred.split(":")[0] and not is_translategemma(name):
                ordered.append(name)
    for name in names:
        if not is_translategemma(name):
            ordered.append(name)
    seen: set[str] = set()
    for name in ordered:
        if name not in seen:
            seen.add(name)
            return name
    return None


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
        "options": {"temperature": 0.2},
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


def _near_dup_cs(left: str, right: str) -> bool:
    a = _bare(left)
    b = _bare(right)
    if not a or not b:
        return True
    if a == b or a.startswith(b) or b.startswith(a):
        return True
    sa, sb = set(a.split()), set(b.split())
    return bool(sa and sb) and len(sa & sb) / max(len(sa), len(sb)) >= 0.6


def clean_translategemma(
    text: str, context_cs: list[str] | None = None
) -> str:
    text = _FENCE_RE.sub("", (text or "").strip()).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'“”":
        text = text[1:-1].strip()
    for prefix in ("Czech:", "CS:", "Česky:", "Translation:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix) :].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if context_cs:
        banned = {_bare(item) for item in context_cs if item}
        lines = [ln for ln in lines if _bare(ln) not in banned]
    while len(lines) > 1 and _near_dup_cs(lines[0], lines[1]):
        keep = lines[0] if len(lines[0]) >= len(lines[1]) else lines[1]
        lines = [keep, *lines[2:]]
    return lines[0] if lines else ""


def local_cue_translation(text: str, prev_en: str, prev_cs: str) -> str | None:
    stripped = (text or "").strip()
    if _bare(stripped) == "period":
        return "Tečka."
    cur = _bare(stripped)
    if (
        not cur
        or len(cur.split()) != 1
        or not prev_en
        or not prev_cs
        or not _bare(prev_en).endswith(cur)
    ):
        return None
    words = re.findall(r"[^\s.,;:!?…]+", prev_cs)
    if not words:
        return None
    word = words[-1]
    ending = "." if stripped.endswith(".") else ""
    return word[0].upper() + word[1:] + ending


def ollama_plain(url: str, model: str, user: str, timeout: float) -> str:
    import httpx

    last_exc: Exception | None = None
    for attempt in range(1, 4):
        try:
            r = httpx.post(
                f"{url.rstrip('/')}/api/chat",
                json={
                    "model": model,
                    "stream": False,
                    "options": {"temperature": 0.1},
                    "messages": [{"role": "user", "content": user}],
                },
                timeout=httpx.Timeout(timeout, connect=15.0),
            )
            r.raise_for_status()
            text = clean_translategemma(
                ((r.json().get("message") or {}).get("content") or ""),
            )
            if not text:
                raise ValueError("Ollama returned empty translation")
            return text
        except (httpx.TimeoutException, httpx.TransportError, ValueError) as exc:
            last_exc = exc
            apple_device.log(f"Ollama request failed ({exc}); retry {attempt}/3")
            time.sleep(2 * attempt)
    raise last_exc or RuntimeError("Ollama request failed")


def translategemma_prompt(
    text: str, context: list[tuple[str, str]], gender: str
) -> str:
    parts = [TRANSLATEGEMMA_PREAMBLE]
    if gender == "female":
        parts.append(
            "The first-person narrator is a woman. Use feminine agreement in every "
            "first-person form: byla jsem, řekla jsem, chodívala jsem, malá. "
            "Never byl jsem, řekl jsem, chodíval jsem, malý, or 'jako kluk'. "
            "Example: 'When I was a kid, I used to go to church every Sunday.' → "
            "'Když jsem byla malá, chodívala jsem každou neděli do kostela.'"
        )
    elif gender == "male":
        parts.append(
            "The first-person narrator is a man. Use masculine agreement: "
            "byl jsem, řekl jsem, chodíval jsem. "
            "Example: 'When I was a kid, I used to go to church every Sunday.' → "
            "'Když jsem byl malý, chodíval jsem každou neděli do kostela.'"
        )
    if context:
        parts.append("Previous subtitle lines (do not retranslate them):")
        for en, cs in context[-3:]:
            parts.append(f"EN: {en}")
            parts.append(f"CS: {cs}")
    parts.append(
        "Produce only the Czech translation, without any additional explanations "
        "or commentary. Please translate the following English text into Czech:"
    )
    if gender == "female":
        parts.append("Remember: the speaker is a woman (ženský rod).")
    elif gender == "male":
        parts.append("Remember: the speaker is a man (mužský rod).")
    return "\n".join(parts) + "\n\n\n" + (text or "").strip()


def translategemma_translate(
    url: str,
    model: str,
    texts: list[str],
    timeout: float,
    gender: str,
    progress_path: Path | None = None,
) -> list[str]:
    out: list[str] = []
    if progress_path and progress_path.is_file():
        try:
            saved = json.loads(progress_path.read_text(encoding="utf-8"))
            if isinstance(saved, list) and all(isinstance(item, str) for item in saved):
                out = saved[: len(texts)]
                if out:
                    apple_device.log(
                        f"Resuming TranslateGemma at {len(out) + 1}/{len(texts)}"
                    )
        except Exception as exc:
            apple_device.log(f"Ignoring TranslateGemma progress ({exc})")
            out = []
    for i in range(len(out), len(texts)):
        text = texts[i]
        prev_en = texts[i - 1] if i else ""
        prev_cs = out[i - 1] if i else ""
        local = local_cue_translation(text, prev_en, prev_cs) if text.strip() else ""
        if not text.strip():
            out.append("")
        elif local:
            apple_device.log(f"TranslateGemma {i + 1}/{len(texts)} (local)")
            out.append(local)
        else:
            ctx = list(zip(texts[max(0, i - 3) : i], out[max(0, i - 3) : i]))
            apple_device.log(f"TranslateGemma {i + 1}/{len(texts)}")
            raw = ollama_plain(
                url, model, translategemma_prompt(text, ctx, gender), timeout
            )
            out.append(clean_translategemma(raw, [cs for _, cs in ctx]))
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8"
            )
    for i, text in enumerate(texts[: len(out)]):
        local = local_cue_translation(
            text, texts[i - 1] if i else "", out[i - 1] if i else ""
        )
        if local:
            out[i] = local
    return capitalize_echoes(texts, out)


def parse_json_value(text: str) -> object:
    text = _FENCE_RE.sub("", (text or "").strip())
    if not text:
        raise ValueError("Empty model output")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON in model output")


def ollama_json(url: str, model: str, system: str, user: str, timeout: float) -> object:
    import httpx

    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    r = httpx.post(f"{url.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    content = ((r.json().get("message") or {}).get("content") or "")
    return parse_json_value(content)


def speaker_instruction(gender: str) -> str:
    if gender == "female":
        return (
            "Vypravěč je žena. První osoba je v ženském rodě "
            "(byla jsem, řekla jsem, chodívala jsem, malá — ne byl jsem, ne jako kluk). "
            "Příklad: 'When I was a kid, I used to go to church every Sunday.' → "
            "'Když jsem byla malá, chodívala jsem každou neděli do kostela.'; "
            "'every Sunday.' → 'Každou neděli.'; "
            "'By now, I haven't been in years.' → 'Léta už tam nechodím.'"
        )
    if gender == "male":
        return (
            "Vypravěč je muž. První osoba je v mužském rodě "
            "(byl jsem, řekl jsem, chodíval jsem — ne řekla jsem). "
            "Oslovení diváka v mužském rodě jen tam, kde angličtina míří na muže; "
            "'you' obecně překládej neutrálně (vy/jste). "
            "Příklad: 'When I was a kid, I used to go to church every Sunday.' → "
            "'Když jsem byl malý, chodíval jsem každou neděli do kostela.'; "
            "'every Sunday.' → 'Každou neděli.'; "
            "'By now, I haven't been in years.' → 'Léta už tam nechodím.'"
        )
    return (
        "Rod vypravěče není jistý; drž ho konzistentní podle okolních vět."
    )


def empty_glossary(gender: str) -> dict:
    return {"speaker_gender": gender, "terms": []}


def detect_speaker_gender_llm(
    url: str, model: str, texts: list[str], timeout: float
) -> tuple[str, str]:
    blob = "\n".join(t for t in texts if t.strip())
    if len(blob) > 12000:
        blob = blob[:12000]
    data = ollama_json(url, model, GENDER_PROMPT, "Transcript:\n" + blob, timeout)
    if not isinstance(data, dict):
        return "unknown", ""
    gender = str(data.get("speaker_gender") or "").strip().lower()
    evidence = str(data.get("evidence") or "").strip()
    if gender in ("male", "female") and evidence:
        return gender, evidence
    return "unknown", evidence


def estimate_gender_from_audio(path: str) -> str:
    try:
        import librosa
        import numpy as np
    except Exception as exc:
        apple_device.log(f"Speaker gender from vocals skipped ({exc})")
        return "unknown"
    try:
        y, sr = librosa.load(path, sr=16000, mono=True, duration=45.0)
        f0, _, _ = librosa.pyin(y, fmin=75, fmax=300, sr=sr)
        voiced = f0[np.isfinite(f0)]
    except Exception as exc:
        apple_device.log(f"Speaker gender from vocals failed ({exc})")
        return "unknown"
    if voiced.size < 30:
        return "unknown"
    med = float(np.median(voiced))
    p75 = float(np.percentile(voiced, 75))
    apple_device.log(f"Speaker f0 median={med:.1f} Hz p75={p75:.1f} Hz")
    if med >= 160 or p75 >= 185:
        return "female"
    if med <= 145 and p75 <= 165:
        return "male"
    return "unknown"


def resolve_speaker_gender(
    args: argparse.Namespace, texts: list[str], tags: dict
) -> str:
    requested = (args.speaker_gender or "auto").strip().lower()
    if requested in ("male", "female"):
        apple_device.log(f"Speaker gender override={requested}")
        return requested
    audio_gender = "unknown"
    if args.audio and Path(args.audio).is_file():
        audio_gender = estimate_gender_from_audio(args.audio)
        apple_device.log(f"Speaker gender from vocals={audio_gender}")
    text_gender = "unknown"
    evidence = ""
    chat = pick_chat_model(tags, args.ollama_model)
    if chat:
        try:
            text_gender, evidence = detect_speaker_gender_llm(
                args.ollama_url, chat, texts, args.ollama_timeout
            )
            extra = f" evidence={evidence!r}" if evidence else ""
            apple_device.log(
                f"Speaker gender from transcript model={chat} "
                f"gender={text_gender}{extra}"
            )
        except Exception as exc:
            apple_device.log(f"Speaker gender from transcript skipped ({exc})")
    if audio_gender in ("male", "female"):
        if text_gender in ("male", "female") and text_gender != audio_gender:
            apple_device.log(
                f"Speaker gender conflict text={text_gender} vocals={audio_gender}; "
                "using vocals"
            )
        return audio_gender
    if text_gender in ("male", "female"):
        return text_gender
    return "unknown"


def extract_glossary(
    url: str,
    model: str,
    texts: list[str],
    timeout: float,
    fallback_gender: str,
) -> dict:
    blob = "\n".join(t for t in texts if t.strip())
    if len(blob) > 24000:
        blob = blob[:24000]
    try:
        data = ollama_json(
            url,
            model,
            GLOSSARY_PROMPT,
            "Transcript:\n" + blob,
            timeout,
        )
    except Exception as exc:
        apple_device.log(f"Glossary extraction skipped ({exc})")
        return empty_glossary(fallback_gender)
    if not isinstance(data, dict):
        return empty_glossary(fallback_gender)
    terms = data.get("terms")
    people = []
    if isinstance(terms, list):
        for item in terms:
            if not isinstance(item, dict):
                continue
            kind = str(item.get("kind") or "person").strip().lower()
            if kind not in ("", "person"):
                continue
            en = str(item.get("en") or "").strip()
            if not en:
                continue
            people.append(
                {
                    "en": en,
                    "cs": str(item.get("cs") or en).strip() or en,
                    "kind": "person",
                    "tts": str(item.get("tts") or "").strip(),
                }
            )
    data["terms"] = people
    inferred = str(data.get("speaker_gender") or "").strip().lower()
    if inferred not in ("male", "female", "unknown"):
        inferred = "unknown"
    if fallback_gender in ("male", "female"):
        data["speaker_gender"] = fallback_gender
    else:
        data["speaker_gender"] = inferred
    apple_device.log(
        f"Glossary terms={len(data.get('terms') or [])} "
        f"speaker_gender={data['speaker_gender']}"
    )
    return data


def ollama_translate_batch(
    url: str,
    model: str,
    texts: list[str],
    context: list[tuple[str, str]],
    timeout: float,
    *,
    gender: str,
) -> list[str]:
    if all(not t.strip() for t in texts):
        return [""] * len(texts)

    user_parts: list[str] = [speaker_instruction(gender)]
    if context:
        user_parts.append(
            "Previous sentences (read-only context, do not translate or output):\n"
            + json.dumps(
                [{"en": en, "cs": cs} for en, cs in context],
                ensure_ascii=False,
            )
        )
    user_parts.append(
        "Translate this JSON array of English sentences into idiomatic spoken Czech. "
        f"Return a JSON array of exactly {len(texts)} Czech strings:"
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


def _bare(text: str) -> str:
    return re.sub(r"[.!?…]+$", "", (text or "").strip()).casefold()


def capitalize_echoes(english: list[str], czech: list[str]) -> list[str]:
    """Echo cues ('every Sunday.') should not stay lowercase fragments."""
    out = list(czech)
    for i in range(1, min(len(english), len(out))):
        cur = _bare(english[i])
        prev = _bare(english[i - 1])
        if not cur or len(cur.split()) > 5 or not prev.endswith(cur):
            continue
        text = (out[i] or "").strip()
        if text and text[0].islower():
            out[i] = text[0].upper() + text[1:]
    return out


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
    glossary_path: Path | None = None,
) -> list[str]:
    backend = args.translation_backend
    device = apple_device.device_str(args.device)
    gender = (args.speaker_gender or "auto").strip().lower()
    glossary = empty_glossary(gender)

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

    gender = resolve_speaker_gender(args, texts, tags)
    glossary = empty_glossary(gender)
    model = pick_ollama_model(tags, args.ollama_model)
    if is_translategemma(model):
        apple_device.log(
            f"Translating with TranslateGemma model={model} speaker_gender={gender}"
        )
        if glossary_path is not None:
            glossary_path.parent.mkdir(parents=True, exist_ok=True)
            glossary_path.write_text(
                json.dumps(glossary, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        progress = (
            glossary_path.with_name("cs.translategemma.json")
            if glossary_path is not None
            else None
        )
        return translategemma_translate(
            args.ollama_url,
            model,
            texts,
            args.ollama_timeout,
            gender,
            progress_path=progress,
        )
    apple_device.log(f"Translating with Ollama model={model} (idiomatic Czech)")
    chat = pick_chat_model(tags, model) or model
    glossary = extract_glossary(
        args.ollama_url, chat, texts, args.ollama_timeout, gender
    )
    if gender in ("male", "female"):
        glossary["speaker_gender"] = gender
    else:
        gender = str(glossary.get("speaker_gender") or "unknown")
        glossary["speaker_gender"] = gender
    if glossary_path is not None:
        glossary_path.parent.mkdir(parents=True, exist_ok=True)
        glossary_path.write_text(
            json.dumps(glossary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        apple_device.log(f"Wrote {glossary_path}")

    translated: list[str] = []
    bs = max(1, args.batch_size)
    ctx_n = CONTEXT_SENTENCES
    for i in range(0, len(texts), bs):
        chunk = texts[i : i + bs]
        ctx_en = texts[max(0, i - ctx_n) : i]
        ctx = list(zip(ctx_en, translated[-len(ctx_en) :] if ctx_en else []))
        try:
            translated.extend(
                ollama_translate_batch(
                    args.ollama_url,
                    model,
                    chunk,
                    ctx,
                    args.ollama_timeout,
                    gender=gender,
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
                    one = ollama_translate_batch(
                        args.ollama_url,
                        model,
                        [cue_text],
                        one_ctx,
                        args.ollama_timeout,
                        gender=gender,
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
    return capitalize_echoes(texts, translated)


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
    glossary_path = Path(args.glossary_out) if args.glossary_out else out_cs.with_name("glossary.json")
    apple_device.log(f"Translating {len(texts)} sentences to Czech")
    czech = translate_cues(texts, args, glossary_path=glossary_path)
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
