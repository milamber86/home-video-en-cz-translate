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

CONTEXT_SENTENCES = 8
POLISH_WINDOW = 10

SYSTEM_PROMPT = (
    "Jsi profesionální audiovizuální překladatel do češtiny. "
    "Piš přirozenou mluvenou češtinu pro dabing YouTube výkladu, ne doslovný kalk. "
    "You translate consecutive subtitle sentences from the same talk. "
    "Keep roughly the same length as the source and finish each thought. "
    "Gramatika: shoda v rodě, čísle a pádě; český slovosled (příklonky, genitiv po číslovce). "
    "Rod podstatných jmen dodržuj (video = střední: toto video; teorie = ženský; "
    "systém = mužský; cena = ženský). "
    "First-person narrator gender is given below; já/řeknu/vysvětlím must agree. "
    "Prefer established Czech wording over awkward synonyms "
    "(frowned upon → nahlíženo s despektem / považováno za neakademické, "
    "not 'pohledávána'; distanced themselves → distancovali se, not 'vzdálili se'; "
    "CEO → generální ředitel). "
    "Složité anglické souvětí přestav do české větné stavby, nekalkuj slovosled. "
    "Use the supplied glossary consistently; fix obvious ASR name typos "
    "(Freriedman→Friedman, Noble→Nobel when the prize is meant). "
    "Do not add explanations, numbering, or timestamps. "
    "Previous sentences given as context are read-only: do not translate or "
    "repeat them in the output. "
    "Return ONLY a JSON array of Czech strings with exactly the same length "
    "and order as the input sentences array."
)

POLISH_PROMPT = (
    "Jsi rodilý český jazykový redaktor dabingu. "
    "Přepiš každou českou větu tak, jak by ji řekl rodilý mluvčí ve výkladovém videu. "
    "Oprav rod, číslo, pád a shodu s vypravěčem. "
    "Odstraň anglické kalky a neohrabané vazby. "
    "Nahraď významově vedlejší synonyma ustáleným českým výrazem. "
    "Složité konstrukce zjednoduš do přirozené češtiny, význam neměň a nic nepřidávej. "
    "Délka zůstane zhruba stejná. Dodrž slovníček. "
    "Příklady špatně→správně: 'této názoru'→'tohoto názoru'; "
    "'současnou ekonomickou systémem'→'současným ekonomickým systémem'; "
    "'tato videí'→'toto video'; 'Jeich strategie'→'Jejich strategie'; "
    "'předseda výkonného výboru'→'generální ředitel' pokud zdroj říká CEO. "
    "Return ONLY a JSON array of Czech strings with exactly the same length "
    "and order as the input drafts."
)

GLOSSARY_PROMPT = (
    "From the English YouTube transcript extract a terminology glossary for "
    "Czech translation. Correct obvious ASR misspellings of names. "
    "For each person, org, recurring term, or abbreviation give the Czech "
    "written form and, when TTS would mispronounce an English name, a Czech "
    "phonetic spelling (tts). "
    "Return ONLY JSON: {\"speaker_gender\": \"male|female|unknown\", "
    "\"terms\": [{\"en\": \"\", \"cs\": \"\", \"kind\": "
    "\"person|org|place|term|abbr\", \"gender\": \"m|f|n|\", \"tts\": \"\"}]}"
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
    p.add_argument("--batch-size", type=int, default=6)
    p.add_argument("--ollama-timeout", type=float, default=240.0)
    p.add_argument(
        "--speaker-gender",
        choices=("male", "female", "unknown"),
        default="male",
        help="Gender of the first-person narrator for Czech agreement",
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
            "(řekla jsem, vysvětlím, byla jsem — ne řekl jsem)."
        )
    if gender == "male":
        return (
            "Vypravěč je muž. První osoba je v mužském rodě "
            "(řekl jsem, vysvětlím, byl jsem — ne řekla jsem). "
            "Oslovení diváka v mužském rodě jen tam, kde angličtina míří na muže; "
            "'you' obecně překládej neutrálně (vy/jste)."
        )
    return (
        "Rod vypravěče není jistý; drž ho konzistentní podle slovníčku "
        "a okolních vět."
    )


def format_glossary(glossary: dict | None) -> str:
    if not glossary:
        return ""
    terms = glossary.get("terms") if isinstance(glossary, dict) else None
    if not isinstance(terms, list) or not terms:
        return ""
    lines = []
    for item in terms:
        if not isinstance(item, dict):
            continue
        en = (item.get("en") or "").strip()
        cs = (item.get("cs") or "").strip()
        if not en or not cs:
            continue
        extra = []
        kind = (item.get("kind") or "").strip()
        gender = (item.get("gender") or "").strip()
        if kind:
            extra.append(kind)
        if gender:
            extra.append(f"rod {gender}")
        suffix = f" ({', '.join(extra)})" if extra else ""
        lines.append(f"- {en} → {cs}{suffix}")
    if not lines:
        return ""
    return "Slovníček, dodržuj konzistentně:\n" + "\n".join(lines)


def empty_glossary(gender: str) -> dict:
    return {"speaker_gender": gender, "terms": []}


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
    if not isinstance(terms, list):
        data["terms"] = []
    inferred = str(data.get("speaker_gender") or "").strip().lower()
    if fallback_gender in ("male", "female"):
        data["speaker_gender"] = fallback_gender
    elif inferred in ("male", "female", "unknown"):
        data["speaker_gender"] = inferred
    else:
        data["speaker_gender"] = "unknown"
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
    glossary: dict | None,
) -> list[str]:
    if all(not t.strip() for t in texts):
        return [""] * len(texts)

    user_parts: list[str] = [speaker_instruction(gender)]
    gloss = format_glossary(glossary)
    if gloss:
        user_parts.append(gloss)
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


def ollama_polish_window(
    url: str,
    model: str,
    english: list[str],
    drafts: list[str],
    prior_cs: list[str],
    timeout: float,
    *,
    gender: str,
    glossary: dict | None,
) -> list[str]:
    if all(not t.strip() for t in drafts):
        return list(drafts)
    user_parts: list[str] = [speaker_instruction(gender)]
    gloss = format_glossary(glossary)
    if gloss:
        user_parts.append(gloss)
    if prior_cs:
        user_parts.append(
            "Already-finalized previous Czech (read-only, do not output):\n"
            + json.dumps(prior_cs, ensure_ascii=False)
        )
    pairs = [{"en": en, "cs": cs} for en, cs in zip(english, drafts)]
    user_parts.append(
        "Rewrite the Czech drafts as native spoken Czech. Each object has the "
        f"English source (en) and the Czech draft (cs). Return a JSON array of "
        f"exactly {len(drafts)} revised Czech strings in the same order:\n"
        + json.dumps(pairs, ensure_ascii=False)
    )
    return ollama_chat(
        url, model, POLISH_PROMPT, "\n\n".join(user_parts), drafts, timeout
    )


def polish_document(
    url: str,
    model: str,
    english: list[str],
    drafts: list[str],
    timeout: float,
    *,
    gender: str,
    glossary: dict | None,
) -> list[str]:
    out = list(drafts)
    window = max(1, POLISH_WINDOW)
    for i in range(0, len(out), window):
        chunk_en = english[i : i + window]
        chunk_cs = out[i : i + window]
        prior = out[max(0, i - CONTEXT_SENTENCES) : i]
        try:
            revised = ollama_polish_window(
                url,
                model,
                chunk_en,
                chunk_cs,
                prior,
                timeout,
                gender=gender,
                glossary=glossary,
            )
            out[i : i + window] = revised
        except Exception as exc:
            apple_device.log(f"Polish window {i + 1}-{i + len(chunk_cs)} skipped ({exc})")
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
    gender = (args.speaker_gender or "male").strip().lower()
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

    model = pick_ollama_model(tags, args.ollama_model)
    apple_device.log(f"Translating with Ollama model={model} (glossary + native polish)")
    glossary = extract_glossary(
        args.ollama_url, model, texts, args.ollama_timeout, gender
    )
    gender = str(glossary.get("speaker_gender") or gender)
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
                    glossary=glossary,
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
                        glossary=glossary,
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
    apple_device.log("Polishing Czech as a native voice-over")
    return polish_document(
        args.ollama_url,
        model,
        texts,
        translated,
        args.ollama_timeout,
        gender=gender,
        glossary=glossary,
    )


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
