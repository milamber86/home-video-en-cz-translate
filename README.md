# Local EN→CS video translation pipeline

Ansible-orchestrated pipeline for Apple Silicon (M2 Ultra). It downloads a YouTube video, separates vocals from the bed, translates English subtitles to Czech (sentence-level), synthesizes timed Czech speech, and remuxes video + background + dubbed voice + Czech SRT.

All AI libraries run in a dedicated Python venv. Ansible itself uses system Python (`connection: local`). Coqui TTS (XTTS-v2 clone baseline, female VITS) uses a second venv (`.venv-xtts`) because Coqui pins an older `transformers` than the main pipeline; Pocket TTS Czech (the default clone engine) uses a third (`.venv-pocket`).

## Prerequisites

- macOS on Apple Silicon (`arm64`). Do not run the pipeline Python env under Rosetta.
- [Ansible](https://docs.ansible.com/). The controller may be a Rosetta venv; modules are forced to arm64 via `scripts/ansible_python`.

The **setup** role installs Apple Silicon Homebrew at `/opt/homebrew` if it is missing (Intel `/usr/local` brew can coexist and is ignored). Creating `/opt/homebrew` needs sudo once:

```bash
ansible-galaxy collection install -r requirements.yml
ansible-playbook site.yml -K -e youtube_url='https://www.youtube.com/watch?v=VIDEO_ID'
```

`-K` prompts for the macOS administrator password. Later runs can omit `-K` once `/opt/homebrew` exists.

If `ansible-playbook` fails in **Gathering Facts** with `xcrun` / `libxcrun` / `need 'x86_64'`, the controller is Rosetta and was using `/usr/bin/python3`. This repo wraps the module interpreter in `scripts/ansible_python`.

Ollama is installed and started by the playbook. A Czech F5-TTS checkpoint is optional (see training below).

## Setup

```bash
ansible-galaxy collection install -r requirements.yml
```

## Translate a video

```bash
ansible-playbook site.yml -K -e youtube_url='https://www.youtube.com/watch?v=VIDEO_ID'
```

Useful extra-vars:

| Variable | Default | Notes |
|---|---|---|
| `voice_mode` | `clone` | Clone the source speaker. `auto` uses pretrained XTTS-v2 when those weights are cached. `bundled` is stock Piper/VITS (or Czech F5 if you set `tts_engine=f5` and add `files/voices/czech_default_ref.wav`). |
| `tts_voice_gender` | `male` | `male` = Piper `cs_CZ-jirka-medium`. `female` = Coqui `tts_models/cs/cv/vits`. Ignored for XTTS clone. |
| `tts_engine` | `auto` | Prefers Pocket TTS Czech when `.venv-pocket` exists and a reference is available; then pretrained XTTS-v2 when `model.pth` is cached (comparison baseline). Else Czech F5 if `models/f5_czech` exists. Else Piper/VITS by gender. |
| `speaker_gender` | `auto` | Narrator gender for Czech agreement. `auto` infers from the transcript, then from `vocals.wav` pitch. Override with `male` or `female` |
| `force_translate` | `false` | Redo `subs/en.srt` and `subs/cs.srt` even if they exist. Reuses `subs/whisper.en.srt` unless you delete it or pass `--force-asr` |
| `reflow_enabled` | `true` | Secondary pass that compresses Czech cues which cannot fit their slot |
| `reflow_dry_run` | `false` | Report `subs/reflow.json` without rewriting `cs.srt` |
| `reflow_chat_model` | `qwen2.5:14b` | Chat model used for compression |
| `reflow_chars_per_sec` | `14.0` | Spoken-Czech rate used to predict cue duration; raise it if your voice is faster |
| `tts_qa_attempts` | `3` | Max takes per cue in the ASR verify-and-correct loop (fidelity over speed) |
| `tts_qa_cer` | `0.15` | Accepted character error rate between synthesized speech and its source text |
| `force_tts` | `false` | Redo Czech vocals (wipes `tts/segments`) and remux |
| `output_container` | `mkv` | `mkv` (native SRT) or `mp4` (`mov_text`) |
| `ollama_model` | `translategemma:12b` | Pulled by the `ollama` role. Dedicated EN→CS. `translategemma:27b` is stronger; `qwen2.5:14b` is a general-chat fallback |
| `tts_base_speed` | `1.15` | Minimum pace for every Czech cue (Czech is longer than English) |
| `tts_max_speed` | `1.25` | Cap when a cue still overruns its English slot. Keep close to `tts_base_speed` to avoid jumps |
| `ytdlp_cookies_from_browser` | `""` | e.g. `chrome` if YouTube returns 429 or 403 even with Deno |

Tags: `setup`, `ollama`, `download`, `demucs`, `whisper_translate`, `f5_tts`, `remux`.

Install and start Ollama only:

```bash
ansible-playbook site.yml -K --tags setup,ollama
```

Skip environment setup on later runs:

```bash
ansible-playbook site.yml --skip-tags setup -e youtube_url='...'
```

Redo translation + TTS + remux (keep the download and Demucs stems):

```bash
ansible-playbook site.yml --skip-tags setup,ollama,download,demucs \
  -e youtube_url='https://www.youtube.com/watch?v=VIDEO_ID' \
  -e force_translate=true -e force_tts=true
```

Female stock voice:

```bash
ansible-playbook site.yml --skip-tags setup,ollama,download,demucs \
  -e youtube_url='https://www.youtube.com/watch?v=VIDEO_ID' \
  -e tts_voice_gender=female -e force_tts=true
```

Output lands in `work/<youtube_id>/output/<youtube_id>.cs.mkv` (or `.mp4`).

## Translation

English comes from **mlx-whisper on `vocals.wav` only** (cached as `subs/whisper.en.srt`). YouTube captions are not merged; they were shifting times and mixing speakers. Leftover karaoke repeats are still dropped. Speaker gender is inferred (`auto`: vocals pitch first, then the transcript if it has a clear quote) and passed into every translate prompt so Czech first-person agreement stays feminine or masculine. Override with `-e speaker_gender=female` or `male`. At TTS time, numbers and abbreviations are expanded into spoken Czech. If Ollama is down or a general-chat model returns bad JSON, it falls back to Marian (`Helsinki-NLP/opus-mt-tc-big-en-ces_slk`) on MPS.

The default translator is [TranslateGemma](https://ai.google.dev/gemma/docs/translategemma/model-card) 12B (~8 GB): dedicated EN→CS, official plain-text prompt, one cue at a time with a short previous-line context and the detected speaker gender. `translategemma:27b` is stronger (~17 GB). `qwen2.5:14b` is a general-chat fallback (`-e ollama_model=qwen2.5:14b`). A second rewrite pass is not used: it made the Czech more literal.

The `ollama` role pulls `translategemma:12b` if it is missing.

**Reflow pass (`roles/reflow`, after translation):** Czech runs ~15% longer than English, so some cues cannot fit their slot even at `tts_max_speed`. The reflow stage measures the time each cue may actually use (from the isolated `vocals.wav`, so leading/trailing silence inside a cue counts), predicts the spoken duration at `reflow_chars_per_sec`, and asks the chat model (`reflow_chat_model`, default `qwen2.5:14b`) to compress only the lines that would otherwise be sped up past the cap or overlap the next cue. A line is accepted only when the compressed form fits; otherwise it is retried once with a stricter instruction, and the original is kept if that still fails. The pristine translation is preserved as `subs/cs.orig.srt` (the pass always reflows from it, so it is idempotent and re-runs automatically when translation changes), and `subs/reflow.json` records per-cue slots, estimates, and before/after speeds. Inspect without touching `cs.srt` via `-e reflow_dry_run=true`; disable with `-e reflow_enabled=false`.

**Fine-tuning (Inkling + Tinker):** Possible later, not the first move. Inkling is a huge generalist MoE; Tinker is a cloud LoRA API. Fine-tuning needs thousands of *human* EN–CS spoken-dub pairs. The current `cs.srt` is model output — training on it would lock in today's mistakes. If you later collect gold cues, SFT TranslateGemma or a mid-size Qwen on Tinker beats SFT Inkling. NLLB / MADLAD / Marian are dedicated MT but more literal than a dub needs.

**Highest quality if you leave local-only:** DeepL or a strong cloud LLM with a tight spoken-Czech prompt.

## Czech speech

Official F5-TTS (`F5TTS_v1_Base`) is Chinese+English only and is **not** used for Czech inference.

| Condition | Engine |
|---|---|
| `.venv-pocket` exists (setup creates it) and a reference is available | **Pocket TTS Czech** (Kyutai, `vvolhejn/pocket-tts-czech`) — 100M-param cloning model, faster than realtime on CPU, MIT/CC-BY-4.0 |
| Pocket venv absent, pretrained XTTS-v2 cached (`~/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2/model.pth`) | XTTS-v2 clones the narrator from a clean vocal clip; guest lines use that line’s vocals (or a guest ref). Kept as the comparison baseline |
| No Pocket/XTTS weights, Czech F5 checkpoint in `models/f5_czech` | Fine-tuned Czech F5. Clear with a Czech reference; English `vocals.wav` as the prompt is mostly unintelligible |
| None of the above, `tts_voice_gender=male` | Piper `cs_CZ-jirka-medium` (setup downloads the ONNX into `models/piper/`) |
| `tts_voice_gender=female` | Coqui Czech Common Voice VITS (`tts_models/cs/cv/vits`) in `.venv-xtts` |

Setup prefetches Pocket TTS weights into `models/pocket/` and XTTS-v2. Force an engine with `-e tts_engine=pocket` (or `xtts`, `f5`, `piper`, `vits`). There is no official female Piper Czech voice. VITS is weaker than Jirka.

Every cue goes through a **verify-and-correct loop** (`tts_qa`, default on). Each take is transcribed with mlx-whisper (the same model family as the transcription step) and compared against the spoken source text on four signals: character error rate (`--qa-cer`, default 0.15), whole-word omissions, a truncated tail (the missing-last-word/syllable signature), and implausibly short audio (silent ending). Discrepancies trigger regeneration with a different sampling temperature per attempt (up to `tts_qa_attempts`, default 3) and the best take wins; trimming trailing silence along the aligned words still kills appended echo/reference artifacts, and trimming never cuts into voiced audio (which is what used to clip final syllables). If attempts still leave whole words missing at the end, the missing ending is re-synthesized and appended (revert-protected). Disable with `tts_qa=false` or `--no-qa`. XTTS tails additionally get the energy-based trim (trailing silence and the short echo burst the model often appends). Pocket TTS is a sentence-level model and is driven one sentence per call.

Dubbed lines are placed against the **original** speaker, not the subtitle edge: when `vocals.wav` is available, its energy profile gives the speech onset and offset inside each cue, so the Czech audio starts when the source speech starts (within ~0.1 s) and its slot is the real spoken span plus a bounded share of the silence that follows — never reaching into the next cue's speech. Every cue is still time-stretched by `tts_base_speed` (default 1.15×) so Czech can keep up with English; a line that overruns only goes up to `tts_max_speed` (1.25×), and the reflow pass shortens the text before that happens. Without `vocals.wav`, the fallback is the cue duration plus gap spill (up to 70% of the silence before the next cue). Clone refs stay inside each cue's timestamps (no 4s bleed into the next speaker). Narrator lines share one ref; guest lines use that guest's audio.

## Czech F5-TTS (optional, hours-long)

Fine-tune from `F5TTS_v1_Base` (never from scratch):

```bash
ansible-playbook train_czech_tts.yml -K
```

Defaults: VoxPopuli Czech (`facebook/voxpopuli`, config `cs`; ~62 transcribed hours), 20 hours of 1–12 s clips, **CPU** training (MPS training is opt-in and can produce silent audio). Overnight-scale on M2 Ultra. After success, `models/f5_czech/model_last.pt` is used in `auto` only when Pocket TTS and pretrained XTTS-v2 are unavailable, or always when you set `tts_engine=f5`.

Mozilla Common Voice is no longer hosted on Hugging Face (Mozilla Data Collective as of October 2025). To train on a CV tarball you downloaded yourself, unpack it to `metadata.csv` + `wavs/` and pass `-e f5_train_local_dir=/path/to/that/dir`.

```bash
ansible-playbook train_czech_tts.yml -e f5_train_device=mps -e f5_train_force=true
```

Then:

```bash
ansible-playbook site.yml --skip-tags setup,ollama,download,demucs \
  -e youtube_url='https://www.youtube.com/watch?v=VIDEO_ID' \
  -e force_tts=true
```

## Device policy

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before any `import torch`.
- Demucs, Marian, and Czech F5 inference use `torch.device("mps")` when available.
- Pocket TTS Czech (default clone) runs on CPU in `.venv-pocket`; Coqui XTTS-v2 (baseline) and VITS run on CPU in `.venv-xtts`.
- Whisper uses **mlx-whisper** on Metal (`mlx-community/whisper-large-v3-mlx`). `openai-whisper` + MPS is unreliable; `faster-whisper` is CPU-only on Mac.

## Layout

```
roles/          Ansible roles
scripts/        Python entry points invoked by roles
files/voices/   Optional bundled F5 reference WAV + transcript (only if a Czech F5 ckpt exists)
work/<id>/      Per-video artifacts (subs/cs.orig.srt + subs/reflow.json from the reflow pass)
models/piper/   Piper cs_CZ-jirka-medium (gitignored)
models/f5_czech/  Trained Czech F5 checkpoint (gitignored)
.venv-xtts/     Isolated Coqui env for XTTS-v2 and VITS (gitignored)
.venv-pocket/   Isolated Kyutai Pocket TTS env (gitignored)
models/pocket/  Pocket TTS prefetch marker (gitignored)
```

## Development

Pipeline Python dependencies are declared in `pyproject.toml` and pinned in `uv.lock`; the `setup` role installs exactly the locked versions with `uv sync --frozen`. After changing dependencies, run `uv lock` and commit `uv.lock`. The training playbook still editable-installs `vendor/F5-TTS` on demand; a later `uv sync` reverts the pipeline venv to the locked wheel.

Tests and lint only need the light dependencies (`pytest`, `ruff`, `mypy`, `srt`, `num2words`, `numpy`, `soundfile`):

```bash
pytest
ruff check scripts tests
mypy scripts
```

CI (`.github/workflows/ci.yml`) runs the test suite, ruff, mypy, `ansible-playbook --syntax-check` for both playbooks, and verifies `uv.lock` is current.
