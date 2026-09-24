# Local EN→CS video translation pipeline

Ansible-orchestrated pipeline for Apple Silicon (M2 Ultra). It downloads a YouTube video, separates vocals from the bed, translates English subtitles to Czech (sentence-level), synthesizes timed Czech speech, and remuxes video + background + dubbed voice + Czech SRT.

All AI libraries run in a dedicated Python venv. Ansible itself uses system Python (`connection: local`). Coqui TTS (XTTS-v2 clone, female VITS) uses a second venv (`.venv-xtts`) because Coqui pins an older `transformers` than the main pipeline.

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
| `tts_engine` | `auto` | Prefers pretrained XTTS-v2 (Czech included) when `model.pth` is cached. Else Czech F5 if `models/f5_czech` exists. Else Piper/VITS by gender. |
| `speaker_gender` | `auto` | Narrator gender for Czech agreement. `auto` infers from the transcript, then from `vocals.wav` pitch. Override with `male` or `female` |
| `force_translate` | `false` | Redo `subs/en.srt` and `subs/cs.srt` even if they exist |
| `force_tts` | `false` | Redo Czech vocals (wipes `tts/segments`) and remux |
| `output_container` | `mkv` | `mkv` (native SRT) or `mp4` (`mov_text`) |
| `ollama_model` | `translategemma:12b` | Pulled by the `ollama` role. Dedicated EN→CS. `translategemma:27b` is stronger; `qwen2.5:14b` is a general-chat fallback |
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

English cues are packed into **complete sentences** (YouTube rolling captions are unrolled first). Speaker gender is inferred (`auto`: vocals pitch first, then the transcript if it has a clear quote) and passed into every translate prompt so Czech first-person agreement stays feminine or masculine. Override with `-e speaker_gender=female` or `male`. At TTS time, numbers and abbreviations are expanded into spoken Czech. If Ollama is down or a general-chat model returns bad JSON, it falls back to Marian (`Helsinki-NLP/opus-mt-tc-big-en-ces_slk`) on MPS.

The default translator is [TranslateGemma](https://ai.google.dev/gemma/docs/translategemma/model-card) 12B (~8 GB): dedicated EN→CS, official plain-text prompt, one cue at a time with a short previous-line context and the detected speaker gender. `translategemma:27b` is stronger (~17 GB). `qwen2.5:14b` is a general-chat fallback (`-e ollama_model=qwen2.5:14b`). A second rewrite pass is not used: it made the Czech more literal.

The `ollama` role pulls `translategemma:12b` if it is missing.

**Fine-tuning (Inkling + Tinker):** Possible later, not the first move. Inkling is a huge generalist MoE; Tinker is a cloud LoRA API. Fine-tuning needs thousands of *human* EN–CS spoken-dub pairs. The current `cs.srt` is model output — training on it would lock in today's mistakes. If you later collect gold cues, SFT TranslateGemma or a mid-size Qwen on Tinker beats SFT Inkling. NLLB / MADLAD / Marian are dedicated MT but more literal than a dub needs.

**Highest quality if you leave local-only:** DeepL or a strong cloud LLM with a tight spoken-Czech prompt.

## Czech speech

Official F5-TTS (`F5TTS_v1_Base`) is Chinese+English only and is **not** used for Czech inference.

| Condition | Engine |
|---|---|
| Pretrained XTTS-v2 cached (`~/Library/Application Support/tts/tts_models--multilingual--multi-dataset--xtts_v2/model.pth`) | XTTS-v2 clones `vocals.wav` into Czech (`tts_engine=auto`) |
| No XTTS weights, Czech F5 checkpoint in `models/f5_czech` | Fine-tuned Czech F5. Clear with a Czech reference; English `vocals.wav` as the prompt is mostly unintelligible |
| Neither of the above, `tts_voice_gender=male` | Piper `cs_CZ-jirka-medium` (setup downloads the ONNX into `models/piper/`) |
| `tts_voice_gender=female` | Coqui Czech Common Voice VITS (`tts_models/cs/cv/vits`) in `.venv-xtts` |

Setup prefetches XTTS-v2. Force F5 with `-e tts_engine=f5`. There is no official female Piper Czech voice. VITS is weaker than Jirka.

## Czech F5-TTS (optional, hours-long)

Fine-tune from `F5TTS_v1_Base` (never from scratch):

```bash
ansible-playbook train_czech_tts.yml -K
```

Defaults: VoxPopuli Czech (`facebook/voxpopuli`, config `cs`; ~62 transcribed hours), 20 hours of 1–12 s clips, **CPU** training (MPS training is opt-in and can produce silent audio). Overnight-scale on M2 Ultra. After success, `models/f5_czech/model_last.pt` is used only when pretrained XTTS-v2 is missing or you set `tts_engine=f5`.

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
- Coqui XTTS-v2 (default clone) and VITS run on CPU in `.venv-xtts`.
- Whisper uses **mlx-whisper** on Metal (`mlx-community/whisper-large-v3-mlx`). `openai-whisper` + MPS is unreliable; `faster-whisper` is CPU-only on Mac.

## Layout

```
roles/          Ansible roles
scripts/        Python entry points invoked by roles
files/voices/   Optional bundled F5 reference WAV + transcript (only if a Czech F5 ckpt exists)
work/<id>/      Per-video artifacts
models/piper/   Piper cs_CZ-jirka-medium (gitignored)
models/f5_czech/  Trained Czech F5 checkpoint (gitignored)
.venv-xtts/     Isolated Coqui env for XTTS-v2 and VITS (gitignored)
```
