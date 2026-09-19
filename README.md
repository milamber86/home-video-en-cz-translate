# Local EN→CS video translation pipeline

Ansible-orchestrated pipeline for Apple Silicon (M2 Ultra). It downloads a YouTube video, separates vocals from the bed, translates English subtitles to Czech (sentence-level), synthesizes timed Czech speech, and remuxes video + background + dubbed voice + Czech SRT.

All AI libraries run in a dedicated Python venv. Ansible itself uses system Python (`connection: local`). XTTS-v2 uses a second venv (`.venv-xtts`) because Coqui pins an older `transformers` than the main pipeline.

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
| `voice_mode` | `clone` | `clone` uses XTTS-v2 with a clip from original vocals; `bundled` uses Piper `cs_CZ-jirka-medium` |
| `tts_engine` | `auto` | `auto` = Czech F5 checkpoint if present, else XTTS for clone / Piper for bundled. Never uses ZH+EN F5-base for Czech. |
| `force_translate` | `false` | Redo `subs/en.srt` and `subs/cs.srt` even if they exist |
| `force_tts` | `false` | Redo Czech vocals and remux |
| `output_container` | `mkv` | `mkv` (native SRT) or `mp4` (`mov_text`) |
| `ollama_model` | `qwen2.5:14b` | Must exist after the `ollama` role pulls it |
| `ytdlp_cookies_from_browser` | `""` | e.g. `chrome` if YouTube returns 429 |

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

Output lands in `work/<youtube_id>/output/<youtube_id>.cs.mkv` (or `.mp4`).

First XTTS run downloads `xtts_v2` weights (~1.8 GB) into the Hugging Face cache.

## Translation

English cues are packed into **complete sentences** (YouTube rolling captions are unrolled first). Ollama translates those sentences with the previous 1–2 English+Czech sentences as read-only context. If the API is down, JSON is malformed, or counts mismatch, it falls back to Marian (`Helsinki-NLP/opus-mt-tc-big-en-ces_slk`) on MPS.

## Czech speech

Official F5-TTS (`F5TTS_v1_Base`) is Chinese+English only and is **not** used for Czech inference.

| Condition | Engine |
|---|---|
| `models/f5_czech/model_last.safetensors` (or `.pt`) + `vocab.txt` | Fine-tuned Czech F5 |
| `voice_mode=clone` (default) | XTTS-v2, `language=cs`, speaker from `vocals.wav` (CPU, isolated `.venv-xtts`) |
| `voice_mode=bundled` | Piper `cs_CZ-jirka-medium` (setup downloads the ONNX into `models/piper/`) |

## Czech F5-TTS (optional, hours-long)

Fine-tune from `F5TTS_v1_Base` (never from scratch):

```bash
ansible-playbook train_czech_tts.yml -K
```

Defaults: Common Voice 17 Czech, 20 hours of 1–12 s clips, **CPU** training (MPS training is opt-in and can produce silent audio). Overnight-scale on M2 Ultra. After success, `models/f5_czech/model_last.safetensors` is picked up automatically by the `f5_tts` role.

```bash
ansible-playbook train_czech_tts.yml -e f5_train_device=mps -e f5_train_force=true
```

## Device policy

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before any `import torch`.
- Demucs, Marian, and Czech F5 inference use `torch.device("mps")` when available.
- XTTS-v2 runs on CPU in `.venv-xtts`.
- Whisper uses **mlx-whisper** on Metal (`mlx-community/whisper-large-v3-mlx`). `openai-whisper` + MPS is unreliable; `faster-whisper` is CPU-only on Mac.

## Layout

```
roles/          Ansible roles
scripts/        Python entry points invoked by roles
files/voices/   Optional bundled F5 reference WAV + transcript (only if a Czech F5 ckpt exists)
work/<id>/      Per-video artifacts
models/piper/   Piper cs_CZ-jirka-medium (gitignored)
models/f5_czech/  Trained Czech F5 checkpoint (gitignored)
.venv-xtts/     Isolated Coqui / XTTS-v2 env (gitignored)
```
