# Local EN→CS video translation pipeline

Ansible-orchestrated pipeline for Apple Silicon (M2 Ultra). It downloads a YouTube video, separates vocals from the bed, translates English subtitles to Czech, synthesizes timed Czech speech with F5-TTS, and remuxes video + background + dubbed voice + Czech SRT.

All AI libraries run in a dedicated Python venv. Ansible itself uses system Python (`connection: local`).

## Prerequisites

- macOS on Apple Silicon (`arm64`, not Rosetta)
- [Homebrew](https://brew.sh/)
- [Ansible](https://docs.ansible.com/) (`brew install ansible`)

Ollama is installed and started by the playbook. A Czech F5-TTS checkpoint is optional (see training below).

## Setup

```bash
ansible-galaxy collection install -r requirements.yml
```

## Translate a video

```bash
ansible-playbook site.yml -e youtube_url='https://www.youtube.com/watch?v=VIDEO_ID'
```

Useful extra-vars:

| Variable | Default | Notes |
|---|---|---|
| `voice_mode` | `clone` | `clone` extracts a reference clip from the original vocals; `bundled` uses `files/voices/czech_default_ref.wav` |
| `output_container` | `mkv` | `mkv` (native SRT) or `mp4` (`mov_text`) |
| `ollama_model` | `qwen2.5:14b` | Must exist after the `ollama` role pulls it |
| `ytdlp_cookies_from_browser` | `""` | e.g. `chrome` if YouTube returns 429 |

Tags: `setup`, `ollama`, `download`, `demucs`, `whisper_translate`, `f5_tts`, `remux`.

Install and start Ollama only:

```bash
ansible-playbook site.yml --tags setup,ollama
```

Skip environment setup on later runs:

```bash
ansible-playbook site.yml --skip-tags setup -e youtube_url='...'
```

Output lands in `work/<youtube_id>/output/<youtube_id>.cs.mkv` (or `.mp4`).

## Translation

The `whisper_translate` role prefers Ollama at `http://127.0.0.1:11434`. If the API is down, JSON is malformed, or cue counts mismatch, it falls back to Marian (`Helsinki-NLP/opus-mt-tc-big-en-ces_slk`) on MPS.

## Czech F5-TTS (optional, hours-long)

Official F5-TTS is ZH+EN. Until you train a local checkpoint, inference uses the base model plus grapheme fixes (`ů→ú`, `ď→d`) and pronunciation will be imperfect.

Fine-tune from `F5TTS_v1_Base` (never from scratch):

```bash
ansible-playbook train_czech_tts.yml
```

Defaults: Common Voice 17 Czech, 20 hours of 1–12 s clips, **CPU** training (MPS training is opt-in and can produce silent audio). Overnight-scale on M2 Ultra. After success, `models/f5_czech/model_last.safetensors` is picked up automatically by the `f5_tts` role.

```bash
ansible-playbook train_czech_tts.yml -e f5_train_device=mps -e f5_train_force=true
```

## Device policy

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before any `import torch`.
- Demucs, Marian, and F5-TTS inference use `torch.device("mps")` when available.
- Whisper uses **mlx-whisper** on Metal (`mlx-community/whisper-large-v3-mlx`). `openai-whisper` + MPS is unreliable; `faster-whisper` is CPU-only on Mac.

## Layout

```
roles/          Ansible roles
scripts/        Python entry points invoked by roles
files/voices/   Optional bundled Czech reference WAV + transcript
work/<id>/      Per-video artifacts
models/f5_czech/  Trained Czech F5 checkpoint (gitignored)
```
