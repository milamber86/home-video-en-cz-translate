# Bundled Czech F5 voice reference

These files are used only when you explicitly run Czech F5 with `tts_engine=f5`
and `voice_mode=bundled`. The default is `voice_mode=clone` with `tts_engine=auto`:
pretrained XTTS-v2 (Czech included) clones `vocals.wav` when those weights are cached.

When a Czech F5 checkpoint is present and you want F5 without cloning the source
video, place two files here:

1. `czech_default_ref.wav` — 5–12 seconds of clean, mono-or-stereo speech, ~24 kHz or 48 kHz PCM/WAV. Leave a short trailing silence.
2. `czech_default_ref.txt` — exact transcript of that clip (UTF-8). Do not leave it empty; F5-TTS otherwise loads a second Whisper model.

Clone-from-original (`voice_mode=clone`) uses pretrained XTTS-v2 plus a clip from
`vocals.wav`. If you force `tts_engine=f5` in clone mode, the clip is transcribed
with Whisper (do not point `--ref-text` at `czech_default_ref.txt`). Without
cached XTTS weights, clone falls back to Czech F5 if present, else Piper/VITS.

Do not commit copyrighted voice recordings.
