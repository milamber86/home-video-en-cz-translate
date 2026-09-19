# Bundled Czech F5 voice reference

These files are used only when you explicitly run Czech F5 with `voice_mode=bundled`
and a checkpoint exists under `models/f5_czech/`. Stock TTS (`voice_mode=bundled`,
the default) uses Piper (male) or Coqui VITS (female) and ignores this directory.

When a Czech F5 checkpoint is present and you want F5 without cloning the source
video, place two files here:

1. `czech_default_ref.wav` — 5–12 seconds of clean, mono-or-stereo speech, ~24 kHz or 48 kHz PCM/WAV. Leave a short trailing silence.
2. `czech_default_ref.txt` — exact transcript of that clip (UTF-8). Do not leave it empty; F5-TTS otherwise loads a second Whisper model.

Clone-from-original (`voice_mode=clone`) uses the Czech F5 checkpoint plus a clip
from `vocals.wav`. Without that checkpoint, clone falls back to the stock Piper/VITS
voice for `tts_voice_gender`.

Do not commit copyrighted voice recordings.
