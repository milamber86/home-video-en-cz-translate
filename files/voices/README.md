# Bundled Czech voice reference

Used only when `voice_mode=bundled`.

Place two files here:

1. `czech_default_ref.wav` — 5–12 seconds of clean, mono-or-stereo speech, ~24 kHz or 48 kHz PCM/WAV. Leave a short trailing silence.
2. `czech_default_ref.txt` — exact transcript of that clip (UTF-8). Do not leave it empty; F5-TTS otherwise loads a second Whisper model.

Clone-from-original (`voice_mode=clone`, the default) does not need these files.

Do not commit copyrighted voice recordings.
