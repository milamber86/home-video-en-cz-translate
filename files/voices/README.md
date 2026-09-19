# Bundled Czech voice reference

Used only when `voice_mode=bundled` **and** a Czech F5 checkpoint exists under `models/f5_czech/`.

Without that checkpoint, `voice_mode=bundled` uses Piper (`cs_CZ-jirka-medium`) and these files are unused.

When a Czech F5 checkpoint is present, place two files here:

1. `czech_default_ref.wav` — 5–12 seconds of clean, mono-or-stereo speech, ~24 kHz or 48 kHz PCM/WAV. Leave a short trailing silence.
2. `czech_default_ref.txt` — exact transcript of that clip (UTF-8). Do not leave it empty; F5-TTS otherwise loads a second Whisper model.

Clone-from-original (`voice_mode=clone`, the default) uses XTTS-v2 (or Czech F5 if trained) and does not need these files.

Do not commit copyrighted voice recordings.
