---
name: listen-video
description: Hear a video's soundtrack - speech with timestamps, meme/SFX sounds, music, hits, pauses, and exact matches from your own sound pack. Use to break down someone's reels or check your own edit by ear.
---

# listen-video

Turns a video's audio into a per-second timeline (`report.md`, `report.json`, `timeline_XX.png`).
Code: https://github.com/levizdev/listen-video

## Setup (once per machine/sandbox)
```bash
git clone https://github.com/levizdev/listen-video ~/listen-video && cd ~/listen-video && bash setup.sh
```
- Needs Python 3.10+, ffmpeg and access to huggingface.co (first run downloads ~1 GB of models to `$HF_HOME`, default `~/hf`). After that set `HF_HUB_OFFLINE=1`.
- If pip fails with `Permission denied` on some system path, a global `PYTHONPATH` is leaking in: `unset PYTHONPATH` (setup.sh does it).
- Install takes 3-8 min; run it in the background if your shell has a command timeout.

## Run
```bash
cd ~/listen-video && .venv/bin/python listen.py VIDEO OUTDIR [--pack SOUNDS_DIR] [--lang ru] [--no-asr] [--vad]
```
- ~1.5-2 min of CPU per 30 s of video (4 cores). For anything longer than ~30 s run in the background and poll.
- `--pack` = folder with the user's sound effects; reports exactly where each file is used.
- Only the pack search (seconds, no models): `.venv/bin/python fp.py VIDEO SOUNDS_DIR`.
- Spectrogram PNGs are large; downscale before viewing (`convert in.png -resize 1200x -quality 70 out.jpg`).

## Read the result
Read `OUTDIR/report.md` first (one row per second: speech · AudioSet sounds 🎵=music · CLAP guess · pack match · ● hit · ⏸ silence).
- Trust: pack match (exact file) > AudioSet event with p >= 0.4 > CLAP guess. Mark CLAP-only findings as "probably".
- Words in (parentheses) have confidence < 0.3 — usually Whisper hallucinating on songs. Sung lyrics are not the author's speech.
- A CLAP guess with `?` was made over speech; CLAP often mistakes a voice for "bruh"/"glass breaking". Treat as unconfirmed.
- An SFX much quieter than the voice is invisible to AudioSet (it answers "Speech"). Absence in the report is not proof of absence — but it also means a phone viewer probably won't hear it either.
- Patterns worth reporting: a silence right before a hit/meme sound ("drop before the punchline"), how often the music changes, what plays at the start and the end, the same sound combo repeating across videos (a running gag), SFX density per minute.
- For the user's own edit: check each planned SFX lands on its second, the voice is not buried under music, no accidental silences.

## Limits (tell the user)
- CLAP labels are guesses, not proof.
- An exact meme-sound name is found only if that file is in the pack; pitched/sped-up versions are not matched.
- Song titles are not identified (use Shazam/songrec).
