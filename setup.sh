#!/usr/bin/env bash
# Creates .venv next to this script with CPU-only torch and all dependencies.
set -euo pipefail
cd "$(dirname "$0")"
command -v ffmpeg >/dev/null || { echo "ffmpeg is required (apt install ffmpeg / brew install ffmpeg)"; exit 1; }
unset PYTHONPATH                      # a global PYTHONPATH can leak foreign packages into the venv
PY=${PYTHON:-python3}
"$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q --index-url https://download.pytorch.org/whl/cpu torch
.venv/bin/pip install -q -r requirements.txt
echo "ok: run  .venv/bin/python listen.py VIDEO OUTDIR [--pack SOUNDS_DIR]"
