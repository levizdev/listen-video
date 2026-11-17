#!/usr/bin/env bash
# Edge cases: every case must finish with "ok" and valid JSON. Uses only generated media (no downloads).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
T=$(mktemp -d); fail=0
ff() { ffmpeg -v error -y "$@"; }
mkdir -p "$T/pack/sub" && touch "$T/pack/readme.txt"
ff -f lavfi -i "anoisesrc=d=0.3:c=white:a=0.9:seed=3,highpass=f=4000" "$T/pack/beep.wav"   # a loud hiss (pure tones are NOT fingerprintable, see README)
ff -f lavfi -i "anoisesrc=d=0.6:c=pink:a=0.8,highpass=f=2000" "$T/pack/sub/шум кириллица.mp3"
# 1 normal: noise bed + planted hiss at 3.0 s and noise burst at 6.0 s
ff -f lavfi -i "anoisesrc=d=10:c=brown:a=0.05:seed=1" -i "$T/pack/beep.wav" -i "$T/pack/sub/шум кириллица.mp3" \
   -filter_complex "[1]adelay=3000:all=1[a];[2]adelay=6000:all=1[b];[0][a][b]amix=3:normalize=0" "$T/mix.wav"
ff -f lavfi -i "color=c=black:s=320x240:d=10" -i "$T/mix.wav" -shortest -c:v libx264 -c:a aac "$T/normal video.mp4"
ff -f lavfi -i "color=c=black:s=320x240:d=3" -c:v libx264 "$T/noaudio.mp4"             # 2 no audio stream
ff -f lavfi -i "anullsrc=r=44100:cl=mono" -t 5 "$T/silent.wav"                           # 3 digital silence
ff -f lavfi -i "sine=f=440:d=0.5" "$T/short.mp3"                                          # 4 shorter than a window
cp "$T/mix.wav" "$T/аудио файл.wav"                                                       # 5 audio only, cyrillic + space
run() { # file, extra args
  out="$T/out_$(basename "$1" | tr ' ' _)"
  if "$PY" listen.py "$1" "$out" --no-png "${@:2}" >"$out.log" 2>&1 && "$PY" -c "import json,sys; json.load(open(sys.argv[1]))" "$out/report.json"; then
    echo "PASS $(basename "$1"): $(tail -1 "$out.log")"
  else echo "FAIL $(basename "$1")"; tail -5 "$out.log"; fail=1; fi
}
run "$T/normal video.mp4" --pack "$T/pack" --no-asr
"$PY" - "$T/out_normal_video.mp4/report.json" <<'P' || fail=1
import json, sys
r = json.load(open(sys.argv[1])); m = {(x['sound'], round(x['t'])) for x in r['pack_matches']}
ok = ('beep', 3) in m and ('шум кириллица', 6) in m
print(('PASS' if ok else 'FAIL') + ' pack matches:', r['pack_matches']); sys.exit(0 if ok else 1)
P
run "$T/noaudio.mp4"
run "$T/silent.wav" --no-asr
"$PY" -c "import json,sys; r=json.load(open(sys.argv[1])); s=r['silences']; ok=bool(s) and s[0][1]-s[0][0]>4; print(('PASS' if ok else 'FAIL')+' silence covers clip:', s); sys.exit(0 if ok else 1)" "$T/out_silent.wav/report.json" || fail=1
run "$T/short.mp3" --no-asr
run "$T/аудио файл.wav" --lang ru
"$PY" listen.py "$T/missing.mp4" "$T/x" >/dev/null 2>&1 && { echo "FAIL missing file accepted"; fail=1; } || echo "PASS missing file rejected"
rm -rf "$T"; exit $fail
