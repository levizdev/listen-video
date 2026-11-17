"""Stress test for fp.py: plant pack sounds into real soundtracks and measure recall / false positives.

    python tests/stress_fp.py PACK_DIR BG1.mp4 [BG2.mp4 ...] [--seed 1] [--grid]

For every background (first 60 s) and gain (0, -6, -12, -18 dB relative to the pack file) every pack
sound (first 4 s of it) is planted at a random free spot, then the mix is checked as is and after an
AAC 64 kbit/s round trip (what Instagram/TikTok compression roughly does). Case "x8": one short sound
repeated 8 times in one clip (a running gag).
A detection counts if the right sound is reported within 0.3 s of where it was planted.
"False positives (clean)" = matches reported on the untouched backgrounds — must be 0 unless the
background really contains a pack sound.
"""
import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import fp  # noqa: E402

SR = fp.SR


def aac(y, br='64k'):
    enc = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(SR), '-ac', '1', '-i', '-',
                          '-c:a', 'aac', '-b:a', br, '-f', 'adts', '-'], input=y.astype(np.float32).tobytes(),
                         capture_output=True).stdout
    dec = subprocess.run(['ffmpeg', '-v', 'error', '-i', '-', '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                         input=enc, capture_output=True).stdout
    return np.frombuffer(dec, np.float32).copy()


def make_cases(pack, bgs, gains, seed):
    raw = {}
    for root, _, files in os.walk(pack):
        for f in files:
            if f.lower().endswith(fp.AUDIO_EXT):
                y = fp._trim(fp.load(os.path.join(root, f)))[0]
                if len(y) > SR * 0.05:
                    raw[os.path.splitext(f)[0]] = y
    rnd = random.Random(seed)
    cases = []
    for bname, bg in bgs.items():
        for g in gains:
            y = bg[:60 * SR].copy()
            planted, t = [], 1.0 + rnd.uniform(0, 1)
            order = sorted(raw)
            rnd.shuffle(order)
            for n in order:
                s = raw[n][:4 * SR] * 10 ** (g / 20)
                i = int(t * SR)
                if i + len(s) >= len(y):
                    break
                y[i:i + len(s)] += s
                planted.append((n, round(t, 3)))
                t += len(s) / SR + rnd.uniform(0.3, 1.5)
            cases.append((bname, g, 'raw', y, planted))
            cases.append((bname, g, 'aac64', aac(y), planted))
        # the same sound used many times (a running gag): 8 x one short sound, 0 dB
        n = rnd.choice([k for k in sorted(raw) if len(raw[k]) < 3 * SR])
        y, planted = bg[:60 * SR].copy(), []
        for j in range(8):
            t = 1.0 + j * (len(y) / SR - 4) / 8 + rnd.uniform(0, 1)
            i = int(t * SR)
            s = raw[n][:min(len(raw[n]), len(y) - i)]
            y[i:i + len(s)] += s
            planted.append((n, round(t, 3)))
        cases.append((bname, 'x8', 'aac64', aac(y), planted))
    return cases


def evaluate(pack, bgs, cases):
    DB, info = fp.build(pack)
    rec, fps, missed = {}, 0, {}
    for bname, g, enc, y, planted in cases:
        m = fp.match(y, DB, info)
        ok = [n for n, t in planted if any(o['sound'] == n and abs(o['t'] - t) < 0.3 for o in m)]
        bad = [o for o in m if not any(o['sound'] == n and abs(o['t'] - t) < 0.3 for n, t in planted)]
        fps += len(bad)
        rec.setdefault(g, []).append(len(ok) / max(1, len(planted)))
        for n, _ in planted:
            if n not in ok:
                missed[(g, n)] = missed.get((g, n), 0) + 1
    clean = []
    for bname, bg in bgs.items():
        clean += [(bname, o['sound'], o['t'], o['score']) for o in fp.match(bg, DB, info)]
    return {g: round(float(np.mean(v)), 3) for g, v in rec.items()}, fps, clean, missed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pack')
    ap.add_argument('bgs', nargs='+')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--grid', action='store_true', help='also try other peak/threshold settings')
    a = ap.parse_args()
    bgs = {os.path.basename(b): fp.load(b) for b in a.bgs}
    gains = [0, -6, -12, -18]
    cases = make_cases(a.pack, bgs, gains, a.seed)
    print(f'{len(cases)} mixes, {sum(len(c[4]) for c in cases)} planted sounds')
    grid = [None]
    if a.grid:
        grid += [dict(PEAK_F=f, PEAK_T=t, PEAK_PCT=p, FAN=fan, MIN_ABS=ma, K_NOISE=k)
                 for f, t, p, fan in [(15, 7, 80, 10), (11, 5, 75, 10), (9, 5, 70, 10), (11, 5, 70, 15), (9, 3, 70, 15)]
                 for ma, k in [(8, 3.0), (8, 4.0), (8, 5.0), (8, 6.0)]]
    defaults = {k: getattr(fp, k) for k in ('PEAK_F', 'PEAK_T', 'PEAK_PCT', 'FAN', 'MIN_ABS', 'K_NOISE', 'NOISE_PCT')}
    for cfg in grid:
        for k, v in {**defaults, **(cfg or {})}.items():
            setattr(fp, k, v)
        fp.match.__defaults__ = (fp.MIN_ABS, fp.K_NOISE, 2)
        t0 = time.time()
        rec, fps, clean, missed = evaluate(a.pack, bgs, cases)
        label = 'default' if cfg is None else cfg
        print(f'{label}\n  recall by gain dB: {rec}  false+ in mixes: {fps}  false+ on clean: {len(clean)}'
              f'  ({time.time() - t0:.0f}s)', flush=True)
        if cfg is None:
            if clean:
                print('  clean matches:', clean[:10])
            worst = sorted(missed.items(), key=lambda kv: (str(kv[0][0]), -kv[1]))
            print('  missed (gain, sound): count ->', worst[:20])
    for k, v in defaults.items():
        setattr(fp, k, v)


if __name__ == '__main__':
    main()
