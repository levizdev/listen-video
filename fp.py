"""fp.py — Shazam-style landmark fingerprinting: find exact sounds from a reference pack inside a mix.

Usage as a script:  python fp.py MIX_FILE PACK_DIR
Works under speech/music, volume changes and lossy re-encoding (AAC/MP3).
Does NOT survive pitch-shift or speed change of the sound (fingerprints are absolute in frequency and time).
"""
import os
import subprocess
from collections import Counter, defaultdict

import numpy as np
from scipy.ndimage import maximum_filter

SR, NF, HOP = 16000, 1024, 256          # 16 ms frames
FPS = SR / HOP
PEAK_F, PEAK_T, PEAK_PCT, FAN = 11, 5, 75, 10   # peak neighbourhood (bins x frames), level percentile, pairs per peak
MIN_ABS, K_NOISE, NOISE_PCT = 8, 5.0, 97   # detection thresholds; all tuned with tests/stress_fp.py
AUDIO_EXT = ('.mp3', '.wav', '.m4a', '.ogg', '.aac', '.flac', '.opus', '.mp4', '.mov', '.webm')


def load(path, sr=SR):
    """Decode any audio/video file to mono float32 at `sr`. Returns empty array if there is no audio."""
    out = subprocess.run(['ffmpeg', '-v', 'error', '-i', path, '-vn', '-ac', '1', '-ar', str(sr), '-f', 'f32le', '-'],
                         capture_output=True)
    if out.returncode != 0 and not out.stdout:
        probe = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=index',
                                '-of', 'csv=p=0', path], capture_output=True)
        if probe.returncode == 0 and not probe.stdout.strip():
            return np.zeros(0, np.float32)            # file is fine, it just has no audio track
        raise RuntimeError(f'ffmpeg could not decode {path}: {out.stderr.decode(errors="ignore")[-300:]}')
    return np.frombuffer(out.stdout, np.float32).copy()


def peaks(y):
    import librosa
    if len(y) < NF:
        y = np.pad(y, (0, NF - len(y)))
    S = np.abs(librosa.stft(y, n_fft=NF, hop_length=HOP))
    L = np.log(S + 1e-6)
    mx = maximum_filter(L, size=(PEAK_F, PEAK_T))
    thr = np.percentile(L, PEAK_PCT)
    f, t = np.where((L == mx) & (L > thr))
    return sorted(zip(t.tolist(), f.tolist()))


def hashes(pk, fan=None, dtmax=48, dfmax=80):
    """Pair each peak with up to `fan` later peaks -> ((f1, f2, dt), t1)."""
    fan = fan or FAN
    H = []
    n = len(pk)
    for i, (t1, f1) in enumerate(pk):
        c = 0
        for j in range(i + 1, n):
            t2, f2 = pk[j]
            dt = t2 - t1
            if dt < 1:
                continue
            if dt > dtmax:
                break
            if abs(f2 - f1) > dfmax:
                continue
            H.append(((f1, f2, dt), t1))
            c += 1
            if c >= fan:
                break
    return H


def _trim(y):
    """Cut leading silence (relative to the sound's own peak). Returns (trimmed, offset_seconds)."""
    if len(y) == 0:
        return y, 0.0
    thr = max(1e-4, 0.05 * float(np.abs(y).max()))
    idx = np.where(np.abs(y) > thr)[0]
    if len(idx) == 0:
        return y[:0], 0.0
    return y[idx[0]:], idx[0] / SR


def build(pack_dir):
    """Index every audio file in pack_dir (recursively). Name = file name without extension."""
    DB, info = defaultdict(list), {}
    for root, _, files in os.walk(pack_dir):
        for fn in sorted(files):
            if not fn.lower().endswith(AUDIO_EXT):
                continue
            y, off = _trim(load(os.path.join(root, fn)))
            if len(y) < SR * 0.05:
                continue
            H = hashes(peaks(y))
            if len(H) < 5:
                continue
            name = os.path.splitext(fn)[0]
            if name in info:                      # same name in two folders -> keep both
                name = os.path.relpath(os.path.join(root, name), pack_dir)
            info[name] = {'n': len(H), 'off': round(off, 3), 'dur': round(len(y) / SR, 3)}
            for h, t in H:
                DB[h].append((name, t))
    return dict(DB), info


def match(y, DB, info, min_abs=MIN_ABS, k=K_NOISE, tol=2):
    """Find occurrences of pack sounds in mix `y`.

    For each pack sound, count hash hits per time offset (mix_frame - pack_frame). A real occurrence piles up
    on one offset; chance hits are spread thin. Counts within ±tol frames are pooled.
    Peaks are taken strongest-first; a peak is accepted when hits >= max(min_abs, k * noise), where noise is
    the NOISE_PCT-th percentile (+1) of the pooled counts over all other offsets (zeros included), excluding this
    peak's region and occurrences already accepted. We do NOT require a fraction of the sound's hashes:
    editors often use only the first second of a long sound.
    Offsets closer than the sound's own duration to an accepted occurrence are skipped (a repetitive
    sound — snoring, a loop — otherwise matches itself at its own period).
    Returns [{'t': start in the mix (s), 'sound', 'hits', 'frac', 'score'}] sorted by time.
    frac = hits / hashes in the whole pack sound (low frac = only part of the sound was used or it is buried).
    """
    per = defaultdict(Counter)
    for h, tq in hashes(peaks(y)):
        for name, tr in DB.get(h, ()):
            per[name][tq - tr] += 1
    n_mix = len(y) // HOP + 1
    out = []
    for name, C in per.items():
        n = info[name]['n']
        n_snd = int(info[name]['dur'] * FPS) + 1
        lo = -n_snd
        dense = np.zeros(n_mix + n_snd + 1)
        for o, c in C.items():
            if lo <= o <= n_mix:
                dense[o - lo] += c
        pooled = np.convolve(dense, np.ones(2 * tol + 1), mode='same')
        sep = max(int(FPS), n_snd)
        # Accept peaks strongest-first. Each candidate is compared with the noise floor of everything
        # else (its own region and already accepted occurrences excluded), so a real occurrence never
        # raises its own threshold and a sound used 10 times in a reel does not hide its other uses.
        free = np.ones(len(pooled), bool)
        while free.any():
            idx = int(np.argmax(np.where(free, pooled, -1)))
            c = float(pooled[idx])
            if c < min_abs:
                break
            rest = free.copy()
            rest[max(0, idx - sep):idx + sep + 1] = False
            noise = (float(np.percentile(pooled[rest], NOISE_PCT)) if rest.any() else 0.0) + 1.0
            if c < k * noise:
                break
            free = rest
            o = idx + lo
            out.append({'t': round(max(0.0, o / FPS), 2), 'sound': name, 'hits': int(c),
                        'frac': round(c / n, 3), 'score': round(c / noise, 1)})
    return sorted(out, key=lambda m: m['t'])


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        sys.exit('usage: python fp.py MIX_FILE PACK_DIR')
    DB, info = build(sys.argv[2])
    print('pack:', {k: v['n'] for k, v in info.items()})
    for m in match(load(sys.argv[1]), DB, info):
        print(m)
