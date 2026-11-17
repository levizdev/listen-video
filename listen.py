#!/usr/bin/env python3
"""listen.py — "ears" for an AI agent: turn the soundtrack of a video into a per-second timeline.

    python listen.py INPUT OUTDIR [--pack DIR] [--lang auto|ru|en|...] [--no-asr] [--asr-model small]
                     [--vad] [--prompts extra_prompts.txt] [--no-png]

What it extracts:
  1. speech, word by word with timestamps (faster-whisper) + per-word confidence
  2. sound events from AudioSet's 527 classes (AST), 1 s windows every 0.5 s
  3. where music plays (AudioSet "Music" > 0.35 for >= 1.5 s)
  4. zero-shot guesses for meme / editing sounds (CLAP vs. text prompts) — guesses, not proof
  5. exact occurrences of files from your own sound pack (fp.py, Shazam-style fingerprints)
  6. hits (strong onsets), silences >= 0.35 s, tempo
Writes OUTDIR/report.json, OUTDIR/report.md and OUTDIR/timeline_XX.png (mel spectrogram, 30 s per image).
"""
import argparse
import json
import math
import os
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ.setdefault('HF_HOME', os.path.expanduser('~/hf'))

import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fp  # noqa: E402

AST_ID = 'MIT/ast-finetuned-audioset-10-10-0.4593'
CLAP_ID = 'laion/clap-htsat-unfused'
WIN, STEP, BATCH = 1.0, 0.5, 16
SPEECH_GATE = 0.5          # AudioSet 'Speech' probability above which CLAP guesses are treated as doubtful
SPEECHY = {'Speech', 'Male speech, man speaking', 'Female speech, woman speaking', 'Narration, monologue',
           'Conversation', 'Child speech, kid speaking', 'Inside, small room', 'Silence', 'Music'}
PROMPTS = [
    'a vine boom bass hit sound effect', 'a record scratch sound effect', 'a loud explosion', 'an air horn',
    'a laugh track, people laughing', 'crowd cheering', 'phonk music with cowbell', 'a dramatic orchestral hit',
    'a whoosh transition sound', 'a cartoon boing sound', 'a man saying bruh', 'a fart sound',
    'crickets chirping in silence', 'a cash register ka-ching', 'a camera shutter click', 'glass breaking',
    'a sad violin melody', 'a choir singing aaah', 'a dubstep wobble bass', 'an 8-bit video game sound',
    'keyboard typing', 'a notification ding', 'a deep bass drop', 'a metal pipe falling clang',
    'a person saying wow', 'an ominous low drone', 'a heartbeat', 'applause', 'a baby crying',
    'a person screaming', 'a car horn', 'a police siren', 'a water splash', 'solo piano music', 'a rap beat',
    'a pop song with female vocals', 'classical music']
NEUTRAL = ['a woman talking', 'a man talking', 'quiet room tone']   # absorb plain speech/room tone


def log(*a):
    print('[listen]', *a, file=sys.stderr, flush=True)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inp', help='video or audio file')
    ap.add_argument('out', help='output directory')
    ap.add_argument('--pack', help='folder with reference sounds to find exactly (mp3/wav/...)')
    ap.add_argument('--lang', default='auto', help='speech language: auto, ru, en, ...')
    ap.add_argument('--no-asr', action='store_true', help='skip speech recognition')
    ap.add_argument('--asr-model', default='small', help='faster-whisper model: tiny/base/small/medium/large-v3')
    ap.add_argument('--vad', action='store_true', help='Whisper VAD filter (fewer hallucinations on music, may drop sung words)')
    ap.add_argument('--prompts', help='text file with extra CLAP prompts, one per line')
    ap.add_argument('--no-png', action='store_true', help='skip spectrogram images')
    return ap.parse_args()


def merge(evs, gap=0.6):
    """Merge per-window labels into [t0, t1] segments per label."""
    out = []
    for e in sorted(evs, key=lambda e: (e['label'], e['t'])):
        if out and out[-1]['label'] == e['label'] and e['t'] - out[-1]['t1'] <= gap:
            out[-1]['t1'] = e['t'] + WIN
            out[-1]['p'] = max(out[-1]['p'], e['p'])
            if e.get('over_speech'):
                out[-1]['over_speech'] = True
        else:
            out.append({'label': e['label'], 't0': e['t'], 't1': e['t'] + WIN, 'p': e['p']})
            if e.get('over_speech'):
                out[-1]['over_speech'] = True
    return sorted(out, key=lambda e: e['t0'])


def windows(D):
    starts = list(np.arange(0, max(D - WIN, 0) + 1e-6, STEP))
    if D > WIN and starts[-1] + WIN < D - 0.05:          # make sure the tail is covered
        starts.append(D - WIN)
    return [round(float(t), 3) for t in starts]


def asr(a, y16, rep):
    from faster_whisper import WhisperModel
    log('speech:', a.asr_model)
    m = WhisperModel(a.asr_model, device='auto', compute_type='int8')
    segs, info = m.transcribe(y16, language=None if a.lang == 'auto' else a.lang, word_timestamps=True,
                              beam_size=5, vad_filter=a.vad, condition_on_previous_text=False)
    words = []
    for s in segs:
        for w in (s.words or []):
            words.append({'w': w.word.strip(), 's': round(w.start, 2), 'e': round(w.end, 2),
                          'p': round(float(w.probability), 2)})
    rep['language'] = info.language
    rep['language_p'] = round(float(info.language_probability), 2)
    return words


def ast_tags(y16, starts, D):
    import torch
    from transformers import AutoFeatureExtractor, ASTForAudioClassification
    log('sound events: AST')
    fe = AutoFeatureExtractor.from_pretrained(AST_ID)
    model = ASTForAudioClassification.from_pretrained(AST_ID).eval()
    L = model.config.id2label
    rows = []
    with torch.no_grad():
        for i in range(0, len(starts), BATCH):
            batch = [y16[int(t * 16000):int((t + WIN) * 16000)] for t in starts[i:i + BATCH]]
            x = fe(batch, sampling_rate=16000, return_tensors='pt')
            p = torch.sigmoid(model(**x).logits).numpy()
            rows += list(zip(starts[i:i + BATCH], p))
    music_idx = [k for k, v in L.items() if v == 'Music'][0]
    speech_idx = [k for k, v in L.items() if v == 'Speech'][0]
    speech_p = {t: float(row[speech_idx]) for t, row in rows}
    evs = []
    for t, row in rows:
        for k in np.argsort(row)[::-1][:6]:
            lab, pr = L[int(k)], float(row[k])
            if pr >= 0.2 and lab not in SPEECHY:
                evs.append({'t': t, 'label': lab, 'p': round(pr, 2)})
    events = [dict(e, t0=round(e['t0'], 2), t1=round(min(D, e['t1']), 2)) for e in merge(evs)]
    music, cur = [], None
    for t, row in rows:
        if row[music_idx] > 0.35:
            cur = [t, t + WIN] if cur is None else [cur[0], t + WIN]
        elif cur:
            music.append(cur)
            cur = None
    if cur:
        music.append(cur)
    music = [{'t0': round(s, 2), 't1': round(min(D, e), 2)} for s, e in music if e - s >= 1.5]
    return events, music, speech_p


def clap_guess(y48, starts, D, extra, speech_p):
    import torch
    from transformers import ClapModel, ClapProcessor
    log('meme-sound guesses: CLAP')
    model = ClapModel.from_pretrained(CLAP_ID).eval()
    proc = ClapProcessor.from_pretrained(CLAP_ID)
    prompts = PROMPTS + extra + NEUTRAL

    def emb(x):  # transformers >= 5 returns an output object instead of a tensor
        x = getattr(x, 'pooler_output', x)
        return x / x.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        te = emb(model.get_text_features(**proc(text=prompts, return_tensors='pt', padding=True)))
        zs = []
        for i in range(0, len(starts), BATCH):
            chunks = [y48[int(t * 48000):int((t + WIN) * 48000)] for t in starts[i:i + BATCH]]
            ae = emb(model.get_audio_features(**proc(audio=chunks, sampling_rate=48000, return_tensors='pt')))
            sim = (ae @ te.T).numpy()
            for t, s in zip(starts[i:i + BATCH], sim):
                pr = np.exp((s - s.max()) * 30)
                pr /= pr.sum()
                k = int(np.argmax(pr))
                if prompts[k] in NEUTRAL:
                    continue
                # over clear speech CLAP often mislabels the voice itself ("bruh", "glass breaking"):
                # demand more confidence there and flag it
                over = speech_p.get(t, 0.0) >= SPEECH_GATE
                if pr[k] >= (0.6 if over else 0.35):
                    zs.append({'t': t, 'label': prompts[k], 'p': round(float(pr[k]), 2), 'over_speech': over})
    return [dict(e, t0=round(e['t0'], 2), t1=round(min(D, e['t1']), 2)) for e in merge(zs)]


def dynamics(y16, rep):
    import librosa
    hop = 512
    rms = librosa.feature.rms(y=y16, hop_length=hop)[0]
    tt = librosa.frames_to_time(np.arange(len(rms)), sr=16000, hop_length=hop)
    on = librosa.onset.onset_strength(y=y16, sr=16000, hop_length=hop)
    hits = []
    if on.max() > 0:
        pk = librosa.util.peak_pick(on, pre_max=8, post_max=8, pre_avg=20, post_avg=20,
                                    delta=float(np.percentile(on, 97)) * 0.6, wait=10)
        hits = [round(float(tt[i]), 2) for i in pk if i < len(tt) and on[i] > np.percentile(on, 98)]
    db = 20 * np.log10(rms + 1e-9)
    thr = max(float(np.percentile(db, 60)) - 25, -70.0)   # relative to the clip, but digital silence always counts
    sil, s0 = [], None
    for t, v in zip(tt, db):
        if v < thr and s0 is None:
            s0 = t
        elif v >= thr and s0 is not None:
            if t - s0 >= 0.35:
                sil.append([round(float(s0), 2), round(float(t), 2)])
            s0 = None
    end = len(y16) / 16000
    if s0 is not None and end - s0 >= 0.35:              # silence that runs to the end of the clip
        sil.append([round(float(s0), 2), round(end, 2)])
    rep['hits'], rep['silences'] = hits, sil
    try:
        rep['tempo_bpm'] = round(float(np.atleast_1d(librosa.beat.beat_track(y=y16, sr=16000)[0])[0]), 1)
    except Exception:
        pass


def write_md(rep, path, min_word_p=0.3):
    D = rep['duration']
    words = rep.get('words', [])
    head = [f"# Слух: {rep['file']}",
            f"Длительность {D:.1f} с · язык {rep.get('language', '-')} · темп ~{rep.get('tempo_bpm', '-')} BPM · "
            f"событий {len(rep.get('sound_events', []))} · совпадений с паком {len(rep.get('pack_matches', []))}",
            '', 'Слова с уверенностью < 0.3 — в скобках (часто галлюцинации Whisper на музыке). '
            'CLAP-догадка со знаком ? сделана поверх речи — проверяй на слух.', '',
            '| t, с | Речь | Звуки (AudioSet) | CLAP-догадка | Совпадение с паком | Удар |', '|---|---|---|---|---|---|']
    for s in range(int(math.ceil(D))):
        sp = ' '.join(w['w'] if w.get('p', 1) >= min_word_p else f"({w['w']})" for w in words if s <= w['s'] < s + 1)
        ev = ', '.join(sorted({e['label'] for e in rep.get('sound_events', []) if e['t0'] < s + 1 and e['t1'] > s}))
        cg = ', '.join(sorted({e['label'] + ('?' if e.get('over_speech') else '')
                               for e in rep.get('clap_guess', []) if e['t0'] < s + 1 and e['t1'] > s}))
        mt = ', '.join(f"{m['sound']} ({m['frac']})" for m in rep.get('pack_matches', []) if s <= m['t'] < s + 1)
        ht = '●' if any(s <= h < s + 1 for h in rep.get('hits', [])) else ''
        mu = ' 🎵' if any(m['t0'] <= s < m['t1'] for m in rep.get('music_segments', [])) else ''
        sl = ' ⏸' if any(a <= s + 0.5 < b for a, b in rep.get('silences', [])) else ''
        lines = f"| {s} | {sp}{sl} | {ev}{mu} | {cg} | {mt} | {ht} |"
        head.append(lines.replace('|  |', '| |'))
    open(path, 'w').write('\n'.join(head) + '\n')


def write_png(y16, rep, out):
    import librosa
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    D = rep['duration']
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y16, sr=16000, n_mels=96, hop_length=320), ref=np.max)
    for k, c0 in enumerate(np.arange(0, D, 30)):
        c1 = min(D, c0 + 30)
        fig, ax = plt.subplots(figsize=(16, 4.5), dpi=80)
        ax.imshow(S[:, int(c0 * 50):int(c1 * 50)], origin='lower', aspect='auto', extent=[c0, c1, 0, 96], cmap='magma')
        for e in rep.get('sound_events', []):
            if c0 <= e['t0'] < c1:
                ax.text(e['t0'], 90, e['label'][:22], color='cyan', fontsize=7, rotation=60, va='top')
        for m in rep.get('pack_matches', []):
            if c0 <= m['t'] < c1:
                ax.axvline(m['t'], color='lime', lw=1.5)
                ax.text(m['t'], 5, m['sound'], color='lime', fontsize=8, rotation=90)
        for h in rep.get('hits', []):
            if c0 <= h < c1:
                ax.axvline(h, color='white', lw=.5, alpha=.5)
        ax.set_xlabel('секунды')
        ax.set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(out, f'timeline_{k:02d}.png'))
        plt.close(fig)


def main():
    a = parse()
    if not os.path.isfile(a.inp):
        sys.exit(f'no such file: {a.inp}')
    os.makedirs(a.out, exist_ok=True)
    y16, y48 = fp.load(a.inp, 16000), fp.load(a.inp, 48000)
    D = len(y16) / 16000
    rep = {'file': os.path.basename(a.inp), 'duration': round(D, 2)}
    if D < 0.1:
        rep['error'] = 'no audio stream (or shorter than 0.1 s)'
        json.dump(rep, open(os.path.join(a.out, 'report.json'), 'w'), ensure_ascii=False, indent=1)
        open(os.path.join(a.out, 'report.md'), 'w').write(f"# Слух: {rep['file']}\n\nВ файле нет звука.\n")
        print('ok', a.out, 'NO AUDIO')
        return
    import torch
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    starts = windows(D)
    rep['words'] = [] if a.no_asr else asr(a, y16, rep)
    rep['sound_events'], rep['music_segments'], speech_p = ast_tags(y16, starts, D)
    extra = [l.strip() for l in open(a.prompts, encoding='utf-8') if l.strip()] if a.prompts else []
    rep['clap_guess'] = clap_guess(y48, starts, D, extra, speech_p)
    rep['pack_matches'] = []
    if a.pack:
        if not os.path.isdir(a.pack):
            sys.exit(f'--pack is not a folder: {a.pack}')
        log('pack matching:', a.pack)
        DB, info = fp.build(a.pack)
        rep['pack'] = {k: v['dur'] for k, v in info.items()}
        rep['pack_matches'] = fp.match(y16, DB, info)
    dynamics(y16, rep)
    json.dump(rep, open(os.path.join(a.out, 'report.json'), 'w'), ensure_ascii=False, indent=1)
    write_md(rep, os.path.join(a.out, 'report.md'))
    if not a.no_png:
        write_png(y16, rep, a.out)
    print('ok', a.out, len(rep['words']), 'words', len(rep['sound_events']), 'events',
          len(rep['pack_matches']), 'pack matches')


if __name__ == '__main__':
    main()
