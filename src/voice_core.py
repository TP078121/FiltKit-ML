"""Voice helpers: monophonic decoding, pYIN (non-ML) baseline, vocadito reader, group-wise split.

Cache, labels and scoring are shared with the guitar module (guitar_core.py), because a sung melody
is handled like any other single-instrument CQT -> notes problem.
"""
import random
import re
from pathlib import Path

import numpy as np

from config import ROOT, N_PITCHES, SAMPLE_RATE
from guitar_core import FPS, note_f1, notes_to_guitar_midi          # re-used by the voice scripts

VOICE_MIN_MIDI = 36            # C2  (bass voices reach about E2 = 40)
VOICE_MAX_MIDI = 88            # E6  (well above a soprano's high C6 = 84)
VOCADITO_DIR = ROOT / "data" / "vocadito"
VOICE_PAIRS_CACHE = ROOT / "data" / "cache" / "voice_pairs"


def vocadito_cache(annotator="A1"):
    return ROOT / "data" / "cache" / f"vocadito_{annotator}"


# ---------------------------------------------------------------- monophonic decoding of the ML model
def mono_decode(frame_p, onset_p, frame_thr=0.5, onset_thr=0.5, min_frames=3,
                lo=VOICE_MIN_MIDI, hi=VOICE_MAX_MIDI, smooth=5):
    """Probabilities (T, 88) -> list of (start_sec, end_sec, midi_pitch), ONE note at a time.
    Per frame the most likely pitch is taken (if its probability > frame_thr), the pitch track is
    median-smoothed, and a note is cut where the pitch changes, where a new onset of the same pitch
    appears (repeated note), or where the voice stops. Notes shorter than min_frames are dropped
    (3 frames = about 100 ms)."""
    from scipy.ndimage import median_filter
    T = frame_p.shape[0]
    if T == 0:
        return []
    k0, k1 = max(0, lo - 21), min(N_PITCHES - 1, hi - 21)
    fp = frame_p[:, k0:k1 + 1]
    pitch = np.where(fp.max(axis=1) > frame_thr, fp.argmax(axis=1), -1)
    if smooth > 1:
        pitch = median_filter(pitch, size=smooth, mode="nearest")
    onset = onset_p[:, k0:k1 + 1] > onset_thr

    notes, cur, start = [], -1, 0
    for t in range(T + 1):
        p = int(pitch[t]) if t < T else -1
        split = p != cur
        if not split and p >= 0 and t > start:
            split = bool(onset[t, p] and not onset[t - 1, p])      # same pitch struck again
        if split:
            if cur >= 0 and t - start >= min_frames:
                notes.append((start / FPS, t / FPS, cur + k0 + 21))
            cur, start = p, t
    return notes


# ---------------------------------------------------------------- pYIN baseline (no machine learning)
def f0_track_to_notes(midi, hop_sec, min_dur=0.08, max_gap=0.06, jump=0.75):
    """midi: float array of MIDI pitch per frame, NaN where unvoiced -> list of (start, end, int pitch).
    A new note starts after a gap longer than max_gap seconds or a pitch jump above `jump` semitones."""
    notes, cur, t0, last = [], [], 0.0, 0.0

    def flush():
        if cur and (last + hop_sec - t0) >= min_dur:
            notes.append((t0, last + hop_sec, int(round(float(np.median(cur))))))

    for i, m in enumerate(midi):
        if np.isnan(m):
            continue
        t = i * hop_sec
        if cur and ((t - last - hop_sec) > max_gap or abs(m - np.median(cur[-5:])) > jump):
            flush()
            cur = []
        if not cur:
            t0 = t
        cur.append(float(m))
        last = t
    flush()
    return notes


def pyin_notes(y, sr=SAMPLE_RATE, hop=160):
    """Classical pitch tracking (librosa pYIN) + the simple note segmentation above."""
    import librosa
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"),
                            sr=sr, frame_length=1024, hop_length=hop)
    return f0_track_to_notes(librosa.hz_to_midi(f0), hop / sr)


# ---------------------------------------------------------------- vocadito
def read_vocadito_notes(path, pitch_unit="hz"):
    """Note csv -> (N, 3) array [start, end, midi_pitch] (pitch may be fractional: singers are not
    always in tune, and scoring allows 50 cents). The column order is detected automatically
    (the pitch column has the larger values); header lines and ',' / space / tab separators are fine."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        parts = [p for p in re.split(r"[,\s;]+", line.strip()) if p]
        if len(parts) < 3:
            continue
        try:
            rows.append([float(p) for p in parts[:3]])
        except ValueError:
            continue                                   # header line
    if not rows:
        raise ValueError(f"no numeric note rows found in {path}")
    r = np.array(rows, dtype=np.float64)
    pitch, dur = (r[:, 1], r[:, 2]) if np.median(r[:, 1]) >= np.median(r[:, 2]) else (r[:, 2], r[:, 1])
    ok = (dur > 0) & (pitch > 0)
    r, pitch, dur = r[ok], pitch[ok], dur[ok]
    midi = 69.0 + 12.0 * np.log2(pitch / 440.0) if pitch_unit == "hz" else pitch
    med = float(np.median(midi))
    if not 40 <= med <= 96:
        raise ValueError(f"{path}: median pitch is MIDI {med:.1f}, which is not a singing range. "
                         f"Try --pitch-unit {'midi' if pitch_unit == 'hz' else 'hz'}.")
    return np.stack([r[:, 0], r[:, 0] + dur, midi], axis=1)


# ---------------------------------------------------------------- split without leakage
def split_by_group(stems, seed=0, fractions=(0.8, 0.1)):
    """stem -> 'train' / 'validation' / 'test'. A song recorded in two keys (en001a, en001b) is one group,
    so the same song never lands in two splits."""
    key = lambda s: re.sub(r"(?<=\d)[ab]$", "", s)
    groups = sorted({key(s) for s in stems})
    random.Random(seed).shuffle(groups)
    n_train = int(round(len(groups) * fractions[0]))
    n_val = int(round(len(groups) * fractions[1]))
    if len(groups) >= 3:
        n_val = max(1, n_val)
        n_train = min(n_train, len(groups) - n_val - 1)
    where = {g: "train" if i < n_train else ("validation" if i < n_train + n_val else "test")
             for i, g in enumerate(groups)}
    return {s: where[key(s)] for s in stems}