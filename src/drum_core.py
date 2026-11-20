"""Drum classes, label building, cache index, peak picking and metrics.

Only numpy and csv are used here (no torch, librosa or pretty_midi at import time),
so this file can be tested anywhere.
"""
import csv
from pathlib import Path

import numpy as np

from config import DRUM_FPS

# ---------------------------------------------------------------- classes
DRUM_CLASSES = ["kick", "snare", "hihat_closed", "hihat_open", "tom", "crash", "ride"]
N_DRUM = len(DRUM_CLASSES)

# General MIDI drum pitch -> our class. This follows the Roland TD-17 / Groove MIDI
# convention as I understand it. Verify it with check_drum_pitches.py.
_GROUPS = {
    "kick": [35, 36],
    "snare": [37, 38, 40],                   # 37 = cross stick
    "hihat_closed": [22, 42, 44],            # 44 = pedal hi-hat
    "hihat_open": [26, 46],
    "tom": [41, 43, 45, 47, 48, 50, 58],
    "crash": [49, 52, 55, 57],
    "ride": [51, 53, 59],
}
PITCH_TO_CLASS = {p: DRUM_CLASSES.index(c) for c, ps in _GROUPS.items() for p in ps}

# MIDI pitch used when writing a predicted class back to a .mid file
CLASS_TO_PITCH = [36, 38, 42, 46, 45, 49, 51]


# ---------------------------------------------------------------- labels
def events_from_notes(notes):
    """notes: iterable of (start_seconds, midi_pitch).
    Returns float array (N, 2) = [time_seconds, class_index], sorted by time.
    Pitches that are not in the mapping are skipped."""
    ev = [(t, PITCH_TO_CLASS[p]) for t, p in notes if p in PITCH_TO_CLASS]
    if not ev:
        return np.zeros((0, 2))
    ev = np.array(ev, dtype=np.float64)
    return ev[np.argsort(ev[:, 0], kind="stable")]


def midi_to_events(midi_path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    return events_from_notes((n.start, n.pitch) for inst in pm.instruments for n in inst.notes)


def events_to_targets(events, n_frames, spread=1):
    """(T, N_DRUM) onset targets: 1.0 on the onset frame, 0.5 on +-spread neighbours."""
    y = np.zeros((n_frames, N_DRUM), dtype=np.float32)
    for t, k in events:
        f = int(round(t * DRUM_FPS))
        k = int(k)
        for d in range(-spread, spread + 1):
            g = f + d
            if 0 <= g < n_frames:
                y[g, k] = max(y[g, k], 1.0 if d == 0 else 0.5)
    return y


# ---------------------------------------------------------------- cache index
def cache_name(row):
    """Unique file-safe name for one E-GMD row."""
    stem = row["audio_filename"].rsplit(".", 1)[0]
    return stem.replace("/", "__").replace("\\", "__")


def cache_paths(cache_dir, name):
    d = Path(cache_dir)
    return d / f"{name}_mel.npy", d / f"{name}_onset.npy", d / f"{name}_ref.npy"


def piece_cached(cache_dir, name):
    return all(p.exists() for p in cache_paths(cache_dir, name))


def write_index(cache_dir, entries):
    """entries: iterable of (name, split). Only pieces that are fully cached are listed."""
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    rows = [(n, s) for n, s in entries if piece_cached(d, n)]
    with open(d / "index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["name", "split"])
        w.writerows(rows)
    return len(rows)


def pieces_for(cache_dir, split=None):
    p = Path(cache_dir) / "index.csv"
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [r["name"] for r in rows
            if (split is None or r["split"] == split) and piece_cached(cache_dir, r["name"])]


# ---------------------------------------------------------------- decoding
def pick_peaks(prob, thr=0.5, window=2):
    """prob (T, N_DRUM) probabilities -> events (N, 2) [time_seconds, class].
    thr: one number or one number per class. A hit is a local maximum within
    +-window frames that is above the threshold."""
    prob = np.asarray(prob, dtype=np.float64)
    T, K = prob.shape
    if T == 0:
        return np.zeros((0, 2))
    thr = np.broadcast_to(np.asarray(thr, dtype=np.float64), (K,))
    out = []
    for k in range(K):
        p = prob[:, k]
        padded = np.pad(p, (window, window), constant_values=-1.0)
        local_max = np.lib.stride_tricks.sliding_window_view(padded, 2 * window + 1).max(axis=1)
        idx = np.where((p >= local_max) & (p > thr[k]))[0]
        last = -10 ** 9
        for t in idx:
            if t - last > window:          # also collapses flat plateaus into one hit
                out.append((t / DRUM_FPS, k))
                last = t
    if not out:
        return np.zeros((0, 2))
    out = np.array(out, dtype=np.float64)
    return out[np.argsort(out[:, 0], kind="stable")]


# ---------------------------------------------------------------- metrics
def onset_counts(ref_times, est_times, tol=0.05):
    """Greedy one-to-one matching within +-tol seconds. Returns (tp, fp, fn)."""
    ref = np.sort(np.asarray(ref_times, dtype=np.float64))
    est = np.sort(np.asarray(est_times, dtype=np.float64))
    i = j = tp = 0
    while i < len(ref) and j < len(est):
        d = est[j] - ref[i]
        if abs(d) <= tol:
            tp += 1
            i += 1
            j += 1
        elif d < 0:
            j += 1
        else:
            i += 1
    return tp, len(est) - tp, len(ref) - tp


def piece_counts(ref, est, tol=0.05):
    """ref, est: (N, 2) event arrays. Returns int array (N_DRUM, 3) of tp, fp, fn."""
    c = np.zeros((N_DRUM, 3), dtype=np.int64)
    for k in range(N_DRUM):
        c[k] = onset_counts(ref[ref[:, 1] == k, 0], est[est[:, 1] == k, 0], tol)
    return c


def f1_from(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return 2 * p * r / max(p + r, 1e-9)


def summarize(counts):
    """counts: summed (N_DRUM, 3). Returns (per_class_f1 list, micro_f1, macro_f1).
    Macro only averages classes that actually occur in the reference."""
    per_class = [float(f1_from(*counts[k])) for k in range(N_DRUM)]
    micro = float(f1_from(counts[:, 0].sum(), counts[:, 1].sum(), counts[:, 2].sum()))
    present = [k for k in range(N_DRUM) if counts[k, 0] + counts[k, 2] > 0]
    macro = float(np.mean([per_class[k] for k in present])) if present else 0.0
    return per_class, micro, macro
