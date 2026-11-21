"""Shared guitar helpers: labels, GuitarSet/Slakh readers, cache index, decoding, scoring.

Uses only numpy/json/csv at import time (pretty_midi, yaml and mir_eval are imported
inside the functions that need them), so most of it can be tested anywhere.
"""
import csv
import json
from pathlib import Path

import numpy as np

from config import ROOT, N_PITCHES, HOP_LENGTH, SAMPLE_RATE, BINS_PER_OCTAVE

FPS = SAMPLE_RATE / HOP_LENGTH                      # 31.25 frames per second, same as piano

GUITARSET_DIR = ROOT / "data" / "guitarset"         # extracted GuitarSet zips go here
GUITARSET_CACHE = ROOT / "data" / "cache" / "guitarset"
SLAKH_CACHE = ROOT / "data" / "cache" / "slakh_guitar"

# Guitar range. Standard tuning starts at E2 = MIDI 40; 38 leaves room for drop-D tuning.
# The 24th fret of the high E string is E6 = MIDI 88.
GUITAR_MIN_MIDI = 38
GUITAR_MAX_MIDI = 88

# GuitarSet has 6 players. We split BY PLAYER so the test player is never seen in training.
GUITARSET_SPLITS = {"train": {"00", "01", "02", "03"}, "validation": {"04"}, "test": {"05"}}


def guitarset_split(track_id):
    player = track_id.split("_")[0]                 # e.g. "00_BN1-129-Eb_comp" -> "00"
    for split, players in GUITARSET_SPLITS.items():
        if player in players:
            return split
    return "train"


# ---------------------------------------------------------------- labels
def notes_to_rolls(notes, n_frames):
    """notes: (N, 3) array [start_sec, end_sec, midi_pitch].
    Returns (frame_roll, onset_roll), both (N_PITCHES, n_frames) float32.
    Same rules as labels.midi_to_rolls: onset label = 2 frames at each note start."""
    frame = np.zeros((N_PITCHES, n_frames), dtype=np.float32)
    onset = np.zeros((N_PITCHES, n_frames), dtype=np.float32)
    for s, e, p in np.asarray(notes, dtype=np.float64).reshape(-1, 3):
        k = int(round(p)) - 21                      # index 0 = MIDI 21 (A0)
        if not 0 <= k < N_PITCHES:
            continue
        a = int(round(s * FPS))
        b = max(a + 1, int(round(e * FPS)))
        if a >= n_frames:
            continue
        frame[k, a:min(b, n_frames)] = 1
        onset[k, a:min(a + 2, n_frames)] = 1
    return frame, onset


def shift_pitch(x, yf, yo, s):
    """Pitch-shift augmentation. x (n_bins, T) CQT, yf/yo (T, N_PITCHES) labels.
    One semitone = BINS_PER_OCTAVE/12 CQT bins = one label column. Edges are zero filled."""
    if s == 0:
        return x, yf, yo
    bins = s * (BINS_PER_OCTAVE // 12)
    x2 = np.zeros_like(x)
    yf2, yo2 = np.zeros_like(yf), np.zeros_like(yo)
    if s > 0:
        x2[bins:] = x[:-bins]
        yf2[:, s:] = yf[:, :-s]
        yo2[:, s:] = yo[:, :-s]
    else:
        x2[:bins] = x[-bins:]
        yf2[:, :s] = yf[:, -s:]
        yo2[:, :s] = yo[:, -s:]
    return x2, yf2, yo2


# ---------------------------------------------------------------- GuitarSet
def load_jams_notes(path):
    """All 'note_midi' annotations (one per string) of a GuitarSet .jams file (it is JSON).
    Returns (N, 3) array [start, end, pitch], sorted by start time."""
    with open(path, encoding="utf-8") as f:
        jam = json.load(f)
    notes = []
    for ann in jam["annotations"]:
        if ann.get("namespace") != "note_midi":
            continue
        data = ann["data"]
        if isinstance(data, dict):                  # "dense" JAMS layout: dict of lists
            rows = zip(data["time"], data["duration"], data["value"])
        else:                                       # usual layout: list of observations
            rows = ((d["time"], d["duration"], d["value"]) for d in data)
        for t, d, v in rows:
            if d > 0:
                notes.append((float(t), float(t) + float(d), float(v)))
    notes.sort()
    return np.array(notes, dtype=np.float64).reshape(-1, 3)


# ---------------------------------------------------------------- Slakh
def _parse_stems_simple(text):
    """Tiny fallback reader for the 'stems:' block of Slakh's metadata.yaml (no PyYAML needed)."""
    stems, cur, in_stems, stem_indent = {}, None, False, None
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        line = raw.strip()
        if indent == 0:
            in_stems = line.startswith("stems:")
            cur, stem_indent = None, None
            continue
        if not in_stems:
            continue
        if stem_indent is None:
            stem_indent = indent
        if indent == stem_indent and line.endswith(":"):
            cur = line[:-1]
            stems[cur] = {}
        elif cur is not None and ":" in line:
            key, val = [t.strip() for t in line.split(":", 1)]
            if val.lower() in ("true", "false"):
                val = val.lower() == "true"
            stems[cur][key] = val
    return stems


def read_slakh_stems(meta_path):
    """{stem_id: {'inst_class':..., 'audio_rendered': bool, 'midi_saved': bool, 'is_drum': bool, ...}}"""
    text = Path(meta_path).read_text(encoding="utf-8")
    try:
        import yaml
        return {k: dict(v) for k, v in yaml.safe_load(text)["stems"].items()}
    except ImportError:
        return _parse_stems_simple(text)


def guitar_stem_ids(meta_path):
    """Stems that are rendered guitar with a saved MIDI file."""
    out = []
    for sid, m in read_slakh_stems(meta_path).items():
        if (m.get("inst_class") == "Guitar" and m.get("audio_rendered", True)
                and m.get("midi_saved", True) and not m.get("is_drum", False)):
            out.append(sid)
    return sorted(out)


def read_midi_notes(midi_path):
    """All notes of a MIDI file -> (N, 3) array [start, end, pitch]."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [(n.start, n.end, n.pitch) for inst in pm.instruments for n in inst.notes]
    return np.array(sorted(notes), dtype=np.float64).reshape(-1, 3)


def find_audio(base):
    """base without extension -> existing .flac or .wav path, else None."""
    for ext in (".flac", ".wav"):
        p = Path(str(base) + ext)
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------- cache
def cache_paths(cache_dir, name):
    d = Path(cache_dir)
    return (d / f"{name}_cqt.npy", d / f"{name}_frame.npy",
            d / f"{name}_onset.npy", d / f"{name}_notes.npy")


def piece_cached(cache_dir, name):
    return all(p.exists() for p in cache_paths(cache_dir, name))


def save_piece(cache_dir, name, cqt, notes):
    """cqt (N_BINS, T); notes (N, 3). Writes cqt last, so 'all files exist' means finished."""
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cqt_p, fr_p, on_p, nt_p = cache_paths(cache_dir, name)
    frame, onset = notes_to_rolls(notes, cqt.shape[1])
    np.save(fr_p, frame.astype(np.uint8))
    np.save(on_p, onset.astype(np.uint8))
    np.save(nt_p, np.asarray(notes, dtype=np.float64).reshape(-1, 3))
    np.save(cqt_p, cqt.astype(np.float16))


def write_index(cache_dir, entries):
    """entries: iterable of (name, split). Only fully cached pieces are listed."""
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


# ---------------------------------------------------------------- decoding and scoring
def decode_notes(frame_p, onset_p, frame_thr=0.5, onset_thr=0.5,
                 lo=GUITAR_MIN_MIDI, hi=GUITAR_MAX_MIDI):
    """Probabilities (T, 88) -> sorted list of (start_sec, end_sec, midi_pitch).
    A note starts where onset_p crosses the threshold and lasts while frame_p stays above
    its threshold, or until the next onset of the same pitch. Pitches outside the
    guitar range [lo, hi] are ignored."""
    T = frame_p.shape[0]
    notes = []
    for k in range(max(0, lo - 21), min(N_PITCHES, hi - 21 + 1)):
        on = onset_p[:, k] > onset_thr
        starts = np.where(on & ~np.concatenate([[False], on[:-1]]))[0]
        if len(starts) == 0:
            continue
        fr = frame_p[:, k] > frame_thr
        for i, a in enumerate(starts):
            limit = starts[i + 1] if i + 1 < len(starts) else T
            b = a + 1
            while b < limit and fr[b]:
                b += 1
            notes.append((a / FPS, b / FPS, k + 21))
    return sorted(notes)


def note_f1(ref_notes, est_notes):
    """Note-level F1 with mir_eval: (onset-only F1, onset+offset F1).
    Onset tolerance 50 ms; offsets must be within 20% of the note length (min 50 ms)."""
    import mir_eval
    ref = np.asarray(ref_notes, dtype=np.float64).reshape(-1, 3)
    est = np.asarray(est_notes, dtype=np.float64).reshape(-1, 3)
    ref = ref[ref[:, 1] > ref[:, 0]]
    est = est[est[:, 1] > est[:, 0]]
    if len(ref) == 0 or len(est) == 0:
        return 0.0, 0.0
    hz = lambda p: 440.0 * 2.0 ** ((p - 69.0) / 12.0)
    ri, rp, ei, ep = ref[:, :2], hz(ref[:, 2]), est[:, :2], hz(est[:, 2])
    onset = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)[2]
    full = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep)[2]
    return float(onset), float(full)


def notes_to_guitar_midi(notes, path, program=25, velocity=80):
    """Write notes as a MIDI file with a guitar sound (program 25 = acoustic guitar, steel)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program, name="Guitar")
    for s, e, p in notes:
        inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=int(p), start=float(s), end=float(e)))
    pm.instruments.append(inst)
    pm.write(str(path))