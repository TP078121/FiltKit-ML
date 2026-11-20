import numpy as np
import torch
import pretty_midi
import mir_eval

from config import *
from maestro_dataset import read_rows, cache_name, cached_paths
from midi_utils import roll_to_notes_onset


def predict_full(model, cqt, device, chunk=2048, ctx=16):
    """Run a whole piece through the model in chunks (with a little overlap
    so chunk edges don't cause artifacts). cqt: (264, T) -> two (T, 88) arrays."""
    T = cqt.shape[1]
    frames, onsets = [], []
    with torch.no_grad():
        for s in range(0, T, chunk):
            a = max(0, s - ctx)
            b = min(T, s + chunk + ctx)
            x = torch.from_numpy(np.ascontiguousarray(cqt[:, a:b]))[None, None].to(device)
            f, o = model(x)
            f = torch.sigmoid(f)[0].cpu().numpy()
            o = torch.sigmoid(o)[0].cpu().numpy()
            lo = s - a
            hi = lo + min(chunk, T - s)
            frames.append(f[lo:hi])
            onsets.append(o[lo:hi])
    return np.concatenate(frames), np.concatenate(onsets)


def note_f1(ref_midi_path, est_notes):
    """Returns (onset-only F1, onset+offset F1) using mir_eval."""
    pm = pretty_midi.PrettyMIDI(str(ref_midi_path))
    ref = [(n.start, n.end, n.pitch) for i in pm.instruments for n in i.notes if n.end > n.start]
    est = [(s, e, p) for s, e, p in est_notes if e > s]
    if not ref or not est:
        return 0.0, 0.0
    ref_i = np.array([[s, e] for s, e, _ in ref])
    ref_p = np.array([pretty_midi.note_number_to_hz(p) for _, _, p in ref])
    est_i = np.array([[s, e] for s, e, _ in est])
    est_p = np.array([pretty_midi.note_number_to_hz(p) for _, _, p in est])
    onset = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, offset_ratio=None)[2]
    full = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p)[2]
    return onset, full


def evaluate_split(model, device, split="validation", max_pieces=10,
                   frame_thr=0.5, onset_thr=0.5):
    """Average note-level F1 over the first max_pieces cached pieces of a split.
    Returns (onset_f1, onset_offset_f1, n_pieces) or None if nothing is cached."""
    rows = [r for r in read_rows(split)
            if all(p.exists() for p in cached_paths(cache_name(r)))]
    rows = rows[:max_pieces]
    if not rows:
        return None

    was_training = model.training
    model.eval()
    on_scores, full_scores = [], []
    for r in rows:
        cqt = np.load(cached_paths(cache_name(r))[0]).astype(np.float32)
        frame_p, onset_p = predict_full(model, cqt, device)
        notes = roll_to_notes_onset(frame_p, onset_p, frame_thr, onset_thr)
        on, full = note_f1(MAESTRO_DIR / r["midi_filename"], notes)
        on_scores.append(on)
        full_scores.append(full)
    if was_training:
        model.train()
    return float(np.mean(on_scores)), float(np.mean(full_scores)), len(rows)