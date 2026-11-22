"""Find out WHY a voice score is low: octave / pitch-offset problems.

  python src/voice_diagnose.py labels --cache-dir data/cache/voice_pairs --audio-dir data/csd --n 8
      pYIN on the training recordings vs their MIDI labels: is the label pitch the real sung pitch?
  python src/voice_diagnose.py model --weights checkpoints/voice_smoke/best.pt --cache-dir data/cache/vocadito_A1 --split test
      trained model vs the reference notes

Both print the note F1 when every estimated note is moved up/down by k semitones. If the best k is
+-12 (or another constant), there is a systematic pitch offset.
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from guitar_core import cache_paths, note_f1, pieces_for

SHIFTS = [-24, -12, -7, -5, 0, 5, 7, 12, 24]


def shift_notes(notes, k):
    return [(s, e, p + k) for s, e, p in notes]


def evenly(names, n):
    if n is None or len(names) <= n:
        return list(names)
    return [names[i] for i in np.linspace(0, len(names) - 1, n).round().astype(int)]


def matched_intervals(ref, est, window=0.1):
    """Pitch difference (estimate - reference, in whole semitones) for every reference note that has an
    estimated note starting within +-window seconds (the closest one). Unmatched notes are skipped."""
    est = sorted(est)
    if not est:
        return []
    starts = np.array([e[0] for e in est])
    out = []
    for s, _, p in np.asarray(ref, dtype=np.float64).reshape(-1, 3):
        i = int(np.searchsorted(starts, s))
        j = min((j for j in (i - 1, i) if 0 <= j < len(est)), key=lambda j: abs(starts[j] - s))
        if abs(starts[j] - s) <= window:
            out.append(int(round(est[j][2] - p)))
    return out


def report(pairs, who):
    """pairs: list of (reference notes, estimated notes)."""
    print(f"\n{who}: onset F1 when all estimated notes are shifted by k semitones")
    print(f"{'k':>4s} {'onset F1':>9s}")
    scores = {}
    for k in SHIFTS:
        scores[k] = float(np.mean([note_f1(ref, shift_notes(est, k))[0] for ref, est in pairs]))
        print(f"{k:+4d} {scores[k]:9.3f}")
    ints = [d for ref, est in pairs for d in matched_intervals(ref, est)]
    n_ref = sum(len(ref) for ref, _ in pairs)
    print(f"\n{len(ints)} of {n_ref} reference notes have an estimate starting within 100 ms. Pitch differences:")
    for d, c in Counter(ints).most_common(6):
        print(f"  {d:+3d} semitones: {c} notes ({100 * c / max(len(ints), 1):.0f}%)")
    best = max(scores, key=scores.get)
    print(f"\nbest shift k = {best:+d}  (F1 {scores[best]:.3f}, without shifting {scores[0]:.3f})")
    if best == 0:
        print("-> no constant pitch offset.")
    else:
        print(f"-> the estimates are systematically {-best:+d} semitones away from the reference.")
    return best


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    a = sub.add_parser("labels")
    a.add_argument("--cache-dir", required=True)
    a.add_argument("--audio-dir", required=True)
    a.add_argument("--n", type=int, default=8)
    b = sub.add_parser("model")
    b.add_argument("--weights", required=True)
    b.add_argument("--cache-dir", required=True)
    b.add_argument("--split", default="test")
    b.add_argument("--max-pieces", type=int, default=40)
    args = ap.parse_args()

    names = evenly(pieces_for(args.cache_dir) if args.mode == "labels"
                   else pieces_for(args.cache_dir, args.split),
                   args.n if args.mode == "labels" else args.max_pieces)
    if not names:
        raise SystemExit("nothing cached there")
    pairs = []
    if args.mode == "labels":
        from features import load_audio
        from voice_core import pyin_notes
        for name in names:
            wav = next((p for p in Path(args.audio_dir).rglob(f"{name}.*") if p.suffix.lower() in (".wav", ".flac", ".mp3")), None)
            if wav is None:
                continue
            print("pYIN on", name)
            pairs.append((np.load(cache_paths(args.cache_dir, name)[3]), pyin_notes(load_audio(str(wav)))))
        report(pairs, "pYIN (as pitch truth) vs MIDI labels")
    else:
        from guitar_evaluate import load_piece, load_weights, pick_device, predict_full
        from model import PianoCNN
        from voice_core import mono_decode
        device = pick_device()
        model = PianoCNN().to(device)
        load_weights(model, args.weights, device)
        model.eval()
        for name in names:
            cqt, ref = load_piece(args.cache_dir, name)
            fp, op = predict_full(model, cqt, device)
            pairs.append((ref, mono_decode(fp, op)))
        report(pairs, "trained model vs reference notes")


if __name__ == "__main__":
    main()