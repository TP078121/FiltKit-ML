"""Evaluate a trained voice model (note-level F1, monophonic decoding) and tune the thresholds.

  python src/voice_evaluate.py --weights checkpoints/voice1/best.pt --cache-dir data/cache/vocadito_A1 --split test
"""
import argparse

import numpy as np

from config import ROOT
from guitar_core import pieces_for
from guitar_evaluate import evenly, load_piece, load_weights, pick_device, predict_full
from model import PianoCNN
from voice_core import mono_decode, note_f1, vocadito_cache


def evaluate_split(model, device, cache_dir, split="validation", max_pieces=10, frame_thr=0.5, onset_thr=0.5):
    """Mean note-level F1. Returns (onset_f1, onset_offset_f1, n_pieces) or None."""
    names = evenly(pieces_for(cache_dir, split), max_pieces)
    if not names:
        return None
    was_training = model.training
    model.eval()
    on_s, full_s = [], []
    for name in names:
        cqt, ref = load_piece(cache_dir, name)
        fp, op = predict_full(model, cqt, device)
        on, full = note_f1(ref, mono_decode(fp, op, frame_thr, onset_thr))
        on_s.append(on)
        full_s.append(full)
    if was_training:
        model.train()
    return float(np.mean(on_s)), float(np.mean(full_s)), len(names)


def main():
    ap = argparse.ArgumentParser(description="Evaluate voice model and tune thresholds")
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "voice1" / "best.pt"))
    ap.add_argument("--cache-dir", default=str(vocadito_cache("A1")))
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-pieces", type=int, default=40)
    args = ap.parse_args()

    device = pick_device()
    model = PianoCNN().to(device)
    load_weights(model, args.weights, device)
    model.eval()

    names = evenly(pieces_for(args.cache_dir, args.split), args.max_pieces)
    if not names:
        raise SystemExit(f"No cached pieces for split '{args.split}' in {args.cache_dir}")
    print(f"{len(names)} pieces, split '{args.split}'. Computing predictions...")
    data = []
    for name in names:
        cqt, ref = load_piece(args.cache_dir, name)
        data.append((ref, *predict_full(model, cqt, device)))

    print(f"\n{'threshold':>9s} {'onset F1':>9s} {'on+off F1':>10s}   (frame and onset threshold equal)")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        on_s, full_s = [], []
        for ref, fp, op in data:
            on, full = note_f1(ref, mono_decode(fp, op, thr, thr))
            on_s.append(on)
            full_s.append(full)
        print(f"{thr:9.2f} {np.mean(on_s):9.3f} {np.mean(full_s):10.3f}")
    print("\nCompare with the pYIN baseline (voice_baseline.py). Use the same annotator for both.")


if __name__ == "__main__":
    main()