"""Evaluate a trained drum model, and tune the detection threshold.

  python src/drum_evaluate.py --weights checkpoints/drums1/best.pt --split validation
"""
import argparse

import numpy as np
import torch

from config import *
from drum_core import (DRUM_CLASSES, N_DRUM, cache_paths, pieces_for,
                       pick_peaks, piece_counts, summarize)
from drum_model import DrumCRNN


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_full(model, mel, device, chunk=3000, ctx=200):
    """Whole piece through the model in chunks with overlap (the GRU needs context).
    mel (n_mels, T) -> probabilities (T, N_DRUM)."""
    T = mel.shape[1]
    outs = []
    with torch.no_grad():
        for s in range(0, T, chunk):
            a = max(0, s - ctx)
            b = min(T, s + chunk + ctx)
            x = torch.from_numpy(np.ascontiguousarray(mel[:, a:b], dtype=np.float32))[None, None].to(device)
            p = torch.sigmoid(model(x))[0].cpu().numpy()
            lo = s - a
            outs.append(p[lo:lo + min(chunk, T - s)])
    return np.concatenate(outs)


def evenly(names, n):
    """n names spread evenly over the list (so we see different drummers and kits)."""
    if n is None or len(names) <= n:
        return list(names)
    idx = np.linspace(0, len(names) - 1, n).round().astype(int)
    return [names[i] for i in idx]


def load_piece(cache_dir, name):
    mel_p, _, ref_p = cache_paths(cache_dir, name)
    return np.load(mel_p).astype(np.float32), np.load(ref_p)


def evaluate_split(model, device, cache_dir, split="validation", max_pieces=20,
                   thr=0.5, window=2, tol=0.05):
    """Returns (per_class_f1, micro_f1, macro_f1, n_pieces), or None if nothing is cached."""
    names = evenly(pieces_for(cache_dir, split), max_pieces)
    if not names:
        return None
    was_training = model.training
    model.eval()
    total = np.zeros((N_DRUM, 3), dtype=np.int64)
    for name in names:
        mel, ref = load_piece(cache_dir, name)
        prob = predict_full(model, mel, device)
        total += piece_counts(ref, pick_peaks(prob, thr, window), tol)
    if was_training:
        model.train()
    per_class, micro, macro = summarize(total)
    return per_class, micro, macro, len(names)


def main():
    ap = argparse.ArgumentParser(description="Evaluate drum model and tune thresholds")
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "drums1" / "best.pt"))
    ap.add_argument("--cache-dir", default=str(DRUM_CACHE_DIR))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-pieces", type=int, default=30)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.05, help="match tolerance in seconds")
    args = ap.parse_args()

    device = pick_device()
    model = DrumCRNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    names = evenly(pieces_for(args.cache_dir, args.split), args.max_pieces)
    if not names:
        raise SystemExit(f"No cached pieces for split '{args.split}' in {args.cache_dir}")
    print(f"{len(names)} pieces, split '{args.split}'. Computing predictions...")
    data = []
    for name in names:
        mel, ref = load_piece(args.cache_dir, name)
        data.append((ref, predict_full(model, mel, device)))

    grid = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    per_class_by_thr = []
    print(f"\n{'threshold':>9s} {'micro F1':>9s} {'macro F1':>9s}")
    for thr in grid:
        total = np.zeros((N_DRUM, 3), dtype=np.int64)
        for ref, prob in data:
            total += piece_counts(ref, pick_peaks(prob, thr, args.window), args.tol)
        per_class, micro, macro = summarize(total)
        per_class_by_thr.append(per_class)
        print(f"{thr:9.2f} {micro:9.3f} {macro:9.3f}")

    pc = np.array(per_class_by_thr)                  # (n_thr, N_DRUM)
    print("\nbest threshold per class (F1 at that threshold):")
    best = []
    for k, name in enumerate(DRUM_CLASSES):
        j = int(pc[:, k].argmax())
        best.append(grid[j])
        print(f"  {name:14s} thr {grid[j]:.2f}  F1 {pc[j, k]:.3f}")
    print("\nper-class thresholds to use:", best)
    print("(classes with no hits in this split show F1 0.000; tune on validation, report on test)")


if __name__ == "__main__":
    main()
