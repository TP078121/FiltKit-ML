"""Evaluate the piano model on MAESTRO (note-level F1) and tune the thresholds.

  python src/piano_evaluate.py --weights checkpoints/piano_full1/best.pt --split validation --max-pieces 30
  python src/piano_evaluate.py --weights checkpoints/piano_full1/best.pt --split test --max-pieces 177 --frame-thr 0.6 --onset-thr 0.6

Without --frame-thr/--onset-thr it tries one threshold at a time (frame = onset) and prints a table.
Pieces are spread evenly over the split (different composers and years), not just the first few.
Tune the threshold on VALIDATION; run the TEST split only once, at the end, with the chosen value.
"""
import argparse

import numpy as np
import torch

from config import ROOT, MAESTRO_DIR
from evaluate import note_f1, predict_full
from guitar_core import decode_notes
from maestro_dataset import cache_name, cached_paths, read_rows
from model import PianoCNN


def evenly(items, n):
    if n is None or len(items) <= n:
        return list(items)
    return [items[i] for i in np.linspace(0, len(items) - 1, n).round().astype(int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "piano_full1" / "best.pt"))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-pieces", type=int, default=30)
    ap.add_argument("--frame-thr", type=float, default=None)
    ap.add_argument("--onset-thr", type=float, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = PianoCNN().to(device)
    ck = torch.load(args.weights, map_location=device)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)
    model.eval()

    rows = [r for r in read_rows(args.split) if all(p.exists() for p in cached_paths(cache_name(r)))]
    rows = evenly(rows, args.max_pieces)
    if not rows:
        raise SystemExit(f"No cached pieces for split '{args.split}'. Run prepare_maestro.py first.")
    print(f"{len(rows)} pieces, split '{args.split}'. Computing predictions...")
    data = []
    for r in rows:
        cqt = np.load(cached_paths(cache_name(r))[0]).astype(np.float32)
        fp, op = predict_full(model, cqt, device)
        data.append((MAESTRO_DIR / r["midi_filename"], fp, op))

    def score(ft, ot):
        on, full = [], []
        for midi, fp, op in data:
            a, b = note_f1(midi, decode_notes(fp, op, ft, ot, lo=21, hi=108))     # whole piano range
            on.append(a)
            full.append(b)
        return np.array(on), np.array(full)

    if args.frame_thr is not None or args.onset_thr is not None:
        ft = args.frame_thr if args.frame_thr is not None else 0.5
        ot = args.onset_thr if args.onset_thr is not None else 0.5
        on, full = score(ft, ot)
        print(f"frame threshold {ft}, onset threshold {ot}")
        print(f"  onset F1          {on.mean():.3f}  (std over pieces {on.std():.3f}, min {on.min():.3f}, max {on.max():.3f})")
        print(f"  onset+offset F1   {full.mean():.3f}  (std over pieces {full.std():.3f})")
        return

    print(f"\n{'threshold':>9s} {'onset F1':>9s} {'on+off F1':>10s}   (frame and onset threshold equal)")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        on, full = score(thr, thr)
        print(f"{thr:9.2f} {on.mean():9.3f} {full.mean():10.3f}")
    print("\nPick the threshold on validation, then run the test split once with --frame-thr and --onset-thr.")


if __name__ == "__main__":
    main()