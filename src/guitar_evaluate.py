"""Evaluate a trained guitar model (note-level F1) and tune the thresholds.

  python src/guitar_evaluate.py --weights checkpoints/guitar_slakh/best.pt --cache-dir data/cache/guitarset --split test
"""
import argparse

import numpy as np
import torch

from config import ROOT
from guitar_core import GUITARSET_CACHE, cache_paths, decode_notes, note_f1, pieces_for
from model import PianoCNN            # the same CQT -> 88-pitch network, trained on guitar


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_weights(model, path, device):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"] if isinstance(ck, dict) and "model" in ck else ck)


def predict_full(model, cqt, device, chunk=2048, ctx=16):
    """Whole piece in chunks with a little overlap. cqt (264, T) -> two (T, 88) probability arrays."""
    T = cqt.shape[1]
    frames, onsets = [], []
    with torch.no_grad():
        for s in range(0, T, chunk):
            a = max(0, s - ctx)
            b = min(T, s + chunk + ctx)
            x = torch.from_numpy(np.ascontiguousarray(cqt[:, a:b], dtype=np.float32))[None, None].to(device)
            f, o = model(x)
            f = torch.sigmoid(f)[0].cpu().numpy()
            o = torch.sigmoid(o)[0].cpu().numpy()
            lo = s - a
            hi = lo + min(chunk, T - s)
            frames.append(f[lo:hi])
            onsets.append(o[lo:hi])
    return np.concatenate(frames), np.concatenate(onsets)


def evenly(names, n):
    if n is None or len(names) <= n:
        return list(names)
    idx = np.linspace(0, len(names) - 1, n).round().astype(int)
    return [names[i] for i in idx]


def load_piece(cache_dir, name):
    cqt_p, _, _, nt_p = cache_paths(cache_dir, name)
    return np.load(cqt_p).astype(np.float32), np.load(nt_p)


def evaluate_split(model, device, cache_dir, split="validation", max_pieces=10, frame_thr=0.5, onset_thr=0.5):
    """Mean note-level F1 over pieces. Returns (onset_f1, onset_offset_f1, n_pieces) or None."""
    names = evenly(pieces_for(cache_dir, split), max_pieces)
    if not names:
        return None
    was_training = model.training
    model.eval()
    on_s, full_s = [], []
    for name in names:
        cqt, ref = load_piece(cache_dir, name)
        fp, op = predict_full(model, cqt, device)
        on, full = note_f1(ref, decode_notes(fp, op, frame_thr, onset_thr))
        on_s.append(on)
        full_s.append(full)
    if was_training:
        model.train()
    return float(np.mean(on_s)), float(np.mean(full_s)), len(names)


def main():
    ap = argparse.ArgumentParser(description="Evaluate guitar model and tune thresholds")
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "guitar1" / "best.pt"))
    ap.add_argument("--cache-dir", default=str(GUITARSET_CACHE))
    ap.add_argument("--split", default="validation")
    ap.add_argument("--max-pieces", type=int, default=30)
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
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7]:
        on_s, full_s = [], []
        for ref, fp, op in data:
            on, full = note_f1(ref, decode_notes(fp, op, thr, thr))
            on_s.append(on)
            full_s.append(full)
        print(f"{thr:9.2f} {np.mean(on_s):9.3f} {np.mean(full_s):10.3f}")
    print("\nTune on validation, then report the test split ONCE at the end.")


if __name__ == "__main__":
    main()