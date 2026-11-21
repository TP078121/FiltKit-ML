"""Numeric alignment check between labels and audio (instead of judging a plot by eye).

For every cached piece it measures how strongly the spectrogram "jumps" (spectral flux) around each
labelled hit, for time shifts of -10..+10 frames (10 ms each), and reports the shift where the
audio matches the labels best.

Run it on BOTH data sets and compare the two numbers:
  python src/drum_check_offset.py            (synthetic data: aligned by construction = the reference)
  python src/drum_check_offset.py --real     (real E-GMD)
The best shift is not exactly 0 even when everything is aligned (the analysis window sees a hit a
little before it happens), so only the DIFFERENCE between real and synthetic matters.
"""
import argparse

import numpy as np

from config import DRUM_CACHE_DIR, DRUM_FPS, DRUM_SYNTH_DIR
from drum_core import cache_paths, pieces_for


def flux(mel):
    d = np.maximum(0.0, np.diff(mel, axis=1, prepend=mel[:, :1])).sum(axis=0)
    return (d - d.mean()) / (d.std() + 1e-9)


def best_shift(cache_dir, max_lag=10):
    lags = list(range(-max_lag, max_lag + 1))
    score, n = np.zeros(len(lags)), 0
    for name in pieces_for(cache_dir):
        mel_p, _, ref_p = cache_paths(cache_dir, name)
        f = flux(np.load(mel_p).astype(np.float32))
        frames = np.unique(np.round(np.load(ref_p)[:, 0] * DRUM_FPS).astype(int))
        if len(frames) == 0:
            continue
        for i, lag in enumerate(lags):
            idx = frames + lag
            idx = idx[(idx >= 0) & (idx < len(f))]
            if len(idx):
                score[i] += f[idx].mean()
        n += 1
    return lags, score / max(n, 1), n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--max-lag", type=int, default=10)
    args = ap.parse_args()
    cache_dir = DRUM_CACHE_DIR if args.real else DRUM_SYNTH_DIR
    lags, score, n = best_shift(cache_dir, args.max_lag)
    if n == 0:
        raise SystemExit(f"Nothing cached in {cache_dir}")
    best = lags[int(score.argmax())]
    print(f"{n} pieces in {cache_dir}")
    print("shift (frames) -> mean energy jump at label+shift")
    for lag, s in zip(lags, score):
        print(f"{lag:+4d}  {s:6.2f} " + "#" * int(max(s, 0) * 8) + ("   <== best" if lag == best else ""))
    print(f"\nbest shift: {best:+d} frames = {best * 1000 / DRUM_FPS:+.0f} ms")