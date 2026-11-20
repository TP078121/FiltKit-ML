"""Plot a cached piece: log-mel spectrogram on top, onset labels below.
Every labelled hit should line up with a vertical streak in the spectrogram.

  python src/drum_check_alignment.py                      (synthetic test data)
  python src/drum_check_alignment.py --real               (real E-GMD after prepare_egmd.py)
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from config import *
from drum_core import DRUM_CLASSES, N_DRUM, cache_paths, pieces_for

ap = argparse.ArgumentParser()
ap.add_argument("--real", action="store_true", help="use data/cache/egmd instead of the synthetic folder")
ap.add_argument("--name", default=None)
ap.add_argument("--seconds", type=float, default=10.0)
args = ap.parse_args()

cache_dir = DRUM_CACHE_DIR if args.real else DRUM_SYNTH_DIR
names = pieces_for(cache_dir)
if not names:
    raise SystemExit(f"Nothing cached in {cache_dir}")
name = args.name or names[0]
mel_p, on_p, _ = cache_paths(cache_dir, name)
mel = np.load(mel_p).astype(np.float32)
on = np.load(on_p).astype(np.float32)
n = min(mel.shape[1], int(args.seconds * DRUM_FPS))
secs = n / DRUM_FPS
print("piece:", name, " mel", mel.shape, " onsets", on.shape)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(13, 7))
ax[0].imshow(mel[:, :n], origin="lower", aspect="auto", extent=[0, secs, 0, mel.shape[0]])
ax[0].set_title("Log-mel spectrogram")
ax[1].imshow(on[:n].T, origin="lower", aspect="auto", extent=[0, secs, -0.5, N_DRUM - 0.5], cmap="gray_r")
ax[1].set_yticks(range(N_DRUM))
ax[1].set_yticklabels(DRUM_CLASSES)
ax[1].set_title("Onset labels")
ax[1].set_xlabel("seconds")
plt.tight_layout()
(ROOT / "output").mkdir(exist_ok=True)
plt.savefig(ROOT / "output" / "drum_alignment.png", dpi=120)
plt.show()
