"""Plot a cached guitar piece: CQT on top, labels below. Notes should line up with the streaks.

  python src/guitar_check_alignment.py                           (GuitarSet)
  python src/guitar_check_alignment.py --cache-dir data/cache/slakh_guitar
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from config import ROOT
from guitar_core import FPS, GUITARSET_CACHE, cache_paths, pieces_for

ap = argparse.ArgumentParser()
ap.add_argument("--cache-dir", default=str(GUITARSET_CACHE))
ap.add_argument("--name", default=None)
ap.add_argument("--seconds", type=float, default=20.0)
args = ap.parse_args()

names = pieces_for(args.cache_dir)
if not names:
    raise SystemExit(f"Nothing cached in {args.cache_dir}")
name = args.name or names[0]
cqt_p, fr_p, _, _ = cache_paths(args.cache_dir, name)
cqt = np.load(cqt_p).astype(np.float32)
frame = np.load(fr_p).astype(np.float32)
n = min(cqt.shape[1], int(args.seconds * FPS))
secs = n / FPS
print("piece:", name, " cqt", cqt.shape, " roll", frame.shape)

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
ax[0].imshow(cqt[:, :n], origin="lower", aspect="auto", extent=[0, secs, 0, cqt.shape[0]])
ax[0].set_title("CQT (audio)")
ax[1].imshow(frame[:, :n], origin="lower", aspect="auto", extent=[0, secs, 0, 88], cmap="gray_r")
ax[1].set_title("Guitar notes (labels)")
ax[1].set_xlabel("seconds")
plt.tight_layout()
(ROOT / "output").mkdir(exist_ok=True)
plt.savefig(ROOT / "output" / "guitar_alignment.png", dpi=120)
plt.show()