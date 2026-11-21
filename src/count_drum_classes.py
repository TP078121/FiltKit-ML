import numpy as np
from config import DRUM_SYNTH_DIR
from drum_core import DRUM_CLASSES, N_DRUM, cache_paths, pieces_for

for split in ("train", "validation"):
    total = np.zeros(N_DRUM, dtype=int)
    for name in pieces_for(DRUM_SYNTH_DIR, split):
        ref = np.load(cache_paths(DRUM_SYNTH_DIR, name)[2])      # (N, 2): time, class
        for k in range(N_DRUM):
            total[k] += int((ref[:, 1] == k).sum())
    print(split, dict(zip(DRUM_CLASSES, total.tolist())))