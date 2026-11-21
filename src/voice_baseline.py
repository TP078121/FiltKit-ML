"""Non-ML baseline: pYIN pitch tracking + simple note segmentation, scored on vocadito.
This is the number your trained model has to beat.

  python src/voice_baseline.py
"""
import argparse
from pathlib import Path

import numpy as np

from features import load_audio
from guitar_core import cache_paths, pieces_for
from voice_core import VOCADITO_DIR, note_f1, pyin_notes, vocadito_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(VOCADITO_DIR))
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    args = ap.parse_args()

    est_cache = {}
    for ann in args.annotators:
        cache = vocadito_cache(ann)
        names = pieces_for(cache)
        if not names:
            print(f"annotator {ann}: nothing cached, run prepare_vocadito.py first")
            continue
        on_s, full_s = [], []
        for name in names:
            if name not in est_cache:
                wav = next(iter(Path(args.dir).rglob(f"{name}.wav")), None)
                est_cache[name] = pyin_notes(load_audio(str(wav)))
            ref = np.load(cache_paths(cache, name)[3])
            on, full = note_f1(ref, est_cache[name])
            on_s.append(on)
            full_s.append(full)
        print(f"pYIN baseline vs annotator {ann}: onset F1 {np.mean(on_s):.3f}   "
              f"onset+offset F1 {np.mean(full_s):.3f}   ({len(names)} tracks)")


if __name__ == "__main__":
    main()