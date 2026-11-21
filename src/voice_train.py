"""Train the voice model. This re-uses guitar_train.py (same network, same data format) but
validates with the monophonic voice decoder, and has voice defaults.

  python src/voice_train.py --val-cache data/cache/voice_pairs --epochs 30 --steps 500 --batch-size 16 --workers 4 --init-from checkpoints/run1/best.pt --run voice1

Defaults: --train-cache data/cache/voice_pairs   --run voice1   (so it never overwrites a guitar run)
Everything else works exactly as in guitar_train.py (--resume, --lr, --pitch-aug, ...).
"""
import sys

import guitar_train
import voice_evaluate
from voice_core import VOICE_PAIRS_CACHE

guitar_train.evaluate_split = voice_evaluate.evaluate_split     # validate with mono_decode


if __name__ == "__main__":          # required on Windows (DataLoader workers)
    if "--run" not in sys.argv:
        sys.argv += ["--run", "voice1"]
    if "--train-cache" not in sys.argv:
        sys.argv += ["--train-cache", str(VOICE_PAIRS_CACHE)]
    guitar_train.main()