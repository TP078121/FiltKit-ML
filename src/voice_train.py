"""Train the voice model. This re-uses guitar_train.py (same network, same data format) but
validates with the monophonic voice decoder, and has voice defaults.

  python src/voice_train.py --val-cache data/cache/voice_pairs --epochs 30 --steps 300 --batch-size 16 --workers 4 --init-from checkpoints/piano_best.pt

Defaults added here (override them by passing the flag yourself):
  --train-cache data/cache/voice_pairs   --run voice1   --pitch-aug 0.8   --max-shift 12
The large pitch shifts (+-12 semitones) matter: the training singer cannot cover every voice, and without
them the model repeats the octave it knows (octave errors on other singers).
Everything else works exactly as in guitar_train.py (--resume, --lr, ...).
"""
import sys

import guitar_train
import voice_evaluate
from voice_core import VOICE_PAIRS_CACHE

guitar_train.evaluate_split = voice_evaluate.evaluate_split     # validate with mono_decode

DEFAULTS = [("--run", "voice1"), ("--train-cache", str(VOICE_PAIRS_CACHE)),
            ("--pitch-aug", "0.8"), ("--max-shift", "12")]


if __name__ == "__main__":          # required on Windows (DataLoader workers)
    for flag, value in DEFAULTS:
        if flag not in sys.argv:
            sys.argv += [flag, value]
    guitar_train.main()