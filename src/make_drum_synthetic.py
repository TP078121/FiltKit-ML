"""Laptop test WITHOUT downloading the 90 GB audio.
Takes real E-GMD drum patterns (MIDI only) and renders them with a toy drum synth,
so the whole pipeline (labels, features, model, training, decoding, scoring) can be tested.

  python src/make_drum_synthetic.py
"""
import argparse
import random

import numpy as np
import soundfile as sf

from config import *
from drum_core import cache_paths, events_to_targets, midi_to_events, write_index
from drum_features import compute_logmel
from drum_midi import render_drums


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(p for d in (ROOT / "data").glob("e-gmd*") if d.is_dir() for p in d.rglob("*.mid*"))
    print("MIDI files found:", len(files))
    if not files:
        raise SystemExit("No MIDI files. Extract e-gmd-v1.0.0-midi.zip (about 103 MB) into the data folder.")
    random.Random(args.seed).shuffle(files)

    out = DRUM_SYNTH_DIR
    (out / "wav").mkdir(parents=True, exist_ok=True)
    need = args.n_train + args.n_val
    entries, made = [], 0
    for path in files:
        if made >= need:
            break
        try:
            ev = midi_to_events(path)
        except Exception:
            continue
        if len(ev) < 20:
            continue
        t0 = max(0.0, ev[0, 0] - 0.5)
        sel = ev[(ev[:, 0] >= t0) & (ev[:, 0] < t0 + args.seconds - 0.2)].copy()
        sel[:, 0] -= t0
        if len(sel) < 20:
            continue

        name = f"synth_{made:02d}"
        y = render_drums(sel, length_seconds=args.seconds, seed=made, vary=True)
        mel = compute_logmel(y)
        tgt = events_to_targets(sel, mel.shape[1])
        mel_p, on_p, ref_p = cache_paths(out, name)
        np.save(on_p, tgt.astype(np.float16))
        np.save(ref_p, sel)
        np.save(mel_p, mel.astype(np.float16))                    # saved last = "finished"
        sf.write(str(out / "wav" / f"{name}.wav"), y, DRUM_SR)

        split = "train" if made < args.n_train else "validation"
        entries.append((name, split))
        print(f"{name}  {split:10s} {len(sel):4d} hits  from {path.name[:50]}")
        made += 1

    n = write_index(out, entries)
    print(f"\nwrote {n} pieces to {out}")
    print("listen to the wav files in", out / "wav")


if __name__ == "__main__":
    main()
