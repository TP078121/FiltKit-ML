"""Preprocess GuitarSet (real acoustic guitar, 360 excerpts of about 30 s) -> data/cache/guitarset.

Download from https://zenodo.org/records/3371780 :
  annotation.zip        (39.1 MB)
  audio_mono-mic.zip    (656.9 MB)
Extract both into data/guitarset/ (any sub-folders are fine), then:

  python src/prepare_guitarset.py

The split is by player: players 00-03 train, 04 validation, 05 test.
"""
import argparse
from pathlib import Path

import numpy as np

from features import load_audio, compute_cqt
from guitar_core import (GUITARSET_CACHE, GUITARSET_DIR, guitarset_split, load_jams_notes,
                         piece_cached, save_piece, write_index)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(GUITARSET_DIR))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    root = Path(args.dir)
    jams = sorted(root.rglob("*.jams"))
    wavs = {p.name: p for p in root.rglob("*.wav")}
    print(f"found {len(jams)} .jams files and {len(wavs)} .wav files in {root}")
    if not jams or not wavs:
        raise SystemExit("Extract annotation.zip and audio_mono-mic.zip into data/guitarset/ first.")
    if args.limit:
        jams = jams[:args.limit]

    entries, skipped = [], 0
    for i, j in enumerate(jams, 1):
        name = j.stem
        mic = wavs.get(name + "_mic.wav") or next(
            (p for n, p in wavs.items() if n.startswith(name) and "mic" in n), None)
        if mic is None:
            skipped += 1
            continue
        entries.append((name, guitarset_split(name)))
        if piece_cached(GUITARSET_CACHE, name):
            continue
        cqt = compute_cqt(load_audio(str(mic)))
        notes = load_jams_notes(j)
        notes[:, 2] = np.round(notes[:, 2])          # whole MIDI pitches (annotations have fine tuning)
        save_piece(GUITARSET_CACHE, name, cqt, notes)
        if i % 20 == 0 or i == len(jams):
            print(f"[{i}/{len(jams)}] {name}  {len(notes)} notes")

    n = write_index(GUITARSET_CACHE, entries)
    counts = {s: sum(1 for _, sp in entries if sp == s) for s in ("train", "validation", "test")}
    print(f"cached {n} pieces in {GUITARSET_CACHE}  (by split: {counts}; skipped without mic audio: {skipped})")


if __name__ == "__main__":
    main()