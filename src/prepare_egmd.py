"""One-time preprocessing of real E-GMD audio: log-mel features + onset labels -> data/cache/egmd.
Safe to stop and re-run (finished pieces are skipped).

  python src/prepare_egmd.py --fraction 0.2 --workers 4
  python src/prepare_egmd.py --limit 20            (quick timing test)

E-GMD re-records every drum pattern on 43 kits, so a fraction of it (for example 0.2,
spread evenly over splits) already gives lots of variety. The cache needs roughly
100 MB per hour of audio (my estimate).
"""
import argparse
import csv
import math
import random
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from config import *
from drum_core import (cache_name, cache_paths, events_to_targets, midi_to_events,
                       piece_cached, write_index)
from drum_features import compute_logmel, load_audio


def read_rows():
    if not EGMD_CSV.exists():
        raise SystemExit(f"Metadata csv not found: {EGMD_CSV}\nExtract E-GMD into the data folder first.")
    with open(EGMD_CSV, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("audio_filename")]


def process(row):
    name = cache_name(row)
    if piece_cached(DRUM_CACHE_DIR, name):
        return name, "already cached"
    wav = EGMD_DIR / row["audio_filename"]
    mid = EGMD_DIR / row["midi_filename"]
    mel = compute_logmel(load_audio(str(wav)))
    ev = midi_to_events(mid)
    tgt = events_to_targets(ev, mel.shape[1])
    mel_p, on_p, ref_p = cache_paths(DRUM_CACHE_DIR, name)
    np.save(on_p, tgt.astype(np.float16))
    np.save(ref_p, ev)
    np.save(mel_p, mel.astype(np.float16))                      # saved last = "finished"
    return name, "done"


def choose(rows, fraction, limit):
    chosen = []
    for split in ("train", "validation", "test"):
        group = [r for r in rows if r["split"] == split]
        random.Random(split).shuffle(group)                     # same order every run
        chosen += group[:math.ceil(len(group) * fraction)]
    if limit:
        random.Random("limit").shuffle(chosen)
        chosen = chosen[:limit]
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--fraction", type=float, default=0.2, help="fraction of each split to preprocess")
    ap.add_argument("--limit", type=int, default=None, help="only N pieces (timing test)")
    args = ap.parse_args()

    rows = read_rows()
    entries = [(cache_name(r), r["split"]) for r in rows]
    available = [r for r in rows if (EGMD_DIR / r["audio_filename"]).exists()
                 and (EGMD_DIR / r["midi_filename"]).exists()]
    print(f"{len(available)} of {len(rows)} pieces have audio and MIDI on disk")
    if not available:
        raise SystemExit("No audio found. Download/extract E-GMD (or run fetch_egmd_samples.py) first.")

    todo = choose(available, args.fraction, args.limit)
    print(f"preprocessing {len(todo)} pieces (fraction {args.fraction})")
    DRUM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    write_index(DRUM_CACHE_DIR, entries)                        # picks up earlier runs

    try:
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as ex:
                for i, (name, status) in enumerate(ex.map(process, todo), 1):
                    print(f"[{i}/{len(todo)}] {status}: {name[:70]}")
        else:
            for i, row in enumerate(todo, 1):
                name, status = process(row)
                print(f"[{i}/{len(todo)}] {status}: {name[:70]}")
    finally:
        n = write_index(DRUM_CACHE_DIR, entries)
        print(f"index written: {n} cached pieces in {DRUM_CACHE_DIR}")


if __name__ == "__main__":       # required on Windows for multiprocessing
    main()
