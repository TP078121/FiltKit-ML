import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from config import *
from features import load_audio, compute_cqt
from labels import midi_to_rolls
from maestro_dataset import read_rows, cache_name, cached_paths


def process(row):
    name = cache_name(row)
    paths = cached_paths(name)
    if all(p.exists() for p in paths):
        return name, "already cached"

    wav = MAESTRO_DIR / row["audio_filename"]
    mid = MAESTRO_DIR / row["midi_filename"]
    cqt = compute_cqt(load_audio(str(wav)))
    frame, onset = midi_to_rolls(str(mid), cqt.shape[1])

    # cqt is saved last, so "all three files exist" means the piece finished
    np.save(paths[1], frame.astype(np.uint8))
    np.save(paths[2], onset.astype(np.uint8))
    np.save(paths[0], cqt.astype(np.float16))
    return name, "done"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=1, help="parallel processes")
    ap.add_argument("--limit", type=int, default=None, help="only first N available pieces")
    args = ap.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    available = [r for r in rows if (MAESTRO_DIR / r["audio_filename"]).exists()]
    print(f"{len(available)} of {len(rows)} pieces have audio on disk")
    if not available:
        raise SystemExit("No audio files found under data/maestro-v3.0.0. "
                         "Download/extract the MAESTRO audio first.")
    if args.limit:
        available = available[:args.limit]

    if args.workers > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            results = ex.map(process, available)
            for i, (name, status) in enumerate(results, 1):
                print(f"[{i}/{len(available)}] {status}: {name[:70]}")
    else:
        for i, row in enumerate(available, 1):
            name, status = process(row)
            print(f"[{i}/{len(available)}] {status}: {name[:70]}")
    print("finished. cache folder:", CACHE_DIR)


if __name__ == "__main__":      # required on Windows for multiprocessing
    main()