"""Preprocess the GUITAR parts of Slakh2100 -> data/cache/slakh_guitar.

  python src/prepare_slakh_guitar.py --slakh-dir data/slakh2100_flac_redux --fraction 0.3 --workers 4

Slakh2100 is synthesized (MIDI rendered with virtual instruments), so labels are perfect, but
the sound is not a real guitar. Use it to pre-train, then fine-tune on GuitarSet.

--source stem : input = the isolated guitar stem   (default; cleanest)
--source mix  : input = the full band mix, labels = all guitars (harder, learns to ignore other instruments)

Expected folder: <slakh-dir>/{train,validation,test}/TrackXXXXX/{mix.flac, stems/Sxx.flac, MIDI/Sxx.mid, metadata.yaml}
BabySlakh (small test set, no split folders) also works: all its tracks are used as 'train'.
"""
import argparse
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from features import load_audio, compute_cqt
from guitar_core import (SLAKH_CACHE, find_audio, guitar_stem_ids, piece_cached, read_midi_notes,
                         save_piece, write_index)


def find_jobs(slakh_dir, source):
    root = Path(slakh_dir)
    split_dirs = [(s, root / s) for s in ("train", "validation", "test") if (root / s).is_dir()]
    if not split_dirs:
        print("no train/validation/test folders found: treating every Track folder as 'train'")
        split_dirs = [("train", root)]
    jobs = []
    for split, d in split_dirs:
        for track in sorted(d.glob("Track*")):
            meta = track / "metadata.yaml"
            if not meta.exists():
                continue
            ids = guitar_stem_ids(meta)
            if not ids:
                continue
            if source == "stem":
                for sid in ids:
                    audio = find_audio(track / "stems" / sid)
                    midi = track / "MIDI" / f"{sid}.mid"
                    if audio and midi.exists():
                        jobs.append(dict(name=f"{track.name}_{sid}", split=split, audio=str(audio), midis=[str(midi)]))
            else:
                audio = find_audio(track / "mix")
                midis = [str(track / "MIDI" / f"{sid}.mid") for sid in ids if (track / "MIDI" / f"{sid}.mid").exists()]
                if audio and midis:
                    jobs.append(dict(name=f"{track.name}_mix", split=split, audio=str(audio), midis=midis))
    return jobs


def process(job):
    if piece_cached(SLAKH_CACHE, job["name"]):
        return job["name"], "already cached"
    cqt = compute_cqt(load_audio(job["audio"]))
    notes = np.concatenate([read_midi_notes(m) for m in job["midis"]])
    notes = notes[np.argsort(notes[:, 0], kind="stable")]
    save_piece(SLAKH_CACHE, job["name"], cqt, notes)
    return job["name"], f"done ({len(notes)} notes)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh-dir", required=True)
    ap.add_argument("--source", choices=["stem", "mix"], default="stem")
    ap.add_argument("--fraction", type=float, default=0.3, help="fraction of each split to preprocess")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    jobs = find_jobs(args.slakh_dir, args.source)
    print(f"{len(jobs)} guitar pieces found ({args.source} mode)")
    if not jobs:
        raise SystemExit("Nothing found. Check --slakh-dir (it should contain train/validation/test or Track folders).")
    entries = [(j["name"], j["split"]) for j in jobs]

    chosen = []
    for split in ("train", "validation", "test"):
        group = [j for j in jobs if j["split"] == split]
        random.Random(split).shuffle(group)                       # same order every run
        chosen += group[:math.ceil(len(group) * args.fraction)]
    if args.limit:
        random.Random("limit").shuffle(chosen)
        chosen = chosen[:args.limit]
    print(f"preprocessing {len(chosen)} pieces (fraction {args.fraction})")

    write_index(SLAKH_CACHE, entries)                             # picks up earlier runs
    try:
        if args.workers > 1:
            with ProcessPoolExecutor(args.workers) as ex:
                for i, (name, status) in enumerate(ex.map(process, chosen), 1):
                    print(f"[{i}/{len(chosen)}] {status}: {name}")
        else:
            for i, job in enumerate(chosen, 1):
                name, status = process(job)
                print(f"[{i}/{len(chosen)}] {status}: {name}")
    finally:
        print(f"index written: {write_index(SLAKH_CACHE, entries)} cached pieces in {SLAKH_CACHE}")


if __name__ == "__main__":       # required on Windows for multiprocessing
    main()