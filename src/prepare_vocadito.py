"""Preprocess vocadito (40 real solo-vocal excerpts, 2 annotators) -> data/cache/vocadito_A1 and _A2.
It is far too small to train on. It is the TEST set: real singers the model has never heard.

Download vocadito.zip (58.5 MB, CC BY 4.0) from https://zenodo.org/records/5557945/latest
and extract it into data/vocadito/, then:

  python src/prepare_vocadito.py

It prints the first lines of a note file so you can check the format.
"""
import argparse
import re
from pathlib import Path

from features import compute_cqt, load_audio
from guitar_core import piece_cached, save_piece, write_index
from voice_core import VOCADITO_DIR, read_vocadito_notes, vocadito_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(VOCADITO_DIR))
    ap.add_argument("--annotators", nargs="+", default=["A1", "A2"])
    ap.add_argument("--pitch-unit", choices=["hz", "midi"], default="hz")
    args = ap.parse_args()

    root = Path(args.dir)
    wavs = sorted(root.rglob("vocadito_*.wav"), key=lambda p: int(re.sub(r"\D", "", p.stem) or 0))
    print(f"found {len(wavs)} vocadito audio files in {root}")
    if not wavs:
        raise SystemExit("Extract vocadito.zip into data/vocadito/ first.")

    cqts, shown = {}, False
    for ann in args.annotators:
        cache, entries = vocadito_cache(ann), []
        for wav in wavs:
            name = wav.stem
            nf = next(iter(root.rglob(f"{name}_notes{ann}.csv")), None)
            if nf is None:
                print(f"no note file for {name} / {ann}")
                continue
            if not shown:
                print(f"first lines of {nf.name}:")
                for line in nf.read_text(encoding="utf-8").splitlines()[:3]:
                    print("   ", line)
                shown = True
            notes = read_vocadito_notes(nf, args.pitch_unit)
            entries.append((name, "test"))
            if piece_cached(cache, name):
                continue
            if name not in cqts:
                cqts[name] = compute_cqt(load_audio(str(wav)))
            save_piece(cache, name, cqts[name], notes)
        n = write_index(cache, entries)
        print(f"annotator {ann}: cached {n} pieces in {cache}")


if __name__ == "__main__":
    main()