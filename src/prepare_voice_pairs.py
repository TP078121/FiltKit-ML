"""Preprocess singing recordings that come as AUDIO + MIDI pairs with the same file name
(for example the Children's Song Dataset: wav/en001a.wav + mid/en001a.mid) -> data/cache/<out-name>.

  python src/prepare_voice_pairs.py --audio-dir data/csd

Every sub-folder is searched. The split is by song (80% train, 10% validation, 10% test); the two
keys of one song (en001a / en001b) always stay together.
"""
import argparse
from pathlib import Path

from config import ROOT
from features import compute_cqt, load_audio
from guitar_core import piece_cached, read_midi_notes, save_piece, write_index
from voice_core import split_by_group

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--midi-dir", default=None, help="default: same as --audio-dir")
    ap.add_argument("--out-name", default="voice_pairs")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    audio = {p.stem: p for p in sorted(Path(args.audio_dir).rglob("*")) if p.suffix.lower() in AUDIO_EXTS}
    midi = {p.stem: p for p in sorted(Path(args.midi_dir or args.audio_dir).rglob("*"))
            if p.suffix.lower() in (".mid", ".midi")}
    common = sorted(set(audio) & set(midi))
    print(f"{len(audio)} audio files, {len(midi)} MIDI files, {len(common)} pairs with the same file name")
    if not common:
        print("examples of audio names:", list(audio)[:3], " MIDI names:", list(midi)[:3])
        raise SystemExit("No pairs found. Audio and MIDI files must share the same name (without extension).")
    if args.limit:
        common = common[:args.limit]

    cache = ROOT / "data" / "cache" / args.out_name
    split_of = split_by_group(common)
    entries = []
    for i, stem in enumerate(common, 1):
        entries.append((stem, split_of[stem]))
        if piece_cached(cache, stem):
            continue
        notes = read_midi_notes(midi[stem])
        if len(notes) == 0:
            print("no notes in", midi[stem].name, "- skipped")
            entries.pop()
            continue
        save_piece(cache, stem, compute_cqt(load_audio(str(audio[stem]))), notes)
        if i % 10 == 0 or i == len(common):
            print(f"[{i}/{len(common)}] {stem}  {len(notes)} notes")

    n = write_index(cache, entries)
    counts = {s: sum(1 for _, sp in entries if sp == s) for s in ("train", "validation", "test")}
    print(f"cached {n} pieces in {cache}  (by split: {counts})")


if __name__ == "__main__":
    main()