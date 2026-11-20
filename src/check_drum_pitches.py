"""Check that every drum MIDI pitch in E-GMD is covered by our class mapping.
Needs the (small) MIDI-only E-GMD download extracted under data/.

  python src/check_drum_pitches.py
"""
import collections
import random

import pretty_midi

from config import ROOT
from drum_core import DRUM_CLASSES, PITCH_TO_CLASS

files = [p for d in sorted((ROOT / "data").glob("e-gmd*")) if d.is_dir() for p in d.rglob("*.mid*")]
print("MIDI files found:", len(files))
if not files:
    raise SystemExit("No MIDI files. Extract e-gmd-v1.0.0-midi.zip (about 103 MB) into the data folder.")

random.Random(0).shuffle(files)
counts = collections.Counter()
for p in files[:400]:
    pm = pretty_midi.PrettyMIDI(str(p))
    for inst in pm.instruments:
        for n in inst.notes:
            counts[n.pitch] += 1

total = sum(counts.values())
mapped = sum(c for p, c in counts.items() if p in PITCH_TO_CLASS)
print(f"\n{'pitch':>5s} {'count':>8s}  class")
for pitch in sorted(counts):
    cls = DRUM_CLASSES[PITCH_TO_CLASS[pitch]] if pitch in PITCH_TO_CLASS else "** UNMAPPED **"
    print(f"{pitch:5d} {counts[pitch]:8d}  {cls}")
print(f"\n{100 * mapped / max(total, 1):.1f}% of {total} notes are mapped.")
if mapped < total:
    print("Some pitches are UNMAPPED. If one has a large count, add it to _GROUPS in drum_core.py.")
