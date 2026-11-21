"""Guitar audio -> MIDI (guitar sound) and a rough audio re-creation to listen to.

  python src/guitar_transcribe.py my_guitar.wav --weights checkpoints/guitar_ft/best.pt
Later this takes the guitar stem produced by Demucs.
"""
import argparse
from pathlib import Path

import soundfile as sf
import torch

from config import ROOT, SAMPLE_RATE
from features import compute_cqt, load_audio
from guitar_core import GUITAR_MAX_MIDI, GUITAR_MIN_MIDI, decode_notes, notes_to_guitar_midi
from guitar_evaluate import load_weights, pick_device, predict_full
from model import PianoCNN


def main():
    ap = argparse.ArgumentParser(description="Guitar audio -> MIDI")
    ap.add_argument("audio")
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "guitar_ft" / "best.pt"))
    ap.add_argument("--frame-thr", type=float, default=0.5)
    ap.add_argument("--onset-thr", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "output"))
    args = ap.parse_args()

    device = pick_device()
    model = PianoCNN().to(device)
    load_weights(model, args.weights, device)
    model.eval()

    y = load_audio(args.audio)
    fp, op = predict_full(model, compute_cqt(y), device)
    notes = decode_notes(fp, op, args.frame_thr, args.onset_thr)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = Path(args.audio).stem
    notes_to_guitar_midi(notes, out / f"{stem}_guitar.mid")
    if notes:
        try:
            from midi_utils import render_audio          # the simple synth from the piano module
            sf.write(str(out / f"{stem}_guitar_recreated.wav"), render_audio(notes), SAMPLE_RATE)
        except ImportError:
            pass
    print(f"{len(notes)} notes in {len(y) / SAMPLE_RATE:.1f}s of audio "
          f"(pitches limited to MIDI {GUITAR_MIN_MIDI}-{GUITAR_MAX_MIDI}) -> {out / (stem + '_guitar.mid')}")


if __name__ == "__main__":
    main()