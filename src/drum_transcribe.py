"""Drum audio -> MIDI (+ a rough audio re-creation so you can listen).

  python src/drum_transcribe.py my_drums.wav --weights checkpoints/drums1/best.pt
Later this takes the drum stem produced by Demucs.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from config import *
from drum_core import DRUM_CLASSES, N_DRUM, pick_peaks
from drum_evaluate import pick_device, predict_full
from drum_features import load_audio, compute_logmel
from drum_midi import events_to_midi, render_drums
from drum_model import DrumCRNN


def main():
    ap = argparse.ArgumentParser(description="Drum audio -> MIDI")
    ap.add_argument("audio")
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "drums1" / "best.pt"))
    ap.add_argument("--thr", type=float, nargs="+", default=[0.5],
                    help="one threshold, or one per class: " + " ".join(DRUM_CLASSES))
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--out", default=str(ROOT / "output"))
    args = ap.parse_args()
    thr = args.thr[0] if len(args.thr) == 1 else args.thr
    if len(args.thr) not in (1, N_DRUM):
        raise SystemExit(f"--thr needs 1 or {N_DRUM} numbers")

    device = pick_device()
    model = DrumCRNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    y = load_audio(args.audio)
    prob = predict_full(model, compute_logmel(y), device)
    events = pick_peaks(prob, thr, args.window)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = Path(args.audio).stem
    events_to_midi(events, out / f"{stem}_drums.mid")
    if len(events):
        sf.write(str(out / f"{stem}_drums_recreated.wav"),
                 render_drums(events, seed=0, vary=False), DRUM_SR)
    print(f"{len(events)} hits in {len(y) / DRUM_SR:.1f}s of audio -> {out / (stem + '_drums.mid')}")
    for k, name in enumerate(DRUM_CLASSES):
        print(f"  {name:14s} {int((events[:, 1] == k).sum()) if len(events) else 0}")


if __name__ == "__main__":
    main()
