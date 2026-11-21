"""Sung melody -> MIDI.

  python src/voice_transcribe.py my_singing.wav --weights checkpoints/voice1/best.pt
  python src/voice_transcribe.py my_singing.wav --baseline          (pYIN, no machine learning)
Later this takes the vocal stem produced by Demucs.
"""
import argparse
from pathlib import Path

import soundfile as sf

from config import ROOT, SAMPLE_RATE
from features import compute_cqt, load_audio
from voice_core import VOICE_MAX_MIDI, VOICE_MIN_MIDI, mono_decode, notes_to_guitar_midi, pyin_notes


def main():
    ap = argparse.ArgumentParser(description="Sung melody -> MIDI")
    ap.add_argument("audio")
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "voice1" / "best.pt"))
    ap.add_argument("--baseline", action="store_true", help="use pYIN instead of the trained model")
    ap.add_argument("--frame-thr", type=float, default=0.5)
    ap.add_argument("--onset-thr", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "output"))
    args = ap.parse_args()

    y = load_audio(args.audio)
    if args.baseline:
        notes, tag = pyin_notes(y), "pyin"
    else:
        from guitar_evaluate import load_weights, pick_device, predict_full
        from model import PianoCNN
        device = pick_device()
        model = PianoCNN().to(device)
        load_weights(model, args.weights, device)
        model.eval()
        fp, op = predict_full(model, compute_cqt(y), device)
        notes, tag = mono_decode(fp, op, args.frame_thr, args.onset_thr), "model"

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = Path(args.audio).stem
    notes_to_guitar_midi(notes, out / f"{stem}_voice_{tag}.mid", program=53)     # 53 = Voice Oohs
    if notes:
        try:
            from midi_utils import render_audio
            sf.write(str(out / f"{stem}_voice_{tag}_recreated.wav"), render_audio(notes), SAMPLE_RATE)
        except ImportError:
            pass
    pitches = [n[2] for n in notes]
    span = f", pitches MIDI {min(pitches)}-{max(pitches)}" if pitches else ""
    print(f"{len(notes)} notes in {len(y) / SAMPLE_RATE:.1f}s of audio{span} "
          f"(allowed range MIDI {VOICE_MIN_MIDI}-{VOICE_MAX_MIDI}) -> {out / (stem + '_voice_' + tag + '.mid')}")


if __name__ == "__main__":
    main()