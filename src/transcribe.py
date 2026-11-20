import argparse
import torch
import soundfile as sf

from config import *
from features import load_audio, compute_cqt
from model import PianoCNN
from evaluate import predict_full
from midi_utils import roll_to_notes_onset, notes_to_midi, render_audio


def main():
    ap = argparse.ArgumentParser(description="Piano audio -> MIDI")
    ap.add_argument("audio", help="path to a wav/mp3/flac file")
    ap.add_argument("--weights", default=str(ROOT / "checkpoints" / "smoke" / "best.pt"))
    ap.add_argument("--frame-thr", type=float, default=0.5)
    ap.add_argument("--onset-thr", type=float, default=0.5)
    ap.add_argument("--out", default=str(ROOT / "output"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model = PianoCNN().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    y = load_audio(args.audio)
    cqt = compute_cqt(y)
    frame_p, onset_p = predict_full(model, cqt, device)
    notes = roll_to_notes_onset(frame_p, onset_p, args.frame_thr, args.onset_thr)

    from pathlib import Path
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = Path(args.audio).stem
    notes_to_midi(notes, out / f"{stem}_pred.mid")
    if notes:
        sf.write(str(out / f"{stem}_pred.wav"), render_audio(notes), SAMPLE_RATE)
    print(f"{len(notes)} notes in {len(y) / SAMPLE_RATE:.1f}s of audio -> {out / (stem + '_pred.mid')}")


if __name__ == "__main__":
    main()