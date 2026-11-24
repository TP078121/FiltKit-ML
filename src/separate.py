"""Split a song into stems with Demucs (drums, bass, vocals, other - and guitar, piano with the 6-stem model).

  python src/separate.py my_song.mp3
  python src/separate.py my_song.mp3 --model htdemucs_6s          (adds guitar and piano)
  python src/separate.py "C:\\music\\song one.mp3" other.wav --out output/stems

Result: output/stems/<song name>/<model name>/<stem>.wav   (stereo, 44.1 kHz, 16 bit)

Models (Demucs 4.1.0, MIT license):
  htdemucs      4 stems, the default
  htdemucs_ft   4 stems, fine-tuned: about 4x slower, maybe a bit better
  htdemucs_6s   6 stems (adds guitar and piano). The Demucs authors say guitar is okay and the piano
                stem has a lot of bleeding and artifacts, so expect weaker piano/guitar transcriptions.

The first run downloads the model weights (internet needed once). Install with:  pip install -U demucs
GPU memory: about 7 GB with the defaults; if you run out, add --segment 8 (or use --device cpu).
"""
import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from config import ROOT

MODELS = ("htdemucs", "htdemucs_ft", "htdemucs_6s")


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_stem(wav, path, samplerate):
    """Demucs stem tensor (channels, samples) -> 16-bit wav. A stem that would clip is scaled down
    (Demucs' default behaviour) instead of being distorted."""
    a = wav.detach().cpu().numpy().T.astype(np.float32)          # (samples, channels)
    peak = float(np.abs(a).max()) if a.size else 0.0
    if peak > 1.0:
        a = a / peak
    sf.write(str(path), a, samplerate, subtype="PCM_16")


def separate_file(path, out_dir=None, model="htdemucs", device=None, segment=None,
                  shifts=1, overlap=0.25, progress=True, separator=None):
    """Separate one audio file. Returns {stem name: Path of the wav}.
    Pass the same `separator` object for several files to avoid loading the model again."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"audio file not found: {path}")
    if separator is None:
        import demucs.api
        separator = demucs.api.Separator(model=model, device=device or pick_device(), segment=segment,
                                         shifts=shifts, overlap=overlap, progress=progress)
    _, stems = separator.separate_audio_file(path)               # (original, {stem: tensor})
    out = Path(out_dir or ROOT / "output" / "stems") / path.stem / model
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for name, wav in stems.items():
        p = out / f"{name}.wav"
        write_stem(wav, p, separator.samplerate)
        written[name] = p
    return written


def main():
    ap = argparse.ArgumentParser(description="Split songs into stems with Demucs")
    ap.add_argument("audio", nargs="+", help="one or more audio files (mp3, wav, flac, ogg, ...)")
    ap.add_argument("--model", default="htdemucs", choices=MODELS)
    ap.add_argument("--out", default=str(ROOT / "output" / "stems"))
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: the best available)")
    ap.add_argument("--segment", type=float, default=None, help="seconds per chunk; smaller = less GPU memory")
    ap.add_argument("--shifts", type=int, default=1, help="more = slightly better but slower")
    ap.add_argument("--overlap", type=float, default=0.25)
    args = ap.parse_args()

    import demucs.api
    device = args.device or pick_device()
    print(f"loading {args.model} on {device} (the first run downloads the model) ...")
    separator = demucs.api.Separator(model=args.model, device=device, segment=args.segment,
                                     shifts=args.shifts, overlap=args.overlap, progress=True)
    for f in args.audio:
        t0 = time.time()
        written = separate_file(f, args.out, args.model, separator=separator)
        print(f"{Path(f).name}: {', '.join(written)} in {time.time() - t0:.0f}s -> {next(iter(written.values())).parent}")


if __name__ == "__main__":
    main()