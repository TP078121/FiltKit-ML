import glob, random, os
import numpy as np
import pretty_midi
import soundfile as sf
from config import *

random.seed(0)

def crop_midi(pm, seconds):
    notes = sorted((n for inst in pm.instruments for n in inst.notes),
                   key=lambda n: n.start)
    t0 = notes[0].start
    out = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for n in notes:
        if t0 <= n.start < t0 + seconds:
            piano.notes.append(pretty_midi.Note(
                velocity=n.velocity, pitch=n.pitch,
                start=n.start - t0, end=min(n.end, t0 + seconds) - t0))
    out.instruments.append(piano)
    return out

def synth(pm, sr, seconds):
    audio = np.zeros(int(sr * (seconds + 1)))
    for n in pm.instruments[0].notes:
        f0 = 440 * 2 ** ((n.pitch - 69) / 12)
        length = (n.end - n.start) + 0.2
        t = np.arange(int(length * sr)) / sr
        tone = sum((1 / h**1.2) * np.sin(2 * np.pi * f0 * h * t)
                   for h in range(1, 6) if f0 * h < sr / 2)
        env = np.exp(-2.5 * t) * np.minimum(1, t / 0.005)    # attack + decay
        rel = np.clip((length - t) / 0.2, 0, 1)              # release
        tone = tone * env * rel * (n.velocity / 127)
        i = int(n.start * sr)
        audio[i:i + len(tone)] += tone[:len(audio) - i]
    audio = audio[:int(sr * seconds)]
    return 0.9 * audio / (np.abs(audio).max() + 1e-9)

files = glob.glob("data/**/*.mid*", recursive=True)
print("found", len(files), "MIDI files")
os.makedirs("data/synthetic", exist_ok=True)

for i, path in enumerate(random.sample(files, 5)):
    pm = crop_midi(pretty_midi.PrettyMIDI(path), CLIP_SECONDS)
    pm.write(f"data/synthetic/clip_{i}.mid")
    sf.write(f"data/synthetic/clip_{i}.wav", synth(pm, SAMPLE_RATE, CLIP_SECONDS), SAMPLE_RATE)
    print("made clip", i, "from", os.path.basename(path), "-", len(pm.instruments[0].notes), "notes")