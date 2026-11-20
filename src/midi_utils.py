import numpy as np
import pretty_midi
from config import *
from labels import FPS


def roll_to_notes(prob, threshold=0.5, min_frames=2):
    """prob: (T, 88) probabilities. Returns a sorted list of (start_sec, end_sec, midi_pitch).
    Short blips (shorter than min_frames) are dropped as noise."""
    active = prob > threshold
    notes = []
    for k in range(N_PITCHES):
        col = active[:, k].astype(np.int8)
        d = np.diff(np.concatenate([[0], col, [0]]))
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        for a, b in zip(starts, ends):
            if b - a >= min_frames:
                notes.append((a / FPS, b / FPS, k + 21))   # index 0 = MIDI 21 (A0)
    return sorted(notes)


def notes_to_midi(notes, path, velocity=80):
    pm = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)
    for s, e, p in notes:
        piano.notes.append(pretty_midi.Note(velocity=velocity, pitch=int(p),
                                            start=float(s), end=float(e)))
    pm.instruments.append(piano)
    pm.write(str(path))
    return pm


def render_audio(notes, sr=SAMPLE_RATE, tail=1.0):
    """Simple synth so you can listen to the prediction (same sound as the test clips)."""
    total = max((e for _, e, _ in notes), default=1.0) + tail
    audio = np.zeros(int(sr * total))
    for s, e, p in notes:
        f0 = 440 * 2 ** ((p - 69) / 12)
        length = (e - s) + 0.2
        t = np.arange(int(length * sr)) / sr
        tone = sum((1 / h**1.2) * np.sin(2 * np.pi * f0 * h * t)
                   for h in range(1, 6) if f0 * h < sr / 2)
        tone = tone * np.exp(-2.5 * t) * np.minimum(1, t / 0.005) * np.clip((length - t) / 0.2, 0, 1)
        i = int(s * sr)
        audio[i:i + len(tone)] += tone[:len(audio) - i]
    return 0.9 * audio / (np.abs(audio).max() + 1e-9)