import numpy as np
import pretty_midi
from config import *

FPS = SAMPLE_RATE / HOP_LENGTH

def midi_to_roll(midi_path, n_frames):
    pm = pretty_midi.PrettyMIDI(midi_path)
    roll = np.zeros((N_PITCHES, n_frames), dtype=np.float32)
    for inst in pm.instruments:
        for n in inst.notes:
            k = n.pitch - 21                       # MIDI 21 = lowest piano key (A0)
            if 0 <= k < N_PITCHES:
                a = int(round(n.start * FPS))
                b = max(a + 1, int(round(n.end * FPS)))
                roll[k, a:min(b, n_frames)] = 1
    return roll