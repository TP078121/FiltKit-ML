import numpy as np
import pretty_midi
from config import *

FPS = SAMPLE_RATE / HOP_LENGTH

def midi_to_rolls(midi_path, n_frames):
    """Returns (frame_roll, onset_roll), both (88, n_frames).
    onset_roll is 1 for 2 frames at each note start."""
    pm = pretty_midi.PrettyMIDI(midi_path)
    frame = np.zeros((N_PITCHES, n_frames), dtype=np.float32)
    onset = np.zeros((N_PITCHES, n_frames), dtype=np.float32)
    for inst in pm.instruments:
        for n in inst.notes:
            k = n.pitch - 21
            if 0 <= k < N_PITCHES:
                a = int(round(n.start * FPS))
                b = max(a + 1, int(round(n.end * FPS)))
                frame[k, a:min(b, n_frames)] = 1
                if a < n_frames:
                    onset[k, a:min(a + 2, n_frames)] = 1
    return frame, onset