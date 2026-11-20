import numpy as np
import librosa
from config import *

def load_audio(path):
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
    return y

def compute_cqt(y):
    C = librosa.cqt(y, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                    fmin=librosa.note_to_hz(FMIN_NOTE),
                    n_bins=N_BINS, bins_per_octave=BINS_PER_OCTAVE)
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)   # range about -80..0
    return ((C_db + 80) / 80).clip(0, 1).astype(np.float32) # scale to 0..1