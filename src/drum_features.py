import numpy as np
import librosa

from config import *


def load_audio(path):
    y, _ = librosa.load(path, sr=DRUM_SR, mono=True)
    return y


def compute_logmel(y):
    """Audio -> log-mel spectrogram (DRUM_N_MELS, T) scaled to 0..1, 100 frames/second.
    Frame i is centred at time i / DRUM_FPS (librosa default), which matches the labels."""
    S = librosa.feature.melspectrogram(
        y=y, sr=DRUM_SR, n_fft=DRUM_NFFT, hop_length=DRUM_HOP,
        n_mels=DRUM_N_MELS, fmin=DRUM_FMIN, fmax=DRUM_FMAX, power=2.0)
    S_db = librosa.power_to_db(S, ref=np.max, top_db=80.0)       # about -80..0
    return ((S_db + 80.0) / 80.0).clip(0, 1).astype(np.float32)
