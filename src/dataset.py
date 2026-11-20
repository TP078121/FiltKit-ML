import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import *
from features import load_audio, compute_cqt
from labels import midi_to_rolls


class PianoDataset(Dataset):
    """Returns (cqt, roll) pairs:
       cqt  : (1, N_BINS, crop_frames)   spectrogram, one channel
       roll : (crop_frames, N_PITCHES)   which of the 88 keys are active per frame
    """

    def __init__(self, folder=SYNTH_DIR, crop_frames=256,
                 samples_per_epoch=40, random_crop=True):
        self.crop_frames = crop_frames
        self.samples_per_epoch = samples_per_epoch
        self.random_crop = random_crop
        self.items = []

        for wav in sorted(Path(folder).glob("*.wav")):
            mid = wav.with_suffix(".mid")
            if not mid.exists():
                continue
            cqt = compute_cqt(load_audio(str(wav)))               # (264, T)
            frame, onset = midi_to_rolls(str(mid), cqt.shape[1])  # (88, T)
            self.items.append((cqt, frame, onset))

        if not self.items:
            raise RuntimeError(f"No wav/mid pairs found in {folder}")
        print(f"Loaded {len(self.items)} clips from {folder}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        cqt, frame, onset = self.items[idx % len(self.items)]
        T = cqt.shape[1]
        if self.random_crop and T > self.crop_frames:
            start = random.randint(0, T - self.crop_frames)
        else:
            start = 0
        end = start + self.crop_frames

        x = cqt[:, start:end]
        yf = frame[:, start:end]
        yo = onset[:, start:end]

        if x.shape[1] < self.crop_frames:
            pad = self.crop_frames - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)))
            yf = np.pad(yf, ((0, 0), (0, pad)))
            yo = np.pad(yo, ((0, 0), (0, pad)))

        return (torch.from_numpy(x).unsqueeze(0),
                torch.from_numpy(yf.T.copy()),
                torch.from_numpy(yo.T.copy()))