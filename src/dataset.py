import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import *
from features import load_audio, compute_cqt
from labels import midi_to_roll


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
            roll = midi_to_roll(str(mid), cqt.shape[1])           # (88, T)
            self.items.append((cqt, roll))

        if not self.items:
            raise RuntimeError(f"No wav/mid pairs found in {folder}")
        print(f"Loaded {len(self.items)} clips from {folder}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        cqt, roll = self.items[idx % len(self.items)]
        T = cqt.shape[1]

        if self.random_crop and T > self.crop_frames:
            start = random.randint(0, T - self.crop_frames)
        else:
            start = 0
        end = start + self.crop_frames

        x = cqt[:, start:end]
        y = roll[:, start:end]

        # pad if the clip is shorter than the crop
        if x.shape[1] < self.crop_frames:
            pad = self.crop_frames - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)))
            y = np.pad(y, ((0, 0), (0, pad)))

        x = torch.from_numpy(x).unsqueeze(0)       # (1, 264, frames)
        y = torch.from_numpy(y.T.copy())           # (frames, 88)
        return x, y