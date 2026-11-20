import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from config import *
from drum_core import cache_paths, pieces_for


class DrumDataset(Dataset):
    """Random crops from preprocessed pieces (see prepare_egmd.py / make_drum_synthetic.py).
       Returns (mel, target):
         mel    (1, DRUM_N_MELS, crop_frames)
         target (crop_frames, N_DRUM)   1.0 on onset frames, 0.5 on neighbours"""

    def __init__(self, cache_dir=DRUM_CACHE_DIR, split="train", crop_frames=1000,
                 samples_per_epoch=2000, random_crop=True, limit=None):
        self.cache_dir = Path(cache_dir)
        self.crop_frames = crop_frames
        self.samples_per_epoch = samples_per_epoch
        self.random_crop = random_crop
        self.names = pieces_for(self.cache_dir, split)
        if limit:
            self.names = self.names[:limit]
        if not self.names:
            raise RuntimeError(f"No cached drum pieces for split '{split}' in {self.cache_dir}. "
                               f"Run prepare_egmd.py (real data) or make_drum_synthetic.py first.")
        print(f"DrumDataset[{split}]: {len(self.names)} pieces from {self.cache_dir}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        name = random.choice(self.names) if self.random_crop else self.names[idx % len(self.names)]
        mel_p, on_p, _ = cache_paths(self.cache_dir, name)
        mel = np.load(mel_p, mmap_mode="r")                    # (n_mels, T)
        on = np.load(on_p, mmap_mode="r")                      # (T, N_DRUM)
        T = mel.shape[1]

        if self.random_crop and T > self.crop_frames:
            start = random.randint(0, T - self.crop_frames)
        else:
            start = 0
        end = start + self.crop_frames

        x = np.asarray(mel[:, start:end], dtype=np.float32)
        y = np.asarray(on[start:end], dtype=np.float32)
        if x.shape[1] < self.crop_frames:                       # pad short pieces
            pad = self.crop_frames - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)))
            y = np.pad(y, ((0, pad), (0, 0)))

        return torch.from_numpy(x).unsqueeze(0), torch.from_numpy(y)
