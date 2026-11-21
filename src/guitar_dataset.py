import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from guitar_core import cache_paths, pieces_for, shift_pitch


class GuitarDataset(Dataset):
    """Random crops from preprocessed guitar pieces. Same output as the piano dataset:
         cqt        (1, 264, crop_frames)
         frame_roll (crop_frames, 88)   note is held
         onset_roll (crop_frames, 88)   note was just struck
    cache_dirs: one cache folder or a list of them (pieces are pooled).
    pitch_aug : probability of shifting the crop by +-1..max_shift semitones (training only)."""

    def __init__(self, cache_dirs, split="train", crop_frames=256, samples_per_epoch=2000,
                 random_crop=True, pitch_aug=0.0, max_shift=2):
        if isinstance(cache_dirs, (str, Path)):
            cache_dirs = [cache_dirs]
        self.pieces = [(Path(d), n) for d in cache_dirs for n in pieces_for(d, split)]
        if not self.pieces:
            raise RuntimeError(f"No cached guitar pieces for split '{split}' in {list(map(str, cache_dirs))}. "
                               f"Run prepare_guitarset.py / prepare_slakh_guitar.py first.")
        self.crop_frames = crop_frames
        self.samples_per_epoch = samples_per_epoch
        self.random_crop = random_crop
        self.pitch_aug = pitch_aug
        self.shifts = [s for s in range(-max_shift, max_shift + 1) if s != 0]
        print(f"GuitarDataset[{split}]: {len(self.pieces)} pieces from {[str(d) for d in cache_dirs]}")

    def __len__(self):
        return self.samples_per_epoch

    def __getitem__(self, idx):
        d, name = random.choice(self.pieces) if self.random_crop else self.pieces[idx % len(self.pieces)]
        cqt_p, fr_p, on_p, _ = cache_paths(d, name)
        c = np.load(cqt_p, mmap_mode="r")                 # (264, T) float16
        f = np.load(fr_p, mmap_mode="r")                  # (88, T) uint8
        o = np.load(on_p, mmap_mode="r")
        T = c.shape[1]

        if self.random_crop and T > self.crop_frames:
            start = random.randint(0, T - self.crop_frames)
        else:
            start = 0
        end = start + self.crop_frames

        x = np.asarray(c[:, start:end], dtype=np.float32)
        yf = np.asarray(f[:, start:end], dtype=np.float32).T      # (frames, 88)
        yo = np.asarray(o[:, start:end], dtype=np.float32).T
        if x.shape[1] < self.crop_frames:
            pad = self.crop_frames - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)))
            yf = np.pad(yf, ((0, pad), (0, 0)))
            yo = np.pad(yo, ((0, pad), (0, 0)))

        if self.pitch_aug > 0 and random.random() < self.pitch_aug:
            x, yf, yo = shift_pitch(x, yf, yo, random.choice(self.shifts))

        return (torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(yf)),
                torch.from_numpy(np.ascontiguousarray(yo)))