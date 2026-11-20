import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from config import *


def cache_name(row):
    """Unique file-safe name for one piece, e.g. 2004__MIDI-Unprocessed_..."""
    return row["audio_filename"].rsplit(".", 1)[0].replace("/", "__")


def read_rows(split=None):
    """Rows of the official MAESTRO csv. split: 'train', 'validation', 'test' or None."""
    with open(MAESTRO_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if split:
        rows = [r for r in rows if r["split"] == split]
    return rows


def cached_paths(name):
    return [CACHE_DIR / f"{name}_{k}.npy" for k in ("cqt", "frame", "onset")]


class MaestroDataset(Dataset):
    """Same output as PianoDataset: (cqt, frame_roll, onset_roll)
       cqt (1, 264, crop_frames), frame_roll (crop_frames, 88), onset_roll (crop_frames, 88).
       Reads only pieces that were preprocessed by prepare_maestro.py."""

    def __init__(self, split="train", crop_frames=256, samples_per_epoch=2000,
                 random_crop=True, limit=None):
        self.crop_frames = crop_frames
        self.samples_per_epoch = samples_per_epoch
        self.random_crop = random_crop

        self.pieces = [cache_name(r) for r in read_rows(split)
                       if all(p.exists() for p in cached_paths(cache_name(r)))]
        if limit:
            self.pieces = self.pieces[:limit]
        if not self.pieces:
            raise RuntimeError(f"No cached pieces for split '{split}'. "
                               f"Run prepare_maestro.py first (needs the audio files).")
        print(f"MaestroDataset[{split}]: {len(self.pieces)} pieces")

    def __len__(self):
        return self.samples_per_epoch

    def load_piece(self, i):
        """Full piece for evaluation: cqt (264, T) float32, frame (88, T), onset (88, T)."""
        c, f, o = [np.load(p) for p in cached_paths(self.pieces[i])]
        return c.astype(np.float32), f.astype(np.float32), o.astype(np.float32)

    def __getitem__(self, idx):
        if self.random_crop:
            name = random.choice(self.pieces)
        else:
            name = self.pieces[idx % len(self.pieces)]
        c, f, o = [np.load(p, mmap_mode="r") for p in cached_paths(name)]   # no full load
        T = c.shape[1]

        if self.random_crop and T > self.crop_frames:
            start = random.randint(0, T - self.crop_frames)
        else:
            start = 0
        end = start + self.crop_frames

        x = np.asarray(c[:, start:end], dtype=np.float32)
        yf = np.asarray(f[:, start:end], dtype=np.float32)
        yo = np.asarray(o[:, start:end], dtype=np.float32)

        if x.shape[1] < self.crop_frames:
            pad = self.crop_frames - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad)))
            yf = np.pad(yf, ((0, 0), (0, pad)))
            yo = np.pad(yo, ((0, 0), (0, pad)))

        return (torch.from_numpy(x).unsqueeze(0),
                torch.from_numpy(yf.T.copy()),
                torch.from_numpy(yo.T.copy()))