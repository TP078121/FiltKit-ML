import torch.nn as nn
from config import N_BINS, N_PITCHES


class PianoCNN(nn.Module):
    """Input : (batch, 1, N_BINS, frames)
       Output: (frame_logits, onset_logits), each (batch, frames, N_PITCHES)"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.shared = nn.Sequential(nn.Linear(32 * (N_BINS // 4), 256), nn.ReLU())
        self.frame_head = nn.Linear(256, N_PITCHES)
        self.onset_head = nn.Linear(256, N_PITCHES)

    def forward(self, x):
        h = self.conv(x)
        h = h.permute(0, 3, 1, 2).flatten(2)
        h = self.shared(h)
        return self.frame_head(h), self.onset_head(h)