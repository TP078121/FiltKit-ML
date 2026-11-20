import torch.nn as nn
from config import N_BINS, N_PITCHES


class PianoCNN(nn.Module):
    """Input : (batch, 1, N_BINS, frames)  CQT spectrogram
       Output: (batch, frames, N_PITCHES)  logits (before sigmoid)"""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                       # shrink frequency only
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * (N_BINS // 4), 256), nn.ReLU(),
            nn.Linear(256, N_PITCHES),
        )

    def forward(self, x):
        h = self.conv(x)                                # (B, 32, 66, T)
        h = h.permute(0, 3, 1, 2).flatten(2)            # (B, T, 32*66)
        return self.fc(h)                               # (B, T, 88)