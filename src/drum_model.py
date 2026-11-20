import torch.nn as nn

from config import DRUM_N_MELS
from drum_core import N_DRUM


class DrumCRNN(nn.Module):
    """Small CNN + bidirectional GRU.
       Input : (batch, 1, DRUM_N_MELS, frames)   log-mel spectrogram
       Output: (batch, frames, N_DRUM)           onset logits (apply sigmoid for probabilities)
    The CNN looks at short-time spectral shape (kick vs snare vs hi-hat),
    the GRU adds rhythmic context across time."""

    def __init__(self, hidden=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                      # mel 128 -> 64 (time unchanged)
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                      # -> 32
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d((2, 1)),                      # -> 16
        )
        feat = 32 * (DRUM_N_MELS // 8)                 # 32 channels * 16 mel bands = 512
        self.proj = nn.Sequential(nn.Linear(feat, 128), nn.ReLU(), nn.Dropout(0.2))
        self.gru = nn.GRU(128, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * hidden, N_DRUM)

    def forward(self, x):
        h = self.conv(x)                               # (B, 32, 16, T)
        h = h.permute(0, 3, 1, 2).flatten(2)           # (B, T, 512)
        h = self.proj(h)                               # (B, T, 128)
        h, _ = self.gru(h)                             # (B, T, 2*hidden)
        return self.out(h)                             # (B, T, N_DRUM)
