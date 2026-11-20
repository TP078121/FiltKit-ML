import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from dataset import PianoDataset
from model import PianoCNN

torch.manual_seed(0)

# ---------- settings ----------
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 4
POS_WEIGHT = 8.0            # active keys are rare (~2.5%)
ONSET_POS_WEIGHT = 20.0     # onsets are rarer still

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", device)

# ---------- data and model ----------
train_ds = PianoDataset(samples_per_epoch=40)
loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

model = PianoCNN().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,} (about {n_params * 4 / 1e6:.1f} MB)")

frame_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=device))
onset_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ONSET_POS_WEIGHT, device=device))
opt = torch.optim.Adam(model.parameters(), lr=LR)


# ---------- simple frame-level F1 on the full clips ----------
def frame_f1(threshold=0.5):
    model.eval()
    tp = fp = fn = 0
    with torch.no_grad():
        for cqt, roll, _ in train_ds.items:
            x = torch.from_numpy(cqt)[None, None].to(device)                 # (1,1,264,T)
            pred = torch.sigmoid(model(x)[0])[0].cpu().numpy() > threshold   # (T,88)
            true = roll.T > 0.5
            tp += (pred & true).sum()
            fp += (pred & ~true).sum()
            fn += (~pred & true).sum()
    model.train()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return precision, recall, f1


# ---------- training loop ----------
CKPT_DIR = ROOT / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)

model.train()
for epoch in range(1, EPOCHS + 1):
    total = 0.0
    for x, yf, yo in loader:
        x, yf, yo = x.to(device), yf.to(device), yo.to(device)
        frame_logits, onset_logits = model(x)
        loss = frame_loss_fn(frame_logits, yf) + onset_loss_fn(onset_logits, yo)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()

    if epoch == 1 or epoch % 5 == 0:
        p, r, f1 = frame_f1()
        print(f"epoch {epoch:3d}  loss {total / len(loader):.4f}  "
              f"precision {p:.3f}  recall {r:.3f}  F1 {f1:.3f}")

    if epoch % 10 == 0:
        torch.save(model.state_dict(), CKPT_DIR / "piano_cnn.pt")

torch.save(model.state_dict(), CKPT_DIR / "piano_cnn.pt")
print("saved:", CKPT_DIR / "piano_cnn.pt")