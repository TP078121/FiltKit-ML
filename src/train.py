import argparse
import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from maestro_dataset import MaestroDataset
from model import PianoCNN
from evaluate import evaluate_split

POS_WEIGHT = 8.0            # active keys are rare (~2%)
ONSET_POS_WEIGHT = 20.0     # onsets are rarer still


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=500, help="batches per epoch")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=0, help="DataLoader worker processes")
    ap.add_argument("--val-pieces", type=int, default=10, help="pieces used for validation F1")
    ap.add_argument("--resume", action="store_true", help="continue from last.pt")
    ap.add_argument("--run", default="run1", help="name of this run (folder under checkpoints/)")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    run_dir = ROOT / "checkpoints" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.csv"
    last_path = run_dir / "last.pt"
    best_path = run_dir / "best.pt"

    train_ds = MaestroDataset("train", samples_per_epoch=args.steps * args.batch_size)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=(device == "cuda"))

    model = PianoCNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,} (about {n_params * 4 / 1e6:.1f} MB)")

    frame_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(POS_WEIGHT, device=device))
    onset_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ONSET_POS_WEIGHT, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch, best = 1, -1.0
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed after epoch {ck['epoch']} (best onset F1 so far {best:.3f})")

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_onset_f1",
                                    "val_onset_offset_f1", "val_pieces", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, yf, yo in loader:
            x, yf, yo = x.to(device), yf.to(device), yo.to(device)
            frame_logits, onset_logits = model(x)
            loss = frame_loss_fn(frame_logits, yf) + onset_loss_fn(onset_logits, yo)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        train_loss = total / len(loader)

        res = evaluate_split(model, device, "validation", max_pieces=args.val_pieces)
        if res is None:
            on_f1 = full_f1 = float("nan")
            n_val = 0
            improved = True                    # no validation data: just keep the latest
        else:
            on_f1, full_f1, n_val = res
            improved = on_f1 > best
        if improved and res is not None:
            best = on_f1
        if improved:
            torch.save(model.state_dict(), best_path)

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best": best}, last_path)

        secs = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{on_f1:.4f}",
                                    f"{full_f1:.4f}", n_val, f"{secs:.0f}"])
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
              f"val onset F1 {on_f1:.3f}  on+off F1 {full_f1:.3f}  "
              f"({n_val} pieces)  {secs:.0f}s" + ("  *best*" if improved else ""))

    print("done. best weights:", best_path)


if __name__ == "__main__":      # required on Windows (DataLoader workers)
    main()