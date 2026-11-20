"""Train the drum model.

  Laptop test on synthetic data:
    python src/drum_train.py --cache-dir data/cache/drum_synth --epochs 40 --steps 20 --batch-size 8 --val-pieces 3 --run drum_smoke
  Real run (after prepare_egmd.py):
    python src/drum_train.py --epochs 30 --steps 400 --batch-size 16 --workers 4 --run drums1
  Resume after an interruption: same command plus --resume
"""
import argparse
import csv
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import *
from drum_core import DRUM_CLASSES, pieces_for
from drum_dataset import DrumDataset
from drum_evaluate import evaluate_split, pick_device
from drum_model import DrumCRNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(DRUM_CACHE_DIR))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=400, help="batches per epoch")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop-frames", type=int, default=1000, help="1000 frames = 10 seconds")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=5.0, help="weight of onset frames (they are rare)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val-pieces", type=int, default=20)
    ap.add_argument("--run", default="drums1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    run_dir = ROOT / "checkpoints" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path, last_path, best_path = run_dir / "log.csv", run_dir / "last.pt", run_dir / "best.pt"
    with open(run_dir / "meta.json", "w") as f:
        json.dump({"classes": DRUM_CLASSES, "sr": DRUM_SR, "hop": DRUM_HOP, "n_fft": DRUM_NFFT,
                   "n_mels": DRUM_N_MELS, "fmin": DRUM_FMIN, "fmax": DRUM_FMAX,
                   "args": vars(args)}, f, indent=2)

    train_ds = DrumDataset(args.cache_dir, "train", crop_frames=args.crop_frames,
                           samples_per_epoch=args.steps * args.batch_size)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=(device == "cuda"))

    val_split = "validation"
    if not pieces_for(args.cache_dir, "validation"):
        print("note: no validation pieces cached, so validating on the TRAIN split "
              "(fine for a smoke test, not a real score)")
        val_split = "train"

    model = DrumCRNN().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,} (about {n_params * 4 / 1e6:.1f} MB)")

    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch, best = 1, -1.0
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed after epoch {ck['epoch']} (best micro F1 so far {best:.3f})")

    if not log_path.exists():
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "train_loss", "val_micro_f1", "val_macro_f1"]
                                   + [f"f1_{c}" for c in DRUM_CLASSES] + ["val_pieces", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = loss_fn(model(x), y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)     # keeps the GRU stable
            opt.step()
            total += loss.item()
        train_loss = total / len(loader)

        res = evaluate_split(model, device, args.cache_dir, val_split, max_pieces=args.val_pieces)
        per_class, micro, macro, n_val = res
        improved = micro > best
        if improved:
            best = micro
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best": best}, last_path)

        secs = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{micro:.4f}", f"{macro:.4f}"]
                                   + [f"{v:.4f}" for v in per_class] + [n_val, f"{secs:.0f}"])
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  val micro F1 {micro:.3f}  "
              f"macro F1 {macro:.3f}  ({n_val} pieces)  {secs:.0f}s" + ("  *best*" if improved else ""))

    print("done. best weights:", best_path)


if __name__ == "__main__":       # required on Windows (DataLoader workers)
    main()
