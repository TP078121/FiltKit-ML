"""Train the guitar model (same network as piano; best started from the piano weights).

Step 1  pre-train on Slakh (lots of synthetic guitar):
  python src/guitar_train.py --train-cache data/cache/slakh_guitar --val-cache data/cache/slakh_guitar --epochs 30 --steps 500 --batch-size 16 --workers 4 --init-from checkpoints/run1/best.pt --run guitar_slakh

Step 2  fine-tune on real guitar (GuitarSet train players), validate on player 04:
  python src/guitar_train.py --train-cache data/cache/guitarset --val-cache data/cache/guitarset --epochs 20 --steps 200 --batch-size 16 --lr 3e-4 --init-from checkpoints/guitar_slakh/best.pt --run guitar_ft

Interrupted? Run the same command again with --resume.
"""
import argparse
import csv
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import ROOT
from guitar_core import SLAKH_CACHE, pieces_for
from guitar_dataset import GuitarDataset
from guitar_evaluate import evaluate_split, load_weights, pick_device
from model import PianoCNN


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-cache", nargs="+", default=[str(SLAKH_CACHE)])
    ap.add_argument("--val-cache", default=None, help="default: the first train cache")
    ap.add_argument("--val-split", default="validation")
    ap.add_argument("--init-from", default=None, help="start from these weights (e.g. piano best.pt)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=500, help="batches per epoch")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crop-frames", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=8.0)
    ap.add_argument("--onset-pos-weight", type=float, default=20.0)
    ap.add_argument("--pitch-aug", type=float, default=0.5, help="probability of a +-1..2 semitone shift")
    ap.add_argument("--max-shift", type=int, default=2, help="largest pitch shift in semitones (voice: try 12)")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--val-pieces", type=int, default=10)
    ap.add_argument("--run", default="guitar1")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(0)
    device = pick_device()
    print("device:", device)

    run_dir = ROOT / "checkpoints" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path, last_path, best_path = run_dir / "log.csv", run_dir / "last.pt", run_dir / "best.pt"
    with open(run_dir / "meta.json", "w") as f:
        json.dump({"args": vars(args)}, f, indent=2)

    train_ds = GuitarDataset(args.train_cache, "train", crop_frames=args.crop_frames,
                             samples_per_epoch=args.steps * args.batch_size, pitch_aug=args.pitch_aug,
                             max_shift=args.max_shift)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, persistent_workers=args.workers > 0,
                        pin_memory=(device == "cuda"))

    val_cache = args.val_cache or args.train_cache[0]
    val_split = args.val_split
    if not pieces_for(val_cache, val_split):
        print(f"note: no '{val_split}' pieces in {val_cache}; validating on the TRAIN split instead "
              f"(not a real score)")
        val_split = "train"
        if not pieces_for(val_cache, val_split):
            raise SystemExit(f"No cached pieces at all in {val_cache}. Run the prepare script first.")

    model = PianoCNN().to(device)
    if args.init_from and not args.resume:
        load_weights(model, args.init_from, device)
        print("initialised from", args.init_from)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"parameters: {n_params:,} (about {n_params * 4 / 1e6:.1f} MB)")

    frame_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))
    onset_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.onset_pos_weight, device=device))
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
            csv.writer(f).writerow(["epoch", "train_loss", "val_onset_f1", "val_onset_offset_f1",
                                    "val_pieces", "seconds"])

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        for x, yf, yo in loader:
            x, yf, yo = x.to(device), yf.to(device), yo.to(device)
            f_logits, o_logits = model(x)
            loss = frame_loss(f_logits, yf) + onset_loss(o_logits, yo)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        train_loss = total / len(loader)

        on_f1, full_f1, n_val = evaluate_split(model, device, val_cache, val_split, args.val_pieces)
        improved = on_f1 > best
        if improved:
            best = on_f1
            torch.save(model.state_dict(), best_path)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": epoch, "best": best}, last_path)

        secs = time.time() - t0
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{train_loss:.5f}", f"{on_f1:.4f}", f"{full_f1:.4f}", n_val, f"{secs:.0f}"])
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  val onset F1 {on_f1:.3f}  "
              f"on+off F1 {full_f1:.3f}  ({n_val} pieces)  {secs:.0f}s" + ("  *best*" if improved else ""))

    print("done. best weights:", best_path)


if __name__ == "__main__":       # required on Windows (DataLoader workers)
    main()