import matplotlib.pyplot as plt
from config import *
from maestro_dataset import MaestroDataset
from labels import FPS

for split in ("train", "validation", "test"):
    try:
        ds = MaestroDataset(split)
    except RuntimeError as e:
        print(f"{split}: none cached yet")
        continue
    x, yf, yo = ds[0]
    print(f"{split}: sample cqt {tuple(x.shape)}  frame {tuple(yf.shape)}  "
          f"onset {tuple(yo.shape)}  active {float(yf.mean()):.3f}")

# real-audio alignment check on the first training piece
ds = MaestroDataset("train", random_crop=False)
cqt, frame, onset = ds.load_piece(0)
n = int(20 * FPS)
secs = n / FPS
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
ax[0].imshow(cqt[:, :n], origin="lower", aspect="auto", extent=[0, secs, 0, 264])
ax[0].set_title("CQT (real recording)")
ax[1].imshow(frame[:, :n], origin="lower", aspect="auto", extent=[0, secs, 0, 88], cmap="gray_r")
ax[1].set_title("Piano-roll from MIDI")
ax[1].set_xlabel("seconds")
plt.tight_layout()
(ROOT / "output").mkdir(exist_ok=True)
plt.savefig(ROOT / "output" / "real_alignment.png", dpi=120)
plt.show()