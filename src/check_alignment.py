import matplotlib.pyplot as plt
from config import *
from features import load_audio, compute_cqt
from labels import midi_to_roll, FPS

wav_path = SYNTH_DIR / "clip_0.wav"
mid_path = SYNTH_DIR / "clip_0.mid"

cqt = compute_cqt(load_audio(str(wav_path)))
roll = midi_to_roll(str(mid_path), cqt.shape[1])
print("CQT shape:", cqt.shape, " Roll shape:", roll.shape)

secs = cqt.shape[1] / FPS
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(12, 7))
ax[0].imshow(cqt, origin="lower", aspect="auto", extent=[0, secs, 0, 264])
ax[0].set_title("CQT of audio")
ax[1].imshow(roll, origin="lower", aspect="auto", extent=[0, secs, 0, 88], cmap="gray_r")
ax[1].set_title("Piano-roll from MIDI (labels)")
ax[1].set_xlabel("seconds")
plt.tight_layout()
plt.savefig(ROOT / "alignment.png", dpi=120)   # always saves a picture
plt.show()