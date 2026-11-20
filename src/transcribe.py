import numpy as np
import torch
import matplotlib.pyplot as plt
import pretty_midi
import mir_eval
import soundfile as sf

from config import *
from features import load_audio, compute_cqt
from labels import midi_to_roll, FPS
from model import PianoCNN
from midi_utils import roll_to_notes, notes_to_midi, render_audio

THRESHOLD = 0.5
MIN_FRAMES = 2

device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = PianoCNN().to(device)
model.load_state_dict(torch.load(ROOT / "checkpoints" / "piano_cnn.pt", map_location=device))
model.eval()

OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)


def predict(wav_path):
    cqt = compute_cqt(load_audio(str(wav_path)))
    x = torch.from_numpy(cqt)[None, None].to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0].cpu().numpy()      # (T, 88)
    return cqt, prob


def note_scores(ref_mid, est_notes):
    pm = pretty_midi.PrettyMIDI(str(ref_mid))
    ref = [(n.start, n.end, n.pitch) for i in pm.instruments for n in i.notes if n.end > n.start]
    est = [(s, e, p) for s, e, p in est_notes if e > s]
    if not ref or not est:
        return 0.0, 0.0
    ref_i = np.array([[s, e] for s, e, _ in ref])
    ref_p = np.array([pretty_midi.note_number_to_hz(p) for _, _, p in ref])
    est_i = np.array([[s, e] for s, e, _ in est])
    est_p = np.array([pretty_midi.note_number_to_hz(p) for _, _, p in est])
    onset = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p, offset_ratio=None)[2]    # onset only
    full = mir_eval.transcription.precision_recall_f1_overlap(
        ref_i, ref_p, est_i, est_p)[2]                       # onset + offset
    return onset, full


print(f"{'clip':8s} {'ref notes':>10s} {'pred notes':>11s} {'onset F1':>9s} {'on+off F1':>10s}")
first = None
for wav in sorted(SYNTH_DIR.glob("clip_*.wav")):
    ref_mid = wav.with_suffix(".mid")
    cqt, prob = predict(wav)
    notes = roll_to_notes(prob, THRESHOLD, MIN_FRAMES)

    notes_to_midi(notes, OUT / f"{wav.stem}_pred.mid")
    sf.write(str(OUT / f"{wav.stem}_pred.wav"), render_audio(notes), SAMPLE_RATE)

    n_ref = sum(len(i.notes) for i in pretty_midi.PrettyMIDI(str(ref_mid)).instruments)
    onset, full = note_scores(ref_mid, notes)
    print(f"{wav.stem:8s} {n_ref:10d} {len(notes):11d} {onset:9.3f} {full:10.3f}")

    if first is None:
        first = (wav, prob, midi_to_roll(str(ref_mid), cqt.shape[1]))

# comparison plot for the first clip
wav, prob, true = first
secs = prob.shape[0] / FPS
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(12, 9))
ax[0].imshow(true, origin="lower", aspect="auto", extent=[0, secs, 0, 88], cmap="gray_r")
ax[0].set_title("True piano-roll")
ax[1].imshow(prob.T, origin="lower", aspect="auto", extent=[0, secs, 0, 88], cmap="gray_r")
ax[1].set_title("Model probabilities")
ax[2].imshow((prob.T > THRESHOLD), origin="lower", aspect="auto", extent=[0, secs, 0, 88], cmap="gray_r")
ax[2].set_title("After threshold")
ax[2].set_xlabel("seconds")
plt.tight_layout()
plt.savefig(OUT / "prediction_vs_true.png", dpi=120)
plt.show()
print("saved results to", OUT)