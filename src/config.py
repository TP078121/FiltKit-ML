from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent      # the Piano-Transcription folder
MAESTRO_DIR = ROOT / "data" / "maestro-v3.0.0"
SYNTH_DIR = ROOT / "data" / "synthetic"
MAESTRO_CSV = MAESTRO_DIR / "maestro-v3.0.0.csv"
CACHE_DIR = ROOT / "data" / "cache" / "maestro"

# ---------- Piano settings (unchanged) ----------
SAMPLE_RATE = 16000
HOP_LENGTH = 512             # 16000 / 512 = 31.25 frames per second
FMIN_NOTE = "A0"             # lowest piano key
BINS_PER_OCTAVE = 36         # 3 bins per semitone
N_BINS = 264                 # about 7.3 octaves, covers all 88 keys
N_PITCHES = 88
CLIP_SECONDS = 20            # length of each synthetic piano test clip


# ---------- Drums (E-GMD) ----------
def _find_egmd():
    """Use the first folder under data/ whose name starts with 'e-gmd'."""
    base = ROOT / "data"
    for c in sorted(base.glob("e-gmd*")):
        if c.is_dir():
            return c
    return base / "e-gmd-v1.0.0"


EGMD_DIR = _find_egmd()
EGMD_CSV = next(iter(sorted(EGMD_DIR.glob("*.csv"))), EGMD_DIR / "e-gmd-v1.0.0.csv")
DRUM_CACHE_DIR = ROOT / "data" / "cache" / "egmd"         # preprocessed real E-GMD
DRUM_SYNTH_DIR = ROOT / "data" / "cache" / "drum_synth"   # small synthetic laptop test

DRUM_SR = 16000
DRUM_HOP = 160               # 16000 / 160 = 100 frames per second (10 ms)
DRUM_FPS = DRUM_SR / DRUM_HOP
DRUM_NFFT = 1024
DRUM_N_MELS = 128            # must be divisible by 8 (the model pools 3 times)
DRUM_FMIN = 30
DRUM_FMAX = 8000
