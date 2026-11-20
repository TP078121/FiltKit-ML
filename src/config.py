from pathlib import Path

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent      # the piano-transcription folder
MAESTRO_DIR = ROOT / "data" / "maestro-v3.0.0"     # extracted MAESTRO MIDI
SYNTH_DIR = ROOT / "data" / "synthetic"            # generated test clips

# ---------- Audio / feature settings ----------
SAMPLE_RATE = 16000
HOP_LENGTH = 512             # 16000 / 512 = 31.25 frames per second
FMIN_NOTE = "A0"             # lowest piano key
BINS_PER_OCTAVE = 36         # 3 bins per semitone
N_BINS = 264                 # about 7.3 octaves, covers all 88 keys
N_PITCHES = 88
CLIP_SECONDS = 20            # length of each test clip