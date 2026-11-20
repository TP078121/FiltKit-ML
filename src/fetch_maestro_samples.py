from remotezip import RemoteZip
from config import *
from maestro_dataset import read_rows

ZIP_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"

def shortest(split, n):
    return sorted(read_rows(split), key=lambda r: float(r["duration"]))[:n]

chosen = shortest("train", 4) + shortest("validation", 1)
with RemoteZip(ZIP_URL) as z:
    for r in chosen:
        member = f"maestro-v3.0.0/{r['audio_filename']}"
        print("fetching", member, f"({float(r['duration']):.0f} s)")
        z.extract(member, path=ROOT / "data")