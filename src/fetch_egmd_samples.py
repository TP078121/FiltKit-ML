"""Download a SMALL sample of real E-GMD (a few pieces) without the 90 GB zip.
Needs:  pip install remotezip requests

  python src/fetch_egmd_samples.py            (12 train + 3 validation pieces, each from a different kit)
"""
import argparse
import csv
import shutil

import requests
from remotezip import RemoteZip

from config import ROOT

BASE = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/"
ZIP_URL = BASE + "e-gmd-v1.0.0.zip"
CSV_URL = BASE + "e-gmd-v1.0.0.csv"
CSV_NAME = "e-gmd-v1.0.0.csv"
TARGET = ROOT / "data" / "e-gmd-v1.0.0"


def pick(rows, split, n):
    """n pieces of at least 10 s, shortest first, each from a different drum kit."""
    rows = [r for r in rows if r["split"] == split and r.get("audio_filename")
            and float(r["duration"]) >= 10]
    rows.sort(key=lambda r: float(r["duration"]))
    chosen, kits = [], set()
    for r in rows:
        if r["kit_name"] in kits:
            continue
        kits.add(r["kit_name"])
        chosen.append(r)
        if len(chosen) == n:
            break
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-val", type=int, default=3)
    args = ap.parse_args()

    TARGET.mkdir(parents=True, exist_ok=True)
    csv_path = TARGET / CSV_NAME
    if not csv_path.exists():
        print("downloading metadata csv ...")
        r = requests.get(CSV_URL, timeout=120)
        r.raise_for_status()
        csv_path.write_bytes(r.content)
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    chosen = pick(rows, "train", args.n_train) + pick(rows, "validation", args.n_val)
    print(f"selected {len(chosen)} pieces")

    with RemoteZip(ZIP_URL) as z:
        names = z.namelist()
        hit = next((n for n in names if n.endswith(CSV_NAME)), None)
        prefix = hit[:-len(CSV_NAME)] if hit else "e-gmd-v1.0.0/"
        print("folder prefix inside the zip:", repr(prefix))
        for r in chosen:
            for rel in (r["audio_filename"], r["midi_filename"]):
                dest = TARGET / rel
                if dest.exists():
                    continue
                member = prefix + rel
                try:
                    src = z.open(member)
                except KeyError:
                    print("NOT FOUND in zip:", member)
                    print("first entries in the zip:", names[:5])
                    raise SystemExit("Tell me the names above so the path can be fixed.")
                dest.parent.mkdir(parents=True, exist_ok=True)
                with src, open(dest, "wb") as out:
                    shutil.copyfileobj(src, out)
                print("fetched", rel)
    print("done ->", TARGET)


if __name__ == "__main__":
    main()
