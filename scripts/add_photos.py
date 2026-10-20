"""Turn a folder of raw phone photos into labeled-data format.

1. Put your photos in data/<your name>/raw/ (any JPG/PNG/HEIC exported as JPG).
2. Run:  python -m scripts.add_photos <your name>
3. Open data/<your name>/labeled/labels.csv in Excel/Sheets/Numbers and fill in
   material, condition, component, stage (before/after) and pair_id for each row.
   Allowed values are listed in src/labels.py. Leave a cell blank if unsure.

Photos are resized to 1024 px on the long side (plenty for a 224 px model) so
they're small enough to commit to GitHub. Running it again only adds new
photos; rows you already filled in are kept.
"""

import csv
import sys
from pathlib import Path

from PIL import Image

from src.data import DATA_DIR, open_image
from src.labels import CSV_COLUMNS

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_SIDE = 1024


def main(name: str):
    raw, labeled = DATA_DIR / name / "raw", DATA_DIR / name / "labeled"
    if not raw.is_dir():
        sys.exit(f"Put your photos in {raw}/ first.")
    labeled.mkdir(parents=True, exist_ok=True)
    csv_path = labeled / "labels.csv"
    rows = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
    known = {r["filename"] for r in rows}

    added = 0
    for src in sorted(raw.iterdir()):
        if src.suffix.lower() not in EXTS:
            if src.suffix.lower() == ".heic":
                print(f"skipping {src.name}: export HEIC photos as JPG first (Photos app > File > Export)")
            continue
        out_name = src.stem + ".jpg"
        if out_name in known:
            continue
        image = open_image(src)
        image.thumbnail((MAX_SIDE, MAX_SIDE), Image.LANCZOS)
        image.save(labeled / out_name, "JPEG", quality=88)
        rows.append({c: "" for c in CSV_COLUMNS} | {"filename": out_name})
        added += 1

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Added {added} photos. Now fill in the labels in {csv_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python -m scripts.add_photos <your name>")
    main(sys.argv[1])
