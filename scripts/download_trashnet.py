"""Download TrashNet (resized, ~43 MB) into data/trashnet/labeled/ in our CSV format.

TrashNet (Yang & Thung, 2016) has ~2,500 photos labeled by material only, so the
condition and component columns are left blank. Its "trash" class doesn't map
onto our schema and is skipped.

Run: python -m scripts.download_trashnet
"""

import csv
import io
import shutil
import urllib.request
import zipfile
from pathlib import Path

import src  # noqa: F401  (sets up SSL certificates for the download)

URL = "https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip"
OUT = Path(__file__).resolve().parent.parent / "data" / "trashnet" / "labeled"

MATERIAL_MAP = {
    "cardboard": "paper_cardboard",
    "paper": "paper_cardboard",
    "glass": "glass",
    "metal": "metal",
    "plastic": "plastic",
}


def main():
    if (OUT / "labels.csv").exists():
        print(f"Already downloaded: {OUT}")
        return
    print(f"Downloading {URL} ...")
    with urllib.request.urlopen(URL) as r:
        archive = zipfile.ZipFile(io.BytesIO(r.read()))

    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in archive.namelist():
        parts = Path(name).parts
        if len(parts) < 2 or not name.lower().endswith(".jpg") or "__MACOSX" in parts:
            continue
        cls = parts[-2]
        if cls not in MATERIAL_MAP:
            continue
        rel = Path(cls) / parts[-1]
        (OUT / cls).mkdir(exist_ok=True)
        with archive.open(name) as src, open(OUT / rel, "wb") as dst:
            shutil.copyfileobj(src, dst)
        rows.append({
            "filename": rel.as_posix(), "material": MATERIAL_MAP[cls],
            "condition": "", "component": "", "stage": "", "pair_id": "",
        })

    with open(OUT / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["filename"]))
    print(f"Wrote {len(rows)} images and labels.csv to {OUT}")


if __name__ == "__main__":
    main()
