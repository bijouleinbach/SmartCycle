"""Load every labeled image under data/ into one dataset and split it.

Layout (one folder per person or public dataset):

    data/<source>/labeled/labels.csv     columns: see src.labels.CSV_COLUMNS
    data/<source>/labeled/<images...>    paths in `filename` are relative to this folder

Adding your own photos means creating data/<your name>/labeled/, putting the
photos there and adding one CSV row per photo. No code changes needed.
"""

import random
from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms

from src.labels import AXES, CSV_COLUMNS, index_of

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMAGE_SIZE = 224
_MEAN, _STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet stats


def load_index(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Read every labels.csv into one table with absolute paths and a `source` column."""
    frames = []
    for csv_path in sorted(data_dir.glob("*/labeled/labels.csv")):
        df = pd.read_csv(csv_path, dtype=str).fillna("")
        missing = set(CSV_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} is missing columns: {sorted(missing)}")
        df["path"] = [str(csv_path.parent / f) for f in df["filename"]]
        df["source"] = csv_path.parent.parent.name
        for axis in AXES:
            df[f"{axis}_idx"] = [index_of(axis, v) for v in df[axis]]
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No labeled data found under {data_dir}/*/labeled/labels.csv. "
            "Run `python -m scripts.download_trashnet` or add your own photos."
        )
    return pd.concat(frames, ignore_index=True)


def split(df: pd.DataFrame, val=0.15, test=0.15, seed=0, holdout_source: str | None = None):
    """Split into train/val/test.

    Before/after photos with the same pair_id always land in the same split, so
    the model is never tested on an item it saw during training. With
    holdout_source, that whole source becomes the test set (for the
    cross-person variability analysis).
    """
    df = df.copy()
    df["group"] = [
        f"{s}/{p}" if p else f"{s}/{f}" for s, p, f in zip(df["source"], df["pair_id"], df["filename"])
    ]
    if holdout_source:
        test_df = df[df["source"] == holdout_source]
        df = df[df["source"] != holdout_source]
        test = 0.0
    groups = sorted(df["group"].unique())
    random.Random(seed).shuffle(groups)
    n_val, n_test = int(len(groups) * val), int(len(groups) * test)
    val_groups = set(groups[:n_val])
    test_groups = set(groups[n_val : n_val + n_test])
    parts = {
        "train": df[~df["group"].isin(val_groups | test_groups)],
        "val": df[df["group"].isin(val_groups)],
        "test": test_df if holdout_source else df[df["group"].isin(test_groups)],
    }
    return {k: v.reset_index(drop=True) for k, v in parts.items()}


def train_transform():
    # Random crops, flips and lighting changes mimic different phones and rooms.
    return transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def eval_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ])


def open_image(path_or_file) -> Image.Image:
    # exif_transpose applies the phone's rotation flag so portrait photos aren't sideways.
    return ImageOps.exif_transpose(Image.open(path_or_file)).convert("RGB")


class SmartCycleDataset(Dataset):
    """Returns (image tensor, {axis: class index or -1 if unlabeled})."""

    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = self.transform(open_image(row["path"]))
        targets = {axis: torch.tensor(row[f"{axis}_idx"]) for axis in AXES}
        return image, targets
