#!/usr/bin/env python3
"""
SmartCycle - pre-processing and feature analysis (Part II)

WHAT THIS SCRIPT DOES
---------------------
1. Reads labels.csv (one row per photo: file name, material, condition, optional crop box).
2. Adds any NEW photos from the photos folder to labels.csv (see run mode A below).
3. Pre-processes every photo:  crop to the item -> resize -> normalise brightness.
4. Computes three features on the RAW photo and on the PRE-PROCESSED crop:
       color_fraction  - % of pixels that are warm-coloured (brown / yellow / green)
       edge_density    - % of pixels where a Sobel filter finds a strong brightness change
       glare_fraction  - % of pixels that are very bright and almost colourless
5. Saves features.csv and stats.csv (correlation + mutual information).
6. Saves four figures (dataset overview, raw vs pre-processed, brightness histograms,
   features vs outputs).

SETUP (once)
------------
    pip install numpy pandas pillow scipy matplotlib

RUN MODES
---------
Run these from inside the folder that contains photos/ and labels.csv.
Use python3 instead of python if "python" is not found.

A) NORMAL RUN: add new photos to labels.csv AND update all results
       python3 smartcycle_feature_analysis.py --images photos --labels labels.csv --out results
   labels.csv : CHANGED. Photos in the photos folder that are not in labels.csv are added as new rows at
                the bottom (material = unknown, condition = unknown, no crop box). A copy of the old
                file is saved first as labels_backup.csv. Existing rows are never touched.
   results/   : features.csv, stats.csv and the 4 figures are rewritten using ALL photos.
   After it runs: open labels.csv, fill in material, condition and crop box for the new rows,
                save, and run the same command again.

B) LEAVE labels.csv ALONE, but still update all results
       python3 smartcycle_feature_analysis.py --images photos --labels labels.csv --out results --no-sync
   labels.csv : NOT changed.
   results/   : rewritten as in A, but only for photos that are already listed in labels.csv.
                New photos in the folder are ignored until you add them to labels.csv yourself.

C) JUST PRINT THE NEW PHOTOS' FEATURES, change nothing else
       python3 smartcycle_feature_analysis.py --images photos --labels labels.csv --out results --new
   labels.csv : NOT changed.
   Looks only at photos in the folder that are not in labels.csv. Prints their raw and
   pre-processed features next to the average of your labelled clean and contaminated
   photos (for comparison only, not a prediction).
   results/   : only new_photos_features.csv and new_photos_preview.png are written.
                features.csv, stats.csv and the 4 figures are NOT redone.
   Crop box   : the whole photo is used unless you give one:  --box 0.2 0.1 0.8 0.9
                (left top right bottom, as fractions from 0 to 1)
   Specific photos instead of "all new ones" (listed in labels.csv or not):
       python3 smartcycle_feature_analysis.py --images photos --labels labels.csv --out results --only IMG_0700.jpg IMG_0701.jpg

D) CHECK THE CROP BOXES
       python3 smartcycle_feature_analysis.py --images photos --labels labels.csv --out results --check-boxes
   Draws every crop box in red on its photo (results/boxes_check.png) and stops.
   No features or statistics are computed.
   Note: this also adds new photos to labels.csv, like mode A. Add --no-sync to prevent that.

OTHER OPTIONS
-------------
  --show IMG_A.jpg IMG_B.jpg   choose the photos used in the raw-vs-pre-processed and histogram
                               figures (default: the first photo of up to 4 different items)
  --out FOLDER                 where the results are saved (default: results)

OTHER BEHAVIOUR
---------------
- Photo types: JPG and PNG are read. HEIC/HEIF (iPhone originals) are skipped with a warning,
  so convert them to JPG first (Preview > File > Export > JPEG).
- condition = unknown (or blank): the photo still gets features and appears in the overview
  figure, but is left out of the statistics and plots until you fill the condition in.
- Crop box left blank: the whole photo is used.
- item_id: one ID per PHYSICAL item, so all photos of one item share it (this keeps an item's
  photos together when you later split into training and test sets). Auto-added rows use the file name.
- With only a few labelled photos, the statistics are only a demonstration.

labels.csv COLUMNS
------------------
    file         photo file name, e.g. IMG_0538.jpg                                    (required)
    material     plastic / glass / paper / metal / organic ...                         (required)
    condition    clean, or anything else for contaminated (residue, grease, liquid)    (required)
                 Write "unknown" if you have not checked the item yet.
    item_id      one ID per physical item                                              (optional)
    x0,y0,x1,y1  crop box as fractions of the photo, 0 to 1 (left, top, right, bottom).
                 Leave blank to use the whole photo.                                   (optional)
    notes        free text for you; the script ignores it                              (optional)
"""

import argparse
import csv
import math
import os
import shutil
import sys

import matplotlib
matplotlib.use("Agg")                      # draw figures to files, no window needed
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw
from scipy import ndimage as ndi

# --------------------------------------------------------------------------
# Settings. These are the numbers to change if you want to experiment.
# --------------------------------------------------------------------------
LONG_SIDE = 256            # every image is resized so its longer side is this many pixels
TARGET_BRIGHTNESS = 0.5    # brightness normalisation: average brightness is set to this (0 to 1)

HUE_RANGE = (20, 90)       # "warm" hues in degrees: brown ~20-40, yellow ~40-65, green ~65-90
MIN_SAT = 0.25             # a pixel needs at least this much colour to count as warm
MIN_VAL = 0.20             # ...and must not be nearly black

EDGE_THRESHOLD = 0.10      # Sobel gradient size (0 to 1 scale) above which a pixel is an "edge"

GLARE_MIN_VAL = 0.85       # glare = very bright...
GLARE_MAX_SAT = 0.15       # ...and almost colourless

# condition values that mean "I have not checked this item yet" (skipped in statistics and plots)
UNKNOWN_CONDITIONS = {"unknown", "uncertain", "?", "", "nan", "none"}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}     # photo types the script can read
UNSUPPORTED_EXTENSIONS = {".heic", ".heif"}      # iPhone formats: convert these to JPG first
DEFAULT_LABEL_COLUMNS = ["file", "item_id", "material", "condition", "x0", "y0", "x1", "y1", "notes"]
NEW_ROW_NOTE = "auto-added: fill in material, condition and crop box"

FEATURES = ["color_fraction", "edge_density", "glare_fraction"]
FEATURE_NAMES = {
    "color_fraction": "Color fraction",
    "edge_density": "Edge density",
    "glare_fraction": "Glare fraction",
}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


# --------------------------------------------------------------------------
# Image helpers
# --------------------------------------------------------------------------
def load_image(path):
    return Image.open(path).convert("RGB")


def resize_long(im, long_side=LONG_SIDE):
    """Resize keeping the shape, so that the longer side equals long_side."""
    w, h = im.size
    s = long_side / max(w, h)
    return im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)


def crop_fraction(im, box):
    """Crop with a box given as fractions of the photo: (left, top, right, bottom)."""
    w, h = im.size
    x0, y0, x1, y1 = box
    return im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))


def to_array(im):
    """PIL image -> float array with values 0 to 1, shape (height, width, 3)."""
    return np.asarray(im).astype(float) / 255.0


def raw_version(im):
    """The 'raw' data: the whole photo, only resized so it is comparable in size."""
    return to_array(resize_long(im))


def preprocess(im, box):
    """Pre-processing: crop to the item -> resize -> normalise brightness.
    No smoothing (it would blur thin residue). Colour balance is left to the phone,
    because gray-world balancing erased almost all colour in tests."""
    a = to_array(resize_long(crop_fraction(im, box)))
    return np.clip(a * (TARGET_BRIGHTNESS / a.mean()), 0, 1)


# --------------------------------------------------------------------------
# Features (all returned as a percentage of the pixels in the array given)
# --------------------------------------------------------------------------
def compute_features(a):
    hsv = mc.rgb_to_hsv(a)
    hue = hsv[..., 0] * 360.0
    sat = hsv[..., 1]
    val = hsv[..., 2]

    # 1) colour fraction: warm hue, enough colour, not too dark
    warm = (hue >= HUE_RANGE[0]) & (hue <= HUE_RANGE[1]) & (sat > MIN_SAT) & (val > MIN_VAL)

    # 2) edge density: Sobel gradient size above a fixed threshold
    gray = a.mean(axis=2)
    gx = ndi.sobel(gray, axis=1) / 4.0      # divide by 4 so the value is a 0-1 brightness step
    gy = ndi.sobel(gray, axis=0) / 4.0
    edges = np.hypot(gx, gy) > EDGE_THRESHOLD

    # 3) glare fraction: bright and low saturation
    glare = (val > GLARE_MIN_VAL) & (sat < GLARE_MAX_SAT)

    return {
        "color_fraction": warm.mean() * 100,
        "edge_density": edges.mean() * 100,
        "glare_fraction": glare.mean() * 100,
    }


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------
def entropy(labels):
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def mutual_info_median_split(x, y):
    """Mutual information (in bits) between a feature and a label.
    The feature is split at its median (above / below) so we can count.
    Fine for a demo; with a big dataset use sklearn.feature_selection.mutual_info_classif."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y)
    b = (x > np.median(x)).astype(int)
    joint = np.array([f"{i}|{j}" for i, j in zip(b, y)])
    return entropy(b) + entropy(y) - entropy(joint)


def safe_corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------
# Reading labels
# --------------------------------------------------------------------------
def read_labels(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    for col in ("file", "material", "condition"):
        if col not in df.columns:
            sys.exit(f"labels.csv is missing the required column '{col}'")
    if "item_id" not in df.columns:
        df["item_id"] = df["file"]
    for col, default in (("x0", 0.0), ("y0", 0.0), ("x1", 1.0), ("y1", 1.0)):
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    df["material"] = df["material"].astype(str).str.strip().str.lower()
    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    # 'unknown' means not checked yet: kept, but left out of the statistics and plots
    df["labeled"] = ~df["condition"].isin(UNKNOWN_CONDITIONS)
    # anything else that is not 'clean' counts as contaminated (residue, grease, liquid, ...)
    df["contaminated"] = np.where(df["labeled"], (df["condition"] != "clean").astype(float), np.nan)
    return df


def box_of(row):
    return (row["x0"], row["y0"], row["x1"], row["y1"])


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def short_label(row):
    return f"{os.path.splitext(row['file'])[0]}\n{row['material']}, {row['condition']}"


def fig_check_boxes(df, img_dir, out):
    n = len(df)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4 * nrows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axs.ravel(), df.iterrows()):
        im = load_image(os.path.join(img_dir, r["file"]))
        w, h = im.size
        d = ImageDraw.Draw(im)
        x0, y0, x1, y1 = box_of(r)
        d.rectangle((x0 * w, y0 * h, x1 * w, y1 * h), outline=(255, 0, 0), width=max(4, w // 150))
        ax.imshow(resize_long(im, 400))
        ax.set_title(r["file"], fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "boxes_check.png"), dpi=130)
    plt.close()


def fig_dataset(df, img_dir, out):
    n = len(df)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.5, 2.7 * nrows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axs.ravel(), df.iterrows()):
        ax.imshow(resize_long(load_image(os.path.join(img_dir, r["file"])), 400))
        ax.set_title(short_label(r), fontsize=8)
    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(out, "fig1_dataset.png"), dpi=170)
    plt.close()


def fig_raw_vs_pre(show, img_dir, out, name="fig2_raw_vs_preprocessed.png"):
    k = len(show)
    fig, axs = plt.subplots(2, k, figsize=(max(4.0, 1.6 * k + 0.6), 5.0), squeeze=False)
    for j, (_, r) in enumerate(show.iterrows()):
        im = load_image(os.path.join(img_dir, r["file"]))
        axs[0, j].imshow(resize_long(im, 420))
        axs[0, j].set_title(os.path.splitext(r["file"])[0], fontsize=9)
        axs[1, j].imshow(preprocess(im, box_of(r)))
        for i in (0, 1):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    axs[0, 0].set_ylabel("Raw photo", fontsize=11)
    axs[1, 0].set_ylabel("Pre-processed", fontsize=11)
    plt.tight_layout(pad=0.6)
    plt.savefig(os.path.join(out, name), dpi=170)
    plt.close()


def fig_histograms(show, img_dir, out):
    k = len(show)
    ncols = 2
    nrows = math.ceil(k / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.5, 2.4 * nrows + 0.5), sharex=True, squeeze=False)
    bins = np.arange(0, 257, 8)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axs.ravel(), show.iterrows()):
        ax.axis("on")
        im = load_image(os.path.join(img_dir, r["file"]))
        raw = raw_version(im).mean(axis=2).ravel() * 255
        pre = preprocess(im, box_of(r)).mean(axis=2).ravel() * 255
        for v, color, label in ((raw, "#7f7f7f", "Raw photo"), (pre, "#1f77b4", "Pre-processed")):
            h, _ = np.histogram(v, bins=bins)
            ax.step(bins[:-1], h / h.sum() * 100, where="post", color=color, lw=1.6, label=label)
        ax.set_title(os.path.splitext(r["file"])[0], fontsize=10)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=9)
        ax.set_ylabel("Share of pixels (%)", fontsize=10)
        ax.set_xlabel("Brightness (0 = black, 255 = white)", fontsize=10)
    handles, labels = None, None
    for ax in axs.ravel():
        if ax.lines:
            handles, labels = ax.get_legend_handles_labels()
            break
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5, frameon=False)
    plt.tight_layout(rect=(0, 0.06, 1, 1), pad=0.6)
    plt.savefig(os.path.join(out, "fig3_brightness_histograms.png"), dpi=170)
    plt.close()


def fig_features_vs_outputs(feat, out):
    materials = sorted(feat["material"].unique())
    # material colours avoid green and orange, which are used for the condition (clean / contaminated)
    palette = ["#d62728", "#1f77b4", "#8c6d31", "#9467bd", "#17becf", "#e377c2", "#7f7f7f"]
    mat_color = {m: palette[i % len(palette)] for i, m in enumerate(materials)}
    cond_color = {0: "#2ca02c", 1: "#ff7f0e"}
    rng = np.random.default_rng(1)

    fig, axs = plt.subplots(2, 3, figsize=(6.5, 5.5))
    for i, f in enumerate(FEATURES):
        col = "pre_" + f

        # top row: feature vs condition (clean / contaminated), dots coloured by material
        ax = axs[0, i]
        for _, r in feat.iterrows():
            ax.scatter(r["contaminated"] + rng.uniform(-0.12, 0.12), r[col], color=mat_color[r["material"]],
                       s=42, edgecolor="k", lw=0.5, zorder=3)
        for c in (0, 1):
            sub = feat[feat["contaminated"] == c]
            if len(sub):
                ax.hlines(sub[col].mean(), c - 0.3, c + 0.3, color="k", lw=1.4, zorder=2)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Clean", "Contaminated"], fontsize=8)
        ax.set_xlim(-0.6, 1.6)
        ax.set_title(f"{FEATURE_NAMES[f]}\nvs condition", fontsize=10)
        ax.set_xlabel("Condition (output)", fontsize=9)
        ax.set_ylabel("% of item pixels", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=9)

        # bottom row: feature vs material, dots coloured by condition
        ax = axs[1, i]
        for _, r in feat.iterrows():
            ax.scatter(materials.index(r["material"]) + rng.uniform(-0.12, 0.12), r[col],
                       color=cond_color[r["contaminated"]], s=42, edgecolor="k", lw=0.5, zorder=3)
        for xi, m in enumerate(materials):
            ax.hlines(feat.loc[feat["material"] == m, col].mean(), xi - 0.3, xi + 0.3, color="k", lw=1.4, zorder=2)
        ax.set_xticks(range(len(materials)))
        ax.set_xticklabels([m.capitalize() for m in materials], fontsize=8)
        ax.set_xlim(-0.6, len(materials) - 0.4)
        ax.set_title(f"{FEATURE_NAMES[f]}\nvs material", fontsize=10)
        ax.set_xlabel("Material (output)", fontsize=9)
        ax.set_ylabel("% of item pixels", fontsize=9)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=9)

    handles = [Line2D([], [], marker="o", ls="", color=c, markeredgecolor="k", label=m.capitalize())
               for m, c in mat_color.items()]
    handles += [Line2D([], [], marker="o", ls="", color=cond_color[0], markeredgecolor="k", label="Clean"),
                Line2D([], [], marker="o", ls="", color=cond_color[1], markeredgecolor="k", label="Contaminated"),
                Line2D([], [], color="k", lw=1.4, label="Group mean")]
    fig.legend(handles=handles, loc="lower center", ncol=min(6, len(handles)), fontsize=8, frameon=False,
               title="Top row: dots colored by material. Bottom row: dots colored by condition.",
               title_fontsize=8)
    plt.tight_layout(rect=(0, 0.1, 1, 1), pad=0.6)
    plt.savefig(os.path.join(out, "fig4_features_vs_outputs.png"), dpi=170)
    plt.close()


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def sync_labels(labels_path, img_dir):
    """Add every photo in img_dir that is not listed in labels.csv as a new row at the end of the file.
    Existing rows are never touched. A backup copy of labels.csv is saved first. Returns the added names."""
    all_files = sorted(os.listdir(img_dir))
    on_disk = [f for f in all_files if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS]
    heic = [f for f in all_files if os.path.splitext(f)[1].lower() in UNSUPPORTED_EXTENSIONS]
    if heic:
        print(f"\nWarning: {len(heic)} HEIC/HEIF photo(s) in the images folder were ignored "
              f"(the script reads JPG and PNG only): {', '.join(heic[:5])}{' ...' if len(heic) > 5 else ''}")
        print("  Convert them to JPG first (for example, open in Preview > File > Export > JPEG).")

    if not os.path.exists(labels_path):                      # no labels.csv yet: start one
        header = list(DEFAULT_LABEL_COLUMNS)
        known = set()
        with open(labels_path, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh, lineterminator="\n").writerow(header)
    else:
        with open(labels_path, newline="", encoding="utf-8-sig") as fh:
            rows = list(csv.reader(fh))
        if not rows:
            sys.exit(f"{labels_path} is empty. Delete it and run again, and a new one will be created.")
        header = [h.strip() for h in rows[0]]
        lower = [h.lower() for h in header]
        if "file" not in lower:
            sys.exit("labels.csv needs a 'file' column.")
        idx = lower.index("file")
        known = {r[idx].strip() for r in rows[1:] if len(r) > idx}

    to_add = [f for f in on_disk if f not in known]
    if not to_add:
        return []

    backup = os.path.splitext(labels_path)[0] + "_backup.csv"
    shutil.copyfile(labels_path, backup)                      # safety copy before we change anything

    with open(labels_path, "rb") as fh:                       # make sure the file ends with a newline
        data = fh.read()
    needs_newline = len(data) > 0 and not data.endswith((b"\n", b"\r"))

    values = {"material": "unknown", "condition": "unknown", "notes": NEW_ROW_NOTE}
    with open(labels_path, "a", newline="", encoding="utf-8") as fh:
        if needs_newline:
            fh.write("\n")
        w = csv.writer(fh, lineterminator="\n")
        for f in to_add:
            row = []
            for col in (h.lower() for h in header):
                if col == "file":
                    row.append(f)
                elif col == "item_id":
                    row.append(os.path.splitext(f)[0])
                else:
                    row.append(values.get(col, ""))
            w.writerow(row)
    return to_add


def build_feature_table(df, img_dir):
    """Raw and pre-processed features for every photo listed in df.
    df needs the columns: file, item_id, material, condition, labeled, contaminated, x0, y0, x1, y1."""
    rows = []
    for _, r in df.iterrows():
        im = load_image(os.path.join(img_dir, r["file"]))
        raw_f = compute_features(raw_version(im))
        pre_f = compute_features(preprocess(im, box_of(r)))
        row = {"file": r["file"], "item_id": r["item_id"], "material": r["material"],
               "condition": r["condition"], "labeled": r["labeled"], "contaminated": r["contaminated"]}
        row.update({"raw_" + k: v for k, v in raw_f.items()})
        row.update({"pre_" + k: v for k, v in pre_f.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def score_new_photos(df, args):
    """--new / --only mode: print the features of just the new photos, not the whole dataset."""
    if args.only:
        files = args.only
    else:
        on_disk = sorted(f for f in os.listdir(args.images)
                         if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS)
        listed = set(df["file"])
        files = [f for f in on_disk if f not in listed]
    if not files:
        sys.exit("No new photos found: every photo in the images folder is already listed in labels.csv.")
    not_found = [f for f in files if not os.path.exists(os.path.join(args.images, f))]
    if not_found:
        sys.exit("These photos were not found in the images folder:\n  " + "\n  ".join(not_found))

    # crop box: --box if given, else the box in labels.csv if the photo is listed there, else the whole photo
    rows = []
    for f in files:
        known = df[df["file"] == f]
        if args.box:
            box = tuple(args.box)
        elif len(known):
            box = box_of(known.iloc[0])
        else:
            box = (0.0, 0.0, 1.0, 1.0)
        rows.append({"file": f, "item_id": f, "material": "", "condition": "new", "labeled": False,
                     "contaminated": np.nan, "x0": box[0], "y0": box[1], "x1": box[2], "y1": box[3]})
    new = pd.DataFrame(rows)
    new_feat = build_feature_table(new, args.images)
    show_cols = ["file"] + ["raw_" + f for f in FEATURES] + ["pre_" + f for f in FEATURES]
    print("\nNew photos (features in % of pixels; 'raw' = whole photo, 'pre' = cropped + brightness-normalised):")
    print(new_feat[show_cols].round(1).to_string(index=False))
    if not args.box and not any(len(df[df["file"] == f]) for f in files):
        print("\n(No crop box given, so the whole photo was used. Add --box x0 y0 x1 y1 to crop to the item, "
              "for example: --box 0.2 0.1 0.8 0.9)")

    # reference: average pre-processed features of the labelled photos, for comparison only
    ref = build_feature_table(df, args.images)
    ref = ref[ref["labeled"]]
    lines = {}
    for value, name in ((0.0, "clean"), (1.0, "contaminated")):
        sub = ref[ref["contaminated"] == value]
        if len(sub):
            lines[f"{name} photos (n={len(sub)})"] = {f: sub["pre_" + f].mean() for f in FEATURES}
    if lines:
        print("\nFor comparison, average pre-processed features of the labelled photos in labels.csv:")
        print(pd.DataFrame(lines).T.round(1).to_string())
    print("\nThese numbers describe the photo. They are not a prediction of clean or dirty, and with this few "
          "labelled photos the averages above are only a rough guide.")
    print("To include these photos in the full analysis, add one row per photo to labels.csv "
          "(file, item_id, material, condition, crop box).")

    os.makedirs(args.out, exist_ok=True)
    new_feat.to_csv(os.path.join(args.out, "new_photos_features.csv"), index=False)
    fig_raw_vs_pre(new.head(6), args.images, args.out, name="new_photos_preview.png")
    print(f"\nSaved new_photos_features.csv and new_photos_preview.png in: {os.path.abspath(args.out)}")


def main():
    ap = argparse.ArgumentParser(description="SmartCycle pre-processing and feature analysis")
    ap.add_argument("--images", required=True, help="folder that contains the photos")
    ap.add_argument("--labels", required=True, help="path to labels.csv")
    ap.add_argument("--out", default="results", help="folder for the outputs (created if missing)")
    ap.add_argument("--show", nargs="*", default=None,
                    help="photo file names to use in the raw-vs-pre-processed and histogram figures "
                         "(default: first photo of each of up to 4 different items)")
    ap.add_argument("--check-boxes", action="store_true",
                    help="only draw the crop boxes on the photos (boxes_check.png) and stop")
    ap.add_argument("--no-sync", action="store_true",
                    help="do not add new photos to labels.csv automatically")
    ap.add_argument("--new", action="store_true",
                    help="only process photos in the images folder that are NOT listed in labels.csv, "
                         "and print their features")
    ap.add_argument("--only", nargs="+", default=None, metavar="FILE",
                    help="only process these photo file names (listed in labels.csv or not)")
    ap.add_argument("--box", nargs=4, type=float, default=None, metavar=("X0", "Y0", "X1", "Y1"),
                    help="crop box for --new / --only photos, as fractions 0-1: left top right bottom")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- 0) add any new photos in the images folder to labels.csv --------------------
    added = []
    if not (args.new or args.only or args.no_sync):
        added = sync_labels(args.labels, args.images)
        if added:
            print(f"\nAdded {len(added)} new photo(s) to {args.labels}: {', '.join(added)}")
            print("  (a copy of the old file was saved as the _backup.csv next to it)")

    df = read_labels(args.labels)

    missing = [f for f in df["file"] if not os.path.exists(os.path.join(args.images, f))]
    if missing:
        sys.exit("These photos from labels.csv were not found in the images folder:\n  " + "\n  ".join(missing))

    if args.check_boxes:
        fig_check_boxes(df, args.images, args.out)
        print(f"Saved {os.path.join(args.out, 'boxes_check.png')}. Check the red boxes, then run without --check-boxes.")
        return

    if args.new or args.only:
        score_new_photos(df, args)
        return

    # ---- 1) compute the features for every photo ------------------------------------
    feat = build_feature_table(df, args.images)
    feat.to_csv(os.path.join(args.out, "features.csv"), index=False)
    print("\nFeatures (% of pixels):")
    print(feat.round(1).to_string(index=False))

    # ---- 2) statistics --------------------------------------------------------------
    lab = feat[feat["labeled"]].reset_index(drop=True)      # only photos whose condition is known
    lab["contaminated"] = lab["contaminated"].astype(int)
    if len(lab) < 3:
        sys.exit("Fewer than 3 photos have a known condition, so there is nothing to analyse yet.\n"
                 "Open labels.csv, fill in material, condition and the crop box for your photos, then run again.")
    y_cond = lab["contaminated"].to_numpy()
    y_mat = lab["material"].to_numpy()
    stat_rows = []
    for f in FEATURES:
        raw, pre = lab["raw_" + f], lab["pre_" + f]
        stat_rows.append({
            "feature": f,
            "corr_condition_raw": safe_corr(raw, y_cond),
            "corr_condition_pre": safe_corr(pre, y_cond),
            "MI_condition_raw_bits": mutual_info_median_split(raw, y_cond),
            "MI_condition_pre_bits": mutual_info_median_split(pre, y_cond),
            "MI_material_raw_bits": mutual_info_median_split(raw, y_mat),
            "MI_material_pre_bits": mutual_info_median_split(pre, y_mat),
        })
    stats = pd.DataFrame(stat_rows)
    stats.to_csv(os.path.join(args.out, "stats.csv"), index=False)
    print("\nCorrelation with condition (0 = clean, 1 = contaminated) and mutual information (bits):")
    print(stats.round(2).to_string(index=False))
    print(f"\nMost any feature could tell us: condition {entropy(y_cond):.2f} bits, material {entropy(y_mat):.2f} bits")
    print("\nFeature-feature correlation (pre-processed):")
    print(lab[["pre_" + f for f in FEATURES]].corr().round(2).to_string())

    n_items = lab["item_id"].nunique()
    n_unknown = len(feat) - len(lab)
    print(f"\nNote: {len(lab)} photos with a known condition from {n_items} physical items, "
          f"{int(y_cond.sum())} contaminated photos"
          + (f" ({n_unknown} more photos are marked unknown and were skipped)" if n_unknown else "") + ". "
          "With so few photos these numbers are only a demonstration. "
          "For a real result, use at least a few hundred photos and split by item_id.")

    # ---- 3) figures -----------------------------------------------------------------
    if args.show:
        show = df[df["file"].isin(args.show)]
    else:
        show = df.drop_duplicates("item_id").head(4)
    fig_dataset(df, args.images, args.out)
    fig_raw_vs_pre(show, args.images, args.out)
    fig_histograms(show, args.images, args.out)
    fig_features_vs_outputs(lab, args.out)
    print(f"\nSaved features.csv, stats.csv and 4 figures in: {os.path.abspath(args.out)}")
    if added:
        print(f"\nTO DO: open {args.labels} and fill in material, condition and the crop box (x0,y0,x1,y1) for: "
              + ", ".join(added))
        print("Until the condition is filled in, those photos are left out of the statistics and plots. "
              "Then run the same command again to update the results.")
    still_unknown_material = lab[lab["material"] == "unknown"]
    if len(still_unknown_material):
        print("\nWarning: these photos have a condition but material is still 'unknown': "
              + ", ".join(still_unknown_material["file"]) + ". Fill in the material for correct material results.")


if __name__ == "__main__":
    main()
