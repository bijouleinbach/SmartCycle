"""Hand-crafted image features + feature analysis (Assignment 2).

Pre-processing (images are 2D signals, Lecture 4):
  1. rotate using the phone's EXIF flag
  2. resize to 256 px and center-crop 224 px      (windowing: keep the item, drop border background)
  3. gray-world color normalization                 (removes the color cast of indoor lighting)
  4. 3x3 median filter                              (suppresses sensor noise, keeps edges)

Features (Lecture 5 statistics + frequency domain, Lecture 7 domain knowledge):
  brightness_mean/std/skew/kurtosis   statistics of pixel brightness (HSV value, 0-1)
  saturation_mean                     how vivid colors are (0-1)
  colorfulness                        Hasler & Suesstrunk colorfulness metric
  warm_fraction                       share of pixels with orange/brown hues (food, sauce, grease)
  glare_fraction                      share of near-white, unsaturated pixels (specular highlights)
  edge_density                        share of pixels on a strong edge (Sobel)
  hf_energy_ratio                     share of 2D-FFT energy at high spatial frequencies (fine texture)
  sharpness                           variance of the Laplacian (low = blurry)

Run: python -m src.features              (writes reports/features/)
     python -m src.features --source bijou --target condition
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import RFE, mutual_info_classif  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from src.data import IMAGE_SIZE, load_index, open_image  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "reports" / "features"

FEATURES = {
    "brightness_mean": "Mean brightness (0-1)",
    "brightness_std": "Brightness std. dev. (0-1)",
    "brightness_skew": "Brightness skewness (unitless)",
    "brightness_kurtosis": "Brightness kurtosis (unitless)",
    "saturation_mean": "Mean saturation (0-1)",
    "colorfulness": "Colorfulness (Hasler-Süsstrunk, unitless)",
    "warm_fraction": "Warm-hue pixels (% of image)",
    "glare_fraction": "Glare pixels (% of image)",
    "edge_density": "Strong-edge pixels (% of image)",
    "hf_energy_ratio": "High-frequency energy (% of spectrum)",
    "sharpness": "Laplacian variance (sharpness, unitless)",
}


def preprocess(image: Image.Image, steps=("crop", "grayworld", "median")) -> Image.Image:
    if "crop" in steps:
        w, h = image.size
        s = 256 / min(w, h)
        image = image.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
        w, h = image.size
        left, upper = (w - IMAGE_SIZE) // 2, (h - IMAGE_SIZE) // 2
        image = image.crop((left, upper, left + IMAGE_SIZE, upper + IMAGE_SIZE))
    if "grayworld" in steps:
        a = np.asarray(image, dtype=np.float32)
        means = a.reshape(-1, 3).mean(0)
        a = np.clip(a * (means.mean() / np.maximum(means, 1e-6)), 0, 255)
        image = Image.fromarray(a.astype(np.uint8))
    if "median" in steps:
        image = image.filter(ImageFilter.MedianFilter(3))
    return image


def _conv(a, k):
    p = np.pad(a, 1, mode="edge")
    out = np.zeros_like(a)
    for dy in range(3):
        for dx in range(3):
            out += k[dy, dx] * p[dy : dy + a.shape[0], dx : dx + a.shape[1]]
    return out


SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
LAPLACE = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)


def extract(image: Image.Image) -> dict:
    """Features of one pre-processed RGB image."""
    rgb = np.asarray(image, dtype=np.float32)
    hsv = np.asarray(image.convert("HSV"), dtype=np.float32) / 255.0
    hue, sat, val = hsv[..., 0] * 360, hsv[..., 1], hsv[..., 2]
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32) / 255.0

    mu, sd = val.mean(), val.std() + 1e-8
    z = (val - mu) / sd
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    rg, yb = r - g, 0.5 * (r + g) - b
    colorfulness = np.hypot(rg.std(), yb.std()) + 0.3 * np.hypot(rg.mean(), yb.mean())

    gx, gy = _conv(gray, SOBEL_X), _conv(gray, SOBEL_X.T)
    grad = np.hypot(gx, gy)

    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(gray - gray.mean()))) ** 2
    n = gray.shape[0]
    yy, xx = np.mgrid[:n, :n] - n // 2
    radius = np.hypot(yy, xx) / (n / 2)  # 1.0 = Nyquist frequency
    hf = spectrum[radius > 0.25].sum() / max(spectrum.sum(), 1e-8)

    return {
        "brightness_mean": float(mu),
        "brightness_std": float(val.std()),
        "brightness_skew": float((z**3).mean()),
        "brightness_kurtosis": float((z**4).mean()),
        "saturation_mean": float(sat.mean()),
        "colorfulness": float(colorfulness),
        "warm_fraction": float((((hue >= 10) & (hue <= 50)) & (sat > 0.35) & (val > 0.2)).mean() * 100),
        "glare_fraction": float(((val > 0.92) & (sat < 0.12)).mean() * 100),
        "edge_density": float((grad > 0.5).mean() * 100),
        "hf_energy_ratio": float(hf * 100),
        "sharpness": float(_conv(gray, LAPLACE).var() * 1000),
    }


def build_table(df: pd.DataFrame, target: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        feats = extract(preprocess(open_image(row["path"])))
        rows.append({"path": row["path"], "source": row["source"], target: row[target], **feats})
    return pd.DataFrame(rows)


def plot_raw_vs_preprocessed(df, target, n=3):
    """Q1: raw image, pre-processed image, and their brightness histograms."""
    picks = [df[df[target] == c].iloc[0] for c in sorted(df[target].unique())[:n]]
    fig, axes = plt.subplots(n, 3, figsize=(13, 4.2 * n))
    for i, row in enumerate(picks):
        raw = open_image(row["path"])
        pre = preprocess(raw)
        for j, (im, title) in enumerate([(raw, "Raw"), (pre, "Pre-processed")]):
            ax = axes[i, j]
            ax.imshow(im)
            ax.set_title(f"{title}: {row[target]} ({Path(row['path']).name})", fontsize=13)
            ax.set_xlabel("x (pixels)", fontsize=12)
            ax.set_ylabel("y (pixels)", fontsize=12)
        ax = axes[i, 2]
        for im, label, style in [(raw, "raw", "-"), (pre, "pre-processed", "--")]:
            v = np.asarray(im.convert("HSV"))[..., 2].ravel()
            hist, edges = np.histogram(v, bins=64, range=(0, 255), density=True)
            ax.plot(edges[:-1], hist * 100, style, label=label)
        ax.set_title("Brightness distribution", fontsize=13)
        ax.set_xlabel("Pixel brightness (0-255)", fontsize=12)
        ax.set_ylabel("Share of pixels (% per bin)", fontsize=12)
        ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "q1_raw_vs_preprocessed.png", dpi=130)
    plt.close(fig)


def plot_features_vs_output(table, target, features):
    """Q2: box plot of each feature for each output class."""
    classes = sorted(table[target].unique())
    fig, axes = plt.subplots(1, len(features), figsize=(5.2 * len(features), 4.6))
    for ax, f in zip(np.atleast_1d(axes), features):
        ax.boxplot([table.loc[table[target] == c, f] for c in classes], showfliers=False)
        ax.set_xticks(range(1, len(classes) + 1), [c.replace("_", "/") for c in classes], rotation=20, fontsize=11)
        ax.set_xlabel(target.capitalize() + " (output class)", fontsize=12)
        ax.set_ylabel(FEATURES[f], fontsize=12)
        ax.set_title(f.replace("_", " "), fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "q2_features_vs_output.png", dpi=130)
    plt.close(fig)


def plot_tsne(table, target):
    x = StandardScaler().fit_transform(table[list(FEATURES)])
    emb = TSNE(n_components=2, random_state=0, init="pca", perplexity=30).fit_transform(x)
    fig, ax = plt.subplots(figsize=(7, 6))
    for c in sorted(table[target].unique()):
        m = (table[target] == c).to_numpy()
        ax.scatter(emb[m, 0], emb[m, 1], s=8, alpha=0.6, label=c.replace("_", "/"))
    ax.set_xlabel("t-SNE dimension 1 (unitless)", fontsize=12)
    ax.set_ylabel("t-SNE dimension 2 (unitless)", fontsize=12)
    ax.set_title("Hand-crafted features, t-SNE", fontsize=13)
    ax.legend(fontsize=10, markerscale=2)
    fig.tight_layout()
    fig.savefig(OUT / "tsne_handcrafted.png", dpi=130)
    plt.close(fig)


def cnn_embeddings(paths):
    """Frozen ImageNet MobileNetV2 features (no fine-tuning) for a linear probe."""
    import torch

    from src.data import eval_transform
    from src.model import MultiHeadClassifier, pick_device

    device = pick_device()
    model = MultiHeadClassifier(pretrained=True).to(device).eval()
    tf, out = eval_transform(), []
    with torch.no_grad():
        for i in range(0, len(paths), 64):
            x = torch.stack([tf(open_image(p)) for p in paths[i : i + 64]]).to(device)
            out.append(torch.flatten(model.pool(model.features(x)), 1).cpu().numpy())
    return np.concatenate(out)


def analyze(table, target, probe=True):
    x, y = table[list(FEATURES)].to_numpy(), table[target].to_numpy()
    mi = mutual_info_classif(StandardScaler().fit_transform(x), y, random_state=0)
    # correlation ratio eta: share of a feature's variance explained by the class (works for categorical outputs)
    eta = {}
    for f in FEATURES:
        groups = [table.loc[table[target] == c, f] for c in np.unique(y)]
        total = table[f].var(ddof=0) * len(table)
        between = sum(len(g) * (g.mean() - table[f].mean()) ** 2 for g in groups)
        eta[f] = float(np.sqrt(between / total)) if total else 0.0
    rf = RandomForestClassifier(n_estimators=300, random_state=0).fit(x, y)
    rfe = RFE(LogisticRegression(max_iter=2000), n_features_to_select=5).fit(StandardScaler().fit_transform(x), y)
    corr = table[list(FEATURES)].corr()

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    scores = {
        "majority_class": float((y == pd.Series(y).mode()[0]).mean()),
        "logreg_handcrafted": float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), x, y, cv=cv).mean()),
        "rf_handcrafted": float(cross_val_score(RandomForestClassifier(n_estimators=300, random_state=0), x, y, cv=cv).mean()),
    }
    if probe:
        emb = cnn_embeddings(list(table["path"]))
        scores["logreg_cnn_probe"] = float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.1)), emb, y, cv=cv).mean())

    ranking = pd.DataFrame({
        "feature": list(FEATURES),
        "mutual_information_bits": mi / np.log(2),
        "correlation_ratio_eta": [eta[f] for f in FEATURES],
        "rf_importance": rf.feature_importances_,
        "rfe_selected": rfe.support_,
    }).sort_values("mutual_information_bits", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ranking["feature"][::-1], ranking["mutual_information_bits"][::-1])
    ax.set_xlabel(f"Mutual information with {target} (bits)", fontsize=12)
    ax.set_title(f"Feature ranking by mutual information ({target})", fontsize=13)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "q4_mutual_information.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURES)), list(FEATURES), rotation=60, ha="right", fontsize=10)
    ax.set_yticks(range(len(FEATURES)), list(FEATURES), fontsize=10)
    for i in range(len(FEATURES)):
        for j in range(len(FEATURES)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="Pearson correlation (unitless)")
    ax.set_title("Feature-feature correlation (redundancy check)", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "q4_feature_correlation.png", dpi=130)
    plt.close(fig)
    return ranking, corr, scores


def main(source=None, target="material", per_class=150, probe=True):
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_index()
    if source:
        df = df[df["source"] == source]
    df = df[df[target] != ""]
    df = df.sample(frac=1, random_state=0).groupby(target).head(per_class)
    print(f"{len(df)} images, {target} classes: {df[target].value_counts().to_dict()}")

    table = build_table(df, target)
    table.to_csv(OUT / "features.csv", index=False)
    plot_raw_vs_preprocessed(df.reset_index(drop=True), target)
    ranking, corr, scores = analyze(table, target, probe)
    # Plot the 4 best-ranked features, skipping any that duplicate one already picked (|r| >= 0.8).
    top = []
    for f in ranking["feature"]:
        if all(abs(corr.at[f, t]) < 0.8 for t in top):
            top.append(f)
        if len(top) == 4:
            break
    plot_features_vs_output(table, target, top)
    plot_tsne(table, target)

    summary = {
        "n_images": len(table), "target": target, "classes": table[target].value_counts().to_dict(),
        "ranking": ranking.round(4).to_dict(orient="records"),
        "class_medians": table.groupby(target)[list(FEATURES)].median().round(3).to_dict(),
        "high_correlations": [
            (a, b, round(float(corr.at[a, b]), 2)) for i, a in enumerate(FEATURES) for b in list(FEATURES)[i + 1 :]
            if abs(corr.at[a, b]) >= 0.8
        ],
        "cv_accuracy_5fold": scores,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(ranking.to_string(index=False))
    print(json.dumps(scores, indent=2))
    print(f"saved to {OUT}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", help="only use data/<source> (e.g. your own photos)")
    p.add_argument("--target", default="material", choices=["material", "condition", "component"])
    p.add_argument("--per-class", type=int, default=150, help="max images per class (keeps classes balanced)")
    p.add_argument("--no-probe", action="store_true", help="skip the CNN linear probe")
    a = p.parse_args()
    main(a.source, a.target, a.per_class, not a.no_probe)
