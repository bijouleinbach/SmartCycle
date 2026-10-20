"""Evaluate the model and compare SmartCycle against both baselines.

Writes to reports/:
  metrics.json                        all numbers below
  confusion_<axis>.png                confusion matrix per label axis
  accuracy_by_source.png              accuracy per person/dataset (variability across phones/lighting)

Systems compared (from the proposal):
  Baseline 1   material only -> bin (ignores condition and parts)
  Baseline 2   material + condition + parts -> rules, one pass, no re-scan
  SmartCycle   same as Baseline 2, plus verifying the "after" photo before the final bin

Decision metrics need images labeled on all three axes, and the re-scan
metrics need before/after pairs (same pair_id). TrashNet has neither, so those
sections stay empty until your own labeled photos are added.

Run: python -m src.eval              (uses the same test split as training)
     python -m src.eval --holdout-source bijou
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix  # noqa: E402

from src import rules  # noqa: E402
from src.data import load_index, open_image, split  # noqa: E402
from src.labels import AXES  # noqa: E402
from src.model import Predictor  # noqa: E402

REPORTS = Path(__file__).resolve().parent.parent / "reports"


def predict_all(predictor, df):
    preds, seconds = [], []
    for path in df["path"]:
        start = time.perf_counter()
        p = predictor.predict(open_image(path))
        seconds.append(time.perf_counter() - start)
        preds.append({axis: p[axis]["label"] for axis in AXES})
    return preds, seconds


def action(decision: rules.Decision) -> str:
    """Comparable summary of a decision: the bin, or the set of prep steps."""
    return f"bin:{decision.bin}" if decision.ready else "prep:" + "|".join(sorted(decision.steps))


def decision_metrics(rows):
    """rows: list of (true_decision, predicted_decision)."""
    if not rows:
        return None
    n = len(rows)
    dirty = [(t, p) for t, p in rows if not t.ready]
    clean = [(t, p) for t, p in rows if t.ready]
    return {
        "n": n,
        "action_accuracy": sum(action(t) == action(p) for t, p in rows) / n,
        "problem_identification": sum(t.ready == p.ready for t, p in rows) / n,
        # dirty item judged ready: contamination reaches the bin
        "missed_contamination_rate": (sum(p.ready for _, p in dirty) / len(dirty)) if dirty else None,
        # clean item sent back for prep: annoys the user
        "unnecessary_rejection_rate": (sum(not p.ready for _, p in clean) / len(clean)) if clean else None,
    }


def baseline1(labels):
    material = labels["material"]
    return rules.Decision(ready=True, bin=rules.CLEAN_BIN[material], reason="Baseline 1: material only")


def full(labels):
    return rules.decide(labels["material"], labels["condition"], labels["component"])


def evaluate(holdout_source=None, seed=0):
    REPORTS.mkdir(exist_ok=True)
    predictor = Predictor()
    test = split(load_index(), seed=seed, holdout_source=holdout_source)["test"]
    preds, seconds = predict_all(predictor, test)
    out = {"n_test": len(test), "trained_axes": predictor.trained_axes,
           "mean_inference_seconds": sum(seconds) / len(seconds), "axes": {}, "by_source": {}}

    # Per-axis accuracy + confusion matrices, only on images labeled for that axis.
    for axis, classes in AXES.items():
        idx = [i for i, v in enumerate(test[axis]) if v]
        if not idx:
            continue
        y_true = [test[axis][i] for i in idx]
        y_pred = [preds[i][axis] for i in idx]
        present = [c for c in classes if c in y_true or c in y_pred]
        out["axes"][axis] = {
            "n": len(idx),
            "accuracy": sum(a == b for a, b in zip(y_true, y_pred)) / len(idx),
            "trained": axis in predictor.trained_axes,
        }
        cm = confusion_matrix(y_true, y_pred, labels=present)
        fig, ax = plt.subplots(figsize=(1.2 * len(present) + 2, 1.2 * len(present) + 1.5))
        ConfusionMatrixDisplay(cm, display_labels=present).plot(ax=ax, cmap="Greens", colorbar=False)
        ax.set_title(f"{axis} (n={len(idx)})")
        plt.xticks(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(REPORTS / f"confusion_{axis}.png", dpi=150)
        plt.close(fig)

        for source in sorted(test["source"].unique()):
            s_idx = [i for i in idx if test["source"][i] == source]
            if s_idx:
                acc = sum(test[axis][i] == preds[i][axis] for i in s_idx) / len(s_idx)
                out["by_source"].setdefault(source, {})[axis] = {"n": len(s_idx), "accuracy": acc}

    _plot_by_source(out["by_source"])

    # Single-photo decisions: only images labeled on every axis.
    complete = [i for i in range(len(test)) if all(test[a][i] for a in AXES)]
    truth = {i: full({a: test[a][i] for a in AXES}) for i in complete}
    out["single_scan"] = {
        "baseline1": decision_metrics([(truth[i], baseline1(preds[i])) for i in complete]),
        "baseline2_and_smartcycle": decision_metrics([(truth[i], full(preds[i])) for i in complete]),
    }

    # Re-scan loop: before/after pairs. The correct final answer depends on the
    # *after* photo's true labels (the user may not have prepared it properly).
    # Baseline 2 trusts the user did the prep; SmartCycle checks the after photo.
    pairs = []
    by_pair = {}
    for i in complete:
        if test["pair_id"][i] and test["stage"][i] in ("before", "after"):
            by_pair.setdefault((test["source"][i], test["pair_id"][i]), {})[test["stage"][i]] = i
    for p in by_pair.values():
        if "before" in p and "after" in p:
            pairs.append((p["before"], p["after"]))
    b2, sc = [], []
    for before, after in pairs:
        t = truth[after]
        first = full(preds[before])
        if first.ready:  # both systems stop after one scan
            b2.append((t, first))
            sc.append((t, first))
        else:
            assumed = rules.Decision(ready=True, bin=rules.CLEAN_BIN[preds[before]["material"]], reason="assumed prepped")
            b2.append((t, assumed))
            sc.append((t, full(preds[after])))
    out["rescan_pairs"] = {"n_pairs": len(pairs), "baseline2": decision_metrics(b2), "smartcycle": decision_metrics(sc)}

    (REPORTS / "metrics.json").write_text(json.dumps(out, indent=2))
    _print(out)
    return out


def _plot_by_source(by_source):
    if not by_source:
        return
    axes = sorted({a for v in by_source.values() for a in v})
    sources = sorted(by_source)
    fig, ax = plt.subplots(figsize=(max(4, 1.6 * len(sources) * len(axes)), 3.5))
    width = 0.8 / len(axes)
    for k, axis in enumerate(axes):
        xs = [i + k * width for i in range(len(sources))]
        ys = [by_source[s].get(axis, {}).get("accuracy", 0) for s in sources]
        ax.bar(xs, ys, width, label=axis)
    ax.set_xticks([i + width * (len(axes) - 1) / 2 for i in range(len(sources))], sources)
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by data source")
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPORTS / "accuracy_by_source.png", dpi=150)
    plt.close(fig)


def _print(out):
    print(f"test images: {out['n_test']}   mean inference: {out['mean_inference_seconds'] * 1000:.0f} ms")
    for axis, m in out["axes"].items():
        flag = "" if m["trained"] else "  (head not trained)"
        print(f"  {axis:10s} accuracy {m['accuracy']:.3f}  n={m['n']}{flag}")
    untrained = [a for a in AXES if a not in out["trained_axes"]]
    if untrained and out["single_scan"]["baseline1"]:
        print(f"WARNING: {untrained} heads are untrained (random guesses), so the decision metrics below aren't meaningful yet.")
    for name, section in [("single scan", out["single_scan"]), ("re-scan pairs", out["rescan_pairs"])]:
        print(f"{name}:")
        for system, m in section.items():
            if system == "n_pairs":
                continue
            print(f"  {system:26s} " + (json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in m.items()}) if m else "n/a (needs fully labeled photos)"))
    print(f"saved to {REPORTS}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--holdout-source")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    evaluate(args.holdout_source, args.seed)
