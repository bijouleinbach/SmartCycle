"""AI model: transfer-learning image classifier with one output head per label axis.

A MobileNetV2 pretrained on ImageNet extracts image features, and three small
linear heads predict material, condition and removable component. Images
missing a label on some axis (e.g. TrashNet has no condition labels) simply
don't contribute to that head's loss.

Train:    python -m src.model --epochs 8
Options:  --holdout-source bijou   test on one person's photos only
"""

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from src.data import SmartCycleDataset, eval_transform, load_index, split, train_transform
from src.labels import AXES

CHECKPOINT = Path(__file__).resolve().parent.parent / "checkpoints" / "smartcycle.pt"

# An axis needs at least this many labeled training images before its head is
# trusted. Below that, predictions are marked untrained and the app asks the user.
MIN_LABELED = 30


class MultiHeadClassifier(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v2(weights=weights)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        dim = backbone.last_channel
        self.dropout = nn.Dropout(0.2)
        self.heads = nn.ModuleDict({axis: nn.Linear(dim, len(c)) for axis, c in AXES.items()})

    def forward(self, x):
        z = self.dropout(torch.flatten(self.pool(self.features(x)), 1))
        return {axis: head(z) for axis, head in self.heads.items()}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon Macs
        return torch.device("mps")
    return torch.device("cpu")


def masked_loss(logits, targets):
    """Sum of per-axis cross-entropy, skipping images with no label (-1) on that axis."""
    total, loss_fn = 0.0, nn.CrossEntropyLoss(ignore_index=-1)
    for axis, out in logits.items():
        t = targets[axis]
        if (t >= 0).any():
            total = total + loss_fn(out, t)
    return total


@torch.no_grad()
def accuracy(model, loader, device):
    model.eval()
    correct = {a: 0 for a in AXES}
    count = {a: 0 for a in AXES}
    for images, targets in loader:
        logits = model(images.to(device))
        for axis in AXES:
            t = targets[axis].to(device)
            mask = t >= 0
            correct[axis] += (logits[axis].argmax(1)[mask] == t[mask]).sum().item()
            count[axis] += mask.sum().item()
    return {a: (correct[a] / count[a] if count[a] else None) for a in AXES}


def train(epochs=8, batch_size=32, lr=1e-3, holdout_source=None, seed=0, num_workers=2):
    torch.manual_seed(seed)
    device = pick_device()
    parts = split(load_index(), seed=seed, holdout_source=holdout_source)
    train_df = parts["train"]
    trained_axes = [
        a for a in AXES
        if (train_df[f"{a}_idx"] >= 0).sum() >= MIN_LABELED and train_df.loc[train_df[f"{a}_idx"] >= 0, f"{a}_idx"].nunique() >= 2
    ]
    print(f"device={device}  train={len(train_df)}  val={len(parts['val'])}  test={len(parts['test'])}")
    print(f"axes with enough labels to train: {trained_axes}")
    if not trained_axes:
        raise SystemExit("Not enough labeled data on any axis to train.")

    loaders = {
        name: DataLoader(
            SmartCycleDataset(df, train_transform() if name == "train" else eval_transform()),
            batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers,
        )
        for name, df in parts.items()
    }

    model = MultiHeadClassifier().to(device)
    # The pretrained backbone learns slower than the new heads so it keeps its ImageNet features.
    optimizer = torch.optim.AdamW([
        {"params": model.features.parameters(), "lr": lr * 0.1},
        {"params": model.heads.parameters(), "lr": lr},
    ])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val, history = -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        start, running, n = time.time(), 0.0, 0
        for images, targets in loaders["train"]:
            images = images.to(device)
            targets = {a: t.to(device) for a, t in targets.items()}
            loss = masked_loss(model(images), targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * len(images)
            n += len(images)
        scheduler.step()
        val_acc = accuracy(model, loaders["val"], device)
        mean_val = _mean([val_acc[a] for a in trained_axes])
        history.append({"epoch": epoch, "train_loss": running / n, "val_acc": val_acc})
        print(f"epoch {epoch}/{epochs}  loss {running / n:.3f}  val {_fmt(val_acc)}  ({time.time() - start:.0f}s)")
        if mean_val > best_val:
            best_val = mean_val
            save(model, trained_axes, history, holdout_source)

    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["state_dict"])
    test_acc = accuracy(model, loaders["test"], device)
    print(f"test {_fmt(test_acc)}  (best checkpoint saved to {CHECKPOINT})")
    return test_acc


def save(model, trained_axes, history, holdout_source):
    CHECKPOINT.parent.mkdir(exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "classes": AXES,
        "trained_axes": trained_axes,
        "history": history,
        "holdout_source": holdout_source,
    }, CHECKPOINT)


class Predictor:
    """Loads the checkpoint once and predicts labels for a single PIL image."""

    def __init__(self, checkpoint: Path = CHECKPOINT):
        self.device = pick_device()
        ckpt = torch.load(checkpoint, map_location=self.device)
        if ckpt["classes"] != AXES:
            raise ValueError("Checkpoint was trained with a different label schema; retrain it.")
        self.trained_axes = ckpt["trained_axes"]
        self.model = MultiHeadClassifier(pretrained=False).to(self.device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.transform = eval_transform()

    @torch.no_grad()
    def predict(self, image):
        x = self.transform(image).unsqueeze(0).to(self.device)
        logits = self.model(x)
        result = {}
        for axis, classes in AXES.items():
            probs = torch.softmax(logits[axis][0], 0).cpu().tolist()
            best = max(range(len(classes)), key=probs.__getitem__)
            result[axis] = {
                "label": classes[best],
                "confidence": probs[best],
                "probs": dict(zip(classes, probs)),
                "trained": axis in self.trained_axes,
            }
        return result


def load_predictor():
    """Return a Predictor, or None if no model has been trained yet."""
    return Predictor() if CHECKPOINT.exists() else None


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _fmt(acc):
    return "  ".join(f"{a}={v:.3f}" for a, v in acc.items() if v is not None) or "n/a"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train the SmartCycle classifier.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--holdout-source", help="use this data/<source> folder as the test set")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    test_acc = train(args.epochs, args.batch_size, args.lr, args.holdout_source, args.seed)
    print(json.dumps(test_acc))
