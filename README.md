# SmartCycle
AI-powered recycling contamination detection and disposal guidance system.

Take a photo of an item and SmartCycle tells you which bin it goes in, or what
to do first (empty it, rinse it, take the lid off). After you prepare the item,
re-scan it and SmartCycle checks that it's actually ready before giving the
final bin:

```
Capture image ──▶ AI model + rules ──▶ Ready? ──yes──▶ Final bin
   ▲                                     │
   │                                     no
   │                                     ▼
   └────── re-scan to verify ◀── Prep instruction
```

See [`docs/ITERATION_PLAN.md`](docs/ITERATION_PLAN.md) for the full project plan.

## Quick start

Requires Python 3.11+ (tested on 3.13, macOS).

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m scripts.download_trashnet   # 2,390 public photos (~43 MB) -> data/trashnet/
python -m src.model                   # train (~5 min on an M-series Mac, longer on CPU)
python -m app.server                  # start the web app
```

Then open **http://localhost:8000**.

- **On a laptop:** scan the QR code with your phone's camera app. Your phone opens the same scan session, so the photo you take on the phone shows up on the laptop. You can also upload a photo from the computer instead.
- **On a phone:** open the address the server prints (e.g. `http://10.0.0.5:8000`) and tap **Take photo**.

Phone and laptop must be on the **same Wi-Fi** while the server runs on a laptop. Some campus networks block devices from reaching each other. If the QR code doesn't load on your phone, try a phone hotspot.

## What's here

| Path | What it does | Owner (plan) |
| --- | --- | --- |
| `src/labels.py` | Label schema: material / condition / removable component | Bijou |
| `src/data.py` | Loads every `data/<source>/labeled/labels.csv`, splits train/val/test (keeps before/after pairs together, can hold out one person) | Aparajita |
| `src/model.py` | **AI model**: MobileNetV2 (ImageNet-pretrained) + one head per label axis. Training CLI + `Predictor` | Bijou |
| `src/rules.py` | **Rule engine** (no AI): labels → bin, or prep steps. **Rules are placeholders**, see below | Tian |
| `src/pipeline.py` | The detect → prep → re-scan → verify loop (`Session`), plus a terminal version | Tian |
| `src/eval.py` | Accuracy, confusion matrices, accuracy by person, Baseline 1 vs Baseline 2 vs SmartCycle | Aparajita |
| `app/` | Web app (FastAPI + one HTML page): QR hand-off from laptop to phone, upload, live results | — |
| `scripts/download_trashnet.py` | Fetches TrashNet into our CSV format | — |
| `scripts/add_photos.py` | Shrinks your raw phone photos and creates a label spreadsheet for them | — |
| `tests/` | `pytest` tests for the rules, split, loop and web app (no model or data needed) | — |

The AI model and the rule engine are deliberately separate modules, so the report can describe and justify the model on its own (Assignment 3), and changing recycling rules never requires retraining.

## Current status (MVP)

- **Material:** trained on TrashNet (plastic, paper/cardboard, glass, metal). TrashNet has no food/organic photos, so **organic** isn't learned yet.
- **Condition and removable component:** not trained yet, because no public dataset labels them. Until our own photos exist, the app asks the user for these two instead of guessing. It also asks whenever the model is under 60% confident. The whole loop still works end to end.
- **Recycling rules:** generic US-curbside placeholders (`RULES_SOURCE` in `src/rules.py`). Replace them with one real local program (task 0.2).

## Adding our own photos

1. Put your photos in `data/<your name>/raw/`. That folder is ignored by git, because originals are too big.
2. Run `python -m scripts.add_photos <your name>`. It writes resized copies plus `labels.csv` to `data/<your name>/labeled/`.
3. Fill in `labels.csv` in Excel/Sheets/Numbers. Allowed values (from `src/labels.py`):

| Column | Values |
| --- | --- |
| `material` | plastic, paper_cardboard, glass, metal, organic |
| `condition` | clean, food_residue, grease, liquid, mixed_materials |
| `component` | none, lid, sleeve_label, straw |
| `stage` | before (as found), after (after preparing it) |
| `pair_id` | same value for the before and after photo of one item, e.g. `yogurt1` |

Leave a cell blank if you're unsure; that photo is just skipped for that column.

4. Retrain with `python -m src.model`. Condition/component heads switch on automatically once there are at least 30 labeled examples (`MIN_LABELED`), and the app stops asking for them.

Before/after pairs are what let us evaluate the re-scan step, which is the part that distinguishes SmartCycle from the baselines. Shoot both whenever you can.

## Evaluation

```bash
python -m src.eval                          # test split shared with training
python -m src.eval --holdout-source bijou   # test on one person's photos only
python -m src.model --holdout-source bijou  # ...after training without them
```

Writes `reports/metrics.json`, `reports/confusion_<axis>.png` and `reports/accuracy_by_source.png`. Decision metrics (missed contamination, unnecessary rejection, Baseline 1/2 vs SmartCycle) need photos labeled on all three axes, and the re-scan comparison needs before/after pairs, so those show `n/a` until our own data is in.

## Other ways to run it

```bash
python -m src.pipeline before.jpg after.jpg   # the loop in the terminal; extra photos are re-scans
pytest                                        # run tests
PORT=9000 python -m app.server                # different port
PUBLIC_URL=https://your-host python -m app.server   # when deployed online, so the QR code uses that address
```

## Data credits
- TrashNet: G. Thung and M. Yang, *Classification of Trash for Recyclability Status*, 2016. https://github.com/garythung/trashnet
