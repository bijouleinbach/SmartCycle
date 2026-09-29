# SmartCycle Iteration Plan

Sep 29, 2026 · prepared by @Bijou Leinbach

> Live version with interactive diagrams: https://claude.ai/code/artifact/c6ea511e-371e-4a1a-ade6-a02570577f60

## Project recap

**SmartCycle** is the closed-loop recycling-readiness system you're building with Tian and Aparajita for SE 498 (*AI-Driven Intelligent Systems*, Prof. Yiwen Dong). In this course's framing, **modeling/AI is the required core** — sensing and actuation are the "bonus" layer — so grading weight favors a rigorous model and evaluation over a polished app.

Revised pipeline (after proposal-presentation feedback): a camera image goes through **material + visible-condition + removable-component recognition**, then a **rule engine** checks the result against local recycling rules. If the item is ready, the system returns a final bin + reason. If not, it returns a preparation instruction (empty / rinse / separate), the user re-submits an image, and the system **verifies readiness** before issuing the final recommendation.

Your proposed contribution is *not* waste classification alone — Longo et al., Lim & Lee, WasteAssistant, and Oscar Sort already cover pieces of that. The gap you identified is a **transparent, evaluated detect → correct → verify loop**, measured against the two baselines your team already defined:

- **Baseline 1** — item recognition → final bin
- **Baseline 2** — item + condition + rules → one-pass instruction or bin
- **SmartCycle** — detect → recommend preparation → re-scan/confirm → verify readiness → final bin

## Deliverable map

| Deliverable | Status | Group part | Individual part |
| --- | --- | --- | --- |
| Proposal (1 pg) | Done | Motivation, goal, lit review, approach, novelty | — |
| Proposal presentation | Done | 5-component talk | — |
| Assignment 1 | Not started | Updated proposal incorporating feedback | Sensing & data-acquisition writeup per teammate: setup diagram, sample raw image, camera "sensor" specs, noise sources, physical process |
| Assignment 2 | Not started | Pre-processing plan & candidate features | Feature-output analysis per teammate: plots, trends, correlation/mutual information |
| Assignment 3 | Not started | System diagram, AI/ML method + justification | Evaluation per teammate: decision boundaries/confusion matrix, performance by condition, train/test error |
| Final presentation (15 min) | Not started | All 7 required components | — |
| Final report (6 pg, ACM format) | Not started | Full paper | Contribution statement (individual roles) |

**Why this matters for the plan:** Part II of Assignments 1–3 is done *individually* — each of you needs your own sensing setup and dataset slice, even though you share one system. The natural move is for each teammate to photograph items under their own conditions (different phone, lighting, background). That satisfies the individual requirement **and** gives you the cross-person performance-variability data Assignment 3 explicitly asks for.

## System architecture

```
Capture image ──▶ AI + rules ──▶ Ready? ──yes──▶ Final bin
   ▲                                 │
   │                                 no
   │                                 ▼
   └────── user re-scans ◀── Prep instruction
```

Each stage maps to a graded component:

1. **Capture image** — phone/camera, initial scan or re-scan.
2. **AI + rules** — the AI model predicts material, visible condition, and removable components; a separate deterministic rule engine looks up the applicable local recycling rule. Keep these two as distinct modules in your code and your report — the rubric wants the *AI model* described and justified on its own (architecture, training, assumptions), separate from rule logic.
3. **Ready?** — decision point.
4. **Final bin** — terminal state: bin + plain-language reason.
5. **Prep instruction** — empty/rinse/separate; the user re-submits an image and the loop returns to step 1 for verification.

### What "sensing" means here (individual, Assignment 1 Part II)

Your "sensor" is a phone/webcam camera, so translate the assignment's sensor-characterization questions into vision terms — each teammate answers these for their **own** camera/setup:

- **Setup diagram + sample raw data:** a photo of your capture setup (distance, angle, background) and 2–3 sample images.
- **Sensitivity, bandwidth, range** → camera resolution, dynamic range in low light, field of view, minimum/maximum working distance.
- **Sampling frequency** → you're single-shot, not continuous — justify why one frame per decision point is sufficient (vs. video).
- **Noise** → motion blur, glare off wet/plastic surfaces, color cast under indoor lighting, shadow occlusion of contamination.
- **Physical process** → how grease/liquid/food residue changes an item's visual appearance, and what lighting/background variation does to that signal.

Using three different phones/lighting setups across the team isn't a compromise here — it's useful variation for the cross-condition evaluation Assignment 3 asks for.

## Data & labeling plan

**Label schema** (multi-label per image):

| Axis | Values |
| --- | --- |
| Material | plastic, cardboard/paper, glass, metal, food waste/organic |
| Condition | clean, food residue, grease, liquid present, mixed materials |
| Removable component | none, lid, sleeve/label, straw |

**Per-teammate target:** ≥150 images across categories, varied lighting/background, including:

- "Before" shots (contaminated, as found)
- "After" shots (post-preparation, e.g. rinsed/emptied) — these are what makes the re-scan/verification step trainable and testable, not just the initial classification.

**Labeling process:** agree on the schema above as a group *before* anyone starts shooting, keep a shared spreadsheet or CSV mapping filename → labels, and spot-check a sample of each other's labels for consistency (this becomes your inter-labeler reliability note if a TA asks about label quality).

## Model & evaluation plan

**Model:** transfer-learning image classifier (e.g., MobileNetV2, ResNet18, or EfficientNet-B0 fine-tuned in PyTorch/Keras) predicting material + condition + component labels. Keep the **rule engine as a separate, deterministic module** (a lookup table: material × condition × local rule → action) — this split makes the required "AI model" story clean for Assignment 3 (architecture, training/testing, assumptions) without conflating it with hand-written logic.

Consider pretraining or augmenting with a public dataset (e.g., TrashNet, TACO) before fine-tuning on your own photos — three people's phone photos alone will be a small, imbalanced dataset.

**Baselines to report against** (already defined in your proposal):

| System | Pipeline |
| --- | --- |
| Baseline 1 | item recognition → final bin |
| Baseline 2 | item + condition + rules → one-pass instruction/bin |
| SmartCycle | detect → recommend prep → re-scan/confirm → verify → final bin |

**Metrics** (map directly onto Assignment 3 Part II):

- Confusion matrix per label axis (material, condition, component)
- Correctness of: problem identification, preparation instruction, post-prep verification, final recommendation
- Successful user-correction rate; missed-contamination rate; unnecessary-rejection rate
- Accuracy broken out **by teammate/lighting/phone** — this is your "performance variability across factors" plot, and it falls out naturally from each person collecting their own data
- End-to-end response time

## 4-week schedule

Goal: ship every deliverable about a week early to leave room for feedback.

| Week | Group work | Individual work (each teammate) | Owner notes |
| --- | --- | --- | --- |
| 1 | Repo scaffold, labeling schema agreed, rule-engine skeleton | Start photographing own item set; draft sensing setup writeup | Everyone creates their own branch off `main` (yours, `bijou`, is already pushed) |
| 2 | Finalize features to extract; integrate first model prototype | Finish own dataset + labels; feature-output plots; **submit Assignment 1** | Merge to `main` via PR before submitting |
| 3 | Train SmartCycle model + both baselines; wire up rule engine end-to-end | Run own evaluation slice (confusion matrix, per-condition accuracy); **draft Assignments 2 & 3** | Everyone's data feeds the cross-person variability analysis |
| 4 (buffer) | Incorporate any professor/TA feedback from earlier deliverables; write final report; rehearse final presentation | Polish own individual write-ups | This week only exists because you're front-loading — protect it, don't let Weeks 1–3 slip into it |

## Git workflow

- `main` stays protected and buildable. Your branch (`bijou`) is already pushed — Tian and Aparajita should each create their own (`tian`, `aparajita`) off `main`.
- Merge into `main` via pull request at each weekly milestone (end of Week 1, 2, 3), with at least one teammate reviewing before merging.
- Suggested folder structure:

```
SmartCycle/
  data/
    tian/raw  tian/labeled
    bijou/raw  bijou/labeled
    aparajita/raw  aparajita/labeled
  src/
    model.py      # AI classifier: train/eval
    rules.py       # deterministic rule engine
    pipeline.py    # ties capture -> model -> rules -> decision together
  assignments/
    assignment1/ {tian,bijou,aparajita}.md
    assignment2/ ...
    assignment3/ ...
  report/          # ACM LaTeX final report
  README.md
```

- Keep the README current with setup + reproduction steps as you go, not at the end — the rubric explicitly deducts points if the GitHub repo can't reproduce the project, and this is also the version a professor would look at if she considers publishing it.

## Risks & what "publish-ready" means here

- **Small, imbalanced dataset.** Three people's phone photos won't cover every material/condition/lighting combination. Mitigate by pretraining or augmenting with a public dataset (TrashNet, TACO) and using your own photos for fine-tuning + evaluation.
- **Inconsistent labels across teammates.** Agree on the labeling schema before anyone shoots, and spot-check each other's labels early — catching this in Week 1 is cheap; catching it in Week 3 is not.
- **Time sunk into app polish instead of rigor.** This course grades and rewards quantitative rigor (confusion matrices, honest error analysis, reproducibility) over UI polish — and that's also what would make the work worth publishing. If time gets tight, protect the evaluation work before the interface.
- **What "publish-ready" actually means:** a reproducible repo (README + requirements + data setup anyone can follow), honest reporting of failure cases and limitations (not just wins), careful citation of the four systems you're positioning against, and a report/presentation that reads as one coherent system rather than three stitched-together parts.
