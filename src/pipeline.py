"""SmartCycle loop: detect -> recommend preparation -> re-scan -> verify -> final bin.

The model (src.model) predicts labels, the rule engine (src.rules) turns them
into a decision, and a Session keeps track of attempts so a re-scan is checked
against the earlier scan.

If the model hasn't been trained on an axis yet, or isn't confident, the
session asks the user for that label instead of guessing. Those answers are
also handy for spotting where the model needs more data.

Try it from the terminal (each extra photo is treated as a re-scan):
    python -m src.pipeline before.jpg after.jpg
"""

import sys
import time
from dataclasses import asdict, dataclass, field

from src import rules
from src.data import open_image
from src.labels import AXES

CONFIDENCE_THRESHOLD = 0.6  # below this, ask the user to confirm the label
MAX_ATTEMPTS = 3  # after this many scans that still need prep, fall back to trash


@dataclass
class StepResult:
    status: str  # "need_input" | "prep" | "final"
    attempt: int
    prediction: dict  # axis -> {label, confidence, trained, ...} from the model, or {}
    labels: dict = field(default_factory=dict)  # labels the rules were applied to
    ask: list = field(default_factory=list)  # axes the user needs to answer (need_input)
    decision: dict | None = None
    verification: list = field(default_factory=list)  # messages comparing to the previous scan
    seconds: float = 0.0

    def to_dict(self):
        return asdict(self)


class Session:
    """One item going through the loop. Call scan() for each photo."""

    def __init__(self, predictor):
        self.predictor = predictor  # src.model.Predictor, or None if no model is trained yet
        self.history: list[StepResult] = []
        self._pending = None  # model prediction for the photo awaiting user answers
        self.done = False

    @property
    def attempt(self):
        return len(self.history) + 1

    def scan(self, image, answers: dict | None = None) -> StepResult:
        """Analyze a new photo. `answers` pre-fills labels the user already gave."""
        start = time.perf_counter()
        prediction = self.predictor.predict(image) if self.predictor else {}
        self._pending = prediction
        return self._resolve(prediction, answers or {}, time.perf_counter() - start)

    def answer(self, answers: dict) -> StepResult:
        """Supply the labels asked for by a need_input result for the same photo."""
        if self._pending is None:
            raise RuntimeError("Nothing is waiting for an answer.")
        return self._resolve(self._pending, answers, 0.0)

    def _resolve(self, prediction, answers, seconds):
        labels, ask = {}, []
        for axis, classes in AXES.items():
            if axis in answers:
                if answers[axis] not in classes:
                    raise ValueError(f"Unknown {axis} {answers[axis]!r}")
                labels[axis] = answers[axis]
                continue
            p = prediction.get(axis)
            if p and p["trained"] and p["confidence"] >= CONFIDENCE_THRESHOLD:
                labels[axis] = p["label"]
            else:
                ask.append(axis)
        if ask:
            return StepResult("need_input", self.attempt, prediction, labels, ask, seconds=seconds)

        self._pending = None
        decision = rules.decide(labels["material"], labels["condition"], labels["component"])
        verification = self._verify(labels, decision)

        if not decision.ready and self.attempt >= MAX_ATTEMPTS:
            decision = rules.Decision(
                ready=True,
                bin=rules.TRASH,
                reason=(f"Still not ready after {MAX_ATTEMPTS} scans. When in doubt, throw it out: "
                        "one contaminated item can spoil a whole recycling load."),
            )

        status = "final" if decision.ready else "prep"
        result = StepResult(status, self.attempt, prediction, labels, [], asdict(decision), verification, seconds)
        self.history.append(result)
        self.done = decision.ready
        return result

    def _verify(self, labels, decision):
        if not self.history:
            return []
        prev = self.history[-1]
        msgs = []
        if prev.labels["material"] != labels["material"]:
            msgs.append(f"Heads up: this looks like {labels['material']}, but the last scan looked like "
                        f"{prev.labels['material']}. Make sure it's the same item.")
        if decision.ready:
            msgs.append("Verified: the item is ready now.")
        else:
            fixed = [s for s in prev.decision["steps"] if s not in decision.steps]
            remaining = [s for s in decision.steps if s in prev.decision["steps"]]
            msgs += [f"Done: {s}" for s in fixed]
            msgs += [f"Still needed: {s}" for s in remaining]
        return msgs


def _cli(paths):
    from src.model import load_predictor

    predictor = load_predictor()
    if predictor is None:
        print("No trained model found, so you'll be asked for every label. Train with: python -m src.model")
    session = Session(predictor)
    for path in paths:
        print(f"\n=== Scan {session.attempt}: {path}")
        result = session.scan(open_image(path))
        while result.status == "need_input":
            answers = {}
            for axis in result.ask:
                guess = result.prediction.get(axis, {})
                hint = f" (model guess: {guess['label']} {guess['confidence']:.0%})" if guess else ""
                options = AXES[axis]
                choice = input(f"{axis}{hint}? {options}: ").strip()
                while choice not in options:
                    choice = input(f"Please type one of {options}: ").strip()
                answers[axis] = choice
            result = session.answer({**result.labels, **answers})
        d = result.decision
        for msg in result.verification:
            print("  " + msg)
        if result.status == "final":
            print(f"  FINAL: {d['bin'].upper()} - {d['reason']}")
            break
        print(f"  NOT READY - {d['reason']}")
        for s in d["steps"]:
            print(f"   - {s}")
        for n in d["notes"]:
            print(f"     ({n})")
    if not session.done:
        print("\nTake another photo after preparing the item and pass it as the next argument.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m src.pipeline PHOTO [RESCAN_PHOTO ...]")
    _cli(sys.argv[1:])
