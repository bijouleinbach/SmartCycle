"""Deterministic rule engine: (material, condition, component) -> what to do.

This module contains no AI. It's a lookup table so the report can describe the
model and the rules separately, and so swapping in a different city's rules
never requires retraining.

TODO (task 0.2 / 1.2, Tian): the RULES below are generic US-curbside
placeholders. Replace them with one real local program and record the source.
"""

from dataclasses import dataclass, field

from src.labels import COMPONENTS, CONDITIONS, MATERIALS

RULES_SOURCE = "PLACEHOLDER: generic US curbside guidance, not a real local program"

RECYCLING = "recycling"
COMPOST = "compost"
TRASH = "trash"

# What a clean item of each material goes in.
CLEAN_BIN = {
    "plastic": RECYCLING,
    "paper_cardboard": RECYCLING,
    "glass": RECYCLING,
    "metal": RECYCLING,
    "organic": COMPOST,
}

# For each (material, condition) that isn't clean: either a preparation step the
# user can fix ("prep", instruction) or a final bin because it can't be fixed
# ("bin", bin, reason). Pairs not listed fall back to DEFAULT_PREP.
CONDITION_RULES = {
    ("plastic", "food_residue"): ("prep", "Scrape out the food and give it a quick rinse."),
    ("plastic", "grease"): ("prep", "Wipe out the grease with a paper towel, then rinse."),
    ("plastic", "liquid"): ("prep", "Empty out the liquid."),
    ("plastic", "mixed_materials"): ("prep", "Separate the non-plastic parts from the plastic."),
    ("paper_cardboard", "food_residue"): ("prep", "Tear off or scrape away the parts with food on them."),
    ("paper_cardboard", "grease"): ("bin", COMPOST, "Greasy paper can't be recycled because the oil ruins the paper pulp."),
    ("paper_cardboard", "liquid"): ("bin", TRASH, "Soaked paper fibers are too damaged to recycle."),
    ("paper_cardboard", "mixed_materials"): ("prep", "Remove any plastic windows, tape or foil."),
    ("glass", "food_residue"): ("prep", "Scrape out the food and rinse."),
    ("glass", "grease"): ("prep", "Wipe out the grease, then rinse."),
    ("glass", "liquid"): ("prep", "Empty out the liquid."),
    ("glass", "mixed_materials"): ("prep", "Separate the glass from the other materials."),
    ("metal", "food_residue"): ("prep", "Scrape out the food and rinse."),
    ("metal", "grease"): ("prep", "Wipe out the grease, then rinse."),
    ("metal", "liquid"): ("prep", "Empty out the liquid."),
    ("metal", "mixed_materials"): ("prep", "Separate the metal from the other materials."),
}
DEFAULT_PREP = ("prep", "Clean the item so it's empty and free of food.")

# Removable parts: instruction, and where the removed part goes.
COMPONENT_RULES = {
    "lid": ("Take the lid off.", "Put the lid in recycling if it's plastic or metal, otherwise trash."),
    "sleeve_label": ("Remove the sleeve or shrink-wrap label.", "Put the sleeve in the trash."),
    "straw": ("Remove the straw.", "Put the straw in the trash."),
}

# Organic waste always goes to compost, regardless of condition or parts.
ALWAYS_BIN = {"organic": (COMPOST, "Food and other organic waste goes in compost.")}


@dataclass
class Decision:
    ready: bool  # True: `bin` is the final answer. False: follow `steps` and re-scan.
    bin: str | None
    reason: str
    steps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def decide(material: str, condition: str, component: str) -> Decision:
    """Apply the rules to one set of predicted labels."""
    _check(material, MATERIALS, "material")
    _check(condition, CONDITIONS, "condition")
    _check(component, COMPONENTS, "component")

    if material in ALWAYS_BIN:
        bin_, reason = ALWAYS_BIN[material]
        return Decision(ready=True, bin=bin_, reason=reason)

    steps, notes = [], []
    if condition != "clean":
        rule = CONDITION_RULES.get((material, condition), DEFAULT_PREP)
        if rule[0] == "bin":
            # Unfixable: the condition decides the bin, no matter the parts.
            return Decision(ready=True, bin=rule[1], reason=rule[2])
        steps.append(rule[1])

    if component != "none":
        instruction, note = COMPONENT_RULES[component]
        steps.append(instruction)
        notes.append(note)

    if steps:
        return Decision(
            ready=False,
            bin=None,
            reason=f"This {_name(material)} item needs preparing before it can be recycled.",
            steps=steps,
            notes=notes,
        )

    return Decision(
        ready=True,
        bin=CLEAN_BIN[material],
        reason=f"Clean {_name(material)} with no removable parts goes in {CLEAN_BIN[material]}.",
    )


def _check(value, allowed, axis):
    if value not in allowed:
        raise ValueError(f"Unknown {axis} {value!r}. Allowed: {allowed}")


def _name(material: str) -> str:
    return material.replace("_", "/")
