"""Label schema shared by data loading, the model, the rule engine and evaluation.

Every image gets one value per axis. If a value isn't known for an image (for
example, public datasets like TrashNet only label material), leave it blank in
the CSV and that axis is skipped for that image during training and evaluation.

Change the schema here and only here: the rest of the code reads from these lists.
"""

MATERIALS = ["plastic", "paper_cardboard", "glass", "metal", "organic"]
CONDITIONS = ["clean", "food_residue", "grease", "liquid", "mixed_materials"]
COMPONENTS = ["none", "lid", "sleeve_label", "straw"]

AXES = {
    "material": MATERIALS,
    "condition": CONDITIONS,
    "component": COMPONENTS,
}

# Columns every labeled.csv must have. `filename` is relative to the CSV's folder.
# `stage` is "before" (as found) or "after" (after preparing it), and `pair_id`
# links a before photo to its after photo for evaluating the re-scan step.
CSV_COLUMNS = ["filename", "material", "condition", "component", "stage", "pair_id"]

STAGES = ["before", "after"]


def index_of(axis: str, value) -> int:
    """Return the class index for a label value, or -1 if it's blank/unknown."""
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return -1
    value = str(value).strip()
    if not value:
        return -1
    classes = AXES[axis]
    if value not in classes:
        raise ValueError(f"Unknown {axis} label {value!r}. Allowed: {classes}")
    return classes.index(value)
