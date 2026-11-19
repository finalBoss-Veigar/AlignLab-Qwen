"""One shared group assignment across SFT and DPO, before any sampling."""
import random


def assign_splits(rows, *, seed=42, validation_fraction=0.2):
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    groups = sorted({row["group_id"] for row in rows})
    random.Random(seed).shuffle(groups)
    n_valid = min(len(groups) - 1, max(1, round(len(groups) * validation_fraction))) if len(groups) > 1 else 0
    validation = set(groups[:n_valid])
    for row in rows:
        row["split"] = "validation" if row["group_id"] in validation else "train"
    return rows


def sample_rows(rows, *, budget, seed):
    if budget < 0:
        raise ValueError("budget must be >= 0")
    rows = sorted(rows, key=lambda row: (row["group_id"], row["provenance"]["source_row_id"]))
    random.Random(seed).shuffle(rows)
    return rows[:budget]


def assert_no_leakage(rows, eval_rows):
    train = {row["group_id"] for row in rows if row["split"] == "train"}
    validation = {row["group_id"] for row in rows if row["split"] == "validation"}
    test = {row["group_id"] for row in eval_rows}
    if train & validation or train & test or validation & test:
        raise ValueError("Duplicate groups leak across train/validation/test")
