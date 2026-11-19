"""Data statistics are observations; missing distributions are null."""
from collections import Counter
import statistics


def distribution(values):
    if not values:
        return {"n": 0, "min": None, "median": None, "p95": None, "max": None, "mean": None}
    values = sorted(values)
    return {"n": len(values), "min": values[0], "median": statistics.median(values),
            "p95": values[min(len(values) - 1, int((len(values) - 1) * 0.95))],
            "max": values[-1], "mean": statistics.mean(values)}


def language_hint(text):
    """Script heuristic only; not a trained language identifier."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return "unknown"
    if sum(char.isascii() for char in letters) / len(letters) > 0.9:
        return "mostly_latin_heuristic"
    if sum("\u4e00" <= char <= "\u9fff" for char in letters) / len(letters) > 0.5:
        return "mostly_han_heuristic"
    return "mixed_or_other_heuristic"


def quality_report(rows, counts):
    lengths = [r["token_lengths"] for r in rows]
    fields = sorted({k for d in lengths for k, v in d.items() if isinstance(v, (float, int))})
    raw = counts["raw"]
    return {"counts": counts,
            "filter_rate": counts["filtered"] / raw if raw else None,
            "duplicate_rate": counts["deduplicated"] / raw if raw else None,
            "token_lengths": {key: distribution([d[key] for d in lengths if key in d]) for key in fields},
            "sources": dict(Counter(r["provenance"]["dataset_id"] for r in rows)),
            "languages": dict(Counter(language_hint(r["prompt"]) for r in rows)),
            "language_method": "Unicode script heuristic, not language proficiency evidence",
            "splits": dict(Counter(r["split"] for r in rows)),
            "task_types": dict(Counter(r["task_type"] for r in rows))}
