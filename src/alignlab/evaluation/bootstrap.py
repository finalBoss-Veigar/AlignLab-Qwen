"""Percentile bootstrap that resamples prompt clusters, not individual answers."""
import random
from collections import defaultdict


def percentile(values, p):
    values = sorted(values)
    index = (len(values) - 1) * p
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def cluster_bootstrap(records, *, n_bootstrap=2000, seed=42, confidence=0.95):
    if n_bootstrap < 1 or not 0 < confidence < 1:
        raise ValueError("invalid bootstrap settings")
    clusters = defaultdict(list)
    for row in records:
        if row["outcome"] in ("win", "loss", "tie"):
            clusters[row["prompt_id"]].append(row["outcome"])
    if not clusters:
        return {"win_rate_ci": None, "tie_score_ci": None, "reason": "no judged prompts", "n_prompts": 0}
    groups = list(clusters.values())
    rng = random.Random(seed)
    win_rates, tie_scores = [], []
    for _ in range(n_bootstrap):
        sample = [outcome for _ in groups for outcome in rng.choice(groups)]
        wins, ties = sample.count("win"), sample.count("tie")
        win_rates.append(wins / len(sample))
        tie_scores.append((wins + 0.5 * ties) / len(sample))
    tail = (1 - confidence) / 2
    return {"win_rate_ci": [percentile(win_rates, tail), percentile(win_rates, 1-tail)],
            "tie_score_ci": [percentile(tie_scores, tail), percentile(tie_scores, 1-tail)],
            "n_prompts": len(groups), "n_bootstrap": n_bootstrap, "seed": seed,
            "confidence": confidence, "unit": "prompt cluster",
            "reason": "few prompts; percentile intervals may be degenerate and do not establish generalization" if len(groups) < 30 else None}
