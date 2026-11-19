"""Human A/B review with reproducible, balanced side assignment and a private key."""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from .bootstrap import cluster_bootstrap
from .generation import comparable_pairs
from .io import digest, read_jsonl, write_json, write_jsonl

LABELS = {"A": "A", "A胜": "A", "B": "B", "B胜": "B", "tie": "tie", "平局": "tie", "unjudgeable": "unjudgeable", "无法判断": "unjudgeable", "": "pending"}
FAILURE_TAGS = {"repetition", "missed_instruction", "incorrect_refusal", "factual_error", "other"}


def export_blind(rows_left, rows_right, output_dir, *, seed=42, limit=None):
    pairs = comparable_pairs(rows_left, rows_right)
    if any(a["source_split"] not in ("test_prefs", "fixture_test") for a, _ in pairs):
        raise ValueError("blind review requires isolated test_prefs or explicitly synthetic fixture_test")
    if any(a["source_split"] == "fixture_test" and not a["synthetic"] for a, _ in pairs):
        raise ValueError("fixture_test must be synthetic")
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")
    rng = random.Random(seed)
    # Sample prompts first; preserve all repeated answers for selected prompts.
    prompt_ids = sorted({a["prompt_id"] for a, _ in pairs})
    rng.shuffle(prompt_ids)
    selected = set(prompt_ids if limit is None else prompt_ids[:limit])
    pairs = [pair for pair in pairs if pair[0]["prompt_id"] in selected]
    rng.shuffle(pairs)
    left_on_a = [i % 2 == 0 for i in range(len(pairs))]
    rng.shuffle(left_on_a)
    public, keys, annotations = [], [], []
    for index, ((left, right), lhs_a) in enumerate(zip(pairs, left_on_a)):
        case_id = f"case-{index+1:05d}"
        a, b = (left, right) if lhs_a else (right, left)
        row = {"case_id": case_id, "prompt": left["prompt"], "A": a["response"], "B": b["response"]}
        public.append(row)
        keys.append({"case_id": case_id, "prompt_id": left["prompt_id"], "sample_id": str(left.get("sample_id", "0")),
                     "left_on_a": lhs_a, "public_sha256": digest(row), "left": left, "right": right})
        annotations.append({"case_id": case_id, "label": "", "reason": "", "failure_tags": []})
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for filename in ("candidates.jsonl", "private_key.json", "annotations.jsonl"):
        if (out / filename).exists():
            raise FileExistsError(f"refusing to overwrite review material: {out / filename}")
    write_jsonl(out / "candidates.jsonl", public)
    write_json(out / "private_key.json", {"seed": seed, "left_model": rows_left[0]["model_id"], "right_model": rows_right[0]["model_id"],
                                        "synthetic": rows_left[0]["synthetic"], "cases": keys})
    write_jsonl(out / "annotations.jsonl", annotations)
    (out / "RUBRIC.md").write_text("# 人工回答偏好盲评\n\n只向标注人发送 candidates.jsonl、annotations.jsonl 和本规则，不发送 private_key.json。\n"
        "在阅读 prompt 后按指令遵循、事实准确、相关性和有用性判断 A/B，不能只因更长就选优。\n"
        "label 填 A、B、tie 或 unjudgeable（亦支持 A胜/B胜/平局/无法判断）；未评分留空。\n"
        "reason 记录依据；failure_tags 可填 repetition / missed_instruction / incorrect_refusal / factual_error / other。\n"
        "无法核验的事实或超出知识范围应填无法判断。无法判断和未评分不进入胜率分母，两者分别报告。\n"
        "允许失败、平局和无提升；人工标签不应根据模型身份修改。\n", encoding="utf-8")
    return {"status": "pending_human_review", "count": len(public), "n_prompts": len(selected),
            "candidates": str(out / "candidates.jsonl"), "annotations": str(out / "annotations.jsonl"),
            "private_key": str(out / "private_key.json"), "win_rate": None, "reason": "no human annotations yet"}


def score_records(records, *, seed=42, n_bootstrap=2000):
    if any(r.get("outcome") not in ("win", "loss", "tie", "pending", "unjudgeable") or not r.get("prompt_id") for r in records):
        raise ValueError("invalid scored record outcome or prompt_id")
    counts = Counter(r["outcome"] for r in records)
    n = counts["win"] + counts["loss"] + counts["tie"]
    return {"W": counts["win"], "L": counts["loss"], "T": counts["tie"], "N": n,
            "unjudgeable": counts["unjudgeable"], "pending": counts["pending"],
            "win_rate": counts["win"] / n if n else None,
            "tie_score": (counts["win"] + .5 * counts["tie"]) / n if n else None,
            "reason": None if n else "no scored comparisons; pending and unjudgeable excluded",
            "denominator": "W+L+T; pending and unjudgeable excluded",
            "bootstrap": cluster_bootstrap(records, seed=seed, n_bootstrap=n_bootstrap)}


def _length_bucket(value):
    if value is None:
        return "unknown_tokens"
    return "0-127" if value < 128 else "128-511" if value < 512 else "512+"


def import_blind(public_path, key_path, annotation_path, *, seed=42, n_bootstrap=2000):
    public_rows = read_jsonl(public_path)
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    keys = {r["case_id"]: r for r in key["cases"]}
    public = {r["case_id"]: r for r in public_rows}
    if len(public) != len(public_rows) or len(keys) != len(key["cases"]) or public.keys() != keys.keys():
        raise ValueError("review key coverage or unique IDs invalid")
    for case_id, row in public.items():
        if digest(row) != keys[case_id]["public_sha256"]:
            raise ValueError("candidate content changed after export")
    annotations = read_jsonl(annotation_path)
    indexed = {}
    for annotation in annotations:
        case_id = annotation.get("case_id")
        if case_id not in keys or case_id in indexed:
            raise ValueError("unknown or duplicate annotation case_id")
        label = annotation.get("label", "")
        if label not in LABELS:
            raise ValueError(f"unsupported annotation label: {label}")
        tags = annotation.get("failure_tags", [])
        if not isinstance(tags, list) or not set(tags) <= FAILURE_TAGS:
            raise ValueError("unknown failure tag")
        indexed[case_id] = annotation
    records, bad_cases = [], []
    buckets = defaultdict(list)
    for case_id, private in keys.items():
        annotation = indexed.get(case_id, {"label": "", "reason": "", "failure_tags": []})
        label = LABELS[annotation.get("label", "")]
        outcome = label
        if label in ("A", "B"):
            outcome = "win" if (label == "A") == private["left_on_a"] else "loss"
        left, right = private["left"], private["right"]
        record = {"case_id": case_id, "prompt_id": private["prompt_id"], "sample_id": private["sample_id"],
                  "outcome": outcome, "failure_tags": annotation.get("failure_tags", [])}
        records.append(record)
        response_lengths = (left.get("response_tokens"), right.get("response_tokens"))
        length_relation = "unknown_tokens" if None in response_lengths else "equal" if response_lengths[0] == response_lengths[1] else "left_longer" if response_lengths[0] > response_lengths[1] else "left_shorter"
        for group, bucket in (("prompt_tokens", _length_bucket(left.get("prompt_tokens"))),
                              ("left_response_tokens", _length_bucket(left.get("response_tokens"))),
                              ("response_length_relation", length_relation), ("task_type", left.get("task_type", "unknown"))):
            buckets[(group, bucket)].append(record)
        if outcome == "loss" or record["failure_tags"]:
            bad_cases.append({**record, "prompt": left["prompt"], "left_response": left["response"], "right_response": right["response"],
                              "reason": annotation.get("reason", ""), "source_split": left["source_split"],
                              "model_revision": left["model_revision"], "synthetic": left["synthetic"]})
    result = score_records(records, seed=seed, n_bootstrap=n_bootstrap)
    result.update({"left_model": key["left_model"], "right_model": key["right_model"], "synthetic": key["synthetic"],
                   "annotation_sha256": digest(annotations), "records": records, "bad_cases": bad_cases,
                   "strata": {f"{group}/{bucket}": score_records(rows, seed=seed, n_bootstrap=n_bootstrap)
                              for (group, bucket), rows in buckets.items()}})
    return result
