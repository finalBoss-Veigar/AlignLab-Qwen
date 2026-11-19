"""Auditable local pipeline. Raw inputs are read-only; each output directory is new."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

from .cleaning import check_quality
from .dedup import duplicate_groups
from .formats import assert_roundtrip, dataset_info, to_llamafactory
from .schema import DataError, dpo_record, first_prompt, prompt_hash, provenance, sft_record
from .split import assert_no_leakage, assign_splits, sample_rows
from .statistics import quality_report
from .tokens import FixtureTokenizer, fit_record


EXPECTED_DATASETS = {"sft": "HuggingFaceH4/ultrachat_200k", "dpo": "HuggingFaceH4/ultrafeedback_binarized"}
REQUIRED_EVAL = {
    ("HuggingFaceH4/ultrachat_200k", "test_sft"),
    ("HuggingFaceH4/ultrachat_200k", "test_gen"),
    ("HuggingFaceH4/ultrafeedback_binarized", "test_prefs"),
    ("HuggingFaceH4/ultrafeedback_binarized", "test_sft"),
    ("HuggingFaceH4/ultrafeedback_binarized", "test_gen"),
    ("google/IFEval", "train"),
}


def read_rows(path):
    path = Path(path)
    if path.suffix == ".json":
        rows = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list):
            raise ValueError("JSON source must contain an array")
        return rows
    # str.splitlines also splits Unicode U+2028/U+2029/U+0085 inside valid JSON
    # string values. JSONL record boundaries are physical newline bytes only.
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path, values):
    Path(path).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values), encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eval_rows(path, synthetic):
    rows = []
    for index, row in enumerate(read_rows(path)):
        prompt = row.get("prompt")
        if prompt is None and row.get("messages"):
            prompt = first_prompt(row["messages"])
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Evaluation prompt is invalid: row {index}")
        metadata = provenance(row, index, dataset_id="synthetic/evaluation", revision="synthetic-v1",
                              source_split="test", source_file=str(Path(path).resolve()), synthetic=synthetic)
        rows.append({"kind": "evaluation", "prompt": prompt, "prompt_hash": prompt_hash(prompt),
                     "messages": row.get("messages", [{"role": "user", "content": prompt}]),
                     "provenance": metadata})
    if not synthetic:
        sidecar = Path(str(path) + ".meta.json")
        if not sidecar.exists():
            raise ValueError("Formal evaluation exclusions require .meta.json coverage sidecar")
        coverage = json.loads(sidecar.read_text(encoding="utf-8"))
        actual = Counter((r["provenance"]["dataset_id"], r["provenance"]["source_split"]) for r in rows)
        stated = {(r["dataset_id"], r["source_split"]): r for r in coverage.get("splits", [])}
        if coverage.get("sha256") != file_hash(path):
            raise ValueError("Evaluation exclusion file hash differs from coverage sidecar")
        for key in REQUIRED_EVAL:
            item = stated.get(key, {})
            if not item.get("complete") or item.get("rows") != actual[key] or actual[key] == 0:
                raise ValueError(f"Incomplete official evaluation exclusions: {key}")
    return rows


def run_pipeline(sft_path, dpo_path, eval_path, output_dir, *, seed=42,
                 validation_fraction=0.2, sft_budget=20000, dpo_budget=10000,
                 max_tokens=1024, tokenizer=None, near_threshold=0.85,
                 quality_filter=True, synthetic=False):
    paths = {"sft": Path(sft_path).resolve(), "dpo": Path(dpo_path).resolve(), "evaluation": Path(eval_path).resolve()}
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use a new empty output directory to preserve previous evidence: {output}")
    if any(path == output or output in path.parents for path in paths.values()):
        raise ValueError("Output directory must not contain any raw input file")
    if not synthetic and (tokenizer is None or getattr(tokenizer, "synthetic", False)):
        raise ValueError("Formal preprocessing requires an actual pinned model tokenizer")
    tokenizer = tokenizer or FixtureTokenizer()
    if not synthetic and not (getattr(tokenizer, "alignlab_revision", None) or getattr(tokenizer, "revision", None)):
        raise ValueError("Actual tokenizer revision must be recorded")
    if min(sft_budget, dpo_budget) < 0 or not 0 < validation_fraction < 1 or max_tokens < 8:
        raise ValueError("Invalid data budget, validation fraction or token limit")
    eval_rows = _eval_rows(paths["evaluation"], synthetic)
    if not eval_rows:
        raise ValueError("At least one isolated evaluation prompt is required")
    rows, rejected = [], []
    counts = {kind: {key: 0 for key in ("raw", "filtered", "valid", "evaluation_overlap", "deduplicated",
                                           "after_deduplication", "train_before_sampling", "validation",
                                           "sampled_train", "budget_excluded", "final")}
              for kind in ("sft", "dpo")}
    for kind in ("sft", "dpo"):
        for index, source in enumerate(read_rows(paths[kind])):
            counts[kind]["raw"] += 1
            if not isinstance(source, dict):
                counts[kind]["filtered"] += 1
                rejected.append({"kind": kind, "source_row_id": str(index), "source_file": str(paths[kind]),
                                 "reason": "record_not_object", "stage": "validation_and_cleaning"})
                continue
            metadata = source.get("provenance", {})
            try:
                metadata = provenance(source, index,
                    dataset_id=f"synthetic/{kind}" if synthetic else EXPECTED_DATASETS[kind],
                    revision="synthetic-v1" if synthetic else "unknown",
                    source_split="train_sft" if kind == "sft" else "train_prefs",
                    source_file=str(paths[kind]), synthetic=synthetic)
                if not synthetic and (metadata["dataset_id"] != EXPECTED_DATASETS[kind]
                                       or metadata["source_split"] != ("train_sft" if kind == "sft" else "train_prefs")):
                    raise DataError("unexpected_training_source")
                row = (sft_record if kind == "sft" else dpo_record)(source, metadata)
                check_quality(row, enabled=quality_filter)
                row = fit_record(row, tokenizer, max_tokens)
                rows.append(row)
                counts[kind]["valid"] += 1
            except DataError as exc:
                if not synthetic and str(exc) in {"unpinned_or_synthetic_provenance", "missing_provenance", "unexpected_training_source"}:
                    raise ValueError(f"Formal source provenance error at {paths[kind]}:{index}: {exc}") from exc
                counts[kind]["filtered"] += 1
                rejected.append({"kind": kind, "source_row_id": str(index), "provenance": metadata,
                                 "reason": str(exc), "stage": "validation_and_cleaning"})
    # Group every valid train-source row jointly with all evaluation rows BEFORE split or budget.
    all_rows = rows + eval_rows
    _, duplicate_stats = duplicate_groups(all_rows, threshold=near_threshold)
    held_out = {row["group_id"] for row in eval_rows}
    retained, seen = [], set()
    for row in rows:
        kind = row["kind"]
        key = (kind, row["group_id"])
        if row["group_id"] in held_out:
            reason, counter = "evaluation_exact_or_near_overlap", "evaluation_overlap"
        elif key in seen:
            reason, counter = "duplicate_prompt_group", "deduplicated"
        else:
            seen.add(key)
            retained.append(row)
            counts[kind]["after_deduplication"] += 1
            continue
        counts[kind][counter] += 1
        rejected.append({"kind": kind, "provenance": row["provenance"], "prompt_hash": row["prompt_hash"],
                         "group_id": row["group_id"], "reason": reason, "stage": "deduplication_and_isolation"})
    assign_splits(retained, seed=seed, validation_fraction=validation_fraction)
    assert_no_leakage(retained, eval_rows)
    selected = []
    for kind, budget in (("sft", sft_budget), ("dpo", dpo_budget)):
        train = [r for r in retained if r["kind"] == kind and r["split"] == "train"]
        validation = [r for r in retained if r["kind"] == kind and r["split"] == "validation"]
        sampled = sample_rows(train, budget=budget, seed=seed)
        counts[kind].update(train_before_sampling=len(train), validation=len(validation),
                            sampled_train=len(sampled), budget_excluded=len(train) - len(sampled),
                            final=len(sampled) + len(validation))
        selected.extend(sampled + validation)
    assert_no_leakage(selected, eval_rows)
    output.mkdir(parents=True, exist_ok=True)
    for row in selected:
        assert_roundtrip(row)
    for kind in ("sft", "dpo"):
        for split in ("train", "validation"):
            subset = [r for r in selected if r["kind"] == kind and r["split"] == split]
            _write_json(output / f"{kind}_{split}.json", [to_llamafactory(r) for r in subset])
            _write_jsonl(output / f"{kind}_{split}.internal.jsonl", subset)
    _write_json(output / "dataset_info.json", dataset_info())
    _write_jsonl(output / "rejected.jsonl", rejected)
    _write_jsonl(output / "evaluation_exclusions.jsonl", eval_rows)
    _write_jsonl(output / "split_assignments.jsonl", [
        {"kind": r["kind"], "group_id": r["group_id"], "prompt_hash": r["prompt_hash"], "split": r["split"],
         "provenance": r["provenance"]} for r in retained])
    reports = {kind: quality_report([r for r in selected if r["kind"] == kind], counts[kind])
               for kind in ("sft", "dpo")}
    for kind, report in reports.items():
        _write_json(output / f"{kind}_quality_report.json", report)
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
                "synthetic": synthetic, "synthetic_fixture": synthetic,
                "purpose": "SYNTHETIC FIXTURE 流程测试，非模型效果" if synthetic else "formal_preprocessing",
                "seed": seed, "inputs": {k: {"path": str(p), "sha256": file_hash(p)} for k, p in paths.items()},
                "tokenizer": {"name": getattr(tokenizer, "name_or_path", type(tokenizer).__name__),
                              "revision": getattr(tokenizer, "alignlab_revision", getattr(tokenizer, "revision", None)),
                              "synthetic": bool(getattr(tokenizer, "synthetic", False))},
                "settings": {"max_tokens": max_tokens, "validation_fraction": validation_fraction,
                             "sft_budget": sft_budget, "dpo_budget": dpo_budget, "quality_filter": quality_filter,
                             "truncation": "oldest_complete_turn_pairs_only; never truncate a response"},
                "counts": counts, "duplicates": duplicate_stats, "evaluation_rows": len(eval_rows),
                "rejection_reasons": dict(Counter(r["reason"] for r in rejected)),
                "leakage_check": "passed", "lf_roundtrip": "passed", "output_dir": str(output),
                "source_revisions": sorted({(r["provenance"]["dataset_id"], r["provenance"]["revision"],
                                             r["provenance"]["source_split"]) for r in all_rows}),
                "artifacts": {p.name: {"sha256": file_hash(p), "bytes": p.stat().st_size}
                              for p in sorted(output.iterdir()) if p.is_file()}}
    _write_json(output / "data_manifest.json", manifest)
    title = manifest["purpose"]
    (output / "data_report.html").write_text(
        "<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>AlignLab 数据报告</title>"
        "<style>body{font-family:system-ui;max-width:1000px;margin:40px auto}pre{white-space:pre-wrap}"
        ".notice{background:#fff3cd;padding:20px}</style><h1>AlignLab 数据质量报告</h1><div class='notice'>"
        + html.escape(title) + "</div><p>仅报告实际处理样本；未生成任何训练成绩。语言为文字脚本启发式。</p><pre>"
        + html.escape(json.dumps({"counts": counts, "quality": reports, "deduplication": duplicate_stats,
                                  "rejections": manifest["rejection_reasons"]}, ensure_ascii=False, indent=2))
        + "</pre></html>", encoding="utf-8")
    return manifest


def run_fixture_pipeline(project_root, output_dir=None):
    root = Path(project_root)
    output = output_dir or root / "data" / "processed" / ("fixture-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f"))
    fixtures = root / "data" / "fixtures"
    return run_pipeline(fixtures / "sft.jsonl", fixtures / "dpo.jsonl", fixtures / "evaluation.jsonl",
                        output, synthetic=True, max_tokens=128)
