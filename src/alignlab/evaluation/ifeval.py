"""Adapter around Google's official IFEval implementation; no replacement rules."""
import os
import subprocess
import sys
import time
from pathlib import Path

from .io import digest, read_jsonl, write_json, write_jsonl

OFFICIAL_REVISION = "f8d6d14a1fc6eaef606d7d1e1fa4e96ee09afb81"
DATASET_REVISION = "966cd89545d6b6acfd7638bc708b98261ca58e84"


def validate_inputs(inputs, responses):
    if not inputs:
        raise ValueError("IFEval inputs are empty; cannot score")
    prompts, keys = set(), set()
    normalized = []
    for row in inputs:
        prompt, instructions, kwargs = row.get("prompt"), row.get("instruction_id_list"), row.get("kwargs")
        if not isinstance(prompt, str) or not prompt.strip() or prompt in prompts:
            raise ValueError("IFEval prompt must be nonempty and unique")
        if type(row.get("key")) is not int or row["key"] in keys:
            raise ValueError("IFEval key must be a unique integer")
        if not isinstance(instructions, list) or not instructions or not all(isinstance(s, str) and s for s in instructions):
            raise ValueError("IFEval instruction_id_list is invalid")
        if not isinstance(kwargs, list) or len(kwargs) != len(instructions) or not all(isinstance(k, dict) for k in kwargs):
            raise ValueError("IFEval kwargs must correspond one-to-one with instruction IDs")
        # HF tabular conversion may insert null fields absent from official JSONL.
        normalized.append({"key": row["key"], "prompt": prompt, "instruction_id_list": instructions,
                           "kwargs": [{k: v for k, v in kw.items() if v is not None} for kw in kwargs]})
        prompts.add(prompt)
        keys.add(row["key"])
    response_prompts = set()
    for row in responses:
        if row.get("prompt") in response_prompts or not isinstance(row.get("response"), str):
            raise ValueError("IFEval responses must be unique strings")
        response_prompts.add(row.get("prompt"))
    if prompts != response_prompts:
        raise ValueError("IFEval responses must have exactly the input prompts; no silent drops")
    return normalized


def parse_official_outputs(output_dir, *, expected_prompts=None):
    result = {"implementation": "google-research/instruction_following_eval", "implementation_revision": OFFICIAL_REVISION,
              "source_verified": False, "scope": "output_aggregation_only; use run_official to verify source and dataset coverage"}
    coverage = None
    for mode in ("strict", "loose"):
        rows = read_jsonl(Path(output_dir) / f"eval_results_{mode}.jsonl")
        if not rows:
            raise ValueError("empty official output is not a metric")
        prompts = [r.get("prompt") for r in rows]
        if len(set(prompts)) != len(prompts) or any(not isinstance(p, str) for p in prompts):
            raise ValueError("invalid official prompt coverage")
        if expected_prompts is not None and set(prompts) != set(expected_prompts):
            raise ValueError("official output coverage differs from inputs")
        if coverage is not None and coverage != set(prompts):
            raise ValueError("strict and loose output coverage differs")
        coverage = set(prompts)
        instruction_count = instruction_correct = prompt_correct = 0
        for row in rows:
            flags = row.get("follow_instruction_list")
            ids = row.get("instruction_id_list")
            if not isinstance(flags, list) or not flags or not isinstance(ids, list) or len(flags) != len(ids) or not all(type(f) is bool for f in flags):
                raise ValueError("invalid official per-instruction flags")
            if type(row.get("follow_all_instructions")) is not bool or row["follow_all_instructions"] != all(flags):
                raise ValueError("inconsistent official prompt/instruction flags")
            instruction_count += len(flags)
            instruction_correct += sum(flags)
            prompt_correct += row["follow_all_instructions"]
        result[mode] = {"prompt_level": prompt_correct / len(rows), "instruction_level": instruction_correct / instruction_count,
                        "n_prompts": len(rows), "n_instructions": instruction_count}
    return result


def run_official(repo_dir, input_path, response_path, output_dir, *, dry_run=True, expected_revision=OFFICIAL_REVISION, timeout=1800):
    repo = Path(repo_dir).resolve()
    out = Path(output_dir).resolve()
    inputs, responses = read_jsonl(input_path), read_jsonl(response_path)
    normalized = validate_inputs(inputs, responses)
    command = [sys.executable, "-m", "instruction_following_eval.evaluation_main",
               f"--input_data={out / 'official_inputs.jsonl'}", f"--input_response_data={Path(response_path).resolve()}",
               f"--output_dir={out}"]
    result = {"status": "dry_run_checked_not_executed", "command": command, "cwd": str(repo),
              "implementation_revision": expected_revision, "input_sha256": digest(inputs),
              "response_sha256": digest(responses), "n_prompts": len(inputs),
              "metrics": None, "reason": "official implementation has not run"}
    if dry_run:
        result["source_available"] = (repo / "instruction_following_eval/evaluation_main.py").is_file()
        return result
    response_synthetic = any(r.get("synthetic") is True for r in responses)
    if response_synthetic and "fixtures" not in out.parts:
        raise ValueError("synthetic responses must be evaluated in a fixtures directory")
    if expected_revision != OFFICIAL_REVISION:
        raise ValueError("change and revalidate pinned adapter explicitly before using a different IFEval revision")
    actual = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if actual != expected_revision:
        raise ValueError(f"IFEval source revision mismatch: {actual}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "instruction_following_eval"], cwd=repo, capture_output=True, text=True, check=True).stdout
    if dirty.strip():
        raise ValueError("IFEval source has local modifications; clean pinned source is required")
    official_path = repo / "instruction_following_eval/data/input_data.jsonl"
    official_inputs = read_jsonl(official_path)
    official_normalized = validate_inputs(official_inputs, [{"prompt": r["prompt"], "response": ""} for r in official_inputs])
    full_dataset = digest(sorted(normalized, key=lambda r: r["key"])) == digest(sorted(official_normalized, key=lambda r: r["key"]))
    result["dataset_scope"] = "full_official_dataset" if full_dataset else "subset_or_custom_development_inputs"
    result["official_n_prompts"] = len(official_normalized)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / name).exists() for name in ("official_inputs.jsonl", "run.json", "eval_results_strict.jsonl", "eval_results_loose.jsonl")):
        raise FileExistsError("use a fresh IFEval output directory to prevent stale metrics")
    write_jsonl(out / "official_inputs.jsonl", normalized)
    started = time.monotonic()
    result["status"] = "running"
    write_json(out / "run.json", result)
    try:
        env = dict(os.environ, PYTHONUTF8="1")
        with (out / "official.log").open("w", encoding="utf-8") as log:
            proc = subprocess.run(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=timeout)
        result["exit_code"] = proc.returncode
        if proc.returncode != 0:
            raise RuntimeError(f"official IFEval exited {proc.returncode}; see {out / 'official.log'}")
        result["metrics"] = parse_official_outputs(out, expected_prompts=[r["prompt"] for r in inputs])
        result["metrics"].update(source_verified=True, scope=result["dataset_scope"], official_n_prompts=len(official_normalized),
                                 synthetic=response_synthetic,
                                 response_provenance="generation_records" if all("synthetic" in r and "model_revision" in r for r in responses) else "caller_supplied_prompt_response_only")
        result["status"], result["reason"] = "completed", None
        write_json(out / "metrics.json", result["metrics"])
    except Exception as error:
        result.update(status="failed", reason=str(error), metrics=None)
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic() - started
        write_json(out / "run.json", result)
    return result
