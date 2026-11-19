"""Explicit opt-in public dataset download, independent of model weights."""
from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path

from .pipeline import REQUIRED_EVAL, file_hash
from .schema import first_prompt, prompt_hash


REVISIONS = {
    "HuggingFaceH4/ultrachat_200k": "8049631c405ae6576f93f445c6b8166f76f5505a",
    "HuggingFaceH4/ultrafeedback_binarized": "3949bf5f8c17c394422ccfab0c31ea9c20bdeb85",
    "google/IFEval": "966cd89545d6b6acfd7638bc708b98261ca58e84",
}


def download_sources(output_dir, *, allow_download=False, sft_limit=None, dpo_limit=None, local_shards=False):
    """Never called by smoke. Exhaust ALL evaluation splits even with train limits.

    Limits bound the observed training-source pool and are recorded, never described
    as processing the full upstream dataset. Interrupted .partial files are retained.
    """
    if not allow_download:
        raise ValueError("Dataset download requires explicit allow_download=True")
    if any(value is not None and value < 1 for value in (sft_limit, dpo_limit)):
        raise ValueError("Download limits must be positive")
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Raw download destination must be new or empty")
    from datasets import load_dataset
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    shard_evidence = {}
    def stream(dataset_id, split):
        if not local_shards:
            yield from load_dataset(dataset_id, split=split, revision=REVISIONS[dataset_id], streaming=True)
            return
        # Download full pinned files before parsing to avoid remote Parquet range
        # requests hanging behind proxies. Still consume only the requested rows.
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq
        revision = REVISIONS[dataset_id]
        names = HfApi().list_repo_files(dataset_id, repo_type='dataset', revision=revision)
        names = [n for n in names if n.startswith('data/' + split + '-') and n.endswith('.parquet')] if dataset_id != 'google/IFEval' else ['ifeval_input_data.jsonl']
        if not names:
            raise ValueError(f'No pinned source files for {dataset_id}/{split}')
        for name in sorted(names):
            print(f'Download shard: {dataset_id}/{name}', flush=True)
            path = hf_hub_download(dataset_id, name, repo_type='dataset', revision=revision)
            shard_evidence[f'{dataset_id}/{name}'] = {'revision': revision, 'sha256': file_hash(path), 'bytes': Path(path).stat().st_size}
            if name.endswith('.jsonl'):
                with open(path, encoding='utf-8') as f:
                    for line in f:
                        if line.strip(): yield json.loads(line)
            else:
                for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
                    yield from batch.to_pylist()
    def metadata(dataset_id, split, index):
        return {"dataset_id": dataset_id, "revision": REVISIONS[dataset_id], "source_split": split,
                "source_row_id": str(index), "source_file": f"hf://datasets/{dataset_id}@{REVISIONS[dataset_id]}/{split}",
                "synthetic": False}
    for kind, dataset_id, split, limit in (
        ("sft", "HuggingFaceH4/ultrachat_200k", "train_sft", sft_limit),
        ("dpo", "HuggingFaceH4/ultrafeedback_binarized", "train_prefs", dpo_limit),
    ):
        target = output / f"{kind}.jsonl"
        partial = target.with_suffix(".jsonl.partial")
        count = 0
        with partial.open("x", encoding="utf-8") as handle:
            for index, row in enumerate(stream(dataset_id, split)):
                if limit is not None and count >= limit:
                    break
                row["provenance"] = metadata(dataset_id, split, index)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
        partial.rename(target)
        counts[kind] = {"rows": count, "download_limit": limit, "full_source_requested": limit is None,
                        "dataset_id": dataset_id, "revision": REVISIONS[dataset_id], "source_split": split,
                        "sha256": file_hash(target)}
    eval_file = output / "evaluation.jsonl"
    eval_partial = eval_file.with_suffix(".jsonl.partial")
    prompt_dir = output / "eval_prompts"
    prompt_dir.mkdir()
    prompt_paths = {"test_prefs": prompt_dir / "test_prefs.jsonl", "ifeval": prompt_dir / "ifeval.jsonl",
                    "ifeval_input": prompt_dir / "ifeval_input.jsonl"}
    splits = []
    with ExitStack() as stack:
        handle = stack.enter_context(eval_partial.open("x", encoding="utf-8"))
        prompt_handles = {key: stack.enter_context(Path(str(path) + ".partial").open("x", encoding="utf-8"))
                          for key, path in prompt_paths.items()}
        for dataset_id, split in sorted(REQUIRED_EVAL):
            count = 0
            for index, row in enumerate(stream(dataset_id, split)):
                prompt = row.get("prompt")
                if prompt is None:
                    prompt = first_prompt(row.get("messages") or row.get("chosen"))
                item = {"prompt": prompt, "provenance": metadata(dataset_id, split, index)}
                # Include all user turns as exclusion aliases where the source has them.
                if isinstance(row.get("messages"), list):
                    item["messages"] = row["messages"]
                elif isinstance(row.get("chosen"), list):
                    item["messages"] = row["chosen"]
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                if dataset_id == "google/IFEval" or (dataset_id.endswith("ultrafeedback_binarized") and split == "test_prefs"):
                    kind = "ifeval" if dataset_id == "google/IFEval" else "test_prefs"
                    generation_prompt = {**metadata(dataset_id, split, index), "prompt": prompt,
                        "prompt_id": f"{kind}:{index}", "prompt_hash": prompt_hash(prompt),
                        "task_type": "instruction_following" if kind == "ifeval" else "unknown"}
                    if kind == "test_prefs" and isinstance(row.get("chosen"), list):
                        generation_prompt["messages"] = row["chosen"][:-1]
                    if kind == "ifeval":
                        required = ("key", "prompt", "instruction_id_list", "kwargs")
                        if any(key not in row for key in required):
                            raise ValueError("Official IFEval row lacks required key/prompt/instruction_id_list/kwargs")
                        generation_prompt.update({key: row[key] for key in ("key", "instruction_id_list", "kwargs")})
                        official_row = {key: row[key] for key in required}
                        prompt_handles["ifeval_input"].write(json.dumps(official_row, ensure_ascii=False) + "\n")
                    prompt_handles[kind].write(json.dumps(generation_prompt, ensure_ascii=False) + "\n")
                count += 1
            splits.append({"dataset_id": dataset_id, "source_split": split, "revision": REVISIONS[dataset_id],
                           "rows": count, "complete": True})
    eval_partial.rename(eval_file)
    for path in prompt_paths.values():
        Path(str(path) + ".partial").rename(path)
    (output / "evaluation.jsonl.meta.json").write_text(
        json.dumps({"splits": splits, "sha256": file_hash(eval_file)}, indent=2), encoding="utf-8")
    manifest = {"training_sources": counts, "evaluation_exclusions": splits,
                "download_backend": "local_pinned_shards" if local_shards else "streaming", "source_files": shard_evidence,
                "generation_prompts": {key: {"path": str(path), "sha256": file_hash(path)} for key, path in prompt_paths.items()},
                "model_weights_downloaded": False, "output_dir": str(output)}
    (output / "download_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
