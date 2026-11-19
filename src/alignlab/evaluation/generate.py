"""Explicit local-only model generation for matched offline comparisons."""
import hashlib
from datetime import datetime, timezone
import json
import os
import platform
import sys
import time
from pathlib import Path

from .generation import validate_generations
from .io import digest, read_jsonl, write_json
from ..paths import project_root
from ..tracking.manifest import command


def _save_status(path, status):
    temporary = path.with_suffix(".tmp")
    write_json(temporary, status)
    temporary.replace(path)


def generate_local(model_path, tokenizer_path, prompts_path, output_dir, *, model_revision,
                   tokenizer_revision, size_class, max_new_tokens=512, max_input_tokens=2048,
                   device="cpu", dry_run=True, timeout_seconds=1800):
    if size_class not in ("1.5B", "7B") or not model_revision or not tokenizer_revision:
        raise ValueError("explicit model/tokenizer revision and 1.5B/7B size class are required")
    if max_new_tokens < 1 or max_input_tokens < 1 or timeout_seconds <= 0:
        raise ValueError("token limits must be positive")
    prompts = read_jsonl(prompts_path)
    if not prompts or any(not isinstance(p.get("prompt"), str) or not p["prompt"].strip() or
                          not p.get("prompt_id") for p in prompts):
        raise ValueError("expected isolated test prompts with prompt_id, prompt and source_split")
    for prompt in prompts:
        if prompt.get("synthetic") is not False or prompt.get("dataset_id", "").startswith("synthetic"):
            raise ValueError("real model evaluation requires explicitly synthetic=false; fixtures cannot be relabeled")
        if not prompt.get("dataset_id") or not prompt.get("revision") or prompt.get("source_row_id") is None:
            raise ValueError("real evaluation prompts require dataset_id, revision and source_row_id provenance")
        is_ifeval = prompt["dataset_id"] == "google/IFEval" and prompt.get("source_split") in ("train", "ifeval")
        is_prefs = prompt["dataset_id"] == "HuggingFaceH4/ultrafeedback_binarized" and prompt.get("source_split") == "test_prefs"
        if not (is_ifeval or is_prefs):
            raise ValueError("only isolated UltraFeedback test_prefs and evaluation-only google/IFEval train are allowed")
    if len({p["prompt_id"] for p in prompts}) != len(prompts):
        raise ValueError("duplicate prompt IDs")
    if len({p["prompt"] for p in prompts}) != len(prompts):
        raise ValueError("duplicate prompt text requires one stable prompt ID; deduplicate before generation")
    for prompt in prompts:
        messages = prompt.get("messages") or [{"role": "user", "content": prompt["prompt"]}]
        if not isinstance(messages, list) or not messages or len(messages) > 64:
            raise ValueError("generation messages must be a nonempty list of at most 64 messages")
        expected = "user"
        for index, message in enumerate(messages):
            if not isinstance(message, dict) or not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError("generation messages need nonempty string content")
            if index == 0 and message.get("role") == "system":
                continue
            if message.get("role") != expected:
                raise ValueError("generation messages have invalid role order")
            expected = "assistant" if expected == "user" else "user"
        if messages[-1].get("role") != "user" or messages[-1]["content"] != prompt["prompt"]:
            raise ValueError("generation context must end with the exact evaluation user prompt, never a reference answer")
    status = {"status": "dry_run_checked_not_generated", "model_path": str(model_path), "n_prompts": len(prompts),
              "model_revision": model_revision, "tokenizer_revision": tokenizer_revision, "size_class": size_class,
              "local_model_available": Path(model_path).is_dir(), "generation_count": None,
              "input_sha256": digest(prompts), "device_requested": device, "timeout_seconds": timeout_seconds,
              "max_input_tokens": max_input_tokens,
              "decoding": {"do_sample": False, "max_new_tokens": max_new_tokens,
                           "num_beams": 1, "min_new_tokens": 0, "repetition_penalty": 1.05}}
    if dry_run:
        return status
    if not Path(model_path).is_dir() or not Path(tokenizer_path).is_dir():
        raise FileNotFoundError("local model and tokenizer directories are required; no automatic weight download")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    partial_path = out / "generations.partial.jsonl"
    if any((out / name).exists() for name in ("generations.jsonl", "generation_run.json", partial_path.name)):
        raise FileExistsError("use a fresh generation output directory")
    started = time.monotonic()
    code = command(["git", "rev-parse", "HEAD"], cwd=project_root())
    dirty = command(["git", "status", "--porcelain"], cwd=project_root())
    status.update(status="running", generation_count=0, partial_output_path=str(partial_path),
                  complete_evaluation=False, started_at=datetime.now(timezone.utc).isoformat(),
                  code_commit=code["stdout"] if code["exit_code"] == 0 else None,
                  code_dirty=dirty["stdout"] if dirty["exit_code"] == 0 else None,
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  command=[sys.executable, *sys.argv], working_directory=str(Path.cwd()))
    _save_status(out / "generation_run.json", status)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, trust_remote_code=False)
        if not tokenizer.chat_template:
            raise ValueError("a real pinned chat template is required")
        stop_ids = sorted({value for value in (tokenizer.eos_token_id, tokenizer.pad_token_id) if value is not None})
        if not stop_ids:
            raise ValueError("explicit tokenizer EOS/stop IDs required")
        status["decoding"]["eos_token_id"] = stop_ids
        model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
                                                   torch_dtype="auto").to(device).eval()
        template_sha = hashlib.sha256(str(tokenizer.chat_template).encode()).hexdigest()
        rows = []
        for prompt in prompts:
            remaining = timeout_seconds - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("local generation deadline exceeded")
            messages = prompt.get("messages") or [{"role": "user", "content": prompt["prompt"]}]
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(device)
            if ids.shape[-1] > max_input_tokens:
                raise ValueError(f"prompt {prompt['prompt_id']} exceeds input limit; no silent truncation")
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            begin = time.monotonic()
            with torch.inference_mode():
                output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), do_sample=False,
                                        max_new_tokens=max_new_tokens, max_time=remaining,
                                        num_beams=1, min_new_tokens=0, repetition_penalty=1.05,
                                        eos_token_id=stop_ids, use_cache=True,
                                        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
            if time.monotonic() - started > timeout_seconds:
                raise TimeoutError("local generation deadline exceeded; partial answers are not scored")
            answer = output[0, ids.shape[-1]:]
            rows.append({"prompt_id": str(prompt["prompt_id"]), "prompt": prompt["prompt"], "response": tokenizer.decode(answer, skip_special_tokens=True),
                         "model_id": str(Path(model_path).resolve()), "model_revision": model_revision, "tokenizer_revision": tokenizer_revision,
                         "chat_template_sha256": template_sha, "source_split": prompt["source_split"], "synthetic": False,
                         "context_sha256": digest(messages),
                         "size_class": size_class, "decoding": status["decoding"], "task_type": prompt.get("task_type", "unknown"),
                         "prompt_tokens": int(ids.shape[-1]), "response_tokens": int(answer.shape[-1]),
                         "elapsed_seconds": elapsed, "ttft_seconds": None, "ttft_reason": "nonstreaming generation",
                         "source": {key: prompt.get(key) for key in ("dataset_id", "revision", "source_row_id", "prompt_hash",
                             "source_repository", "hf_source_revision", "hf_source_row_id")}})
            # Persist only fully finished answers. Partial files are not full evaluation results.
            with partial_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(rows[-1], ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            status["generation_count"] = len(rows)
            _save_status(out / "generation_run.json", status)
        validate_generations(rows)
        partial_path.replace(out / "generations.jsonl")
        status.update(status="completed", generation_count=len(rows), output_path=str(out / "generations.jsonl"),
                      complete_evaluation=True, partial_output_path=None,
                      device=str(next(model.parameters()).device), torch_version=torch.__version__,
                      hardware=torch.cuda.get_device_name(next(model.parameters()).device) if device.startswith("cuda") else platform.processor(),
                      concurrency=1, chat_template_sha256=template_sha)
    except BaseException as error:
        status.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        status["ended_at"] = datetime.now(timezone.utc).isoformat()
        _save_status(out / "generation_run.json", status)
    return status
