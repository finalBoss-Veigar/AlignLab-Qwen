"""Subprocess wrapper preserving run records and full-checkpoint fingerprints."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml
from .configs import COMMIT, validate_config, verify_sources


def digest_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(config, data_manifest):
    stable = {k: v for k, v in config.items() if k != "resume_from_checkpoint"}
    return hashlib.sha256(json.dumps({"config": stable, "data": data_manifest}, sort_keys=True).encode()).hexdigest()


def validate_resume(checkpoint, expected_fingerprint):
    p = Path(checkpoint)
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "alignlab_fingerprint.json"]
    missing = [name for name in required if not (p / name).is_file()]
    if not list(p.glob("rng_state*.pth")):
        missing.append("rng_state*.pth")
    if not any((p / x).exists() for x in ("adapter_model.safetensors", "adapter_model.bin", "model.safetensors", "model.safetensors.index.json")):
        missing.append("model/adapter weights")
    if missing:
        raise ValueError(f"Incomplete training checkpoint: {missing}; adapter loading is not resume")
    saved = json.loads((p / "alignlab_fingerprint.json").read_text(encoding="utf-8"))
    if saved.get("fingerprint") != expected_fingerprint:
        raise ValueError("Resume data/config fingerprint mismatch")
    return True


def _path(root, value):
    p = Path(value)
    return p if p.is_absolute() else root / p


def validate_training_data(data_dir, config):
    data_dir = Path(data_dir)
    data = json.loads((data_dir / "data_manifest.json").read_text(encoding="utf-8"))
    tokenizer_meta = data.get("tokenizer", {})
    if data.get("synthetic") is not False or data.get("synthetic_fixture") or tokenizer_meta.get("synthetic") is not False:
        raise ValueError("Production training requires explicitly non-synthetic data and tokenizer")
    if data.get("settings", {}).get("max_tokens") != config["cutoff_len"]:
        raise ValueError("Data token cutoff differs from training cutoff; reprocess explicitly")
    if data.get("leakage_check") != "passed" or not data.get("evaluation_rows"):
        raise ValueError("Evaluation isolation evidence missing")
    if config.get("model_revision") and tokenizer_meta.get("revision") != config["model_revision"]:
        raise ValueError("Tokenizer and model revision mismatch")
    artifacts = data.get("artifacts", {})
    required = ["dataset_info.json", f"{config['stage']}_train.json", f"{config['stage']}_validation.json"]
    if any(name not in artifacts for name in required):
        raise ValueError("Data artifact checksum manifest incomplete")
    for name, info in artifacts.items():
        path = (data_dir / name).resolve()
        if not path.is_relative_to(data_dir.resolve()) or digest_file(path) != info["sha256"]:
            raise ValueError("Data artifact modified since preprocessing")
    counts = data.get("counts", {}).get(config["stage"], {})
    if not counts.get("sampled_train") or not counts.get("validation"):
        raise ValueError("Nonempty training and validation samples required")
    return data


def _run(config_path, project_root, dry_run, mode, resume=None):
    from alignlab.tracking import Run
    root = Path(project_root).resolve()
    source = _path(root, config_path)
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    with Run(mode, config, output_root=root / "runs" / ("checks" if dry_run else "experiments")) as run:
        proof = validate_config(config, "export" if mode == "export" else "train")
        verify_sources(root)
        if dry_run:
            return run.finish("dry_run_passed", checks=proof, training_completed=False,
                reason="Static source-field/recipe checks only; GPU parser, data, tokenizer, weights and memory not validated")
        if sys.platform != "linux":
            raise RuntimeError("Real training/export recipe targets Linux NVIDIA; use dry-run here")
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("llamafactory") is None:
            raise RuntimeError("Separate CUDA training environment required")
        for package, expected in {"transformers": "4.51.3", "peft": "0.15.2", "trl": "0.9.6"}.items():
            if importlib.metadata.version(package) != expected:
                raise ValueError(f"{package} must match audited constraint {expected}")
        import torch
        if mode == "train" and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if mode == "train" and torch.cuda.device_count() != 1:
            raise ValueError("Select exactly one GPU with CUDA_VISIBLE_DEVICES for this single-card recipe")
        if int(env_world := os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError(f"This launcher validates single-process recipes only, WORLD_SIZE={env_world}")
        if mode == "train" and config.get("bf16") and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 unsupported; explicitly edit config to fp16 and record new candidate")
        checkout = root / "vendor/LlamaFactory"
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=checkout, text=True).strip()
        if actual != COMMIT:
            raise ValueError("Framework checkout differs from pin")
        import llamafactory
        if not Path(llamafactory.__file__).resolve().is_relative_to(checkout.resolve()):
            raise ValueError("Install pinned vendor checkout editable in training environment")
        env = os.environ.copy()
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        # Offline prevents launch wrappers from unexpectedly downloading multi-GB model files.
        if mode == "train":
            data_dir = _path(root, config["dataset_dir"])
            data = validate_training_data(data_dir, config)
            fp = fingerprint(config, data)
            output = _path(root, config["output_dir"])
            if resume and config.get("resume_from_checkpoint") and str(resume) != str(config["resume_from_checkpoint"]):
                raise ValueError("CLI and YAML resume checkpoints conflict")
            resume = resume or config.get("resume_from_checkpoint")
            if resume:
                validate_resume(_path(root, resume), fp)
                config["resume_from_checkpoint"] = str(_path(root, resume))
            elif output.exists() and any(output.iterdir()):
                raise ValueError("Output nonempty; provide a complete validated checkpoint or new output_dir")
            if config["stage"] == "dpo":
                merged = _path(root, config["model_name_or_path"])
                provenance = json.loads((merged / "alignlab_export_manifest.json").read_text(encoding="utf-8"))
                if provenance.get("stage") != "sft" or provenance.get("quantized_merge") is not False:
                    raise ValueError("DPO base must be audited nonquantized SFT export")
                if data["tokenizer"].get("revision") != provenance["training"].get("model_revision"):
                    raise ValueError("DPO data tokenizer revision differs from SFT endpoint tokenizer")
                for name, digest in provenance["files"].items():
                    if digest_file(merged / name) != digest:
                        raise ValueError("Merged SFT model checksum mismatch")
                run.value["sft_endpoint"] = provenance
            run.value.update(data_revision=data, effective_batch_size=config["per_device_train_batch_size"] * config["gradient_accumulation_steps"] * int(env.get("WORLD_SIZE", "1")), gpu_count=torch.cuda.device_count(), world_size=int(env.get("WORLD_SIZE", "1")))
        else:
            adapter = _path(root, config["adapter_name_or_path"])
            if not (adapter / "adapter_config.json").exists():
                raise ValueError("SFT adapter absent")
            parent = json.loads((adapter / "alignlab_training_manifest.json").read_text(encoding="utf-8"))
            if parent.get("stage") not in ("sft", "dpo") or parent.get("status") != "completed":
                raise ValueError("Export requires completed SFT/DPO provenance")
            original_config = parent["config"]
            if any(original_config.get(k) != config.get(k) for k in ("model_name_or_path", "model_revision")):
                raise ValueError("Export base differs from adapter's recorded training base/revision")
            if parent["stage"] == "dpo":
                base = _path(root, config["model_name_or_path"])
                sft_base = json.loads((base / "alignlab_export_manifest.json").read_text(encoding="utf-8"))
                if sft_base.get("stage") != "sft" or sft_base.get("quantized_merge") is not False:
                    raise ValueError("DPO export must merge into high-precision SFT endpoint")
                for name, checksum in sft_base["files"].items():
                    if digest_file(base / name) != checksum:
                        raise ValueError("SFT base modified since export")
            output = _path(root, config["export_dir"])
            if output.exists() and any(output.iterdir()):
                raise ValueError("Export destination nonempty")
            fp = None
        runtime_config = run.path / "actual_config.yaml"
        runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        run.value.update(config=config, framework_commit=COMMIT)
        run.save()
        cmd = [sys.executable, "-m", "alignlab.training.worker", mode, str(runtime_config), str(run.path)]
        if fp:
            cmd += [fp]
        with (run.path / "process.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        telemetry_path = run.path / "telemetry.json"
        if telemetry_path.exists():
            run.value.update(json.loads(telemetry_path.read_text(encoding="utf-8")))
        if result.returncode:
            run.finish("failed", exit_code=result.returncode, log_file=str(run.path / "process.log"))
            raise RuntimeError(f"{mode} failed ({result.returncode}); logs preserved: {run.path}")
        if mode == "export":
            metadata = {"stage": parent["stage"], "quantized_merge": False, "framework_commit": COMMIT,
                "source_config": config, "training": parent, "files": {str(p.relative_to(output)): digest_file(p) for p in output.rglob("*") if p.is_file()}}
            (output / "alignlab_export_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        else:
            state = json.loads((output / "trainer_state.json").read_text(encoding="utf-8"))
            (output / "alignlab_training_manifest.json").write_text(json.dumps({"stage": config["stage"], "status": "completed", "fingerprint": fp, "run_id": run.id, "model_revision": config.get("model_revision"), "config": config}, indent=2), encoding="utf-8")
            run.value["training_steps"] = state.get("global_step")
        return run.finish("completed", training_completed=mode == "train", metrics_file=str(output / "train_results.json") if mode == "train" else None, artifacts=[str(output)], log_file=str(run.path / "process.log"))


def run_training(config_path, project_root, dry_run=True, resume=None):
    return _run(config_path, project_root, dry_run, "train", resume)


def run_export(config_path, project_root, dry_run=True):
    return _run(config_path, project_root, dry_run, "export")
