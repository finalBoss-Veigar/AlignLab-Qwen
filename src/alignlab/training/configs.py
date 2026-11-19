"""Generate official YAML fields, validated against captured upstream AST definitions."""
import hashlib
import json
from pathlib import Path
import yaml

COMMIT = "ca75f1edf3cb50343ed1c98605141c3e22075b5f"
MODEL_REVISIONS = {"7B": "a09a35458c702b33eeacc393d103063234e8bc28", "1.5B": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"}


def schema():
    return json.loads(Path(__file__).with_name("upstream_schema.json").read_text(encoding="utf-8"))


def source_digest(content):
    """Ignore checkout CRLF/LF conversion only; preserve every other source byte."""
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def verify_sources(root):
    root = Path(root)
    evidence = schema()
    if evidence.get("source_hash_normalization") != "CRLF_to_LF_only":
        raise ValueError("Unknown upstream source hash normalization")
    missing = []
    for name, digest in evidence["files"].items():
        p = root / name
        if not p.exists() or source_digest(p.read_bytes()) != digest:
            missing.append(name)
    if missing:
        raise ValueError(f"Upstream source files absent or modified: {missing}")
    return True


def validate_config(config, mode="train"):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    fields = schema()["fields"]
    allowed = {k: v for k, v in fields.items() if not v["file"].endswith("evaluation_args.py")}
    if mode == "export":
        allowed = {k: v for k, v in allowed.items() if "training_args" not in v["file"]}
    unknown = sorted(set(config) - set(allowed))
    if unknown:
        raise ValueError(f"Unknown upstream fields: {unknown}")
    for key, value in config.items():
        annotation = fields[key]["type"]
        if value is None and "Optional" in annotation:
            continue
        if annotation in ("bool", "Optional[bool]") and not isinstance(value, bool):
            raise ValueError(f"{key} must be a YAML boolean")
        if annotation in ("int", "Optional[int]") and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{key} must be an integer")
        if annotation in ("float", "Optional[float]") and (not isinstance(value, (float, int)) or isinstance(value, bool)):
            raise ValueError(f"{key} must be numeric")
    if not config.get("model_name_or_path") or config.get("template") != "qwen":
        raise ValueError("Explicit model_name_or_path and qwen template required")
    if config.get("trust_remote_code", False):
        raise ValueError("Qwen2.5 does not require trust_remote_code")
    if mode == "export":
        if config.get("quantization_bit") or config.get("export_quantization_bit"):
            raise ValueError("Merge must load original high-precision base, never quantized weights")
        if not config.get("adapter_name_or_path") or not config.get("export_dir"):
            raise ValueError("Export needs adapter and destination")
    elif mode == "train":
        if config.get("stage") not in ("sft", "dpo") or config.get("finetuning_type") != "lora":
            raise ValueError("Only continued SFT/DPO LoRA is supported")
        if config.get("do_train") is not True:
            raise ValueError("Training configuration requires do_train: true")
        if config.get("packing") or config.get("train_on_prompt") or not config.get("ignore_pad_token_for_loss", True):
            raise ValueError("Packing disabled; prompt and padding labels must be masked")
        if config.get("quantization_bit") != 4 or config.get("quantization_type") != "nf4":
            raise ValueError("Candidate recipe requires 4-bit NF4")
        if bool(config.get("bf16")) == bool(config.get("fp16")):
            raise ValueError("Select exactly one precision: bf16 or fp16")
        for key in ("lora_rank", "lora_alpha", "cutoff_len", "per_device_train_batch_size", "gradient_accumulation_steps", "learning_rate"):
            if isinstance(config.get(key), bool) or not isinstance(config.get(key), (int, float)) or config[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if config.get("save_only_model") is not False or config.get("overwrite_output_dir"):
            raise ValueError("Full checkpoint state required and output overwrite forbidden")
        if not config.get("dataset") or not config.get("dataset_dir") or not config.get("output_dir"):
            raise ValueError("Dataset and output paths required")
        if config.get("stage") == "dpo":
            if config.get("pref_loss") != "sigmoid" or config.get("pref_beta", 0) <= 0:
                raise ValueError("Standard sigmoid DPO and positive beta required")
            if config.get("pref_ftx", 0) or config.get("dpo_label_smoothing", 0) or config.get("ld_alpha") is not None or config.get("pissa_init"):
                raise ValueError("Use unmodified standard DPO")
            if config.get("adapter_name_or_path") or not config.get("create_new_adapter"):
                raise ValueError("DPO requires fresh adapter over merged SFT base")
            if str(config["model_name_or_path"]).startswith("Qwen/") or config.get("ref_model"):
                raise ValueError("DPO reference must be merged SFT base; original Instruct or separate ref forbidden")
    else:
        raise ValueError("mode must be train or export")
    return {"status": "static_fields_passed", "framework_commit": COMMIT,
            "official_parser": "not_run", "gpu_execution": "not_validated",
            "field_evidence": {k: fields[k] for k in config}}


def make_train(size="7B", stage="sft"):
    c = dict(model_name_or_path=f"Qwen/Qwen2.5-{size}-Instruct", model_revision=MODEL_REVISIONS[size],
        trust_remote_code=False, quantization_bit=4, quantization_method="bnb", quantization_type="nf4",
        double_quantization=True, stage=stage, do_train=True, finetuning_type="lora", lora_rank=16,
        lora_alpha=32, lora_target="all", lora_dropout=0.0, template="qwen", cutoff_len=1024,
        dataset=f"alignlab_{stage}_train", eval_dataset=f"alignlab_{stage}_validation", dataset_dir="data/processed/production",
        packing=False, train_on_prompt=False, ignore_pad_token_for_loss=True, mask_history=False,
        preprocessing_num_workers=1, dataloader_num_workers=0, output_dir=f"runs/artifacts/qwen-{size.lower()}-{stage}",
        logging_steps=1, save_steps=100, save_total_limit=2, save_only_model=False, overwrite_output_dir=False,
        report_to="none", per_device_train_batch_size=1, gradient_accumulation_steps=16, learning_rate=1e-4,
        num_train_epochs=1.0, lr_scheduler_type="cosine", warmup_ratio=0.1, bf16=True, fp16=False,
        gradient_checkpointing=True, disable_gradient_checkpointing=False, eval_strategy="steps", eval_steps=100,
        per_device_eval_batch_size=1, seed=42, data_seed=42, resume_from_checkpoint=None)
    if size == "1.5B":
        c.update(max_steps=5, save_steps=5, eval_steps=5, cutoff_len=512, gradient_accumulation_steps=4)
    if stage == "dpo":
        c.update(model_name_or_path=f"runs/artifacts/qwen-{size.lower()}-sft-merged", create_new_adapter=True,
                 pref_loss="sigmoid", pref_beta=0.1, learning_rate=5e-6, dpo_label_smoothing=0.0)
        c.pop("model_revision")  # local merged files have manifest checksums, not upstream revision
    return c


def make_export(size="7B", stage="sft"):
    if stage not in ("sft", "dpo"):
        raise ValueError("Unknown export stage")
    c = dict(model_name_or_path=f"Qwen/Qwen2.5-{size}-Instruct", model_revision=MODEL_REVISIONS[size],
        adapter_name_or_path=f"runs/artifacts/qwen-{size.lower()}-{stage}", template="qwen", finetuning_type="lora",
        trust_remote_code=False, export_dir=f"runs/artifacts/qwen-{size.lower()}-{stage}-merged", export_size=5,
        export_device="cpu", export_legacy_format=False)
    if stage == "dpo":
        c["model_name_or_path"] = f"runs/artifacts/qwen-{size.lower()}-sft-merged"
        c.pop("model_revision")
    return c


def generate_configs(output_dir, project_root=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if project_root:
        verify_sources(project_root)
    result = {}
    for name, config, mode in [("sft_7b.yaml", make_train(), "train"),
            ("dpo_7b.yaml", make_train(stage="dpo"), "train"),
            ("debug_1_5b.yaml", make_train("1.5B"), "train"),
            ("dpo_1_5b.yaml", make_train("1.5B", "dpo"), "train"),
            ("export_7b.yaml", make_export(), "export"), ("export_1_5b.yaml", make_export("1.5B"), "export"),
            ("export_dpo_7b.yaml", make_export(stage="dpo"), "export"),
            ("export_dpo_1_5b.yaml", make_export("1.5B", "dpo"), "export")]:
        evidence = validate_config(config, mode)
        target = output_dir / name
        text = yaml.safe_dump(config, sort_keys=False, allow_unicode=True)
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"Refusing to overwrite edited config: {target}")
        target.write_text(text, encoding="utf-8")
        result[name] = evidence
    (output_dir / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
