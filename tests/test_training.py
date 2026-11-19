import json
from pathlib import Path
import pytest
from alignlab.training.configs import make_train, make_export, validate_config, generate_configs, verify_sources, schema
from alignlab.training.audit import validate_labels
from alignlab.training.runner import validate_resume, fingerprint, validate_training_data, digest_file


def test_all_configs_are_upstream_fields(tmp_path):
    result = generate_configs(tmp_path)
    assert len(result) == 8
    assert all(r["status"] == "static_fields_passed" and r["gpu_execution"] == "not_validated" for r in result.values())
    assert result["dpo_7b.yaml"]["field_evidence"]["pref_beta"]["file"].endswith("finetuning_args.py")


@pytest.mark.parametrize("change", [{"imaginary_cli_flag": 1}, {"packing": True}, {"train_on_prompt": True},
    {"ignore_pad_token_for_loss": False}, {"save_only_model": True}, {"bf16": True, "fp16": True},
    {"lora_rank": 0}, {"cutoff_len": 0}, {"bf16": "false"}, {"task": "mmlu"}])
def test_invalid_training_config_fails(change):
    config = make_train()
    config.update(change)
    with pytest.raises(ValueError):
        validate_config(config)


def test_no_quantized_merge_or_original_instruct_reference():
    c = make_export()
    c["quantization_bit"] = 4
    with pytest.raises(ValueError, match="high-precision"):
        validate_config(c, "export")
    c = make_train(stage="dpo")
    c["model_name_or_path"] = "Qwen/Qwen2.5-7B-Instruct"
    with pytest.raises(ValueError, match="merged SFT"):
        validate_config(c)
    dpo_export = make_export(stage="dpo")
    assert dpo_export["model_name_or_path"].endswith("sft-merged")
    assert dpo_export["adapter_name_or_path"].endswith("-dpo")
    assert dpo_export["export_dir"].endswith("dpo-merged")
    assert validate_config(dpo_export, "export")["status"] == "static_fields_passed"


def test_labels_assistant_only_and_padding():
    assert validate_labels([10, 11, 12, 0], [-100, -100, 12, -100], [1, 1, 1, 0], [0, 0, 1, 0]) == 1
    with pytest.raises(ValueError, match="No valid"):
        validate_labels([1], [-100])
    with pytest.raises(ValueError, match="Padding"):
        validate_labels([1, 0], [1, 0], [1, 0])
    with pytest.raises(ValueError, match="mask"):
        validate_labels([10, 11], [10, 11], expected_mask=[0, 1])


def test_complete_checkpoint_and_fingerprint(tmp_path):
    fp = fingerprint(make_train(), {"sha": "dataset-a"})
    with pytest.raises(ValueError, match="Incomplete"):
        validate_resume(tmp_path, fp)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth", "adapter_model.safetensors"):
        (tmp_path / name).write_text("fixture only")
    (tmp_path / "alignlab_fingerprint.json").write_text(json.dumps({"fingerprint": fp}))
    assert validate_resume(tmp_path, fp)
    with pytest.raises(ValueError, match="mismatch"):
        validate_resume(tmp_path, fingerprint(make_train(), {"sha": "dataset-b"}))


def test_generate_preserves_user_changes(tmp_path):
    generate_configs(tmp_path)
    p = tmp_path / "sft_7b.yaml"
    p.write_text("my edit")
    with pytest.raises(FileExistsError):
        generate_configs(tmp_path)
    assert p.read_text() == "my edit"


def test_upstream_schema_source_hashes():
    assert verify_sources(Path(__file__).resolve().parents[1])


def test_source_hashes_survive_crlf_lf_checkout_but_detect_edits(tmp_path):
    root = Path(__file__).resolve().parents[1]
    source_files = list(schema()["files"])
    for i, name in enumerate(source_files):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = (root / name).read_bytes().replace(b"\r\n", b"\n")
        target.write_bytes(content.replace(b"\n", b"\r\n") if i % 2 else content)
    assert verify_sources(tmp_path)
    changed = tmp_path / source_files[0]
    changed.write_bytes(changed.read_bytes() + b"# changed source\n")
    with pytest.raises(ValueError, match="modified"):
        verify_sources(tmp_path)


def test_training_data_fingerprint_and_revision(tmp_path):
    config = make_train()
    for name in ("dataset_info.json", "sft_train.json", "sft_validation.json"):
        (tmp_path / name).write_text("[]")
    manifest = {"synthetic": False, "tokenizer": {"synthetic": False, "revision": config["model_revision"]},
        "settings": {"max_tokens": config["cutoff_len"]}, "leakage_check": "passed", "evaluation_rows": 2,
        "counts": {"sft": {"sampled_train": 1, "validation": 1}},
        "artifacts": {p.name: {"sha256": digest_file(p)} for p in tmp_path.iterdir()}}
    (tmp_path / "data_manifest.json").write_text(json.dumps(manifest))
    assert validate_training_data(tmp_path, config)["synthetic"] is False
    (tmp_path / "sft_train.json").write_text("changed")
    with pytest.raises(ValueError, match="modified"):
        validate_training_data(tmp_path, config)


def test_training_data_rejects_synthetic(tmp_path):
    (tmp_path / "data_manifest.json").write_text(json.dumps({"synthetic": True}))
    with pytest.raises(ValueError, match="non-synthetic"):
        validate_training_data(tmp_path, make_train())
