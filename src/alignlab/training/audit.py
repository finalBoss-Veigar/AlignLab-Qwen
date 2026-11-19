"""Runtime audits using the pinned official processors and DPO trainer (optional CUDA)."""
import hashlib
import json
from pathlib import Path
import sys
import yaml


def validate_labels(input_ids, labels, attention_mask=None, expected_mask=None):
    if len(input_ids) != len(labels) or not input_ids:
        raise ValueError("input/label lengths inconsistent")
    if not any(label != -100 for label in labels):
        raise ValueError("No valid assistant supervision token")
    for i, (token, label) in enumerate(zip(input_ids, labels)):
        if label != -100 and label != token:
            raise ValueError("Supervised token differs from input token")
        if attention_mask is not None and not attention_mask[i] and label != -100:
            raise ValueError("Padding must be masked")
        if expected_mask is not None and bool(label != -100) != bool(expected_mask[i]):
            raise ValueError("User/system or assistant mask is incorrect")
    return sum(x != -100 for x in labels)


def _frozen_hash(model):
    import torch
    h = hashlib.sha256()
    count = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            if "lora_" not in name:
                raise ValueError(f"Unexpected trainable base parameter {name}")
        else:
            h.update(name.encode())
            h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
            count += p.numel()
    return h.hexdigest(), count


def audit_reference(trainer, batch, learning_rate=5e-6):
    """A discarded one-step official loss probe; never replaces the production trainer."""
    import torch
    model = trainer.model
    model.eval()  # deterministic comparison; official trainer also disables dropout
    before, count = _frozen_hash(model)
    with torch.no_grad():
        policy = trainer.concatenated_forward(model, batch)
        chosen, rejected = policy[:2]
        del policy  # release full-vocabulary logits before the gradient probe
        ref_chosen, ref_rejected = trainer.compute_reference_log_probs(model, batch)
    for a, b in ((chosen, ref_chosen), (rejected, ref_rejected)):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable DPO adapter")
    model.train()  # enable model gradient checkpointing; official trainer disabled dropout
    optimizer = torch.optim.AdamW(params, lr=learning_rate)
    loss, diagnostics = trainer.get_batch_loss_metrics(model, batch)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite official DPO loss")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after, _ = _frozen_hash(model)
    if before != after:
        raise ValueError("Frozen SFT reference parameters changed")
    model.eval()
    after_chosen, after_rejected = trainer.compute_reference_log_probs(model, batch)
    torch.testing.assert_close(ref_chosen, after_chosen, rtol=0, atol=0)
    torch.testing.assert_close(ref_rejected, after_rejected, rtol=0, atol=0)
    return {"initial_policy_reference_logps_equal": True, "frozen_parameters_unchanged": True,
            "frozen_parameter_elements": count, "frozen_sha256": before, "probe_loss": loss.item(),
            "chosen_logps": ref_chosen.tolist(), "rejected_logps": ref_rejected.tolist(),
            "probe_discarded_before_training": True, "diagnostics_not_generation_win_rate": diagnostics}


def run_audit(config, output):
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer, load_model
    from llamafactory.data import get_template_and_fix_tokenizer, get_dataset, PairwiseDataCollatorWithPadding
    from llamafactory.data.processor.supervised import SupervisedDatasetProcessor
    from transformers import DataCollatorForSeq2Seq
    import torch
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    probe_config = dict(config, max_samples=2, output_dir=str(output / "scratch"), eval_strategy="no")
    probe_config.pop("eval_dataset", None)
    probe_config.pop("resume_from_checkpoint", None)
    model_args, data_args, training_args, ft_args, _ = get_train_args(probe_config)
    modules = load_tokenizer(model_args)
    tokenizer = modules["tokenizer"]
    template = get_template_and_fix_tokenizer(tokenizer, data_args)
    processor = SupervisedDatasetProcessor(template=template, tokenizer=tokenizer,
        processor=modules["processor"], data_args=data_args)
    prompt = [{"role": "user", "content": "Return only the word blue."}]
    response = [{"role": "assistant", "content": "blue"}]
    ids, labels = processor._encode_data_example(prompt, response, "You are a concise assistant.", None, [], [], [])
    # Qwen uses non-efficient EOS: source (system/user/header) fully ignored.
    pairs = template.encode_multiturn(tokenizer, prompt + response, "You are a concise assistant.", None)
    expected = []
    for source, target in pairs:
        expected.extend([False] * len(source) + [True] * len(target))
    if len(expected) != len(ids):
        raise ValueError("Audit control unexpectedly truncated; increase cutoff_len")
    validate_labels(ids, labels, expected_mask=expected)
    padded = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)(
        [{"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}])
    validate_labels(padded["input_ids"][0].tolist(), padded["labels"][0].tolist(), padded["attention_mask"][0].tolist())
    result = {"status": "runtime_passed", "tokenizer": config["model_name_or_path"],
        "control_prompt_is_synthetic": True, "token_ids_are_real_tokenizer": True,
        "input_ids": ids, "labels": labels, "decoded_supervision": tokenizer.decode([t for t in labels if t != -100]),
        "prompt_system_padding_mask_verified": True}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    dataset = get_dataset(template, model_args, data_args, training_args,
        stage="rm" if config["stage"] == "dpo" else "sft", **modules)
    item = dataset["train_dataset"][0]
    result["actual_first_training_sample"] = item
    if config["stage"] == "sft":
        validate_labels(item["input_ids"], item["labels"], item["attention_mask"])
    else:
        for prefix in ("chosen", "rejected"):
            validate_labels(item[f"{prefix}_input_ids"], item[f"{prefix}_labels"], item[f"{prefix}_attention_mask"])
        chosen_response = [t for t in item["chosen_labels"] if t != -100]
        rejected_response = [t for t in item["rejected_labels"] if t != -100]
        if chosen_response == rejected_response:
            raise ValueError("Preference difference lost after official tokenization/truncation")
        from llamafactory.train.dpo.trainer import CustomDPOTrainer
        from llamafactory.train.trainer_utils import create_ref_model
        model = load_model(tokenizer, model_args, ft_args, is_trainable=True)
        collator = PairwiseDataCollatorWithPadding(template=template, model=model,
            pad_to_multiple_of=8, label_pad_token_id=-100, **modules)
        ref = create_ref_model(model_args, ft_args)
        if ref is not None:
            raise ValueError("This recipe must use adapter-off merged SFT reference")
        trainer = CustomDPOTrainer(model=model, ref_model=ref, finetuning_args=ft_args,
            args=training_args, data_collator=collator, **dataset, **modules)
        batch = trainer._prepare_inputs(collator([item]))
        result["dpo_reference"] = audit_reference(trainer, batch, config["learning_rate"])
    result["audit_peak_vram_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_initialized() else None
    (output / "runtime_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    run_audit(yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")), sys.argv[2])
