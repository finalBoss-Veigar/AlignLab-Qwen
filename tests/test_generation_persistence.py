"""Fault injection for persistence only; no model performance is measured here."""
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from alignlab.evaluation.generate import generate_local
from alignlab.evaluation.io import read_jsonl, write_jsonl


class FakeIds:
    shape = (1, 2)

    def to(self, device):
        return self


class FakeOutput:
    def __getitem__(self, key):
        return SimpleNamespace(shape=(3,))


@pytest.mark.parametrize("interrupt", [None, RuntimeError("injected failure"), KeyboardInterrupt()])
def test_completed_answers_survive_later_failure_without_claiming_completion(tmp_path, interrupt):
    model_path = tmp_path / "mock-model"
    model_path.mkdir()
    prompts_path = tmp_path / "prompts.jsonl"
    write_jsonl(prompts_path, [
        {"prompt_id": str(i), "prompt": f"Fault injection input {i}", "dataset_id": "google/IFEval",
         "revision": "test-only", "source_split": "train", "source_row_id": i, "synthetic": False}
        for i in range(2)
    ])
    tokenizer = Mock(chat_template="mock template", eos_token_id=2, pad_token_id=0)
    tokenizer.apply_chat_template.return_value = FakeIds()
    tokenizer.decode.return_value = "Mock answer, never a measured result"
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    model.parameters.return_value = iter([SimpleNamespace(device="cpu")])
    model.generate.side_effect = [FakeOutput(), interrupt if interrupt is not None else FakeOutput()]
    torch = SimpleNamespace(inference_mode=contextlib.nullcontext, ones_like=lambda x: x, __version__="mock")
    transformers = SimpleNamespace(AutoTokenizer=Mock(), AutoModelForCausalLM=Mock())
    transformers.AutoTokenizer.from_pretrained.return_value = tokenizer
    transformers.AutoModelForCausalLM.from_pretrained.return_value = model
    out = tmp_path / "output"
    with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
        options = dict(model_revision="mock", tokenizer_revision="mock", size_class="7B", dry_run=False)
        if interrupt is not None:
            with pytest.raises(type(interrupt)):
                generate_local(model_path, model_path, prompts_path, out, **options)
        else:
            generate_local(model_path, model_path, prompts_path, out, **options)
    status = json.loads((out / "generation_run.json").read_text())
    assert status["started_at"] <= status["ended_at"]
    assert len(status["generator_sha256"]) == 64
    assert "code_commit" in status and "code_dirty" in status
    assert status["command"] and status["working_directory"]
    assert status["decoding"]["eos_token_id"] == [0, 2]
    assert model.generate.call_args.kwargs["repetition_penalty"] == 1.05
    assert model.generate.call_args.kwargs["use_cache"] is True
    if interrupt is not None:
        assert status["status"] == ("interrupted" if isinstance(interrupt, KeyboardInterrupt) else "failed")
        assert status["generation_count"] == 1
        assert status["complete_evaluation"] is False
        assert not (out / "generations.jsonl").exists()
        assert [r["prompt_id"] for r in read_jsonl(out / "generations.partial.jsonl")] == ["0"]
        with pytest.raises(FileExistsError):
            generate_local(model_path, model_path, prompts_path, out, **options)
    else:
        assert status["status"] == "completed"
        assert status["generation_count"] == 2
        assert status["complete_evaluation"] is True
        assert not (out / "generations.partial.jsonl").exists()
        assert [r["prompt_id"] for r in read_jsonl(out / "generations.jsonl")] == ["0", "1"]
