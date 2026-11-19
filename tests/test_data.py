import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from alignlab.data.dedup import duplicate_groups, ngrams
from alignlab.data.formats import assert_roundtrip, dataset_info, to_llamafactory
from alignlab.data.pipeline import file_hash, read_rows, run_fixture_pipeline, run_pipeline
from alignlab.data.schema import DataError, dpo_record, prompt_hash, provenance, sft_record, validate_messages
from alignlab.data.sources import download_sources
from alignlab.data.split import assign_splits
from alignlab.data.tokens import FixtureTokenizer, fit_record, fixture_token_labels


ROOT = Path(__file__).resolve().parents[1]


def message(role, content):
    return {"role": role, "content": content}


def sft(prompt="Hello world", answer="A useful response"):
    return sft_record({"messages": [message("user", prompt), message("assistant", answer)]}, {})


def preference(prompt="Explain rain"):
    return {"chosen": [message("user", prompt), message("assistant", "Water drops fall from clouds.")],
            "rejected": [message("user", prompt), message("assistant", "Rain is dry sand.")]}


@pytest.mark.parametrize("messages", [[], [message("assistant", "wrong first role")],
    [message("user", "question"), message("user", "question")],
    [message("user", "question"), message("assistant", " ")],
    [message("tool", "unsupported"), message("assistant", "answer")]])
def test_reject_invalid_conversations(messages):
    with pytest.raises(DataError):
        validate_messages(messages)


def test_normalization_does_not_rewrite_body():
    prompt = "  HELLO\tWorld "
    row = sft(prompt)
    assert row["messages"][0]["content"] == prompt
    assert row["prompt_hash"] == prompt_hash("hello world")


def test_preference_context_and_different_answers():
    row = preference()
    row["rejected"][0]["content"] = "Different context"
    with pytest.raises(DataError, match="context_mismatch"):
        dpo_record(row, {})
    row = preference()
    row["rejected"] = copy.deepcopy(row["chosen"])
    with pytest.raises(DataError, match="identical_preference"):
        dpo_record(row, {})


def test_dpo_does_not_duplicate_prompt_and_roundtrip():
    row = dpo_record(preference(), {})
    assert len(row["messages"]) == 1
    assert row["chosen"]["role"] == "assistant"
    assert_roundtrip(row)
    mapped = to_llamafactory(row)
    assert mapped["messages"] == [message("user", "Explain rain")]
    registry = dataset_info()["alignlab_dpo_train"]
    assert registry["ranking"] is True
    assert registry["formatting"] == "sharegpt"
    assert registry["columns"]["chosen"] == "chosen"
    assert registry["columns"]["rejected"] == "rejected"
    assert registry["tags"]["assistant_tag"] == "assistant"


def test_group_split_shared_by_sft_and_dpo():
    rows = [sft("A unique first question"), sft("A separate mathematical question"),
            dpo_record(preference("A unique first question"), {})]
    duplicate_groups(rows)
    assign_splits(rows)
    assert rows[0]["group_id"] == rows[2]["group_id"]
    assert rows[0]["split"] == rows[2]["split"]


def test_prefix_index_has_no_false_negative_against_bruteforce():
    rng = random.Random(7)
    prompts = ["".join(rng.choices("abcdefgh ", k=35)) for _ in range(40)]
    prompts += [p[:-2] + "xx" for p in prompts[:10]]
    rows = [sft(p) for p in prompts]
    duplicate_groups(rows, threshold=0.65, n=3)
    for i, a in enumerate(prompts):
        for j, b in enumerate(prompts[:i]):
            x, y = ngrams(a, 3), ngrams(b, 3)
            if len(x & y) / len(x | y) >= 0.65:
                assert rows[i]["group_id"] == rows[j]["group_id"]


def test_truncation_preserves_turn_boundaries_and_answers():
    row = sft_record({"messages": [message("system", "Be helpful."),
        message("user", "old " * 50), message("assistant", "old answer " * 20),
        message("user", "Final question"), message("assistant", "Final answer")]}, {})
    fitted = fit_record(row, FixtureTokenizer(), 30)
    assert fitted["messages"] == [row["messages"][0], *row["messages"][-2:]]
    assert fitted["token_lengths"]["assistant_content_tokens"] > 0
    assert row["messages"][1]["content"] == "old " * 50
    validate_messages(fitted["messages"])


def test_dpo_context_truncated_identically():
    common = [message("system", "Be helpful"), message("user", "old " * 60),
              message("assistant", "earlier answer"), message("user", "Final question")]
    row = dpo_record({"chosen": common + [message("assistant", "Good answer")],
                      "rejected": common + [message("assistant", "Bad answer")]}, {})
    fitted = fit_record(row, FixtureTokenizer(), 25)
    assert fitted["messages"] == [common[0], common[-1]]
    assert fitted["chosen"] == row["chosen"] and fitted["rejected"] == row["rejected"]


def test_overlong_final_response_rejected_not_silently_truncated():
    with pytest.raises(DataError, match="overlong_final_turn"):
        fit_record(sft(answer="word " * 100), FixtureTokenizer(), 15)


def test_fixture_assistant_labels_and_padding():
    messages = [message("system", "system"), message("user", "user"), message("assistant", "answer")]
    batch = fixture_token_labels(messages, pad_to=15)
    assert batch["labels"][:7] == [-100] * 7  # system, user, assistant role prefix
    assert batch["labels"][7] != -100
    assert batch["labels"][-6:] == [-100] * 6
    assert len(batch["labels"]) == len(batch["input_ids"]) == 15
    assert batch["synthetic"] is True


def test_fixture_pipeline_counts_and_isolation(tmp_path):
    raw = ROOT / "data" / "fixtures"
    before = {p.name: file_hash(p) for p in raw.glob("*.jsonl")}
    manifest = run_fixture_pipeline(ROOT, tmp_path / "processed")
    assert manifest["synthetic"] is True
    assert manifest["counts"]["sft"]["raw"] == 16
    assert manifest["counts"]["dpo"]["raw"] == 12
    assert manifest["counts"]["sft"]["evaluation_overlap"] == 1
    assert manifest["counts"]["dpo"]["evaluation_overlap"] == 1
    assert manifest["counts"]["sft"]["deduplicated"] == 2
    assert manifest["counts"]["sft"]["filtered"] == 3
    assert manifest["counts"]["dpo"]["filtered"] == 3
    assert manifest["counts"]["sft"]["final"] == 10
    assert manifest["counts"]["dpo"]["final"] == 8
    assignments = read_rows(tmp_path / "processed" / "split_assignments.jsonl")
    partitions = {}
    for row in assignments:
        partitions.setdefault(row["group_id"], set()).add(row["split"])
    assert all(len(values) == 1 for values in partitions.values())
    test_groups = {r["group_id"] for r in read_rows(tmp_path / "processed" / "evaluation_exclusions.jsonl")}
    assert not (set(partitions) & test_groups)
    assert {p.name: file_hash(p) for p in raw.glob("*.jsonl")} == before
    assert "SYNTHETIC FIXTURE" in (tmp_path / "processed" / "data_report.html").read_text(encoding="utf-8")
    for kind in ("sft", "dpo"):
        counts = manifest["counts"][kind]
        assert counts["raw"] == counts["filtered"] + counts["evaluation_overlap"] + counts["deduplicated"] + counts["after_deduplication"]
        assert counts["final"] == counts["sampled_train"] + counts["validation"]


def test_deterministic_outputs_and_no_overwrite(tmp_path):
    one = run_fixture_pipeline(ROOT, tmp_path / "one")
    two = run_fixture_pipeline(ROOT, tmp_path / "two")
    assert one["counts"] == two["counts"]
    for name in ("sft_train.json", "dpo_train.json", "split_assignments.jsonl"):
        assert file_hash(tmp_path / "one" / name) == file_hash(tmp_path / "two" / name)
    with pytest.raises(FileExistsError):
        run_fixture_pipeline(ROOT, tmp_path / "one")


def test_sampling_does_not_inflate_or_duplicate(tmp_path):
    raw = ROOT / "data" / "fixtures"
    manifest = run_pipeline(raw / "sft.jsonl", raw / "dpo.jsonl", raw / "evaluation.jsonl",
                            tmp_path / "tiny", sft_budget=2, dpo_budget=1, synthetic=True)
    assert manifest["counts"]["sft"]["sampled_train"] == 2
    assert manifest["counts"]["dpo"]["sampled_train"] == 1


def test_formal_pipeline_refuses_fixture_tokenizer(tmp_path):
    raw = ROOT / "data" / "fixtures"
    with pytest.raises(ValueError, match="actual pinned"):
        run_pipeline(raw / "sft.jsonl", raw / "dpo.jsonl", raw / "evaluation.jsonl", tmp_path / "formal")


def test_download_is_never_implicit(tmp_path):
    with pytest.raises(ValueError, match="explicit"):
        download_sources(tmp_path / "raw")
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize('separator', ['\u2028', '\u2029', '\u0085'])
def test_jsonl_preserves_unicode_separators_inside_strings(tmp_path, separator):
    rows = [{'prompt': 'before' + separator + 'after'}, {'prompt': 'second physical line'}]
    path = tmp_path / 'unicode.jsonl'
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    assert read_rows(path) == rows


def test_download_export_contract_with_synthetic_transport(monkeypatch, tmp_path):
    """Simulated network rows only; never fetch actual datasets in unit tests."""
    def fake_load(dataset_id, *, split, revision, streaming):
        if dataset_id == "google/IFEval":
            return [{"key": 17, "prompt": "Exactly two words please", "instruction_id_list": ["length_constraints:number_words"],
                     "kwargs": [{"num_words": 2, "relation": "equal to"}]}]
        prompt = f"Question from {dataset_id} split {split}"
        if dataset_id.endswith("ultrachat_200k"):
            return [{"prompt": prompt, "messages": [message("user", prompt), message("assistant", "Good answer")]}]
        return [{"prompt": prompt, **preference(prompt)}]
    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load))
    manifest = download_sources(tmp_path / "raw", allow_download=True)
    from alignlab.data.pipeline import _eval_rows
    exclusions = _eval_rows(tmp_path / "raw" / "evaluation.jsonl", synthetic=False)
    assert len(exclusions) == 6
    prompts = read_rows(tmp_path / "raw" / "eval_prompts" / "ifeval.jsonl")
    assert prompts[0]["synthetic"] is False
    assert prompts[0]["prompt_id"] == "ifeval:0"
    assert prompts[0]["key"] == 17
    assert prompts[0]["kwargs"][0]["num_words"] == 2
    official = read_rows(tmp_path / "raw" / "eval_prompts" / "ifeval_input.jsonl")
    assert set(official[0]) == {"key", "prompt", "instruction_id_list", "kwargs"}
    prefs = read_rows(tmp_path / "raw" / "eval_prompts" / "test_prefs.jsonl")
    assert len(prefs[0]["messages"]) == 1
    assert manifest["model_weights_downloaded"] is False


@pytest.mark.parametrize("revision", ["main", "master", "unknown", "synthetic-v1", "not-a-commit"])
def test_formal_provenance_requires_commit_sha(revision):
    with pytest.raises(DataError, match="unpinned_or_synthetic"):
        provenance({}, 0, dataset_id="HuggingFaceH4/ultrachat_200k", revision=revision,
                   source_split="train_sft", source_file="test.jsonl", synthetic=False)


def test_evaluation_overlap_transitive_and_old_user_turns():
    base = "A long and distinctive question about measuring mountain glaciers in winter"
    rows = [sft(base), sft(base + "?"), sft("An unrelated last user question")]
    rows[2]["original_context_hash_input"] = [base + "!", "An unrelated last user question"]
    duplicate_groups(rows, threshold=0.85)
    assert len({r["group_id"] for r in rows}) == 1
