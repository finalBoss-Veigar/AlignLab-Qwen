"""Token lengths and turn-boundary truncation; no character-count token estimates."""
from __future__ import annotations

import copy
import re

from .schema import DataError


class FixtureTokenizer:
    """Synthetic development tokenizer, NEVER evidence about Qwen token lengths."""

    name_or_path = "synthetic-fixture-tokenizer-v1"
    revision = "synthetic-v1"
    synthetic = True

    def encode(self, text, **kwargs):
        return [i + 1 for i, _ in enumerate(re.findall(r"\w+|[^\w\s]", text))]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        ids = []
        for message in messages:
            ids.extend([101])
            ids.extend(self.encode(message["content"]))
            ids.extend([102])
        return ids


def load_tokenizer(path: str, revision: str):
    """Load only an already-downloaded tokenizer; this call cannot fetch weights."""
    if not revision or revision in {"main", "master"}:
        raise ValueError("A pinned tokenizer revision is required")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, revision=revision, local_files_only=True,
                                              trust_remote_code=False)
    tokenizer.alignlab_revision = revision
    return tokenizer


def length(tokenizer, messages: list[dict]) -> int:
    return len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))


def _remove_oldest_turn(messages):
    start = 1 if messages and messages[0]["role"] == "system" else 0
    # Preserve at least the final user turn (and answer for SFT).
    if len(messages) - start < 3:
        return None
    return messages[:start] + messages[start + 2:]


def fit_record(row: dict, tokenizer, max_tokens: int) -> dict:
    if max_tokens < 8:
        raise ValueError("max_tokens must be >= 8")
    row = copy.deepcopy(row)
    original_messages = copy.deepcopy(row["messages"])
    def lengths():
        if row["kind"] == "sft":
            return {"sequence_tokens": length(tokenizer, row["messages"])}
        return {"chosen_sequence_tokens": length(tokenizer, row["messages"] + [row["chosen"]]),
                "rejected_sequence_tokens": length(tokenizer, row["messages"] + [row["rejected"]])}
    before = lengths()
    while max(lengths().values()) > max_tokens:
        shortened = _remove_oldest_turn(row["messages"])
        if shortened is None:
            raise DataError("overlong_final_turn")
        row["messages"] = shortened
        row["cleaning_reasons"].append("removed_oldest_complete_turn")
    answers = ([m for m in row["messages"] if m["role"] == "assistant"] if row["kind"] == "sft"
               else [row["chosen"], row["rejected"]])
    answer_counts = [len(tokenizer.encode(m["content"], add_special_tokens=False)) for m in answers]
    if not answer_counts or min(answer_counts) < 1:
        raise DataError("no_assistant_content_tokens")
    row["token_lengths"] = {**lengths(), "original": before,
                             "prompt_tokens": length(tokenizer, row["messages"][:-1]
                                                      if row["kind"] == "sft" else row["messages"])}
    if row["kind"] == "dpo":
        row["token_lengths"].update(chosen_response_tokens=answer_counts[0],
                                    rejected_response_tokens=answer_counts[1],
                                    response_length_delta=answer_counts[0] - answer_counts[1])
    else:
        row["token_lengths"]["assistant_content_tokens"] = sum(answer_counts)
    # original prompt remains the leakage key even after context truncation.
    row["original_context_hash_input"] = [m["content"] for m in original_messages if m["role"] == "user"]
    return row


def fixture_token_labels(messages, pad_to=None):
    """Development mask test only; real LF labels are inspected by training code."""
    tokenizer = FixtureTokenizer()
    ids, labels = [], []
    for message in messages:
        values = [101] + tokenizer.encode(message["content"]) + [102]
        ids.extend(values)
        labels.extend([-100] + values[1:] if message["role"] == "assistant" else [-100] * len(values))
    if pad_to is not None:
        if pad_to < len(ids):
            raise ValueError("Padding cannot truncate labels")
        padding = pad_to - len(ids)
        ids += [0] * padding
        labels += [-100] * padding
    return {"input_ids": ids, "labels": labels, "synthetic": True}
