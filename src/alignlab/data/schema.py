"""Strict internal schemas and adapters for the documented HF source fields."""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata


class DataError(ValueError):
    """An invalid record, with a machine-readable reason."""


def normalize(text: str) -> str:
    """For duplicate keys only. Original message contents remain untouched."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_hash(text: str) -> str:
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def validate_messages(messages: list[dict], *, final_role="assistant") -> None:
    if not isinstance(messages, list) or not messages:
        raise DataError("empty_messages")
    index = 0
    if isinstance(messages[0], dict) and messages[0].get("role") == "system":
        index = 1
    if len(messages) <= index:
        raise DataError("missing_user_turn")
    expected = "user"
    for i, message in enumerate(messages):
        if not isinstance(message, dict):
            raise DataError("invalid_message")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise DataError("empty_content")
        if "\x00" in content:
            raise DataError("nul_content")
        if i < index:
            continue
        if message.get("role") != expected:
            raise DataError("illegal_role_order")
        expected = "assistant" if expected == "user" else "user"
    if messages[-1].get("role") != final_role:
        raise DataError("invalid_final_role")


def first_prompt(messages: list[dict]) -> str:
    return next(message["content"] for message in messages if message["role"] == "user")


def provenance(row: dict, row_index: int, *, dataset_id: str, revision: str,
               source_split: str, source_file: str, synthetic: bool) -> dict:
    metadata = copy.deepcopy(row.get("provenance", {}))
    if row.get("prompt_id") is not None:
        metadata.setdefault("source_prompt_id", str(row["prompt_id"]))
    for key, default in {"dataset_id": dataset_id, "revision": revision,
                         "source_split": source_split, "source_row_id": str(row_index),
                         "source_file": source_file, "synthetic": synthetic}.items():
        metadata.setdefault(key, default)
    if not all(isinstance(metadata.get(k), str) and metadata[k] for k in
               ("dataset_id", "revision", "source_split", "source_row_id", "source_file")):
        raise DataError("missing_provenance")
    if not synthetic and (not re.fullmatch(r"[0-9a-f]{40}", metadata["revision"])
                          or metadata["synthetic"] is not False):
        raise DataError("unpinned_or_synthetic_provenance")
    return metadata


def sft_record(row: dict, metadata: dict) -> dict:
    messages = copy.deepcopy(row.get("messages"))
    validate_messages(messages)
    prompt = first_prompt(messages)
    if "prompt" in row and row["prompt"] != prompt:
        raise DataError("prompt_context_mismatch")
    return {"kind": "sft", "messages": messages, "prompt": prompt,
            "prompt_hash": prompt_hash(prompt), "provenance": metadata,
            "cleaning_reasons": [], "task_type": row.get("task_type", "unknown")}


def dpo_record(row: dict, metadata: dict) -> dict:
    chosen, rejected = copy.deepcopy(row.get("chosen")), copy.deepcopy(row.get("rejected"))
    # HF UltraFeedback stores two complete message lists. Internal format stores
    # the shared context exactly once and two assistant message objects.
    if isinstance(chosen, list) and isinstance(rejected, list):
        validate_messages(chosen)
        validate_messages(rejected)
        if chosen[:-1] != rejected[:-1]:
            raise DataError("preference_context_mismatch")
        context, chosen, rejected = chosen[:-1], chosen[-1], rejected[-1]
        if row.get("messages") is not None and row["messages"] != context + [chosen]:
            raise DataError("source_messages_mismatch")
    elif isinstance(chosen, dict) and isinstance(rejected, dict):
        context = copy.deepcopy(row.get("messages"))
    else:
        raise DataError("invalid_preference_shape")
    validate_messages(context, final_role="user")
    for answer in (chosen, rejected):
        if answer.get("role") != "assistant":
            raise DataError("preference_not_assistant")
        validate_messages(context + [answer])
    if normalize(chosen["content"]) == normalize(rejected["content"]):
        raise DataError("identical_preference")
    prompt = first_prompt(context)
    if "prompt" in row and row["prompt"] != prompt:
        raise DataError("prompt_context_mismatch")
    return {"kind": "dpo", "messages": context, "chosen": chosen, "rejected": rejected,
            "prompt": prompt, "prompt_hash": prompt_hash(prompt), "provenance": metadata,
            "cleaning_reasons": [], "task_type": row.get("task_type", "unknown")}


def record_fingerprint(row: dict) -> str:
    payload = {k: row[k] for k in ("kind", "messages", "chosen", "rejected") if k in row}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
