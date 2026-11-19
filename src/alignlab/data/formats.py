"""LLaMA-Factory ShareGPT/OpenAI-role mapping, including ranking data."""
import copy

from .schema import validate_messages


def to_llamafactory(row):
    result = {"messages": copy.deepcopy(row["messages"])}
    if row["kind"] == "dpo":
        result.update(chosen=copy.deepcopy(row["chosen"]), rejected=copy.deepcopy(row["rejected"]))
    return result


def from_llamafactory(row, kind):
    if kind not in {"sft", "dpo"}:
        raise ValueError("kind must be sft or dpo")
    validate_messages(row["messages"], final_role="user" if kind == "dpo" else "assistant")
    result = {"kind": kind, "messages": copy.deepcopy(row["messages"])}
    if kind == "dpo":
        for field in ("chosen", "rejected"):
            validate_messages(row["messages"] + [row[field]])
            result[field] = copy.deepcopy(row[field])
    return result


def dataset_info():
    result = {}
    for kind in ("sft", "dpo"):
        for split in ("train", "validation"):
            name = f"alignlab_{kind}_{split}"
            columns = {"messages": "messages"}
            if kind == "dpo":
                columns.update(chosen="chosen", rejected="rejected")
            result[name] = {"file_name": f"{kind}_{split}.json", "formatting": "sharegpt",
                            "columns": columns,
                            "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                                     "assistant_tag": "assistant", "system_tag": "system"}}
            if kind == "dpo":
                result[name]["ranking"] = True
    return result


def assert_roundtrip(row):
    decoded = from_llamafactory(to_llamafactory(row), row["kind"])
    for key in ("kind", "messages", "chosen", "rejected"):
        if key in row and row[key] != decoded[key]:
            raise ValueError(f"LLaMA-Factory roundtrip mismatch: {key}")
