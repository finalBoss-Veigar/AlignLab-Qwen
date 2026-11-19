"""Validate comparable generation records without loading models."""
from .io import digest

REQUIRED_TEXT = ("prompt_id", "prompt", "model_id", "model_revision", "tokenizer_revision", "chat_template_sha256", "source_split")
COMPARISON_FIELDS = ("tokenizer_revision", "chat_template_sha256", "decoding", "size_class")


def validate_generations(rows, *, allow_empty=False):
    if not isinstance(rows, list) or (not rows and not allow_empty):
        raise ValueError("generation records must be a nonempty list")
    seen = set()
    identities = set()
    prompt_by_id, id_by_prompt = {}, {}
    comparison_configs = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each generation row must be an object")
        for name in REQUIRED_TEXT:
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise ValueError(f"invalid generation field: {name}")
        if not isinstance(row.get("response"), str):
            raise ValueError("response must be a string (empty generated answers remain measurable)")
        if type(row.get("synthetic")) is not bool:
            raise ValueError("synthetic must be explicitly true or false")
        if not row["synthetic"] and not row.get("context_sha256"):
            raise ValueError("real generations must record context_sha256 for matched comparisons")
        if row.get("size_class") not in ("1.5B", "7B", "synthetic"):
            raise ValueError("size_class must identify 1.5B / 7B / synthetic")
        decoding = row.get("decoding")
        if not isinstance(decoding, dict) or not isinstance(decoding.get("max_new_tokens"), int) or decoding["max_new_tokens"] <= 0:
            raise ValueError("decoding must include a positive max_new_tokens")
        if "do_sample" not in decoding or type(decoding["do_sample"]) is not bool:
            raise ValueError("decoding must record do_sample")
        key = (row["prompt_id"], str(row.get("sample_id", "0")))
        if key in seen:
            raise ValueError("duplicate prompt/sample generation")
        seen.add(key)
        if row["prompt_id"] in prompt_by_id and prompt_by_id[row["prompt_id"]] != row["prompt"]:
            raise ValueError("one prompt_id cannot refer to different prompt text")
        if row["prompt"] in id_by_prompt and id_by_prompt[row["prompt"]] != row["prompt_id"]:
            raise ValueError("identical prompt text must use one prompt_id for cluster bootstrap")
        prompt_by_id[row["prompt_id"]], id_by_prompt[row["prompt"]] = row["prompt"], row["prompt_id"]
        comparison_configs.add(digest({name: row.get(name) for name in COMPARISON_FIELDS}))
        identities.add((row["model_id"], row["model_revision"], row["synthetic"]))
        for length_name in ("prompt_tokens", "response_tokens"):
            if row.get(length_name) is not None and (type(row[length_name]) is not int or row[length_name] < 0):
                raise ValueError(f"invalid {length_name}")
    if len(identities) > 1:
        raise ValueError("a generation file must contain one model revision and one synthetic status")
    if len(comparison_configs) > 1:
        raise ValueError("a generation file must use one scale, template, tokenizer and decoding configuration")
    return {"count": len(rows), "fingerprint": digest(rows), "model": list(identities)[0] if identities else None}


def comparable_pairs(left, right):
    validate_generations(left)
    validate_generations(right)
    right_by_key = {(r["prompt_id"], str(r.get("sample_id", "0"))): r for r in right}
    if len(left) != len(right):
        raise ValueError("comparison requires equal coverage, no silent missing-prompt exclusion")
    pairs = []
    for lhs in left:
        key = (lhs["prompt_id"], str(lhs.get("sample_id", "0")))
        rhs = right_by_key.get(key)
        if rhs is None or lhs["prompt"] != rhs["prompt"]:
            raise ValueError("comparison prompts/sample IDs do not match")
        for name in COMPARISON_FIELDS + ("source_split", "synthetic", "context_sha256"):
            if lhs.get(name) != rhs.get(name):
                raise ValueError(f"comparison configuration mismatch: {name}")
        pairs.append((lhs, rhs))
    return pairs
