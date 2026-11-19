"""Count printed list markers in SFT demonstrations for explicit English list requests."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time


KINDS = {
    "numbered": re.compile(r"\b(?:a numbered list|as a numbered list|in numbered (?:points|steps))\b", re.I),
    "bulleted": re.compile(r"\b(?:a (?:bulleted|bullet(?:[- ]point)?) list|(?:in|as|using|use) bullet points)\b", re.I),
}
AMBIGUOUS = re.compile(r"\b(?:not|never|avoid|without|instead of|rather than|don't|do not)\b", re.I)
NUMBER = re.compile(r"^[ \t]*(\d{1,3})[.)][ \t]+\S")
BULLET = re.compile(r"^[ \t]*[-*+•][ \t]+\S")
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")


def request_kind(prompt):
    """Drop mixed requests, code fences and negated turns to favor precision over recall."""
    normalized = prompt.replace("’", "'")
    if AMBIGUOUS.search(normalized) or any(FENCE.match(line) for line in prompt.splitlines()):
        return None
    matches = [kind for kind, pattern in KINDS.items() if pattern.search(normalized)]
    return matches[0] if len(matches) == 1 else None


def marker_count(answer, kind):
    count = 0
    fence = None
    for line in answer.splitlines():
        match = FENCE.match(line)
        if fence is not None:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not line[match.end():].strip():
                fence = None
            continue
        if match:
            fence = match[1]
            continue
        pattern = NUMBER if kind == "numbered" else BULLET
        if pattern.match(line):
            count += 1
    return count


def audit_rows(rows):
    counts = {kind: Counter() for kind in KINDS}
    n_rows = n_pairs = n_identified_rows = 0
    for row in rows:
        n_rows += 1
        turns = row["messages"]
        found = False
        for user, assistant in zip(turns, turns[1:]):
            if (user["role"], assistant["role"]) != ("user", "assistant"):
                continue
            n_pairs += 1
            kind = request_kind(user["content"])
            if kind:
                found = True
                n = marker_count(assistant["content"], kind)
                bucket = ("zero", "one", "two_or_more")[min(n, 2)]
                counts[kind][bucket] += 1
        n_identified_rows += found
    return {
        "dialogues": n_rows, "adjacent_user_assistant_pairs": n_pairs,
        "dialogues_with_any_matched_turn": n_identified_rows,
        "detected_turns": {
            kind: {key: counts[kind][key] for key in ("zero", "one", "two_or_more")}
            for kind in KINDS
        },
    }


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a fresh output path to retain previous results")
    start = time.perf_counter()
    manifest = json.loads((args.data / "data_manifest.json").read_text(encoding="utf-8"))
    result = {
        "id": args.output.stem, "started_at": datetime.now(timezone.utc).isoformat(),
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "worktree_at_start": subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines(),
        "script_sha256": sha256(Path(__file__)), "data_manifest_sha256": sha256(args.data / "data_manifest.json"),
        "argv": ["python", "scripts/audit_list_format.py", *sys.argv[1:]],
        "environment": f"Python {platform.python_version()}, {platform.system()} {platform.machine()}, CPU",
        "seed": None, "splits": {},
    }
    for split in ("train", "validation"):
        path = args.data / f"sft_{split}.internal.jsonl"
        digest = sha256(path)
        if digest != manifest["artifacts"][path.name]["sha256"]:
            raise ValueError(f"{path.name}: checksum differs from data manifest")
        with path.open(encoding="utf-8") as stream:
            stats = audit_rows(json.loads(line) for line in stream if line.strip())
        result["splits"][split] = {"source": path.name, "sha256": digest, **stats}
    result["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, indent=2, ensure_ascii=True) + "\n")
    print(json.dumps(result["splits"], indent=2))


if __name__ == "__main__":
    main()
