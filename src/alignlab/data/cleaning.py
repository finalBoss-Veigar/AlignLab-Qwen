"""Optional quality filtering is separate from mandatory legality and leakage checks."""
from .schema import DataError


def check_quality(row, *, enabled=True):
    if not enabled:
        return
    messages = row["messages"] + ([row["chosen"], row["rejected"]] if row["kind"] == "dpo" else [])
    for message in messages:
        text = message["content"]
        if sum(ord(c) < 32 and c not in "\n\r\t" for c in text) > 0:
            raise DataError("control_characters")
        words = text.split()
        if len(words) >= 20 and len(set(words)) <= 2:
            raise DataError("degenerate_repetition")
