"""Exact normalized hashing + exact character n-gram Jaccard duplicate groups."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import math

from .schema import normalize


def ngrams(text, n=5):
    text = normalize(text)
    return {text[i:i + n] for i in range(max(1, len(text) - n + 1))}


def duplicate_groups(records, *, threshold=0.85, n=5):
    """All user-turn strings are leakage keys, including pre-truncation strings.

    Globally ordered prefix filtering is exact: a Jaccard match must intersect
    prefixes of length |S| - ceil(threshold * |S|) + 1. Full sets verify candidates.
    No approximate search or candidate cap is used.
    """
    if not 0 < threshold <= 1 or n < 1:
        raise ValueError("threshold must be in (0,1], n >= 1")
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    aliases, owners, gramsets = [], [], []
    exact = {}
    exact_edges = near_edges = 0
    for index, row in enumerate(records):
        texts = {row["prompt"], *row.get("original_context_hash_input", [])}
        texts.update(m["content"] for m in row.get("messages", []) if m["role"] == "user")
        for text in sorted(texts):
            normalized = normalize(text)
            if normalized in exact:
                owner = owners[exact[normalized]]
                if find(index) != find(owner):
                    exact_edges += 1
                union(index, owner)
                continue
            grams = ngrams(text, n)
            alias_id = len(aliases)
            aliases.append(normalized)
            owners.append(index)
            gramsets.append(grams)
            exact[normalized] = alias_id
    frequencies = Counter(gram for grams in gramsets for gram in grams)
    postings = defaultdict(list)
    for alias_id, grams in enumerate(gramsets):
        ordered = sorted(grams, key=lambda gram: (frequencies[gram], gram))
        prefix = ordered[:len(grams) - math.ceil(threshold * len(grams)) + 1]
        candidates = {other for gram in prefix for other in postings[gram]}
        for other in candidates:
            size = len(gramsets[other])
            if min(size, len(grams)) < threshold * max(size, len(grams)):
                continue
            intersection = len(grams & gramsets[other])
            similarity = intersection / (size + len(grams) - intersection)
            if similarity >= threshold:
                owner, index = owners[other], owners[alias_id]
                if find(index) != find(owner):
                    near_edges += 1
                union(index, owner)
        for gram in prefix:
            postings[gram].append(alias_id)
    groups = defaultdict(list)
    for index in range(len(records)):
        groups[find(index)].append(index)
    result = []
    for members in groups.values():
        group_id = hashlib.sha256("|".join(sorted({records[i]["prompt_hash"] for i in members})).encode()).hexdigest()
        for index in members:
            records[index]["group_id"] = group_id
        result.append(members)
    return result, {"groups": len(groups), "exact_union_edges": exact_edges,
                    "near_union_edges": near_edges, "ngram_n": n,
                    "jaccard_threshold": threshold, "candidate_search": "exact_global_prefix_index"}
