"""Fetch small primary metadata/cards only. Never downloads weights or dataset shards."""
import concurrent.futures
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "source_snapshots"
SOURCES = [
    ("qwen7b", "models", "Qwen/Qwen2.5-7B-Instruct"),
    ("qwen1_5b", "models", "Qwen/Qwen2.5-1.5B-Instruct"),
    ("ultrachat", "datasets", "HuggingFaceH4/ultrachat_200k"),
    ("ultrafeedback", "datasets", "HuggingFaceH4/ultrafeedback_binarized"),
    ("ifeval_data", "datasets", "google/IFEval"),
]


def fetch(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "AlignLab-source-audit"}), timeout=40) as r:
        return r.read()


def collect(entry):
    key, kind, repo = entry
    result = {"id": repo, "kind": kind, "access_date": "2026-11-13"}
    try:
        meta = json.loads(fetch(f"https://huggingface.co/api/{kind}/{repo}"))
        result.update(revision=meta["sha"], license=meta.get("cardData", {}).get("license"))
        prefix = "datasets/" if kind == "datasets" else ""
        result["url"] = f"https://huggingface.co/{prefix}{repo}"
        card = fetch(f"{result['url']}/resolve/{meta['sha']}/README.md")
        (DEST / f"{key}.md").write_bytes(card)
        (DEST / f"{key}.metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        result["status"] = "verified_metadata_and_card_only"
    except Exception as exc:
        cached = DEST / f"{key}.metadata.json"
        if cached.exists() and (DEST / f"{key}.md").exists():
            meta = json.loads(cached.read_text(encoding="utf-8"))
            prefix = "datasets/" if kind == "datasets" else ""
            result.update(revision=meta["sha"], license=meta.get("cardData", {}).get("license"),
                url=f"https://huggingface.co/{prefix}{repo}", status="verified_cached_metadata_and_card_only",
                retry_error=str(exc))
        else:
            result["status"] = "blocked"
            result["error"] = str(exc)
    return key, result


if __name__ == "__main__":
    DEST.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        results = dict(pool.map(collect, SOURCES))
    try:
        import subprocess
        sha = subprocess.check_output(["git", "ls-remote", "https://github.com/huggingface/alignment-handbook.git", "refs/heads/main"], timeout=40).decode().split()[0]
        (DEST / "alignment_handbook.md").write_bytes(fetch(f"https://raw.githubusercontent.com/huggingface/alignment-handbook/{sha}/README.md"))
        (DEST / "alignment_handbook.LICENSE").write_bytes(fetch(f"https://raw.githubusercontent.com/huggingface/alignment-handbook/{sha}/LICENSE"))
        results["alignment_handbook"] = {"revision": sha, "license": "Apache-2.0", "access_date": "2026-11-13", "url": "https://github.com/huggingface/alignment-handbook", "status": "verified_readme_license_only"}
    except Exception as exc:
        results["alignment_handbook"] = {"status": "blocked", "error": str(exc)}
    # A refresh must not silently replace the committed experiment pins.
    lock_path = ROOT / "docs" / "sources.lock.json"
    target = ROOT / "docs" / "sources.refresh.json" if lock_path.exists() else lock_path
    target.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
