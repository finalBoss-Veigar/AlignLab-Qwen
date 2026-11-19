import hashlib
import json
from pathlib import Path
import yaml


def load_config(path):
    path = Path(path)
    with path.open(encoding="utf-8-sig") as f:
        value = json.load(f) if path.suffix == ".json" else yaml.safe_load(f)
    if not isinstance(value, dict):
        raise ValueError("Config must be a mapping")
    return value


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
