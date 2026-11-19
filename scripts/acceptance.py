"""Run only lightweight acceptance, preserving exact commands, output and exit codes."""
import datetime
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S")
DEST = ROOT / "runs" / "acceptance" / stamp
DEST.mkdir(parents=True, exist_ok=False)
commands = [
    ("import", ["-c", "import alignlab; print(alignlab.__version__)" ]),
    ("dependencies", ["-m", "pip", "check"]),
    ("help", ["-m", "alignlab", "--help"]),
    ("doctor", ["-m", "alignlab", "doctor", "--output", str(DEST / "environment.json")]),
    ("smoke", ["-m", "alignlab", "smoke"]),
    ("tests", ["-m", "pytest", "-q"]),
    ("serve-dry-run", ["-m", "alignlab", "serve", "--dry-run"]),
]
commands.extend((p.stem + "-dry-run", ["-m", "alignlab", "export" if p.stem.startswith("export") else "train", str(p), "--dry-run"])
    for p in sorted((ROOT / "configs").glob("*.yaml")))
records = []
for name, args in commands:
    cmd = [sys.executable, *args]
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180,
                            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    path = DEST / (name + ".txt")
    path.write_text(result.stdout + "\nSTDERR:\n" + result.stderr, encoding="utf-8")
    records.append({"check": name, "command": cmd, "exit_code": result.returncode, "output_file": str(path),
                    "stdout_tail": result.stdout[-500:]})
    print(f"{name}: exit={result.returncode}")
summary = {"date": "2026-11-13", "user_timezone": "Asia/Shanghai", "status": "passed" if all(r["exit_code"] == 0 for r in records) else "failed",
           "hardware_usage": "CPU only; NVIDIA inspected but not used for training", "model_weights_downloaded": False,
           "real_training_executed": False, "official_ifeval_executed": False, "records": records}
(ROOT / "docs" / "acceptance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"status": summary["status"], "checks": len(records), "logs": str(DEST)}, ensure_ascii=False))
raise SystemExit(0 if summary["status"] == "passed" else 1)
