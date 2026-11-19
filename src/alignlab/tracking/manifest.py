"""Durable run records, including failed and dry-run outcomes."""
import datetime as dt
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path
from ..config import fingerprint, write_json
from ..paths import desktop_path, project_root


def command(args, cwd=None):
    try:
        p = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=cwd, timeout=20)
        return {"exit_code": p.returncode, "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as e:
        return {"exit_code": None, "stdout": "", "stderr": str(e)}


def environment():
    root = project_root()
    disk = shutil.disk_usage(root)
    gpu = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    torch_info = {"installed": importlib.util.find_spec("torch") is not None, "cuda_available": False}
    if torch_info["installed"]:
        probe = command([sys.executable, "-c", "import torch,json; print(json.dumps(dict(version=torch.__version__,cuda_build=torch.version.cuda,cuda_available=torch.cuda.is_available(),bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False)))"])
        try:
            torch_info.update(json.loads(probe["stdout"]))
        except (ValueError, TypeError):
            torch_info["probe_error"] = probe
    mem = command(["powershell.exe", "-NoProfile", "-Command", "Get-CimInstance Win32_OperatingSystem | Select-Object TotalVisibleMemorySize,FreePhysicalMemory | ConvertTo-Json -Compress"]) if os.name == "nt" else command(["getconf", "_PHYS_PAGES"])
    try:
        desktop = str(desktop_path())
    except Exception as e:
        desktop = f"unavailable: {e}"
    return {"platform": platform.platform(), "python": sys.version, "python_executable": sys.executable,
            "desktop": desktop, "project_root": str(root), "memory_probe": mem,
            "disk_free_bytes": disk.free, "git": command(["git", "--version"]), "gpu": gpu,
            "cuda_compiler": command(["nvcc", "--version"]), "torch": torch_info,
            "packages": {p: importlib.metadata.version(p) for p in ("alignlab", "PyYAML", "pytest") if importlib.util.find_spec({"alignlab": "alignlab", "PyYAML": "yaml", "pytest": "pytest"}[p])},
            "available_stages": ["doctor", "synthetic_data_pipeline", "unit_tests", "config_static_validation", "blind_review", "empty_model_http_service"],
            "training_status": "not_validated_on_gpu", "model_loaded": False}


class Run:
    def __init__(self, kind, config, *, fixture=False, output_root=None):
        self.started = time.monotonic()
        self.id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:10]
        self.path = Path(output_root or project_root() / "runs" / ("fixtures" if fixture else "experiments")) / self.id
        self.path.mkdir(parents=True, exist_ok=False)
        git = command(["git", "rev-parse", "HEAD"], cwd=project_root())
        self.value = {"run_id": self.id, "kind": kind, "synthetic_fixture": fixture,
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(), "config": config,
            "config_fingerprint": fingerprint(config), "code_commit": git["stdout"] if git["exit_code"] == 0 else None,
            "code_dirty": command(["git", "status", "--porcelain"], cwd=project_root())["stdout"],
            "environment": environment(), "model_revision": config.get("model_revision"),
            "data_revision": config.get("data_revision"), "training_steps": None,
            "peak_vram_bytes": None, "elapsed_seconds": None, "exit_code": None,
            "status": "running", "metrics_file": None, "artifacts": []}
        self.save()

    def save(self):
        write_json(self.path / "run_manifest.json", self.value)

    def finish(self, status, *, exit_code=0, **details):
        if status not in {"completed", "failed", "dry_run_passed", "not_ready"}:
            raise ValueError("invalid final run status")
        self.value.update(details, status=status, exit_code=exit_code, elapsed_seconds=time.monotonic() - self.started)
        self.save()
        return self.value

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        if typ:
            import traceback
            (self.path / "failure.log").write_text("".join(traceback.format_exception(typ, value, tb)), encoding="utf-8")
            previous = self.value.get("exit_code") if self.value.get("status") == "failed" else None
            self.finish("failed", exit_code=previous or 1, error=str(value))
        elif self.value["status"] == "running":
            self.finish("completed")
        return False
