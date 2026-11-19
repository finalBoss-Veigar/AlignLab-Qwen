"""Resolve the actual desktop independently of the current working directory."""
import os
import platform
import subprocess
from pathlib import Path


def desktop_path() -> Path:
    system = platform.system()
    if system == "Windows":
        value = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command",
            "[Console]::OutputEncoding=[Text.Encoding]::UTF8; [Environment]::GetFolderPath('Desktop')"],
            timeout=15).decode("utf-8-sig").strip()
        return Path(value)
    if "microsoft" in platform.release().lower():
        value = subprocess.check_output(["powershell.exe", "-NoProfile", "-Command",
            "[Environment]::GetFolderPath('Desktop')"], timeout=15).decode().strip()
        return Path(subprocess.check_output(["wslpath", "-u", value], timeout=10).decode().strip())
    if system == "Darwin":
        return Path.home() / "Desktop"
    try:
        value = subprocess.check_output(["xdg-user-dir", "DESKTOP"], timeout=10).decode().strip()
        if value:
            return Path(value)
    except (FileNotFoundError, subprocess.SubprocessError):
        pass
    return Path.home() / "Desktop"


def project_root() -> Path:
    override = os.environ.get("ALIGNLAB_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "pyproject.toml").exists():
        return source_root
    return desktop_path() / "study" / "AlignLab"


def resolve_path(value) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()
