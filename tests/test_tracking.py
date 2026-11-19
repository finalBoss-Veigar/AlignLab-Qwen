import json
import pytest
from alignlab.tracking.manifest import Run


def test_failed_exit_is_preserved_and_has_log(tmp_path, monkeypatch):
    monkeypatch.setattr("alignlab.tracking.manifest.environment", lambda: {"test": True})
    with pytest.raises(RuntimeError):
        with Run("test", {}, fixture=True, output_root=tmp_path) as run:
            run.finish("failed", exit_code=137)
            raise RuntimeError("subprocess OOM")
    record = json.loads((run.path / "run_manifest.json").read_text())
    assert record["status"] == "failed"
    assert record["exit_code"] == 137
    assert "subprocess OOM" in (run.path / "failure.log").read_text()


def test_dry_run_never_marked_training_completed(tmp_path, monkeypatch):
    monkeypatch.setattr("alignlab.tracking.manifest.environment", lambda: {"test": True})
    with Run("test", {}, fixture=True, output_root=tmp_path) as run:
        run.finish("dry_run_passed", training_completed=False)
    assert run.value["status"] == "dry_run_passed"
    assert run.value["training_steps"] is None
