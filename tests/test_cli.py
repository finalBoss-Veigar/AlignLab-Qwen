"""Actual subprocess roundtrips. All answers/labels here are synthetic test data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from alignlab.evaluation.io import read_jsonl, write_jsonl


def synthetic_record(index, model):
    return {"prompt_id": f"synthetic-{index}", "prompt": f"Synthetic test question {index}", "response": f"Synthetic candidate text {model} {index}",
            "model_id": model, "model_revision": "synthetic-test", "tokenizer_revision": "synthetic-test", "chat_template_sha256": "synthetic-test",
            "source_split": "fixture_test", "synthetic": True, "size_class": "synthetic", "decoding": {"do_sample": False, "max_new_tokens": 32}}


class CliIntegrationTests(unittest.TestCase):
    def run_cli(self, *args, success=True):
        result = subprocess.run([sys.executable, "-m", "alignlab", *map(str, args)], capture_output=True,
                                text=True, encoding="utf-8", timeout=30, env=dict(os.environ, PYTHONUTF8="1"))
        if success:
            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(result.stdout)
        self.assertNotEqual(result.returncode, 0)
        return json.loads(result.stderr)

    def test_actual_cli_blind_roundtrip_and_null_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left.jsonl", root / "right.jsonl"
            write_jsonl(left, [synthetic_record(i, "synthetic-left") for i in range(4)])
            write_jsonl(right, [synthetic_record(i, "synthetic-right") for i in range(4)])
            export = self.run_cli("blind-export", left, right, "--output", root / "review")
            self.assertEqual(export["status"], "pending_human_review")
            pending_path = root / "pending.json"
            pending = self.run_cli("blind-import", "--public", export["candidates"], "--key", export["private_key"],
                                   "--annotations", export["annotations"], "--output", pending_path)
            self.assertIsNone(pending["win_rate"])
            self.assertIsNone(pending["tie_score"])
            self.assertEqual(pending["pending"], 4)
            report = self.run_cli("report", "--output", root / "fixtures" / "pending-report", "--fixture", "--preference", pending_path)
            self.assertTrue(Path(report["report.html"]).is_file())
            self.assertIn("SYNTHETIC FIXTURE", Path(report["report.html"]).read_text(encoding="utf-8"))
            self.run_cli("report", "--output", root / "experiments" / "forbidden", "--preference", pending_path, success=False)
            # These labels are manually constructed solely to check scoring code.
            key = json.loads(Path(export["private_key"]).read_text(encoding="utf-8"))
            annotations = read_jsonl(export["annotations"])
            for index, (annotation, case) in enumerate(zip(annotations, key["cases"])):
                annotation["label"] = ("A" if case["left_on_a"] else "B") if index == 0 else "tie" if index == 1 else "unjudgeable" if index == 2 else ""
                annotation["reason"] = "Synthetic CLI test label; no model was evaluated"
            write_jsonl(export["annotations"], annotations)
            scored = self.run_cli("blind-import", "--public", export["candidates"], "--key", export["private_key"],
                                  "--annotations", export["annotations"], "--output", root / "synthetic-scored.json")
            self.assertEqual((scored["W"], scored["T"], scored["N"]), (1, 1, 2))
            self.assertEqual(scored["win_rate"], .5)
            self.assertEqual(scored["tie_score"], .75)
            self.assertTrue(scored["synthetic"])

    def test_actual_cli_empty_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = self.run_cli("report", "--output", Path(tmp) / "fixtures" / "empty", "--fixture")
            metrics = json.loads(Path(report["metrics.json"]).read_text(encoding="utf-8"))
            self.assertIsNone(metrics["preference"]["win_rate"])
            self.assertIsNone(metrics["ifeval"]["strict"]["prompt_level"])

    def test_ifeval_accepts_generation_jsonl_bridge_on_dryrun(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            row = synthetic_record(0, "synthetic-model")
            prompts, responses = root / "input.jsonl", root / "generations.jsonl"
            write_jsonl(prompts, [{"key": 1, "prompt": row["prompt"], "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{}]}])
            write_jsonl(responses, [row])
            result = self.run_cli("ifeval", "--repo", root / "absent-official-source", "--input", prompts,
                                  "--responses", responses, "--output", root / "output")
            self.assertEqual(result["status"], "dry_run_checked_not_executed")
            self.assertIsNone(result["metrics"])
            self.assertFalse(result["source_available"])


if __name__ == "__main__":
    unittest.main()
