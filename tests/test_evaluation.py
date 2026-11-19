import json
import tempfile
import unittest
from pathlib import Path

from alignlab.evaluation.blind import export_blind, import_blind, score_records
from alignlab.evaluation.bootstrap import cluster_bootstrap
from alignlab.evaluation.generation import comparable_pairs, validate_generations
from alignlab.evaluation.generate import generate_local
from alignlab.evaluation.ifeval import parse_official_outputs, validate_inputs
from alignlab.evaluation.io import write_jsonl
from alignlab.evaluation.report import write_report


def generation(prompt_id, model="left", sample_id="0"):
    return {"prompt_id": prompt_id, "sample_id": sample_id, "prompt": f"Synthetic prompt {prompt_id}", "response": f"Synthetic answer {model}",
            "model_id": model, "model_revision": "fixture-v1", "tokenizer_revision": "fixture-v1", "chat_template_sha256": "fixture-template",
            "source_split": "fixture_test", "synthetic": True, "size_class": "synthetic", "decoding": {"do_sample": False, "max_new_tokens": 32},
            "prompt_tokens": 10, "response_tokens": 8}


class EvaluationTests(unittest.TestCase):
    def test_empty_metrics_are_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "fixtures" / "report"
            paths = write_report(out, fixture=True)
            metrics = json.loads(Path(paths["metrics.json"]).read_text(encoding="utf-8"))
            self.assertIsNone(metrics["preference"]["win_rate"])
            self.assertIsNone(metrics["ifeval"]["strict"]["prompt_level"])
            self.assertIn("SYNTHETIC FIXTURE", Path(paths["report.html"]).read_text(encoding="utf-8"))
            self.assertIn("null", Path(paths["summary.csv"]).read_text(encoding="utf-8-sig"))

    def test_fixture_separation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_report(Path(tmp) / "experiments", fixture=True)
            with self.assertRaises(ValueError):
                write_report(Path(tmp) / "experiments", preference={"synthetic": True})
            with self.assertRaises(ValueError):
                write_report(Path(tmp) / "experiments", ifeval={"synthetic": True})

    def test_ttft_cannot_be_total_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_report(tmp, engineering={"ttft_seconds": 1})

    def test_comparison_configuration_must_match(self):
        lhs, rhs = generation("p"), generation("p", "right")
        rhs["decoding"]["max_new_tokens"] = 64
        with self.assertRaises(ValueError):
            comparable_pairs([lhs], [rhs])

    def test_empty_answers_preserved(self):
        row = generation("p")
        row["response"] = ""
        self.assertEqual(validate_generations([row])["count"], 1)

    def test_same_prompt_cannot_inflate_independent_clusters(self):
        first, second = generation("p"), generation("different-id")
        second["prompt"] = first["prompt"]
        with self.assertRaises(ValueError):
            validate_generations([first, second])

    def test_real_generation_rejects_fixtures_before_model_loading(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompts = Path(tmp) / "prompts.jsonl"
            write_jsonl(prompts, [generation("p")])
            with self.assertRaises(ValueError):
                generate_local("absent-model", "absent-tokenizer", prompts, Path(tmp)/"out", model_revision="rev", tokenizer_revision="rev", size_class="7B")

    def test_real_generation_dryrun_checks_provenance_without_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompts = Path(tmp) / "prompts.jsonl"
            row = {"prompt_id": "eval-1", "prompt": "Write lowercase", "dataset_id": "google/IFEval", "revision": "fixed-rev", "source_split": "train", "source_row_id": 1, "synthetic": False}
            write_jsonl(prompts, [row])
            result = generate_local("absent-model", "absent-tokenizer", prompts, Path(tmp)/"out", model_revision="rev", tokenizer_revision="rev", size_class="7B")
            self.assertEqual(result["status"], "dry_run_checked_not_generated")
            self.assertFalse(result["local_model_available"])
            self.assertTrue(result["input_sha256"])
            row["messages"] = [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": "leaked reference answer"}]
            write_jsonl(prompts, [row])
            with self.assertRaises(ValueError):
                generate_local("absent-model", "absent-tokenizer", prompts, Path(tmp)/"out", model_revision="rev", tokenizer_revision="rev", size_class="7B")

    def test_balanced_blind_export_and_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_blind([generation(str(i)) for i in range(5)], [generation(str(i), "right") for i in range(5)], tmp)
            keys = json.loads(Path(result["private_key"]).read_text(encoding="utf-8"))
            self.assertLessEqual(abs(2 * sum(r["left_on_a"] for r in keys["cases"]) - 5), 1)
            public = Path(result["candidates"]).read_text(encoding="utf-8")
            self.assertNotIn("model_id", public)
            metrics = import_blind(result["candidates"], result["private_key"], result["annotations"], n_bootstrap=100)
            self.assertIsNone(metrics["win_rate"])
            self.assertEqual(metrics["pending"], 5)
            self.assertIsNone(metrics["bootstrap"]["win_rate_ci"])

    def test_blind_import_counts_and_key_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_blind([generation(str(i)) for i in range(4)], [generation(str(i), "right") for i in range(4)], tmp)
            keys = json.loads(Path(result["private_key"]).read_text(encoding="utf-8"))["cases"]
            annotations = []
            for index, case in enumerate(keys):
                label = ("A" if case["left_on_a"] else "B") if index == 0 else ("B" if case["left_on_a"] else "A") if index == 1 else "tie" if index == 2 else "unjudgeable"
                annotations.append({"case_id": case["case_id"], "label": label})
            write_jsonl(result["annotations"], annotations)
            metrics = import_blind(result["candidates"], result["private_key"], result["annotations"], n_bootstrap=100)
            self.assertEqual((metrics["W"], metrics["L"], metrics["T"], metrics["N"]), (1, 1, 1, 3))
            self.assertAlmostEqual(metrics["win_rate"], 1/3)
            self.assertEqual(metrics["tie_score"], .5)
            self.assertEqual(metrics["unjudgeable"], 1)
            self.assertTrue(metrics["bad_cases"])

    def test_tampered_candidates_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = export_blind([generation("p")], [generation("p", "right")], tmp)
            public = json.loads(Path(result["candidates"]).read_text(encoding="utf-8"))
            public["A"] = "changed"
            write_jsonl(result["candidates"], [public])
            with self.assertRaises(ValueError):
                import_blind(result["candidates"], result["private_key"], result["annotations"])

    def test_train_split_rejected(self):
        lhs, rhs = generation("p"), generation("p", "right")
        lhs["source_split"] = rhs["source_split"] = "train_prefs"
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            export_blind([lhs], [rhs], tmp)

    def test_bootstrap_boundaries_and_prompt_clusters(self):
        rows = [{"prompt_id": "one", "outcome": "win"}] * 4
        result = cluster_bootstrap(rows, n_bootstrap=100)
        self.assertEqual(result["n_prompts"], 1)
        self.assertEqual(result["win_rate_ci"], [1, 1])
        ties = cluster_bootstrap([{"prompt_id": "one", "outcome": "tie"}], n_bootstrap=100)
        self.assertEqual(ties["win_rate_ci"], [0, 0])
        self.assertEqual(ties["tie_score_ci"], [.5, .5])
        for name in ("win_rate_ci", "tie_score_ci"):
            mixed = cluster_bootstrap(rows + [{"prompt_id": "two", "outcome": "loss"}], n_bootstrap=100)
            self.assertTrue(0 <= mixed[name][0] <= mixed[name][1] <= 1)

    def test_all_unjudgeable_is_not_zero_win_rate(self):
        result = score_records([{"prompt_id": "p", "outcome": "unjudgeable"}])
        self.assertEqual(result["N"], 0)
        self.assertIsNone(result["win_rate"])

    def test_official_ifeval_shape_and_coverage(self):
        inputs = [{"key": 1, "prompt": "hi", "instruction_id_list": ["change_case:english_lowercase"], "kwargs": [{"unknown_column": None}]}]
        normalized = validate_inputs(inputs, [{"prompt": "hi", "response": "hello"}])
        self.assertEqual(normalized[0]["kwargs"], [{}])
        with self.assertRaises(ValueError):
            validate_inputs(inputs, [])
        with self.assertRaises(ValueError):
            validate_inputs([], [])

    def test_official_metric_aggregation_not_rule_implementation(self):
        row = {"prompt": "synthetic parser fixture", "instruction_id_list": ["x", "y"],
               "follow_instruction_list": [True, False], "follow_all_instructions": False}
        with tempfile.TemporaryDirectory() as tmp:
            for mode in ("strict", "loose"):
                write_jsonl(Path(tmp) / f"eval_results_{mode}.jsonl", [row])
            metrics = parse_official_outputs(tmp)
            self.assertEqual(metrics["strict"]["prompt_level"], 0)
            self.assertEqual(metrics["strict"]["instruction_level"], .5)
            row["follow_all_instructions"] = True
            write_jsonl(Path(tmp) / "eval_results_strict.jsonl", [row])
            with self.assertRaises(ValueError):
                parse_official_outputs(tmp)


if __name__ == "__main__":
    unittest.main()
