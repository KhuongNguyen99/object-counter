import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import render_reliability_report as report


def row(target, *, missed=0, extra=0, split="tune", run="baseline-tune", model="yoloe-26n-seg"):
    return {"id": target + model, "run": run, "phase": report.phase(run + "/results.json"),
            "strategy": "baseline", "model": model,
            "threshold": 0.01, "variant": "default", "split": split, "category": "screws",
            "target_id": target, "target_sha256": target, "truth": 2, "predicted": 2,
            "matched": 2 - missed, "missed": missed, "extra": extra,
            "exact": missed == extra == 0, "count_equal": True, "elapsed_ms": 10,
            "status": "ok"}


class ReliabilityReportTests(unittest.TestCase):
    def test_count_equality_is_separate_from_exact_match(self):
        one = row("photo", missed=1, extra=1)
        group = report.aggregate([one])[0]
        self.assertEqual((group["count_equal"], group["exact"], group["missed"], group["extra"]),
                         (1, 0, 1, 1))

    def test_countgd_spotlight_uses_saved_core_records(self):
        baseline = row("original", missed=1, extra=1, model="countgd-plusplus")
        baseline["threshold"] = 0.5
        mixed = row("mixed", missed=2, extra=3, run="mixed-tune", model="countgd-plusplus")
        mixed["threshold"] = 0.5
        summary = report.spotlight([baseline, mixed])
        self.assertEqual(summary["baseline"]["missed"], 1)
        self.assertEqual(summary["other"]["worst"]["target_id"], "mixed")

    def test_case_selection_keeps_distinct_queries_and_core_runs(self):
        rows = [row("photo", missed=2), row("photo", missed=1, model="yoloe-26x-seg"),
                row("other", missed=1), row("strategy-only", missed=8, run="strategy-output/x")]
        self.assertEqual({x["target_id"] for x in report.selected_cases(rows)}, {"photo", "other"})

    def test_normalization_and_sam3_runs_stay_in_separate_phases(self):
        raw = row("same", split="heldout", run="heldout-core", model="countgd-plusplus")
        normalized = row("same", split="heldout", run="normalization-core", model="countgd-plusplus")
        self.assertEqual(len(report.aggregate([raw, normalized])), 2)
        self.assertEqual(normalized["phase"], "JPEG normalization stress")
        self.assertEqual(report.phase("sam3-threshold-0.9/results.json"), "SAM3 tune threshold sweep")

    def test_string_experiment_marks_only_xl_per_object(self):
        base = {"status": "ok", "threshold": .01, "target_id": "sample", "category": "nuts",
                "split": "heldout", "points": [], "prediction": {"count": 0, "elapsed_ms": 1},
                "metrics": {"matched": 0, "missed": 0, "extra": 0, "exact_image": True},
                "settings": {"experiment": "server-normalization",
                             "secondary_reference_strategy": "per-object"}}
        xl = report.compact_record(base | {"model": "yoloe-26x-seg"}, "a", "normalization-core/results.json", "hash", "manifest")
        countgd = report.compact_record(base | {"model": "countgd-plusplus"}, "b", "normalization-core/results.json", "hash", "manifest")
        self.assertIn("per-object", xl["strategy"])
        self.assertEqual(countgd["strategy"], "server-normalization")

    def test_load_results_keeps_errors_and_source_hash(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "baseline-tune/results.json"
            result.parent.mkdir()
            saved = {"records": {"job": {"model": "yoloe-26n-seg", "threshold": .01,
                                           "target_id": "photo", "category": "screws", "split": "tune",
                                           "points": [[1, 1]], "status": "error",
                                           "error": {"message": "forced"}}}, "selected_jobs": 1}
            result.write_text(json.dumps(saved), encoding="utf-8")
            auxiliary = root / "gpu-consistency/results.json"
            auxiliary.parent.mkdir()
            auxiliary.write_text('{"checks": []}', encoding="utf-8")
            rows, sources, missing, issues = report.load_results(root)
            self.assertEqual((len(rows), sources[0]["errors"], issues), (1, 1, []))
            self.assertEqual(rows[0]["source_sha256"], report.digest(result))
            self.assertIn("heldout-core", missing)
            self.assertIn("normalization-core", missing)
            self.assertIsNone(report.local_image(root, "/experiment/../outside.jpg"))
            self.assertIsNone(report.local_image(root, "/experiment/C:\\outside.jpg"))


if __name__ == "__main__":
    unittest.main()
