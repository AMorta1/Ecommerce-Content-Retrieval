import importlib.util
from pathlib import Path
import unittest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "track_experiment.py"
SPEC = importlib.util.spec_from_file_location("track_experiment", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
track_experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(track_experiment)


class ExperimentTrackingTests(unittest.TestCase):
    def make_row(
        self,
        *,
        version: str,
        method: str,
        evaluation_set_id: str = "generation:same",
        hit_rate: str = "0.880000",
        factual_error_rate: str = "0.560000",
    ) -> dict[str, str]:
        row = {field: "" for field in track_experiment.LOG_FIELDS}
        row.update(
            {
                "module": "generation",
                "version": version,
                "method": method,
                "evaluation_set_id": evaluation_set_id,
                "core_attribute_hit_rate": hit_rate,
                "factual_error_sample_rate": factual_error_rate,
            }
        )
        return row

    def test_same_version_is_updated_instead_of_duplicated(self) -> None:
        rows = [self.make_row(version="baseline_v1", method="baseline")]
        replacement = self.make_row(
            version="baseline_v1", method="baseline", hit_rate="0.900000"
        )

        action = track_experiment.upsert_record(rows, replacement)

        self.assertEqual(action, "updated")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["core_attribute_hit_rate"], "0.900000")

    def test_deltas_are_computed_only_on_the_same_evaluation_set(self) -> None:
        baseline = self.make_row(version="baseline_v1", method="baseline")
        comparable = self.make_row(
            version="lora_v1",
            method="lora",
            hit_rate="0.910000",
            factual_error_rate="0.300000",
        )
        different_set = self.make_row(
            version="lora_other_set",
            method="lora",
            evaluation_set_id="generation:different",
            hit_rate="0.950000",
        )

        rows = [baseline, comparable, different_set]
        track_experiment.refresh_comparisons(rows)

        self.assertEqual(comparable["comparison_status"], "compared")
        self.assertEqual(comparable["delta_core_attribute_hit_rate"], "0.030000")
        self.assertEqual(comparable["factual_error_rate_reduction"], "0.260000")
        self.assertEqual(different_set["comparison_status"], "no_matching_baseline")
        self.assertEqual(different_set["delta_core_attribute_hit_rate"], "")

    def test_assistant_draft_metrics_are_rejected(self) -> None:
        report = {
            "status": "human_evaluation_completed",
            "annotation_provenance": {"assistant_draft_rows": 1},
        }

        with self.assertRaisesRegex(ValueError, "AI 初标"):
            track_experiment.require_human_evaluation("generation", report)


if __name__ == "__main__":
    unittest.main()
