from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from summarize_priority_evidence import (
    EvidenceError,
    OPERATOR_STEM,
    _synthetic_fixture,
    summarize,
)


class PriorityEvidenceSummaryTest(unittest.TestCase):
    def test_synthetic_matrix_metrics(self) -> None:
        with tempfile.TemporaryDirectory(prefix="priority-summary-test-") as temporary:
            root = Path(temporary)
            _synthetic_fixture(root)

            summary = summarize(results_root=root, suite="auto")

            self.assertEqual(summary["suite"], "core")
            self.assertEqual(summary["validation"]["exact_run_count"], 18)
            self.assertEqual(len(summary["runs"]), 18)
            self.assertEqual(len(summary["groups"]), 6)
            for run in summary["runs"]:
                self.assertEqual(
                    run["behavior"]["first_90_percent_step"],
                    10_000,
                )
                self.assertEqual(run["behavior"]["peak_step"], 20_000)
                self.assertAlmostEqual(
                    run["behavior"]["post_peak_max_drawdown"],
                    0.15,
                )
                self.assertAlmostEqual(
                    run["operator"]["endpoint"][
                        "alias_held_out_usable_gain_bits"
                    ],
                    1_500.0,
                )
                output = run["causal"]["sites"]["output_final"]
                self.assertAlmostEqual(
                    output["canonical_cycle_median_success"],
                    0.81,
                )
                self.assertAlmostEqual(
                    output["negative_control_max_median"],
                    0.21,
                )

    def test_tampered_successor_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="priority-summary-hash-") as temporary:
            root = Path(temporary)
            _synthetic_fixture(root)
            operator_path = next(root.glob(f"*/{OPERATOR_STEM}.json"))
            payload = json.loads(operator_path.read_text())
            payload["metadata"]["successor_sha256"] = hashlib.sha256(
                b"post-hoc successor"
            ).hexdigest()
            operator_path.write_text(json.dumps(payload) + "\n")

            with self.assertRaisesRegex(
                EvidenceError,
                "preregistered successor fields",
            ):
                summarize(results_root=root, suite="core")

    def test_scale_matrix_is_auto_detected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="priority-summary-scale-") as temporary:
            root = Path(temporary)
            _synthetic_fixture(root, suite="scale")

            summary = summarize(results_root=root, suite="auto")

            self.assertEqual(summary["suite"], "scale")
            self.assertEqual(
                summary["validation"]["presets"],
                ["small", "medium"],
            )
            self.assertEqual(summary["validation"]["exact_run_count"], 18)

    def test_large_matrix_is_exactly_nine_runs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="priority-summary-large-") as temporary:
            root = Path(temporary)
            _synthetic_fixture(root, suite="large")

            summary = summarize(results_root=root, suite="auto")

            self.assertEqual(summary["suite"], "large")
            self.assertEqual(summary["validation"]["presets"], ["large"])
            self.assertEqual(summary["validation"]["exact_run_count"], 9)
            self.assertEqual(len(summary["groups"]), 3)

    def test_capacity_comparison_requires_exact_27_run_union(self) -> None:
        with tempfile.TemporaryDirectory(prefix="priority-summary-capacity-") as temporary:
            root = Path(temporary)
            _synthetic_fixture(root, suite="capacity")

            with self.assertRaisesRegex(EvidenceError, "exactly one complete suite"):
                summarize(results_root=root, suite="auto")

            summary = summarize(results_root=root, suite="capacity")
            self.assertEqual(summary["validation"]["exact_run_count"], 27)
            self.assertEqual(
                summary["validation"]["presets"],
                ["small", "medium", "large"],
            )

    def test_duplicate_identity_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="priority-summary-duplicate-") as temporary:
            root = Path(temporary)
            _synthetic_fixture(root)
            source = next(root.glob("clean-grok-s0-*/config.json"))
            duplicate = root / "duplicate-clean-grok-s0"
            duplicate.mkdir()
            duplicate_config = json.loads(source.read_text())
            duplicate_config["run_name"] = duplicate.name
            (duplicate / "config.json").write_text(
                json.dumps(duplicate_config) + "\n"
            )

            with self.assertRaisesRegex(EvidenceError, "nonunique"):
                summarize(results_root=root, suite="core")


if __name__ == "__main__":
    unittest.main()
