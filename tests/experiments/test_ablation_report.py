"""Tests for the ablation summary script.

The script's whole job is grouping seeds into arms and reporting the spread, so
these check that grouping — not MLflow, which is exercised by actually running
the sweep.
"""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[2] / "experiments" / "ablation_report.py"
_spec = importlib.util.spec_from_file_location("ablation_report", SCRIPT)
assert _spec is not None and _spec.loader is not None
ablation_report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ablation_report)

FRAME_METRIC = ablation_report.FRAME_METRIC


def run(name: str, frame_f1: float, **params) -> SimpleNamespace:
    """A stand-in for an MLflow ``Run`` with just the fields the script reads."""
    defaults = {
        "model.attention_type": "linear",
        "model.add_module_tcn": "False",
        "model.d_model": "32",
        "model.cmt_num_layers": "5",
        "model.num_heads": "8",
    }
    return SimpleNamespace(
        data=SimpleNamespace(
            tags={"mlflow.runName": name},
            params={**defaults, **params},
            metrics={FRAME_METRIC: frame_f1},
        )
    )


class TestArmName(unittest.TestCase):
    def test_the_seed_suffix_is_stripped(self):
        self.assertEqual(ablation_report.arm_name("abl-flash-s42"), "flash")

    def test_two_seeds_share_one_arm_name(self):
        first = ablation_report.arm_name("abl-d64-s42")
        second = ablation_report.arm_name("abl-d64-s44")
        self.assertEqual(first, second)

    def test_a_hyphenated_arm_survives(self):
        # `abl-flash-tcn-s43` is one arm with two frozen choices, not an arm
        # called "flash" with a seed of "tcn".
        self.assertEqual(ablation_report.arm_name("abl-flash-tcn-s43"), "flash-tcn")

    def test_a_name_without_a_seed_is_kept_whole(self):
        self.assertEqual(ablation_report.arm_name("abl-base"), "base")

    def test_a_non_numeric_suffix_is_not_treated_as_a_seed(self):
        self.assertEqual(ablation_report.arm_name("abl-d64-scaled"), "d64-scaled")


class TestCollect(unittest.TestCase):
    def test_seeds_of_one_arm_are_grouped(self):
        runs = [run(f"abl-flash-s{s}", 0.8) for s in (42, 43, 44)]
        arms = ablation_report.collect(runs, [FRAME_METRIC])
        self.assertEqual(len(arms), 1)
        self.assertEqual(len(arms[0].metrics[FRAME_METRIC]), 3)

    def test_runs_without_the_prefix_are_ignored(self):
        # The experiment also holds the ordinary training runs; only the
        # ablation's own runs belong in its table.
        runs = [run("abl-base-s42", 0.8), run("single-rn30", 0.9)]
        arms = ablation_report.collect(runs, [FRAME_METRIC])
        self.assertEqual([a.name for a in arms], ["base"])

    def test_arms_are_ordered_by_the_frame_metric(self):
        runs = [run("abl-a-s42", 0.70), run("abl-b-s42", 0.90), run("abl-c-s42", 0.80)]
        arms = ablation_report.collect(runs, [FRAME_METRIC])
        self.assertEqual([a.name for a in arms], ["b", "c", "a"])

    def test_the_axes_are_carried_through(self):
        runs = [run("abl-d64-s42", 0.8, **{"model.d_model": "64"})]
        arms = ablation_report.collect(runs, [FRAME_METRIC])
        self.assertIn("64", arms[0].axes)

    def test_a_missing_metric_leaves_the_arm_present(self):
        # An arm whose event metrics never logged still belongs in the table
        # with its frame score; dropping it would hide a run silently.
        arms = ablation_report.collect([run("abl-base-s42", 0.8)], [FRAME_METRIC, "absent"])
        self.assertEqual(arms[0].metrics.get("absent", []), [])


class TestSpread(unittest.TestCase):
    def test_it_reports_mean_and_deviation_as_percent(self):
        self.assertIn("80.00", ablation_report.spread([0.8, 0.8, 0.8]))

    def test_a_single_seed_has_zero_deviation(self):
        self.assertIn("0.00", ablation_report.spread([0.8]))

    def test_a_missing_metric_reads_as_a_dash(self):
        self.assertEqual(ablation_report.spread([]), "--")

    def test_the_deviation_is_the_sample_standard_deviation(self):
        # Three seeds at 0.90/0.92/0.94 -> mean 0.92, sd 0.02.
        cell = ablation_report.spread([0.90, 0.92, 0.94])
        self.assertIn("92.00", cell)
        self.assertIn("2.00", cell)


class TestReport(unittest.TestCase):
    def test_an_empty_sweep_says_what_to_run(self):
        # Printed rather than raised: an empty table is the normal state before
        # the first stage, not an error.
        ablation_report.report([], {})

    def test_a_populated_table_prints(self):
        arms = ablation_report.collect([run("abl-base-s42", 0.8)], [FRAME_METRIC])
        ablation_report.report(arms, {})


if __name__ == "__main__":
    unittest.main()
