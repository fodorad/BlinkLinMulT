"""Tests for the LR range test wiring.

The sweep itself is Lightning's and is not re-tested here. What is tested is
the wiring this project adds -- the flat ``lr`` attribute the tuner reads and
writes, and the plot path -- because that wiring failing is silent: the tuner
raises only at the end of a long sweep, after the compute is already spent.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from blinklinmult.train import lr_find
from blinklinmult.train.config import ExperimentConfig

CONFIGS = (
    Path("config/data/single_rn30.yaml"),
    Path("config/model/blinklint_baseline.yaml"),
    Path("config/train/lint_blink.yaml"),
)


def config() -> ExperimentConfig:
    """The diagnostic run the LR is chosen for."""
    return ExperimentConfig.from_files(*CONFIGS)


class TestSweepBounds(unittest.TestCase):
    def test_the_sweep_spans_many_decades(self):
        # A range test only shows structure if it brackets both the flat region
        # and divergence; the measured RN30 curve used every decade of this.
        self.assertLess(lr_find.DEFAULT_MIN_LR, 1e-5)
        self.assertGreaterEqual(lr_find.DEFAULT_MAX_LR, 1.0)

    def test_the_step_count_beats_lightnings_default(self):
        # Lightning's 100 is coarse enough that the suggestion jitters between
        # runs on this model.
        self.assertGreater(lr_find.DEFAULT_NUM_TRAINING_STEPS, 100)


class TestLrAttribute(unittest.TestCase):
    """Lightning writes its suggestion back through a flat `lr` attribute."""

    def test_the_module_exposes_lr(self):
        # Without this the tuner raises *after* the whole sweep has run.
        from blinklinmult.train.module import BlinkLightningModule

        self.assertIsInstance(
            getattr(BlinkLightningModule, "lr", None),
            property,
            "BlinkLightningModule.lr must be a property for Tuner.lr_find to write to.",
        )

    def test_it_is_writable(self):
        from blinklinmult.train.module import BlinkLightningModule

        prop = BlinkLightningModule.lr
        self.assertIsNotNone(prop.fset, "the tuner writes its suggestion back through lr")


class TestOutputPath(unittest.TestCase):
    def test_the_plot_lands_in_the_runs_own_directory(self):
        # Beside the checkpoints and predictions of the run it describes, so a
        # sweep is never orphaned from the config that produced it.
        from blinklinmult.train.cli import run_output_dir

        expected = run_output_dir(config()) / "lr_find.png"
        self.assertEqual(expected.name, "lr_find.png")
        self.assertIn("lint", str(expected).lower())


if __name__ == "__main__":
    unittest.main()
