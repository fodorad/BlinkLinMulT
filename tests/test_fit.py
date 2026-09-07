"""Tests for fitting an operating point.

The 1.x models shipped without one, so this is what makes an event-level
comparison against ``BlinkCNN`` fair. Two properties carry that fairness and are
tested hardest: pooling counts across recordings (so a short clip cannot outvote
a long session), and preferring a single threshold on ties (so hysteresis has to
earn its extra parameter).
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.fit import DEFAULT_LOW_RATIOS, FitError, OperatingPoint, fit_operating_point
from blinklinmult.registry import spec


def _recording(blinks: list[int], length: int = 200, seed: int = 0) -> tuple[np.ndarray, list]:
    """Build a signal with clean blinks in low-level noise.

    Args:
        blinks (list[int]): Frame indices where a blink starts.
        length (int): Total frames.
        seed (int): Noise seed.

    Returns:
        tuple[np.ndarray, list]: The signal and its annotated intervals.
    """
    rng = np.random.default_rng(seed)
    signal = rng.random(length) * 0.15
    shape = [0.3, 0.7, 0.95, 0.9, 0.6, 0.25]
    annotated = []
    for start in blinks:
        for offset, value in enumerate(shape):
            signal[start + offset] = value
        annotated.append((start, start + len(shape) - 1))
    return signal, annotated


class TestFitting(unittest.TestCase):
    """Searching the threshold grid."""

    def test_recovers_clean_blinks(self) -> None:
        """A signal with obvious blinks fits to a perfect score."""
        signal, annotated = _recording([30, 120])
        point = fit_operating_point([signal], [annotated])
        self.assertAlmostEqual(point.f1, 1.0)

    def test_returns_a_usable_threshold(self) -> None:
        """The fitted threshold separates the blinks from the noise.

        Noise here is ``random() * 0.15``, so every noise sample is strictly
        below 0.15 while a blink peaks at 0.95. Any cut in ``[0.15, 0.95]``
        separates them, and the grid's lowest such value is the expected answer.
        """
        signal, annotated = _recording([40])
        point = fit_operating_point([signal], [annotated])
        self.assertGreaterEqual(point.threshold, 0.15)
        self.assertLessEqual(point.threshold, 0.95)

    def test_pools_across_recordings(self) -> None:
        """Several recordings fit jointly rather than one at a time."""
        first, first_truth = _recording([30, 120], seed=1)
        second, second_truth = _recording([50], length=100, seed=2)
        point = fit_operating_point([first, second], [first_truth, second_truth])
        self.assertGreater(point.f1, 0.0)

    def test_low_threshold_is_derived_from_the_ratio(self) -> None:
        """``low_threshold`` is the ratio times the high threshold."""
        point = OperatingPoint(threshold=0.6, low_ratio=0.25, f1=0.5, precision=0.5, recall=0.5)
        self.assertAlmostEqual(point.low_threshold or 0.0, 0.15)

    def test_single_cut_reports_no_low_threshold(self) -> None:
        """With no ratio there is no second threshold to report."""
        point = OperatingPoint(threshold=0.6, low_ratio=None, f1=0.5, precision=0.5, recall=0.5)
        self.assertIsNone(point.low_threshold)

    def test_ties_prefer_the_simpler_rule(self) -> None:
        """``None`` leads the ratio grid so a single cut wins equal scores.

        Hysteresis adds a parameter and its variance; it should be adopted only
        when it actually scores better, not on a rounding difference.
        """
        self.assertIsNone(DEFAULT_LOW_RATIOS[0])


class TestApplying(unittest.TestCase):
    """Writing a fitted point back onto a model spec."""

    def test_produces_an_updated_spec(self) -> None:
        """The returned spec carries the fitted thresholds."""
        point = OperatingPoint(threshold=0.42, low_ratio=0.2, f1=0.8, precision=0.8, recall=0.8)
        updated = point.applied_to(spec("densenet121-union"))
        self.assertAlmostEqual(updated.threshold, 0.42)
        self.assertAlmostEqual(updated.low_ratio or 0.0, 0.2)

    def test_leaves_the_registry_untouched(self) -> None:
        """Applying must not mutate the shared table.

        ``MODELS`` is module-level state; a fit for one experiment silently
        changing every later caller's default would be a nasty bug.
        """
        before = spec("densenet121-union").threshold
        point = OperatingPoint(threshold=0.9, low_ratio=None, f1=1.0, precision=1.0, recall=1.0)
        point.applied_to(spec("densenet121-union"))
        self.assertEqual(spec("densenet121-union").threshold, before)

    def test_keeps_the_rest_of_the_spec(self) -> None:
        """Only the operating point changes; normalisation must survive."""
        original = spec("densenet121-union")
        point = OperatingPoint(threshold=0.3, low_ratio=None, f1=0.5, precision=0.5, recall=0.5)
        updated = point.applied_to(original)
        self.assertEqual(updated.mean, original.mean)
        self.assertEqual(updated.filename, original.filename)


class TestValidation(unittest.TestCase):
    """Inputs that cannot produce a meaningful fit."""

    def test_mismatched_lengths_are_rejected(self) -> None:
        """Signals and annotations must correspond one to one."""
        signal, annotated = _recording([30])
        with self.assertRaises(FitError):
            fit_operating_point([signal, signal], [annotated])

    def test_empty_input_is_rejected(self) -> None:
        """There is nothing to fit on."""
        with self.assertRaises(FitError):
            fit_operating_point([], [])

    def test_no_annotated_blinks_is_rejected(self) -> None:
        """Without a positive example every threshold scores zero.

        Returning an arbitrary "best" would look like a successful fit.
        """
        signal, _ = _recording([30])
        with self.assertRaises(FitError):
            fit_operating_point([signal], [[]])


if __name__ == "__main__":
    unittest.main()
