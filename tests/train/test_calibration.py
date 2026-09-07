"""Tests for per-corpus calibration."""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.train.calibration import (
    CalibrationError,
    apply_temperature,
    expected_calibration_error,
    fit_temperature,
    fit_threshold,
    logit,
)


class TestLogit(unittest.TestCase):
    """The inverse sigmoid stays finite at the asymptotes."""

    def test_it_is_finite_at_zero_and_one(self):
        values = logit(np.array([0.0, 1.0]))
        self.assertTrue(np.all(np.isfinite(values)))

    def test_it_is_zero_at_one_half(self):
        self.assertAlmostEqual(float(logit(np.array([0.5]))[0]), 0.0)

    def test_it_inverts_the_sigmoid(self):
        probability = np.array([0.1, 0.3, 0.7, 0.9])
        restored = 1.0 / (1.0 + np.exp(-logit(probability)))
        np.testing.assert_allclose(restored, probability, atol=1e-6)


class TestApplyTemperature(unittest.TestCase):
    """Temperature moves the boundary without reordering anything."""

    def test_unit_temperature_is_the_identity(self):
        probability = np.array([0.2, 0.5, 0.8])
        np.testing.assert_allclose(apply_temperature(probability, 1.0), probability, atol=1e-6)

    def test_it_preserves_the_ranking(self):
        """The property that makes this safe: AUC cannot change."""
        rng = np.random.default_rng(0)
        probability = rng.uniform(0.001, 0.999, 500)
        for temperature in (0.2, 0.5, 2.0, 8.0):
            calibrated = apply_temperature(probability, temperature)
            self.assertTrue(
                np.array_equal(np.argsort(probability), np.argsort(calibrated)),
                f"temperature {temperature} reordered the scores",
            )

    def test_a_low_temperature_sharpens(self):
        sharpened = apply_temperature(np.array([0.6]), 0.5)
        self.assertGreater(sharpened[0], 0.6)

    def test_a_high_temperature_softens(self):
        softened = apply_temperature(np.array([0.6]), 5.0)
        self.assertLess(softened[0], 0.6)

    def test_it_rejects_a_non_positive_temperature(self):
        with self.assertRaises(CalibrationError):
            apply_temperature(np.array([0.5]), 0.0)


class TestFitThreshold(unittest.TestCase):
    """The cheaper estimator picks the boundary maximising validation F1."""

    def test_it_finds_a_separating_boundary(self):
        probability = np.array([0.05, 0.08, 0.11, 0.80, 0.85, 0.90])
        target = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        threshold, f1 = fit_threshold(probability, target)
        self.assertAlmostEqual(f1, 1.0)
        self.assertTrue(0.11 < threshold <= 0.80)

    def test_it_moves_the_boundary_for_a_compressed_score(self):
        """The RN regime: positives real but scored low."""
        probability = np.concatenate([np.full(200, 0.02), np.full(10, 0.19)])
        target = np.concatenate([np.zeros(200), np.ones(10)])
        threshold, f1 = fit_threshold(probability, target)
        self.assertAlmostEqual(f1, 1.0)
        self.assertLess(threshold, 0.5)

    def test_it_rejects_mismatched_lengths(self):
        with self.assertRaises(CalibrationError):
            fit_threshold(np.array([0.5, 0.5]), np.array([1.0]))

    def test_it_rejects_an_all_negative_split(self):
        with self.assertRaises(CalibrationError):
            fit_threshold(np.array([0.1, 0.2]), np.array([0.0, 0.0]))


class TestFitTemperature(unittest.TestCase):
    """Temperature is fitted by NLL, not by a thresholded score."""

    def test_a_calibrated_input_keeps_temperature_near_one(self):
        rng = np.random.default_rng(1)
        probability = rng.uniform(0.02, 0.98, 4000)
        target = (rng.uniform(size=probability.size) < probability).astype(float)
        temperature, _ = fit_temperature(probability, target)
        self.assertGreater(temperature, 0.6)
        self.assertLess(temperature, 1.7)

    def test_it_sharpens_an_under_confident_score(self):
        """Under-confident positives should pull the temperature below 1."""
        rng = np.random.default_rng(2)
        truth = (rng.uniform(size=4000) < 0.5).astype(float)
        # Squash towards 0.5: confident labels, timid probabilities.
        probability = np.where(truth > 0.5, 0.55, 0.45)
        temperature, _ = fit_temperature(probability, truth)
        self.assertLess(temperature, 1.0)

    def test_it_lowers_the_negative_log_likelihood(self):
        rng = np.random.default_rng(3)
        truth = (rng.uniform(size=2000) < 0.5).astype(float)
        probability = np.where(truth > 0.5, 0.55, 0.45)
        temperature, fitted_loss = fit_temperature(probability, truth)
        _, unit_loss = fit_temperature(probability, truth, temperatures=np.array([1.0]))
        self.assertLess(fitted_loss, unit_loss)
        self.assertNotAlmostEqual(temperature, 1.0)

    def test_it_rejects_mismatched_lengths(self):
        with self.assertRaises(CalibrationError):
            fit_temperature(np.array([0.5, 0.5]), np.array([1.0]))

    def test_it_rejects_empty_input(self):
        with self.assertRaises(CalibrationError):
            fit_temperature(np.array([]), np.array([]))


class TestExpectedCalibrationError(unittest.TestCase):
    """ECE shows miscalibration that F1 alone hides."""

    def test_a_perfectly_calibrated_score_is_near_zero(self):
        rng = np.random.default_rng(4)
        probability = rng.uniform(0.0, 1.0, 40000)
        target = (rng.uniform(size=probability.size) < probability).astype(float)
        self.assertLess(expected_calibration_error(probability, target), 0.05)

    def test_a_systematically_low_score_is_penalised(self):
        """Always predicting 0.1 for a class that occurs half the time."""
        probability = np.full(1000, 0.1)
        target = np.concatenate([np.ones(500), np.zeros(500)])
        self.assertGreater(expected_calibration_error(probability, target), 0.3)

    def test_it_handles_scores_at_exactly_one(self):
        """digitize must not push 1.0 past the last bin."""
        probability = np.ones(10)
        target = np.ones(10)
        self.assertAlmostEqual(expected_calibration_error(probability, target), 0.0)

    def test_it_returns_nan_for_empty_input(self):
        self.assertTrue(np.isnan(expected_calibration_error(np.array([]), np.array([]))))


if __name__ == "__main__":
    unittest.main()
