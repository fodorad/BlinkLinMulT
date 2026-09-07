"""Tests for phase derivation and event-shape measurement.

Phase is read off the signal's **derivative**, not off the annotation's
geometry, because no corpus annotates phase and deriving it from interval
position would assert a closure shape rather than measure one. These tests pin
that: the same value must receive a different phase depending on which way it is
moving.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.train.phases import (
    CLOSED,
    CLOSING,
    OPEN,
    OPENING,
    PhaseError,
    classify_event,
    describe_event,
    phases,
    smooth,
)


def blink(depth: float = 1.0, fall: int = 3, hold: int = 2, rise: int = 4) -> np.ndarray:
    """A synthetic closure: descend, hold, ascend, on an open baseline."""
    return np.concatenate(
        [
            np.zeros(4),
            np.linspace(0.0, depth, fall + 1)[1:],
            np.full(hold, depth),
            np.linspace(depth, 0.0, rise + 1)[1:],
            np.zeros(4),
        ]
    )


class TestSmooth(unittest.TestCase):
    def test_it_keeps_the_length(self):
        self.assertEqual(smooth(np.arange(10.0), 3).size, 10)

    def test_a_window_of_one_is_a_no_op(self):
        values = np.array([0.0, 1.0, 0.0])
        np.testing.assert_array_equal(smooth(values, 1), values)

    def test_it_reduces_single_frame_noise(self):
        spike = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
        self.assertLess(smooth(spike, 3).max(), spike.max())

    def test_the_edges_are_not_pulled_to_zero(self):
        # Zero padding would invent a ramp at each end and so invent a phase.
        flat = np.ones(6)
        np.testing.assert_allclose(smooth(flat, 3), flat)

    def test_an_even_window_is_rejected(self):
        with self.assertRaises(PhaseError):
            smooth(np.arange(5.0), 4)

    def test_an_empty_signal_is_handled(self):
        self.assertEqual(smooth(np.zeros(0), 3).size, 0)


class TestPhases(unittest.TestCase):
    def test_an_open_eye_is_open(self):
        self.assertTrue(np.all(phases(np.zeros(10)) == OPEN))

    def test_a_held_closure_is_closed(self):
        self.assertTrue(np.all(phases(np.ones(10)) == CLOSED))

    def test_the_descent_is_closing(self):
        labelled = phases(blink())
        self.assertIn(CLOSING, labelled[:9])

    def test_the_ascent_is_opening(self):
        labelled = phases(blink())
        self.assertIn(OPENING, labelled[9:])

    def test_the_same_value_differs_by_direction(self):
        # The core claim: 0.5 alone is ambiguous, 0.5-and-falling is not.
        signal = blink()
        labelled = phases(signal)
        midway = np.flatnonzero(np.isclose(signal, 0.5, atol=0.2))
        kinds = {int(labelled[i]) for i in midway}
        self.assertIn(CLOSING, kinds)
        self.assertIn(OPENING, kinds)

    def test_it_labels_every_frame(self):
        self.assertEqual(phases(blink()).size, blink().size)

    def test_an_empty_signal_yields_no_labels(self):
        self.assertEqual(phases(np.zeros(0)).size, 0)

    def test_noise_on_an_open_eye_is_not_motion(self):
        rng = np.random.default_rng(0)
        jitter = rng.normal(0.0, 0.003, 40).clip(0.0, 1.0)
        self.assertTrue(np.all(phases(jitter) == OPEN))


class TestDescribeEvent(unittest.TestCase):
    def test_it_measures_the_span(self):
        signal = blink()
        shape = describe_event(signal, 4, 12)
        self.assertEqual(shape["frames"], 9.0)

    def test_a_full_closure_peaks_high(self):
        self.assertGreater(describe_event(blink(depth=1.0), 4, 12)["peak"], 0.9)

    def test_a_shallow_closure_peaks_low(self):
        self.assertLess(describe_event(blink(depth=0.4), 4, 12)["peak"], 0.5)

    def test_a_single_closure_has_one_peak(self):
        self.assertEqual(describe_event(blink(), 4, 12)["n_peaks"], 1.0)

    def test_a_double_blink_has_two_peaks(self):
        # Two closures inside one elevated run -- indistinguishable to any
        # single threshold, which is the argument for reading the shape.
        signal = np.concatenate([np.zeros(2), [0.9, 0.35, 0.9], np.zeros(2)])
        self.assertEqual(describe_event(signal, 2, 4)["n_peaks"], 2.0)

    def test_asymmetry_is_positive_when_reopening_is_slower(self):
        # Real blinks close faster than they open.
        shape = describe_event(blink(fall=2, hold=1, rise=6), 4, 12)
        self.assertGreater(shape["asymmetry"], 0.0)

    def test_a_span_outside_the_signal_is_rejected(self):
        with self.assertRaises(PhaseError):
            describe_event(np.zeros(5), 3, 9)

    def test_an_inverted_span_is_rejected(self):
        with self.assertRaises(PhaseError):
            describe_event(np.zeros(5), 4, 2)


class TestClassifyEvent(unittest.TestCase):
    def shape(self, **kwargs) -> dict[str, float]:
        base = {
            "frames": 8.0,
            "peak": 0.95,
            "mean": 0.6,
            "n_peaks": 1.0,
            "fall_frames": 3.0,
            "rise_frames": 4.0,
            "asymmetry": 1.0,
        }
        base.update(kwargs)
        return base

    def test_a_full_closure_is_complete(self):
        self.assertEqual(classify_event(self.shape()), "complete")

    def test_two_peaks_make_it_double(self):
        self.assertEqual(classify_event(self.shape(n_peaks=2.0)), "double")

    def test_a_shallow_closure_is_incomplete(self):
        self.assertEqual(classify_event(self.shape(peak=0.45)), "incomplete")

    def test_a_held_closure_is_long(self):
        self.assertEqual(classify_event(self.shape(frames=30.0)), "long")

    def test_a_one_frame_closure_is_brief(self):
        self.assertEqual(classify_event(self.shape(frames=1.0)), "brief")

    def test_double_outranks_depth(self):
        # A double blink whose second closure is shallow is still a double.
        self.assertEqual(classify_event(self.shape(n_peaks=2.0, peak=0.5)), "double")
