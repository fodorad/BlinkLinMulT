"""Tests for the timing harness.

**Nothing here asserts a wall-clock duration.** A test that expects a call to
take under N milliseconds fails on a loaded CI runner for reasons that have
nothing to do with the code, and the usual fix is to loosen the bound until it
proves nothing. These assert structure instead: that warm-up calls really are
discarded, that the percentiles are ordered, and that the per-frame figure
divides by the right number.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.bench.timing import BenchError, Latency, time_callable


class _Counter:
    """A callable that records how often it was invoked."""

    def __init__(self) -> None:
        """Start at zero."""
        self.calls = 0

    def __call__(self) -> None:
        """Count one invocation."""
        self.calls += 1


class TestTimeCallable(unittest.TestCase):
    """Running the harness."""

    def test_warmup_calls_happen_but_are_not_timed(self) -> None:
        """The callable runs ``warmup + repeats`` times, and ``repeats`` land.

        This is the property the whole warm-up idea rests on: the first calls
        must actually execute, so arenas are allocated and kernels chosen, yet
        contribute nothing to the reported figure.
        """
        counter = _Counter()
        latency = time_callable(counter, frames=1, warmup=7, repeats=13)
        self.assertEqual(counter.calls, 20)
        self.assertEqual(latency.samples_ms.size, 13)

    def test_zero_warmup_is_allowed(self) -> None:
        """Sometimes the cold path is the thing being measured."""
        counter = _Counter()
        time_callable(counter, frames=1, warmup=0, repeats=5)
        self.assertEqual(counter.calls, 5)

    def test_samples_are_non_negative(self) -> None:
        """A monotonic clock cannot produce a negative duration."""
        latency = time_callable(_Counter(), frames=1, warmup=1, repeats=10)
        self.assertTrue(bool((latency.samples_ms >= 0).all()))

    def test_frames_are_recorded(self) -> None:
        """The per-frame figure depends on this being what was asked for."""
        latency = time_callable(_Counter(), frames=15, warmup=1, repeats=5)
        self.assertEqual(latency.frames, 15)

    def test_impossible_parameters_are_refused(self) -> None:
        """Zero frames would divide by zero; zero repeats measures nothing."""
        for frames, warmup, repeats in ((0, 1, 5), (1, 1, 0), (1, -1, 5)):
            with (
                self.subTest(frames=frames, warmup=warmup, repeats=repeats),
                self.assertRaises(BenchError),
            ):
                time_callable(_Counter(), frames=frames, warmup=warmup, repeats=repeats)


class TestLatencySummary(unittest.TestCase):
    """Deriving figures from the samples."""

    def _latency(self, samples: list[float], frames: int = 1) -> Latency:
        """Build a Latency from known samples.

        Args:
            samples (list[float]): Millisecond timings.
            frames (int): Frames per call.

        Returns:
            Latency: The summary object.
        """
        return Latency(samples_ms=np.array(samples, dtype=float), warmup=0, frames=frames)

    def test_percentiles_are_ordered(self) -> None:
        """p50 <= p95 <= p99 must hold for any input."""
        latency = self._latency([float(value) for value in range(1, 201)])
        self.assertLessEqual(latency.p50_ms, latency.p95_ms)
        self.assertLessEqual(latency.p95_ms, latency.p99_ms)

    def test_the_median_is_the_median(self) -> None:
        """A known vector, so a percentile mix-up shows up as a wrong number."""
        self.assertAlmostEqual(self._latency([10.0, 20.0, 30.0]).p50_ms, 20.0, places=9)

    def test_per_frame_divides_by_the_window(self) -> None:
        """A 15-frame call at 150 ms costs 10 ms per frame.

        Without this, a sequence model scoring 15 frames per call would look
        15x slower than a frame-wise one doing the same work.
        """
        latency = self._latency([150.0] * 5, frames=15)
        self.assertAlmostEqual(latency.ms_per_frame, 10.0, places=9)

    def test_throughput_matches_the_median(self) -> None:
        """15 frames in 150 ms is 100 frames per second."""
        latency = self._latency([150.0] * 5, frames=15)
        self.assertAlmostEqual(latency.fps, 100.0, places=6)

    def test_a_short_run_does_not_claim_a_p99(self) -> None:
        """Below 100 samples the 99th percentile is the maximum renamed."""
        self.assertFalse(self._latency([1.0] * 20).tail_is_trustworthy)
        self.assertNotIn("p99", self._latency([1.0] * 20).describe())

    def test_a_long_run_reports_its_p99(self) -> None:
        """With enough samples the tail is worth printing."""
        latency = self._latency([1.0] * 200)
        self.assertTrue(latency.tail_is_trustworthy)
        self.assertIn("p99", latency.describe())

    def test_describe_carries_the_headline_figures(self) -> None:
        """A table row must show the cost and the throughput."""
        text = self._latency([150.0] * 200, frames=15).describe()
        self.assertIn("ms/frame", text)
        self.assertIn("fps", text)


if __name__ == "__main__":
    unittest.main()
