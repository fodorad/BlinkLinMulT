"""Tests for the comparison protocol.

The protocol's job is to assemble primitives that are already tested
individually, so these concentrate on the assembly: that the comparison is
genuinely paired, that misaligned inputs are refused rather than silently
subtracted, and that a corpus too small for an interval says so instead of
producing one.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.compare.protocol import (
    Comparison,
    ProtocolError,
    compare_average_precision,
)

ROUNDS = 200
"""Replicates in tests; small enough to stay fast."""


def _store(n_recordings: int, quality: float, seed: int = 0) -> dict:
    """Build a synthetic signal store.

    Args:
        n_recordings (int): Recordings to create.
        quality (float): How well the signal separates the positives. At 1.0 the
            score is the label; at 0.0 it is noise.
        seed (int): Seed for the noise.

    Returns:
        dict: In the shape :func:`load_signals` returns.
    """
    rng = np.random.default_rng(seed)
    store = {}
    for index in range(n_recordings):
        truth = np.zeros(60, dtype=np.float32)
        truth[10:14] = 1.0
        truth[40:44] = 1.0
        noise = rng.random(60).astype(np.float32)
        signal = (quality * truth + (1.0 - quality) * noise).astype(np.float32)
        store[f"rec_{index:02d}"] = {
            "signal": signal,
            "truth": truth,
            "mask": np.ones(60, dtype=bool),
        }
    return store


class TestPairing(unittest.TestCase):
    """The comparison must be paired across models."""

    def test_a_better_model_scores_higher(self) -> None:
        """A sanity floor: the protocol must order obvious cases correctly."""
        stores = {"good": _store(20, 0.95), "bad": _store(20, 0.05, seed=1)}
        report = compare_average_precision(stores, "synthetic", rounds=ROUNDS)
        self.assertGreater(report.scores["good"].point, report.scores["bad"].point)

    def test_a_clear_difference_is_called_significant(self) -> None:
        """With a large effect and 20 clusters, the pair must separate."""
        stores = {"good": _store(20, 0.95), "bad": _store(20, 0.05, seed=1)}
        report = compare_average_precision(stores, "synthetic", rounds=ROUNDS)
        self.assertTrue(report.comparisons[0].conclusive)

    def test_identical_models_never_separate(self) -> None:
        """A model compared with a copy of itself differs by exactly zero."""
        store = _store(20, 0.7)
        report = compare_average_precision({"a": store, "b": store}, "synthetic", rounds=ROUNDS)
        comparison = report.comparisons[0]
        self.assertAlmostEqual(comparison.interval.point, 0.0, places=12)
        self.assertFalse(comparison.conclusive)
        self.assertEqual(comparison.p_raw, 1.0)


class TestAlignment(unittest.TestCase):
    """Refusing to subtract scores computed on different data."""

    def test_differing_recordings_are_refused(self) -> None:
        """An unpaired subtraction would compare two different corpora."""
        first = _store(10, 0.8)
        second = _store(10, 0.8)
        second["extra"] = second.pop("rec_00")
        with self.assertRaises(ProtocolError) as caught:
            compare_average_precision({"a": first, "b": second}, "synthetic", rounds=ROUNDS)
        self.assertIn("different recordings", str(caught.exception))

    def test_one_model_is_refused(self) -> None:
        """There is no comparison to make."""
        with self.assertRaises(ProtocolError):
            compare_average_precision({"only": _store(10, 0.8)}, "synthetic", rounds=ROUNDS)


class TestFamilyOfComparisons(unittest.TestCase):
    """Four models make six pairs, corrected together."""

    def test_four_models_give_six_pairs(self) -> None:
        """Every unordered pair once, never a model against itself."""
        stores = {name: _store(12, 0.5 + i * 0.1, seed=i) for i, name in enumerate("abcd")}
        report = compare_average_precision(stores, "synthetic", rounds=ROUNDS)
        self.assertEqual(len(report.comparisons), 6)
        pairs = {(c.first, c.second) for c in report.comparisons}
        self.assertEqual(len(pairs), 6)
        self.assertFalse(any(first == second for first, second in pairs))

    def test_correction_never_lowers_a_p_value(self) -> None:
        """Holm across six pairs must make each claim harder, not easier."""
        stores = {name: _store(12, 0.5 + i * 0.1, seed=i) for i, name in enumerate("abcd")}
        report = compare_average_precision(stores, "synthetic", rounds=ROUNDS)
        for comparison in report.comparisons:
            with self.subTest(pair=f"{comparison.first}-{comparison.second}"):
                self.assertGreaterEqual(comparison.p_holm, comparison.p_raw)


class TestSmallCorpora(unittest.TestCase):
    """A corpus that cannot carry an interval must say so."""

    def test_a_single_recording_gets_no_interval(self) -> None:
        """The TalkingFace case: 524 windows, one video."""
        stores = {"a": _store(1, 0.9), "b": _store(1, 0.2, seed=1)}
        report = compare_average_precision(stores, "talkingface-like", rounds=ROUNDS)
        self.assertFalse(report.resolvable)
        self.assertFalse(report.comparisons[0].interval.reportable)

    def test_an_unresolvable_corpus_is_never_conclusive(self) -> None:
        """A missing interval must not read as a significant result."""
        stores = {"a": _store(1, 0.99), "b": _store(1, 0.01, seed=1)}
        report = compare_average_precision(stores, "talkingface-like", rounds=ROUNDS)
        self.assertFalse(report.comparisons[0].conclusive)

    def test_point_estimates_survive(self) -> None:
        """Refusing an interval must not refuse the measurement."""
        stores = {"a": _store(1, 0.9), "b": _store(1, 0.2, seed=1)}
        report = compare_average_precision(stores, "talkingface-like", rounds=ROUNDS)
        self.assertGreater(report.scores["a"].point, report.scores["b"].point)


class TestReportMetadata(unittest.TestCase):
    """The context a number must travel with."""

    def test_cluster_count_is_reported(self) -> None:
        """The honest sample size, not the window count."""
        stores = {"a": _store(14, 0.8), "b": _store(14, 0.6, seed=1)}
        report = compare_average_precision(stores, "synthetic", rounds=ROUNDS)
        self.assertEqual(report.n_clusters, 14)

    def test_positive_rate_is_reported(self) -> None:
        """At ~1-4% it is why average precision leads rather than accuracy."""
        stores = {"a": _store(14, 0.8), "b": _store(14, 0.6, seed=1)}
        report = compare_average_precision(stores, "synthetic", rounds=ROUNDS)
        self.assertAlmostEqual(report.positive_rate, 8 / 60, places=6)

    def test_frame_count_counts_only_valid_frames(self) -> None:
        """Masked positions are not evidence and must not be counted."""
        stores = {"a": _store(5, 0.8), "b": _store(5, 0.6, seed=1)}
        for store in stores.values():
            for record in store.values():
                record["mask"][:10] = False
        report = compare_average_precision(stores, "synthetic", rounds=ROUNDS)
        self.assertEqual(report.n_frames, 5 * 50)


class TestComparisonRendering(unittest.TestCase):
    """How a pair reads in a table."""

    def test_a_conclusive_pair_says_so(self) -> None:
        """The verdict is spelled out, not left to the reader."""
        from blinklinmult.compare.bootstrap import Interval

        comparison = Comparison("a", "b", Interval(0.2, 0.1, 0.3, 20, 100, 1), 0.001, 0.006)
        self.assertIn("significant", comparison.describe())
        self.assertTrue(comparison.conclusive)

    def test_a_significant_interval_with_a_failed_correction_is_not_conclusive(self) -> None:
        """Holm is what six comparisons cost; the stricter reading wins."""
        from blinklinmult.compare.bootstrap import Interval

        comparison = Comparison("a", "b", Interval(0.2, 0.1, 0.3, 20, 100, 1), 0.02, 0.12)
        self.assertFalse(comparison.conclusive)
        self.assertIn("not significant", comparison.describe())


if __name__ == "__main__":
    unittest.main()
