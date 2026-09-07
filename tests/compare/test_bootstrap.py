"""Tests for the paired cluster bootstrap.

Two of these check exact, hand-derivable numbers rather than inequalities, which
is what makes them useful: a vague ``assertLess`` passes for many wrong
implementations, while ``p == 2 / (B + 1)`` passes for essentially one.

The third pins the property the whole package exists for -- that clustered
resampling gives a *wider* interval than naive resampling on correlated data.
Measured on this repo's own arms, the naive interval was 5.9x too narrow and
reversed the conclusion.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.compare.bootstrap import (
    MIN_CLUSTERS_FOR_INTERVAL,
    Interval,
    bootstrap_statistic,
    two_sided_p,
)

ROUNDS = 200
"""Replicates in tests.

Small so the suite stays fast; the exact-p assertions are written against this
value rather than the production default, so they stay correct either way.
"""


def _difference(better: np.ndarray, worse: np.ndarray):
    """Build a paired per-cluster difference statistic.

    Args:
        better (np.ndarray): Per-cluster score of one model.
        worse (np.ndarray): Per-cluster score of the other.

    Returns:
        Callable: Maps cluster indices to the mean paired difference.
    """

    def statistic(indices) -> float:
        picked = np.asarray(list(indices))
        return float(better[picked].mean() - worse[picked].mean())

    return statistic


class TestKnownAnswers(unittest.TestCase):
    """Cases whose correct output can be derived by hand."""

    def test_strict_dominance_gives_the_smallest_possible_p(self) -> None:
        """When A beats B in every cluster, p is exactly ``2 / (B + 1)``.

        Every resample -- whatever clusters it draws -- yields a positive
        difference, so no draw ever lands at or below zero. The estimator's
        floor is then ``2 * (1 + 0) / (B + 1)``, which is the least a finite
        bootstrap can evidence.

        The equality pins three things at once: the ``+1`` correction, the
        two-sided doubling, and the replicate count. Break any one and this
        fails with a specific wrong number.
        """
        better = np.linspace(0.80, 0.90, 20)
        worse = better - 0.10
        _, draws = bootstrap_statistic(_difference(better, worse), 20, ROUNDS, seed=1)
        self.assertAlmostEqual(two_sided_p(draws), 2 / (ROUNDS + 1), places=12)

    def test_identical_models_give_p_of_exactly_one(self) -> None:
        """A model compared with itself differs by zero in every resample.

        Both tail counts are then ``B``, so ``2 * min(1, 1)`` clips to 1.0. A
        sign error or an off-by-one would make a model differ from itself.
        """
        scores = np.linspace(0.3, 0.7, 15)
        interval, draws = bootstrap_statistic(_difference(scores, scores), 15, ROUNDS, seed=1)
        self.assertEqual(two_sided_p(draws), 1.0)
        self.assertAlmostEqual(interval.point, 0.0, places=12)
        self.assertAlmostEqual(interval.low or 0.0, 0.0, places=12)
        self.assertAlmostEqual(interval.high or 0.0, 0.0, places=12)

    def test_a_p_value_can_never_be_zero(self) -> None:
        """Claiming p=0 asserts certainty no finite resampling supports."""
        better = np.full(20, 0.9)
        worse = np.full(20, 0.1)
        _, draws = bootstrap_statistic(_difference(better, worse), 20, ROUNDS, seed=1)
        self.assertGreater(two_sided_p(draws), 0.0)

    def test_dominance_interval_excludes_zero(self) -> None:
        """The interval and the p-value must agree with each other."""
        better = np.linspace(0.80, 0.90, 20)
        interval, _ = bootstrap_statistic(_difference(better, better - 0.1), 20, ROUNDS, seed=1)
        self.assertTrue(interval.excludes_zero)


class TestClusteringWidensTheInterval(unittest.TestCase):
    """The property the package exists to guarantee."""

    def test_correlated_data_gives_a_wider_clustered_interval(self) -> None:
        """Ignoring within-cluster correlation shrinks the interval.

        Each cluster here holds 30 near-identical observations, mimicking the
        ~209 windows per RN30 recording. Resampling the 20 clusters respects
        that structure; resampling all 600 observations pretends there are 600
        independent facts when there are 20.
        """
        rng = np.random.default_rng(0)
        per_cluster, size = 20, 30
        effect = rng.normal(0.05, 0.20, per_cluster)
        clustered = np.repeat(effect, size) + rng.normal(0, 0.001, per_cluster * size)

        def by_cluster(indices) -> float:
            picked = np.asarray(list(indices))
            return float(effect[picked].mean())

        def by_observation(indices) -> float:
            picked = np.asarray(list(indices))
            return float(clustered[picked].mean())

        cluster_ci, _ = bootstrap_statistic(by_cluster, per_cluster, 400, seed=3)
        naive_ci, _ = bootstrap_statistic(by_observation, per_cluster * size, 400, seed=3)

        self.assertTrue(cluster_ci.reportable and naive_ci.reportable)
        clustered_width = float(cluster_ci.high or 0.0) - float(cluster_ci.low or 0.0)
        naive_width = float(naive_ci.high or 0.0) - float(naive_ci.low or 0.0)
        self.assertGreater(
            clustered_width,
            naive_width * 3,
            "clustered resampling must be markedly wider on correlated data",
        )


class TestTooFewClusters(unittest.TestCase):
    """Refusing an interval the data cannot carry."""

    def test_a_single_cluster_gets_no_interval(self) -> None:
        """The TalkingFace case: one recording resamples to itself forever.

        Its spread is exactly zero, which would read as a *precise* estimate.
        """
        scores = np.array([0.8])
        interval, _ = bootstrap_statistic(_difference(scores, scores - 0.1), 1, ROUNDS, seed=1)
        self.assertFalse(interval.reportable)
        self.assertIsNone(interval.low)
        self.assertTrue(interval.reason)

    def test_six_clusters_gets_no_interval(self) -> None:
        """MRL has 6 subjects behind 12 698 windows."""
        scores = np.linspace(0.4, 0.6, 6)
        interval, _ = bootstrap_statistic(_difference(scores, scores - 0.1), 6, ROUNDS, seed=1)
        self.assertFalse(interval.reportable)

    def test_the_threshold_is_the_boundary(self) -> None:
        """One below refuses, exactly at it reports."""
        scores = np.linspace(0.4, 0.6, MIN_CLUSTERS_FOR_INTERVAL)
        statistic = _difference(scores, scores - 0.1)
        below, _ = bootstrap_statistic(statistic, MIN_CLUSTERS_FOR_INTERVAL - 1, ROUNDS, seed=1)
        at, _ = bootstrap_statistic(statistic, MIN_CLUSTERS_FOR_INTERVAL, ROUNDS, seed=1)
        self.assertFalse(below.reportable)
        self.assertTrue(at.reportable)

    def test_an_unreportable_result_is_never_significant(self) -> None:
        """A missing interval must not be mistaken for one excluding zero."""
        scores = np.array([0.9])
        interval, _ = bootstrap_statistic(_difference(scores, scores - 0.5), 1, ROUNDS, seed=1)
        self.assertFalse(interval.excludes_zero)

    def test_the_point_estimate_survives(self) -> None:
        """Refusing an interval must not refuse the measurement itself."""
        scores = np.array([0.8])
        interval, _ = bootstrap_statistic(_difference(scores, scores - 0.3), 1, ROUNDS, seed=1)
        self.assertAlmostEqual(interval.point, 0.3, places=6)


class TestDeterminism(unittest.TestCase):
    """A published interval must reproduce."""

    def test_the_same_seed_gives_the_same_interval(self) -> None:
        """Reproducibility is claimed in the output JSON, so it is tested."""
        scores = np.linspace(0.3, 0.9, 25)
        statistic = _difference(scores, scores - 0.05)
        first, _ = bootstrap_statistic(statistic, 25, ROUNDS, seed=11)
        second, _ = bootstrap_statistic(statistic, 25, ROUNDS, seed=11)
        self.assertEqual(first, second)

    def test_the_seed_is_recorded(self) -> None:
        """A number without its seed cannot be reproduced by a reader."""
        scores = np.linspace(0.3, 0.9, 25)
        interval, _ = bootstrap_statistic(_difference(scores, scores), 25, ROUNDS, seed=11)
        self.assertEqual(interval.seed, 11)
        self.assertEqual(interval.rounds, ROUNDS)
        self.assertEqual(interval.n_clusters, 25)


class TestDescribe(unittest.TestCase):
    """Rendering for a table cell."""

    def test_a_reportable_interval_shows_its_bounds(self) -> None:
        """The interval is the point of the exercise, so it leads."""
        text = Interval(0.5464, -0.067, 0.027, 35, 10, 1).describe()
        self.assertIn("0.5464", text)
        self.assertIn("-0.0670", text)

    def test_a_refusal_says_why(self) -> None:
        """A reader must see the sample size behind a bare point estimate."""
        text = Interval(0.81, None, None, 1, 10, 1, reason="one recording").describe()
        self.assertIn("no CI", text)
        self.assertIn("one recording", text)


if __name__ == "__main__":
    unittest.main()
