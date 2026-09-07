"""Tests for the resampling unit.

The unit of resampling is the one choice in this package that changes a
conclusion while leaving the arithmetic looking correct, so it is pinned harder
than anything else: both eyes of a subject in one cluster, an empty id refused
rather than silently pooled, and a fixed seed giving a fixed draw.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.compare.clusters import (
    ClusterError,
    cluster_of,
    clusters_from_signals,
    resample,
)
from blinklinmult.data.schema import build_sample_id


class TestClusterOf(unittest.TestCase):
    """Mapping a sample to its cluster."""

    def test_a_sample_id_yields_its_recording(self) -> None:
        """The leading component of the id is the cluster."""
        self.assertEqual(cluster_of(build_sample_id("rn30_1", "000881", "left")), "rn30_1")

    def test_both_eyes_share_a_cluster(self) -> None:
        """A subject blinks with both eyes, so the eyes are not independent.

        Resampling them separately would be the window-level error one level
        down, and would look entirely correct while inflating the sample size.
        """
        left = cluster_of(build_sample_id("rn30_1", "000881", "left"))
        right = cluster_of(build_sample_id("rn30_1", "000881", "right"))
        self.assertEqual(left, right)

    def test_windows_of_one_recording_share_a_cluster(self) -> None:
        """209 windows per recording is the measured RN30 average."""
        first = cluster_of(build_sample_id("rn30_1", "000000", "left"))
        later = cluster_of(build_sample_id("rn30_1", "004500", "left"))
        self.assertEqual(first, later)

    def test_different_recordings_are_different_clusters(self) -> None:
        """Otherwise everything pools into one and n collapses to 1."""
        self.assertNotEqual(cluster_of("rn30_1|0|left"), cluster_of("rn30_2|0|left"))

    def test_a_bare_recording_name_is_its_own_cluster(self) -> None:
        """Signal archives key by recording, with no eye side to strip."""
        self.assertEqual(cluster_of("test_10"), "test_10")

    def test_an_empty_id_is_refused(self) -> None:
        """Silently pooling into one nameless cluster would be worse."""
        with self.assertRaises(ClusterError):
            cluster_of("")


class TestClustersFromSignals(unittest.TestCase):
    """Reducing archive keys to clusters."""

    def test_duplicates_collapse(self) -> None:
        """Two eyes and many windows of one recording count once."""
        keys = ["a|0|left", "a|0|right", "a|15|left", "b|0|left"]
        self.assertEqual(clusters_from_signals(keys), ["a", "b"])

    def test_order_is_stable(self) -> None:
        """Sorted, so a seeded run reproduces regardless of dict order."""
        self.assertEqual(clusters_from_signals(["c", "a", "b"]), ["a", "b", "c"])

    def test_no_recordings_is_refused(self) -> None:
        """There is nothing to resample, and a zero-length draw would raise."""
        with self.assertRaises(ClusterError):
            clusters_from_signals([])


class TestResample(unittest.TestCase):
    """The bootstrap draw itself."""

    def test_shape_is_rounds_by_clusters(self) -> None:
        """Each replicate redraws the full corpus."""
        self.assertEqual(resample(35, 100, seed=0).shape, (100, 35))

    def test_indices_stay_in_range(self) -> None:
        """An out-of-range index would silently score the wrong recording."""
        draws = resample(12, 200, seed=0)
        self.assertGreaterEqual(int(draws.min()), 0)
        self.assertLess(int(draws.max()), 12)

    def test_draws_with_replacement(self) -> None:
        """Sampling without replacement would reproduce the corpus exactly."""
        draws = resample(30, 300, seed=0)
        repeated = [len(set(row.tolist())) < 30 for row in draws]
        self.assertTrue(any(repeated), "no replicate repeated an index")

    def test_the_same_seed_gives_the_same_draw(self) -> None:
        """Reproducibility is a published claim, so it is tested."""
        np.testing.assert_array_equal(resample(20, 50, seed=7), resample(20, 50, seed=7))

    def test_different_seeds_differ(self) -> None:
        """Otherwise the seed is not doing anything."""
        self.assertFalse(np.array_equal(resample(20, 50, seed=7), resample(20, 50, seed=8)))

    def test_non_positive_arguments_are_refused(self) -> None:
        """Zero clusters or zero rounds is a caller error, not an empty result."""
        for clusters, rounds in ((0, 10), (10, 0)):
            with self.subTest(clusters=clusters, rounds=rounds), self.assertRaises(ClusterError):
                resample(clusters, rounds, seed=0)


if __name__ == "__main__":
    unittest.main()
