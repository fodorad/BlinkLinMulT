"""The unit of resampling: what counts as one independent observation.

This module exists because getting this wrong is the most common way a model
comparison overstates its own certainty, and it fails silently -- the arithmetic
is identical, only the interval is wrong, and it is wrong in the flattering
direction.

**A recording is one observation. A window is not.** Two 15-frame windows cut
from the same recording share a subject, a camera, an illuminant, an eyelid
shape and a blink rate. They are close to duplicates. Resampling them
independently pretends the corpus holds far more information than it does:
measured on RN30, 7312 windows come from **35 recordings**, and treating the
windows as independent shrank the confidence interval on a model difference by
**5.9x** -- turning an inconclusive result into a significant one.

The same argument applies one level down. The left and right eye of one subject
are two samples, but they are not independent: they blink together. So both eyes
land in the same cluster, which :func:`cluster_of` enforces by construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

SAMPLE_SEPARATOR = "|"
"""Separator inside a sample id, matching ``data.schema.build_sample_id``."""


class ClusterError(ValueError):
    """Raised when clusters cannot be determined from the given identifiers."""


def cluster_of(sample_id: str) -> str:
    """Return the cluster a sample belongs to.

    Sample ids are ``video_id|frame_group|eye_side`` (see
    :func:`~blinklinmult.data.schema.build_sample_id`), so the cluster is the
    leading component. **Both eyes of one subject map to the same cluster**:
    they blink together, so resampling them independently would be the
    window-level error one level down.

    Args:
        sample_id (str): An id built by
            :func:`~blinklinmult.data.schema.build_sample_id`, or a bare
            recording name, which is already its own cluster.

    Returns:
        str: The cluster key.

    Raises:
        ClusterError: If the id is empty, which would silently pool everything
            into one nameless cluster.
    """
    if not sample_id:
        raise ClusterError("An empty sample id has no cluster.")
    return sample_id.split(SAMPLE_SEPARATOR, 1)[0]


def clusters_from_signals(recordings: Sequence[str]) -> list[str]:
    """Reduce recording keys to their sorted, unique clusters.

    Args:
        recordings (Sequence[str]): Keys from
            :func:`~blinklinmult.train.callbacks.load_signals`. Read the archive
            through that function rather than :func:`numpy.load`: it checks
            which target the ``truth`` arrays hold, and reading a
            ``blink_presence`` archive as ``eye_state`` once fabricated 200 045
            phantom false positives.

    Returns:
        list[str]: Unique clusters, sorted so a run is reproducible.

    Raises:
        ClusterError: If no recordings were given, since there is nothing to
            resample.
    """
    if not recordings:
        raise ClusterError("No recordings, so there is nothing to resample.")
    return sorted({cluster_of(name) for name in recordings})


def resample(
    n_clusters: int,
    rounds: int,
    seed: int,
) -> np.ndarray:
    """Draw bootstrap resamples of cluster indices.

    One array for every model to share. Drawing once and evaluating every model
    on the same draw is what makes the comparison **paired**: a resample heavy
    on hard recordings drags all models down together, so the difference between
    them is stable where their absolute scores are not. At n=20-35 clusters that
    between-recording variance dominates, and an unpaired interval would be too
    wide to resolve anything.

    Args:
        n_clusters (int): How many clusters the corpus has.
        rounds (int): Bootstrap replicates.
        seed (int): Seed, recorded in the output so a result can be reproduced.

    Returns:
        np.ndarray: ``(rounds, n_clusters)`` indices, drawn with replacement.

    Raises:
        ClusterError: If either argument is not positive.
    """
    if n_clusters < 1:
        raise ClusterError(f"Need at least one cluster to resample, got {n_clusters}.")
    if rounds < 1:
        raise ClusterError(f"Need at least one round, got {rounds}.")
    generator = np.random.default_rng(seed)
    return generator.integers(0, n_clusters, size=(rounds, n_clusters))
