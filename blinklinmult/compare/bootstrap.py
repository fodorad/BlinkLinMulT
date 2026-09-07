"""The paired cluster bootstrap, and the refusal to report one when it cannot.

The estimator is ordinary; the two decisions around it are what matter.

**Resampling is paired.** One draw of clusters is scored by every model, so the
difference between two models is computed within a draw rather than between two
independent draws. See :func:`~blinklinmult.compare.clusters.resample`.

**Below :data:`MIN_CLUSTERS_FOR_INTERVAL` clusters, no interval is returned at
all.** This is deliberate and is the module's sharpest opinion. A percentile
bootstrap over one cluster resamples that same cluster every time: the spread is
exactly zero and the "interval" has zero width, which reads as a *precise*
estimate when it is the opposite. TalkingFace (one recording, 524 windows, 61
events) is the case this exists for. The alternative -- silently falling back to
window-level resampling to manufacture a number -- is the error this whole
package was written to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

MIN_CLUSTERS_FOR_INTERVAL = 10
"""Fewest clusters that can carry a confidence interval.

Below this the resampling distribution is too coarse for its tails to mean
anything: with ``n`` clusters a resample can only take ``n`` distinct values per
slot, and the 2.5th percentile is estimated from a handful of distinct
compositions. Ten is a judgement, not a theorem -- it admits RN15 (20) and RN30
(35) while excluding TalkingFace (1) and MRL (6 subjects).
"""

DEFAULT_ROUNDS = 10000
"""Bootstrap replicates.

Large enough that the 2.5%/97.5% endpoints rest on ~250 replicates each, so the
Monte Carlo noise on an endpoint is small next to the sampling noise the
interval is reporting. Cheap here because a replicate is array arithmetic over
precomputed per-recording scores, never a re-run of a model.
"""

DEFAULT_SEED = 20260905
"""Seed for the resampling draw, recorded alongside every result."""


@dataclass(frozen=True)
class Interval:
    """A point estimate, with an interval when the data can carry one.

    Args:
        point (float): The statistic on the observed sample.
        low (float | None): Lower bound, or ``None`` when refused.
        high (float | None): Upper bound, or ``None`` when refused.
        n_clusters (int): Independent units behind the estimate. The honest
            sample size, which is why it travels with the number.
        rounds (int): Replicates drawn.
        seed (int): Seed used.
        reason (str): Why no interval was produced; empty when there is one.
    """

    point: float
    low: float | None
    high: float | None
    n_clusters: int
    rounds: int
    seed: int
    reason: str = ""

    @property
    def reportable(self) -> bool:
        """Whether an interval was produced.

        Returns:
            bool: ``True`` when both bounds are present.
        """
        return self.low is not None and self.high is not None

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies wholly above or below zero.

        Returns:
            bool: ``False`` when there is no interval, so an unreportable
            result is never mistaken for a significant one.
        """
        low, high = self.low, self.high
        if low is None or high is None:
            return False
        return (low > 0.0) or (high < 0.0)

    def describe(self) -> str:
        """One line for a table cell.

        Returns:
            str: e.g. ``"0.5464 [-0.0670, +0.0270]"``, or the refusal.
        """
        if not self.reportable:
            return f"{self.point:.4f} (n={self.n_clusters}, no CI: {self.reason})"
        return f"{self.point:.4f} [{self.low:+.4f}, {self.high:+.4f}]"


def _percentile_interval(draws: np.ndarray) -> tuple[float, float]:
    """Two-sided 95% percentile bounds.

    Args:
        draws (np.ndarray): The bootstrap distribution.

    Returns:
        tuple[float, float]: Lower and upper bounds.
    """
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def bootstrap_statistic(
    statistic: Callable[[Sequence[int]], float],
    n_clusters: int,
    rounds: int = DEFAULT_ROUNDS,
    seed: int = DEFAULT_SEED,
) -> tuple[Interval, np.ndarray]:
    """Resample clusters and summarise a statistic over the draws.

    Args:
        statistic (Callable[[Sequence[int]], float]): Maps cluster indices to a
            scalar. Called once per replicate, so it should be arithmetic over
            precomputed per-cluster values rather than anything that re-scores.
        n_clusters (int): Clusters available.
        rounds (int): Replicates.
        seed (int): Seed.

    Returns:
        tuple[Interval, np.ndarray]: The summary, and the raw draws for a
        downstream p-value.
    """
    from blinklinmult.compare.clusters import resample

    observed = float(statistic(list(range(n_clusters))))
    indices = resample(n_clusters, rounds, seed)
    draws = np.array([statistic(row) for row in indices], dtype=float)

    if n_clusters < MIN_CLUSTERS_FOR_INTERVAL:
        return (
            Interval(
                point=observed,
                low=None,
                high=None,
                n_clusters=n_clusters,
                rounds=rounds,
                seed=seed,
                reason=(
                    f"{n_clusters} cluster(s) is below the {MIN_CLUSTERS_FOR_INTERVAL} "
                    "needed for the resampling tails to mean anything"
                ),
            ),
            draws,
        )

    low, high = _percentile_interval(draws)
    return (
        Interval(
            point=observed,
            low=low,
            high=high,
            n_clusters=n_clusters,
            rounds=rounds,
            seed=seed,
        ),
        draws,
    )


def two_sided_p(draws: np.ndarray) -> float:
    """Bootstrap p-value for the null that the difference is zero.

    Uses the ``(1 + count) / (B + 1)`` form rather than a bare proportion. The
    correction matters: without it a difference that never changes sign reports
    ``p = 0``, claiming certainty no finite resampling can support. With it the
    floor is ``2 / (B + 1)``, which is exactly what ``B`` replicates can
    evidence -- and that identity is what
    ``tests/compare/test_bootstrap.py`` pins.

    Args:
        draws (np.ndarray): Bootstrap distribution of the difference.

    Returns:
        float: Two-sided p-value in ``(0, 1]``.
    """
    rounds = draws.size
    below = float((1 + np.sum(draws <= 0.0)) / (rounds + 1))
    above = float((1 + np.sum(draws >= 0.0)) / (rounds + 1))
    return min(1.0, 2.0 * min(below, above))
