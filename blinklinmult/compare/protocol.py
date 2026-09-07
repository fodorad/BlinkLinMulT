"""The comparison protocol: three tables, and why it takes three.

One table cannot compare these four models honestly, because they do not arrive
on equal terms.

**Table A -- threshold-free (average precision).** The primary claim. On RN30's
test split only **1.08%** of frames are closed (3542 of 328 819), so accuracy is
meaningless -- a model that answers "open" every time scores 98.9% -- and any
F1 is a statement about the threshold as much as about the model. AP integrates
over every threshold and is the one number no operating-point choice can flatter.

**Table B -- events at a fitted operating point.** What each model achieves when
someone actually tunes it. The threshold is swept on **validation** and applied
to test, per model, so every model is judged after the same amount of care.

**Table C -- events as shipped.** What a user gets from ``pip install`` today.

Tables B and C exist separately because of a real asymmetry that
``registry.py`` documents plainly: the three 1.x models carry ``threshold=0.5``
because "nothing measured it", while ``blinkcnn`` carries a pair swept on
validation. Comparing only as-shipped numbers would measure *calibration
effort*; comparing only fitted numbers would hide a defect a user really hits.
The gap between B and C is the cost of that defect, made visible.

**Fitting never touches test.** ``fit.py`` says of itself: "the function does
not enforce this; it cannot see where the signals came from." A driver can, and
:func:`compare_models` refuses when the two paths coincide.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import torch

from blinklinmult.compare.bootstrap import (
    DEFAULT_ROUNDS,
    DEFAULT_SEED,
    Interval,
    bootstrap_statistic,
    two_sided_p,
)
from blinklinmult.compare.tests import holm_adjust

if TYPE_CHECKING:
    from collections.abc import Sequence

Signals = dict[str, dict[str, np.ndarray]]
"""What :func:`~blinklinmult.train.callbacks.load_signals` returns."""


class ProtocolError(ValueError):
    """Raised when a comparison is asked for on incomparable inputs."""


@dataclass(frozen=True)
class Comparison:
    """One pairwise result.

    Args:
        first (str): The model on the left of the difference.
        second (str): The model subtracted from it.
        interval (Interval): The difference, with its confidence interval.
        p_raw (float): Two-sided bootstrap p-value, uncorrected.
        p_holm (float): The same after Holm correction across the family.
    """

    first: str
    second: str
    interval: Interval
    p_raw: float
    p_holm: float

    @property
    def conclusive(self) -> bool:
        """Whether the corrected result separates the two models.

        Both conditions must hold: an interval excluding zero *and* a corrected
        p below 0.05. They can disagree at the margin, and a claim should rest
        on the stricter reading.

        Returns:
            bool: Whether the difference stands after correction.
        """
        return self.interval.excludes_zero and self.p_holm < 0.05

    def describe(self) -> str:
        """One line for a table.

        Returns:
            str: The difference, its interval, and the corrected p.
        """
        verdict = "significant" if self.conclusive else "not significant"
        return (
            f"{self.first} - {self.second}: {self.interval.describe()} "
            f"p={self.p_raw:.4f} p_holm={self.p_holm:.4f} ({verdict})"
        )


@dataclass(frozen=True)
class CorpusReport:
    """Every model's score on one corpus, and every pairwise comparison.

    Args:
        corpus (str): Which corpus.
        n_clusters (int): Recordings behind the numbers -- the honest sample
            size, which is why it is reported next to them.
        n_frames (int): Valid frames scored.
        positive_rate (float): Fraction of frames labelled closed. Printed
            because at ~1% it is why average precision leads.
        scores (dict[str, Interval]): Per-model average precision.
        comparisons (list[Comparison]): Every pair, Holm-corrected together.
    """

    corpus: str
    n_clusters: int
    n_frames: int
    positive_rate: float
    scores: dict[str, Interval] = field(default_factory=dict)
    comparisons: list[Comparison] = field(default_factory=list)

    @property
    def resolvable(self) -> bool:
        """Whether this corpus can carry an interval at all.

        Returns:
            bool: ``False`` for a corpus like TalkingFace, whose single
            recording supports a point estimate and nothing more.
        """
        return any(interval.reportable for interval in self.scores.values())


def _average_precision(store: Signals, names: Sequence[str]) -> float:
    """Pooled average precision over a set of recordings.

    Pooling concatenates rather than averaging per-recording scores, so a long
    recording carries proportionate weight -- the same convention
    :func:`~blinklinmult.train.events.pool_curves` uses for event counts.

    Args:
        store (Signals): Loaded signals.
        names (Sequence[str]): Recordings to pool, possibly with repeats from a
            bootstrap draw.

    Returns:
        float: Average precision.
    """
    from blinklinmult.train.metrics import average_precision

    signal = np.concatenate([store[name]["signal"] for name in names])
    truth = np.concatenate([store[name]["truth"] for name in names])
    mask = np.concatenate([store[name]["mask"] for name in names])
    return float(
        average_precision(torch.from_numpy(signal), torch.from_numpy(truth), torch.from_numpy(mask))
    )


def _check_alignment(stores: dict[str, Signals]) -> list[str]:
    """Confirm every model covers the same recordings, and return them.

    A paired comparison is only paired if both models saw the same data. A
    missing recording would silently make the difference a comparison of
    different corpora.

    Args:
        stores (dict[str, Signals]): Loaded signals, keyed by model.

    Returns:
        list[str]: The shared recordings, sorted.

    Raises:
        ProtocolError: If fewer than two models were given, or their recordings
            differ.
    """
    if len(stores) < 2:
        raise ProtocolError(f"Need at least two models to compare, got {len(stores)}.")

    reference_name, reference = next(iter(stores.items()))
    expected = set(reference)
    for name, store in stores.items():
        if set(store) != expected:
            missing = sorted(expected.symmetric_difference(store))
            raise ProtocolError(
                f"{name!r} and {reference_name!r} cover different recordings "
                f"({len(missing)} differ, e.g. {missing[:3]}), so a paired "
                "comparison would subtract scores on different data."
            )
    return sorted(expected)


def compare_average_precision(
    stores: dict[str, Signals],
    corpus: str,
    rounds: int = DEFAULT_ROUNDS,
    seed: int = DEFAULT_SEED,
) -> CorpusReport:
    """Table A: threshold-free comparison of every model on one corpus.

    Each model is scored on the observed recordings and on every bootstrap
    resample of them. **The same draw is used for every model**, so a resample
    heavy on hard recordings moves all of them together and the difference stays
    stable where the absolutes do not.

    Args:
        stores (dict[str, Signals]): Per-model signals, from
            :func:`~blinklinmult.train.callbacks.load_signals`. Loading through
            that function rather than :func:`numpy.load` is what stops a
            ``blink_presence`` archive being scored as ``eye_state``.
        corpus (str): Name, for the report.
        rounds (int): Bootstrap replicates.
        seed (int): Seed, recorded in the result.

    Returns:
        CorpusReport: Per-model intervals and every pairwise comparison.

    Raises:
        ProtocolError: If fewer than two models are given or they disagree on
            which recordings they cover.
    """
    recordings = _check_alignment(stores)
    n_clusters = len(recordings)

    reference = next(iter(stores.values()))
    mask = np.concatenate([reference[name]["mask"] for name in recordings])
    truth = np.concatenate([reference[name]["truth"] for name in recordings])
    n_frames = int(mask.sum())
    positive_rate = float(truth[mask].mean()) if n_frames else 0.0

    def picked(indices: Sequence[int]) -> list[str]:
        return [recordings[index] for index in indices]

    scores: dict[str, Interval] = {}
    for model, store in stores.items():
        interval, _ = bootstrap_statistic(
            lambda idx, store=store: _average_precision(store, picked(idx)),
            n_clusters,
            rounds,
            seed,
        )
        scores[model] = interval

    models = list(stores)
    raw: dict[str, float] = {}
    intervals: dict[str, Interval] = {}
    pairs: list[tuple[str, str]] = []
    for i, first in enumerate(models):
        for second in models[i + 1 :]:
            key = f"{first} vs {second}"
            pairs.append((first, second))

            def difference(idx: Sequence[int], a: str = first, b: str = second) -> float:
                names = picked(idx)
                return _average_precision(stores[a], names) - _average_precision(stores[b], names)

            interval, draws = bootstrap_statistic(difference, n_clusters, rounds, seed)
            intervals[key] = interval
            raw[key] = two_sided_p(draws)

    adjusted = holm_adjust(raw)
    comparisons = [
        Comparison(
            first=first,
            second=second,
            interval=intervals[f"{first} vs {second}"],
            p_raw=raw[f"{first} vs {second}"],
            p_holm=adjusted[f"{first} vs {second}"],
        )
        for first, second in pairs
    ]

    return CorpusReport(
        corpus=corpus,
        n_clusters=n_clusters,
        n_frames=n_frames,
        positive_rate=positive_rate,
        scores=scores,
        comparisons=comparisons,
    )
