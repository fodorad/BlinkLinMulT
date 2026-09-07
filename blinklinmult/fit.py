"""Fit an operating point for a model that never had one.

``BlinkCNN``'s thresholds were fitted on validation during training. The 1.x
models were released as weights alone, so :data:`~blinklinmult.registry.MODELS`
records the plain 0.5 default for them -- a placeholder, not a measurement.

That gap makes *event* comparisons between the generations unfair: v2's operating
point was tuned and theirs was not. This module closes it. Give it scored
recordings with their annotated blinks and it sweeps the same grids the training
callback uses, returning the pair that maximises event F1::

    from blinklinmult.fit import fit_operating_point

    point = fit_operating_point(signals, annotations, fps=30.0)
    detector.spec = point.applied_to(detector.spec)

**Fit on validation, never on test.** A threshold chosen on the same recordings
it is reported on is not a measurement, it is a description of that split -- and
it will not survive contact with new data. The function does not enforce this; it
cannot see where the signals came from.

**Hysteresis usually wins, so it is searched by default.** A single cut has to be
low enough to catch the shallow start of a closure and high enough to ignore
noise, and no one value does both. Two thresholds split those jobs: the high one
decides *whether* a run is a blink, the low one decides *where* it starts and
ends. On ``BlinkCNN`` this was worth 0.19 to 0.52 event F1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import numpy as np

from blinklinmult.train.events import DEFAULT_THRESHOLDS, Interval, match, to_intervals

if TYPE_CHECKING:
    from blinklinmult.registry import ModelSpec

logger = logging.getLogger(__name__)
"""Module-level logger."""

DEFAULT_LOW_RATIOS: tuple[float | None, ...] = (
    None,
    0.5,
    0.4,
    1.0 / 3.0,
    0.25,
    0.2,
    0.15,
    0.125,
    0.1,
)
"""Low-threshold fractions searched, mirroring the training callback's grid.

``None`` comes first so a single threshold wins ties: hysteresis costs a second
parameter, and it should have to *earn* its place rather than be adopted on a
rounding difference.
"""


class FitError(Exception):
    """Raised when there is not enough data to fit an operating point."""


@dataclass(frozen=True)
class OperatingPoint:
    """A fitted decision rule and the score it achieved.

    Args:
        threshold (float): The high threshold -- whether a run is a blink.
        low_ratio (float | None): Low threshold as a fraction of ``threshold``,
            or ``None`` for a single cut.
        f1 (float): Event F1 at this point, on the data it was fitted to.
        precision (float): Event precision there.
        recall (float): Event recall there.
    """

    threshold: float
    low_ratio: float | None
    f1: float
    precision: float
    recall: float

    @property
    def low_threshold(self) -> float | None:
        """The absolute low threshold, or ``None`` for a single cut."""
        return None if self.low_ratio is None else self.threshold * self.low_ratio

    def applied_to(self, model_spec: ModelSpec) -> ModelSpec:
        """Copy a spec with this operating point substituted in.

        Args:
            model_spec (ModelSpec): The spec to update.

        Returns:
            ModelSpec: A new spec carrying the fitted thresholds.
        """
        return replace(model_spec, threshold=self.threshold, low_ratio=self.low_ratio)


def _score_point(
    signals: list[np.ndarray],
    annotations: list[list[Interval]],
    threshold: float,
    low_ratio: float | None,
) -> tuple[float, float, float]:
    """Evaluate one candidate operating point across every recording.

    Counts are pooled before the ratios are taken, so a long recording carries
    proportionally more weight than a short one -- averaging per-recording F1
    would let a 30-frame clip outvote a 30-minute session.

    Args:
        signals (list[np.ndarray]): Per-recording per-frame scores.
        annotations (list[list[Interval]]): Their annotated blinks.
        threshold (float): Candidate high threshold.
        low_ratio (float | None): Candidate low ratio.

    Returns:
        tuple[float, float, float]: ``(f1, precision, recall)``.
    """
    low = None if low_ratio is None else threshold * low_ratio
    true_positive = predicted_total = annotated_total = 0

    for signal, truth in zip(signals, annotations, strict=True):
        mask = np.ones(signal.shape[0], dtype=bool)
        predicted = to_intervals(signal, mask, threshold, low)
        result = match(predicted, truth, criterion="any")
        true_positive += result.true_positives
        predicted_total += result.true_positives + result.false_positives
        annotated_total += result.true_positives + result.false_negatives

    if true_positive == 0:
        return 0.0, 0.0, 0.0
    precision = true_positive / predicted_total
    recall = true_positive / annotated_total
    return 2 * precision * recall / (precision + recall), precision, recall


def fit_operating_point(
    signals: list[np.ndarray],
    annotations: list[list[Interval]],
    thresholds: np.ndarray | None = None,
    low_ratios: tuple[float | None, ...] | None = None,
) -> OperatingPoint:
    """Search for the threshold pair that maximises event F1.

    Args:
        signals (list[np.ndarray]): One per-frame score array per recording, in
            ``[0, 1]``. Produced by :meth:`~blinklinmult.detector.BlinkDetector.score`.
        annotations (list[list[Interval]]): The annotated blink intervals for
            each, in the same order.
        thresholds (np.ndarray | None): High thresholds to try;
            :data:`~blinklinmult.train.events.DEFAULT_THRESHOLDS` by default.
        low_ratios (tuple[float | None, ...] | None): Low ratios to try;
            :data:`DEFAULT_LOW_RATIOS` by default.

    Returns:
        OperatingPoint: The best pair found, with its scores.

    Raises:
        FitError: If the inputs disagree in length, are empty, or hold no
            annotated blink to fit against.
    """
    if len(signals) != len(annotations):
        raise FitError(f"{len(signals)} signals against {len(annotations)} annotation lists.")
    if not signals:
        raise FitError("No recordings to fit on.")
    if not any(annotations):
        raise FitError("No annotated blinks; there is nothing to fit against.")

    grid = DEFAULT_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    ratios = DEFAULT_LOW_RATIOS if low_ratios is None else low_ratios

    best = OperatingPoint(threshold=0.5, low_ratio=None, f1=-1.0, precision=0.0, recall=0.0)
    for ratio in ratios:
        for threshold in grid:
            f1, precision, recall = _score_point(signals, annotations, float(threshold), ratio)
            # Strictly greater: the grids put `None` and low thresholds first, so
            # ties keep the simpler rule already found.
            if f1 > best.f1:
                best = OperatingPoint(float(threshold), ratio, f1, precision, recall)

    logger.info(
        f"Fitted threshold={best.threshold:.2f} low_ratio={best.low_ratio} "
        f"-> F1 {best.f1:.4f} (P {best.precision:.4f}, R {best.recall:.4f})."
    )
    return best
