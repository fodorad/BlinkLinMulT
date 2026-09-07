"""Per-corpus calibration of the eye-state score.

The frame-wise model **ranks** closed frames above open ones on the video
corpora -- threshold-free AUC 0.857 on RN, 0.917 on TalkingFace -- but places
its decision boundary in the wrong place there. Measured, a genuinely closed RN
eye scores a median of 0.19, against 0.98 F1 on the still corpora.

The cause is a class-prior mismatch, not blindness: the sampler shows the model
a ~50% closed world (CEW is 49.2% closed, MRL 81.9%) while RN is 1.3-1.6%. A
model fitted to the first prior systematically under-predicts under the second.

Because the ranking is already good, one scalar per corpus recovers most of the
loss without touching the weights. Two estimators are provided:

* :func:`fit_temperature` -- one temperature on the logit, the standard
  recipe (Guo et al., 2017). Monotone, so it *cannot* change the ranking or the
  AUC; it moves only where the boundary falls.
* :func:`fit_threshold` -- the cheaper baseline: pick the threshold that
  maximises validation F1 directly.

**Prefer the threshold unless temperature clearly beats it.** They address the
same failure, and a threshold is one number a reader can interpret. Temperature
earns its place only when the whole probability curve is needed -- for a
downstream expected-value decision, say -- rather than a single cut.

Every parameter here is fitted on **validation** and applied to test. Fitting on
test would report a number no deployment could reproduce. Corpora with no
validation split (TalkingFace ships 0 valid windows, and MPEblink annotates no
closure at all) therefore cannot be calibrated and must keep the universal
parameter -- which is exactly why the universal one is reported beside it.

Measured result: **calibration does not improve the decision on RN.**
---------------------------------------------------------------------

Over ten recording-level splits of ``fw-focal``'s RN test signals, fitting on
half the recordings and scoring the other half:

=======================  ====================  ================
method                   test F1               beats universal
=======================  ====================  ================
universal 0.37           **0.4439 +/- 0.035**  --
per-corpus threshold     0.4269 +/- 0.027      1 of 10
temperature + threshold  0.4261 +/- 0.027      1 of 10
=======================  ====================  ================

The fitted thresholds are *bimodal* across splits -- 0.30-0.36 on six, 0.55-0.56
on four -- so the validation optimum is unstable and does not transfer. The
temperature, by contrast, is stable at 0.58-0.71 and does what it claims: it
cuts expected calibration error from 0.0355 to 0.0109.

So the probabilities really were mis-scaled, and temperature really does fix
them -- but the *decision* was already near its best at the universal
threshold, and re-fitting per corpus buys nothing while adding a parameter and
its variance.

**This module is therefore not wired into the benchmark.** It is kept because
the ECE numbers are worth reporting and because the video models in Part 3 may
sit in a different regime; it should be re-measured there rather than assumed
useless. The lesson is that the prior mismatch has to be fixed where it is
caused -- in the sampler (Task 4) or by temporal context (Task 6) -- not
patched at the output.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)
"""Module-level logger."""

EPSILON = 1e-6
"""Clamp keeping ``logit`` finite at probabilities of exactly 0 or 1."""

DEFAULT_TEMPERATURES = np.geomspace(0.05, 20.0, 121)
"""Temperature grid searched by :func:`fit_temperature`.

Geometric rather than linear: temperature acts multiplicatively on the logit, so
0.1-to-0.2 is the same size of change as 5-to-10 and a linear grid would spend
most of its points where they alter the score least.
"""

DEFAULT_THRESHOLDS = np.linspace(0.01, 0.99, 99)
"""Threshold grid searched by :func:`fit_threshold`."""


class CalibrationError(Exception):
    """Raised when calibration is asked for something it cannot fit."""


def logit(probability: np.ndarray) -> np.ndarray:
    """Inverse sigmoid, clamped away from the asymptotes.

    Args:
        probability (np.ndarray): Values in ``[0, 1]``.

    Returns:
        np.ndarray: Log-odds, finite even at exactly 0 or 1.
    """
    clipped = np.clip(np.asarray(probability, dtype=np.float64), EPSILON, 1.0 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def apply_temperature(probability: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities by a temperature on the logit.

    Args:
        probability (np.ndarray): Uncalibrated values in ``[0, 1]``.
        temperature (float): Divisor on the logit. Below 1 sharpens, above 1
            softens; 1 is the identity.

    Returns:
        np.ndarray: Calibrated probabilities, same shape.

    Raises:
        CalibrationError: If the temperature is not positive.
    """
    if not temperature > 0:
        raise CalibrationError(f"temperature must be positive, got {temperature}.")
    return 1.0 / (1.0 + np.exp(-logit(probability) / temperature))


def _f1(probability: np.ndarray, target: np.ndarray, threshold: float) -> float:
    """F1 of a single operating point.

    Args:
        probability (np.ndarray): Scores in ``[0, 1]``.
        target (np.ndarray): Binary labels.
        threshold (float): Decision boundary.

    Returns:
        float: F1, zero when nothing is predicted and nothing is annotated.
    """
    predicted = probability >= threshold
    positive = target >= 0.5
    true_positive = float(np.sum(predicted & positive))
    if true_positive == 0.0:
        return 0.0
    precision = true_positive / float(np.sum(predicted))
    recall = true_positive / float(np.sum(positive))
    return 2.0 * precision * recall / (precision + recall)


def fit_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Pick the decision boundary that maximises F1.

    Args:
        probability (np.ndarray): Validation scores in ``[0, 1]``.
        target (np.ndarray): Validation labels.
        thresholds (np.ndarray | None): Grid to search;
            :data:`DEFAULT_THRESHOLDS` by default.

    Returns:
        tuple[float, float]: ``(threshold, its F1)``.

    Raises:
        CalibrationError: If the inputs disagree in length or hold no positive.
    """
    scores = np.asarray(probability, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape:
        raise CalibrationError(f"{scores.size} scores against {labels.size} labels.")
    if not np.any(labels >= 0.5):
        raise CalibrationError("no positive label to fit against.")

    grid = DEFAULT_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=np.float64)
    values = [_f1(scores, labels, float(point)) for point in grid]
    best = int(np.argmax(values))
    return float(grid[best]), float(values[best])


def fit_temperature(
    probability: np.ndarray,
    target: np.ndarray,
    temperatures: np.ndarray | None = None,
) -> tuple[float, float]:
    """Fit one temperature by validation negative log-likelihood.

    NLL rather than F1: temperature is a *probability* correction, and fitting
    it to a thresholded score would tune it to one operating point and leave the
    rest of the curve arbitrary. The threshold is then chosen separately, on the
    calibrated scores.

    Args:
        probability (np.ndarray): Validation scores in ``[0, 1]``.
        target (np.ndarray): Validation labels.
        temperatures (np.ndarray | None): Grid to search;
            :data:`DEFAULT_TEMPERATURES` by default.

    Returns:
        tuple[float, float]: ``(temperature, its mean NLL)``.

    Raises:
        CalibrationError: If the inputs disagree in length or are empty.
    """
    scores = np.asarray(probability, dtype=np.float64).reshape(-1)
    labels = np.asarray(target, dtype=np.float64).reshape(-1)
    if scores.shape != labels.shape:
        raise CalibrationError(f"{scores.size} scores against {labels.size} labels.")
    if scores.size == 0:
        raise CalibrationError("nothing to calibrate.")

    grid = (
        DEFAULT_TEMPERATURES if temperatures is None else np.asarray(temperatures, dtype=np.float64)
    )
    binary = (labels >= 0.5).astype(np.float64)
    base = logit(scores)

    losses = []
    for temperature in grid:
        calibrated = np.clip(1.0 / (1.0 + np.exp(-base / temperature)), EPSILON, 1.0 - EPSILON)
        losses.append(
            float(-np.mean(binary * np.log(calibrated) + (1 - binary) * np.log(1 - calibrated)))
        )
    best = int(np.argmin(losses))
    return float(grid[best]), float(losses[best])


def expected_calibration_error(
    probability: np.ndarray,
    target: np.ndarray,
    bins: int = 15,
) -> float:
    """Gap between confidence and accuracy, averaged over score bins.

    Reported because F1 alone cannot show *why* a threshold had to move: a model
    can be perfectly ranked and badly calibrated at the same time, which is
    exactly the state measured on RN.

    Args:
        probability (np.ndarray): Scores in ``[0, 1]``.
        target (np.ndarray): Binary labels.
        bins (int): Equal-width bins over ``[0, 1]``.

    Returns:
        float: Weighted mean ``|confidence - accuracy|``; 0 is perfect.
    """
    scores = np.asarray(probability, dtype=np.float64).reshape(-1)
    labels = (np.asarray(target, dtype=np.float64).reshape(-1) >= 0.5).astype(np.float64)
    if scores.size == 0:
        return float("nan")

    edges = np.linspace(0.0, 1.0, bins + 1)
    # `right=True` so a score of exactly 1.0 falls in the last bin rather than
    # one past it; `clip` then keeps bin 0 (scores of exactly 0) in range.
    index = np.clip(np.digitize(scores, edges[1:-1], right=True), 0, bins - 1)

    total = 0.0
    for bin_id in range(bins):
        selected = index == bin_id
        count = int(np.sum(selected))
        if count == 0:
            continue
        confidence = float(np.mean(scores[selected]))
        accuracy = float(np.mean(labels[selected]))
        total += (count / scores.size) * abs(confidence - accuracy)
    return total
