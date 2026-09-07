"""Event-level scoring: recovering blinks from a per-frame signal.

This is the evaluation protocol the blink literature uses, and following it is
what makes numbers here comparable with published ones. The window a model sees
is its receptive field and nothing more — no blink is ever assigned to a window.
Events are recovered afterwards:

1. **Sweep** the recording with 50%-overlapping windows, annotation-blind.
2. **Average** the per-frame predictions wherever windows overlap.
3. **Threshold** the averaged signal and merge consecutive above-threshold
   frames into predicted intervals.
4. **Match** those intervals against the annotated blinks.

Steps 3 and 4 are exactly MPEblink's Figure 6 procedure.

**Four matching criteria, all reported.** Papers differ in which they use, and
they are not interchangeable — a detection one frame short of the annotation is
a hit under ``any`` and a miss under ``iou75``:

=========  ==================================================================
``any``    ≥1 frame of intersection — Drutarovsky & Fogelton (2015)
``iou20``  temporal IoU > 0.2 — Fogelton & Beneš (2016)
``iou50``  temporal IoU ≥ 0.50 — MPEblink Blink-AP
``iou75``  temporal IoU ≥ 0.75 — MPEblink Blink-AP
=========  ==================================================================

**Matching is one-to-one**, as in MPEblink's Hungarian assignment, and that
cuts both ways:

* Two predicted intervals on the same blink score one true positive and one
  *false* positive — otherwise a model that fires twice per blink reports twice
  the recall it earned.
* One long prediction spanning two annotated blinks scores one true positive and
  one *false negative*. Failing to separate a close pair is a real failure —
  MPEblink's "consecutive rapid eyeblink" case — and crediting both would hide
  it. A model that resolves the pair gets full credit.

**No single threshold.** The operating point changes every number, so the
headline is an FROC curve: recall against false alarms per minute, swept across
thresholds. A single-threshold table is still reported for readability, but the
curve is what the comparison rests on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
"""Module-level logger."""

CRITERIA: dict[str, float] = {
    "any": 0.0,
    "iou20": 0.2,
    "iou50": 0.5,
    "iou75": 0.75,
}
"""Matching criteria, as minimum temporal IoU.

``any`` is the degenerate case: a threshold of ``0.0`` accepts any non-empty
intersection, so it needs no separate code path. ``iou20`` is *strictly*
greater than 0.2 in the source; the difference from ``>=`` is one frame in
edge cases and is documented on :func:`matches`.
"""

DEFAULT_THRESHOLDS = np.round(np.linspace(0.01, 0.99, 99), 2)
"""Operating points swept for the FROC curve.

0.01 resolution over ``[0.01, 0.99]``. The original 0.05-step sweep over
``[0.05, 0.95]`` was too coarse to distinguish a fitted optimum from a boundary
hit: the frame-wise model's operating point landed on 0.05 -- the sweep's floor
-- for both RN15 and RN30, which is the signature of a search running out of
range rather than finding a minimum. Extending below the old floor makes that
distinguishable, and the finer step recovers calibration that 0.05 buckets lost.
"""

Interval = tuple[int, int]
"""A closed frame interval ``(start, end)``, both inclusive."""


class EventError(ValueError):
    """Raised when a signal and its annotation cannot be scored together."""


@dataclass(frozen=True)
class MatchResult:
    """Counts from matching predictions to annotations under one criterion.

    Args:
        true_positives (int): Annotated blinks that were detected.
        false_positives (int): Predicted intervals matching no annotation.
        false_negatives (int): Annotated blinks that were missed.
        matched (dict[int, list[int]]): Annotation index to the prediction
            indices that hit it, for inspecting *how* a blink was detected.
    """

    true_positives: int
    false_positives: int
    false_negatives: int
    matched: dict[int, list[int]] = field(default_factory=dict)

    @property
    def recall(self) -> float:
        """Fraction of annotated blinks detected.

        Returns:
            float: ``TP / (TP + FN)``; ``0.0`` when nothing was annotated.
        """
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def precision(self) -> float:
        """Fraction of predictions that hit an annotation.

        Returns:
            float: ``TP / (TP + FP)``; ``0.0`` when nothing was predicted.
        """
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall.

        Returns:
            float: ``0.0`` when either is zero.
        """
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0

    def false_alarms_per_minute(self, minutes: float) -> float:
        """False positives normalised by recording length.

        The natural rate on continuous footage: a count alone is not comparable
        between a 3-second clip and a 5-minute recording.

        Args:
            minutes (float): Recording duration.

        Returns:
            float: ``FP / minutes``; ``0.0`` for a zero-length recording.
        """
        return self.false_positives / minutes if minutes > 0 else 0.0


def average_overlapping(
    probabilities: Sequence[float] | np.ndarray,
    frame_ids: Sequence[int] | np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collapse overlapping window predictions into one signal per frame.

    A frame covered by several windows receives the mean of their predictions.
    A frame no window covers has no prediction at all and is **masked**, not
    zeroed — a zero is a confident "no blink", which is a different claim from
    "not evaluated".

    Args:
        probabilities (Sequence[float] | np.ndarray): ``(N,)`` per-frame
            predictions, flattened across windows and eyes.
        frame_ids (Sequence[int] | np.ndarray): ``(N,)`` frame id each
            prediction belongs to.
        n_frames (int): Length of the recording.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(n_frames,)`` averaged signal and its
        ``(n_frames,)`` coverage mask.

    Raises:
        EventError: If the inputs disagree in length, or a frame id is out of
            range for the declared recording length.
    """
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    frames = np.asarray(frame_ids).reshape(-1).astype(np.int64)

    if values.shape != frames.shape:
        raise EventError(
            f"{values.size} predictions against {frames.size} frame ids; they must pair up."
        )
    if n_frames < 1:
        raise EventError(f"n_frames must be >= 1, got {n_frames}.")
    if values.size and (frames.min() < 0 or frames.max() >= n_frames):
        raise EventError(
            f"frame ids span [{frames.min()}, {frames.max()}] but the recording "
            f"declares {n_frames} frames."
        )

    totals = np.zeros(n_frames, dtype=np.float64)
    counts = np.zeros(n_frames, dtype=np.int64)
    np.add.at(totals, frames, values)
    np.add.at(counts, frames, 1)

    mask = counts > 0
    signal = np.zeros(n_frames, dtype=np.float64)
    signal[mask] = totals[mask] / counts[mask]
    return signal, mask


def _runs(flags: np.ndarray) -> list[Interval]:
    """Inclusive ``(start, end)`` spans of consecutive ``True`` values.

    Args:
        flags (np.ndarray): ``(T,)`` boolean.

    Returns:
        list[Interval]: One span per run, in order.
    """
    if not flags.any():
        return []
    # Padding with False on both sides makes the first and last runs fall out
    # of the same diff.
    padded = np.concatenate([[False], flags, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(start), int(stop) - 1) for start, stop in zip(edges[::2], edges[1::2], strict=True)
    ]


def to_intervals(
    signal: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    low_threshold: float | None = None,
) -> list[Interval]:
    """Merge consecutive above-threshold frames into predicted intervals.

    An uncovered frame **breaks** a run rather than extending it: with no
    prediction there, joining across the gap would assert a continuity the
    model never claimed.

    **Hysteresis.** With ``low_threshold`` set, a run must *peak* above
    ``threshold`` to count as an event but extends while it stays above
    ``low_threshold``. A single cut is the crudest possible extractor and is
    what makes the operating point brittle -- it has to be low enough to catch
    the shallow start of a closure and high enough not to fire on noise, and no
    single value does both. Two thresholds separate those jobs: the high one
    decides *whether* this is a blink, the low one decides *where* it begins and
    ends.

    This matters for the physical shape of a blink. The eye-state signal ramps
    down through closing, plateaus while closed, and ramps back up through
    opening; the ramps cross the low threshold well before the high one, so
    hysteresis recovers the true onset and offset rather than clipping to the
    part of the closure that happened to be deepest. It also makes an
    *incomplete* blink -- one that crosses low but never high -- explicitly
    detectable rather than silently truncated.

    Args:
        signal (np.ndarray): ``(T,)`` averaged per-frame predictions.
        mask (np.ndarray): ``(T,)`` coverage mask.
        threshold (float): Frames strictly above this are "blinking". With
            hysteresis this is the *high* threshold, which a run must reach.
        low_threshold (float | None): Extend a run while it stays strictly above
            this. ``None`` -- the default -- keeps the single-threshold
            behaviour exactly, so every existing caller is unchanged. Values at
            or above ``threshold`` are treated as ``None``, since they could
            only shrink a run rather than extend it.

    Returns:
        list[Interval]: Inclusive ``(start, end)`` frame intervals, in order.

    Raises:
        EventError: If the signal and mask disagree in length.
    """
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    covered = np.asarray(mask, dtype=bool).reshape(-1)
    if values.shape != covered.shape:
        raise EventError(f"signal is {values.size} long against a {covered.size} mask.")

    above = covered & (values > threshold)
    if low_threshold is None or low_threshold >= threshold:
        return _runs(above)

    if not above.any():
        return []

    # Every span the signal holds above the low threshold, then only those that
    # reach the high one. Keeping the *extended* span is the point: it is the
    # full closure, where `above` alone is just its deepest part.
    extended = covered & (values > low_threshold)
    return [span for span in _runs(extended) if above[span[0] : span[1] + 1].any()]


def temporal_iou(a: Interval, b: Interval) -> float:
    """Intersection over union of two inclusive frame intervals.

    Args:
        a (Interval): First interval.
        b (Interval): Second.

    Returns:
        float: ``|a ∩ b| / |a ∪ b|`` in frames; ``0.0`` when disjoint.
    """
    start = max(a[0], b[0])
    stop = min(a[1], b[1])
    intersection = max(0, stop - start + 1)
    if intersection == 0:
        return 0.0

    union = (a[1] - a[0] + 1) + (b[1] - b[0] + 1) - intersection
    return intersection / union


BLINK_AP_TIOU = np.linspace(0.5, 0.95, 10)
"""The ten tIoU thresholds MPEblink's Blink-AP averages over.

``np.linspace(0.5, 0.95, 10)`` verbatim from the authors' ``action_ap``, so
``[0]`` is Blink-AP@0.5, ``[5]`` is @0.75 and the mean is the headline
@0.5:0.95. Reproducing the grid exactly is what makes the numbers comparable.
"""


def segment_iou(target: tuple[float, float], candidates: np.ndarray) -> np.ndarray:
    """Temporal IoU of one interval against many, in the authors' convention.

    **Endpoints are exclusive here**, unlike :func:`temporal_iou`: the authors
    compute ``t_end - t_start`` with no ``+1``, so a blink annotated
    ``[10, 14]`` has length 4 rather than 5. Keeping their convention matters --
    on a five-frame blink the two differ by 20%, which is enough to move a
    detection across a tIoU threshold and change the reported AP.

    Both arguments must already be **half-open**; use :func:`to_exclusive` to
    convert this project's inclusive intervals.

    Args:
        target (tuple[float, float]): The predicted interval, half-open.
        candidates (np.ndarray): ``(N, 2)`` annotated intervals, half-open.

    Returns:
        np.ndarray: ``(N,)`` temporal IoU against each candidate.
    """
    if candidates.size == 0:
        return np.zeros(0, dtype=np.float64)

    start = np.maximum(target[0], candidates[:, 0])
    stop = np.minimum(target[1], candidates[:, 1])
    intersection = np.clip(stop - start, 0, None)
    union = (candidates[:, 1] - candidates[:, 0]) + (target[1] - target[0]) - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, intersection / union, 0.0).astype(np.float64)


def interpolated_average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under the precision-recall curve, VOC-style.

    Precision is made monotonically decreasing from the right, then the area is
    summed over the steps where recall changes. This is the authors'
    ``interpolated_prec_rec`` verbatim.

    Args:
        precision (np.ndarray): ``(N,)`` cumulative precision.
        recall (np.ndarray): ``(N,)`` cumulative recall.

    Returns:
        float: The average precision.
    """
    padded_precision = np.hstack([[0.0], precision, [0.0]])
    padded_recall = np.hstack([[0.0], recall, [1.0]])
    for index in range(len(padded_precision) - 2, -1, -1):
        padded_precision[index] = max(padded_precision[index], padded_precision[index + 1])
    changes = np.where(padded_recall[1:] != padded_recall[:-1])[0] + 1
    return float(
        np.sum((padded_recall[changes] - padded_recall[changes - 1]) * padded_precision[changes])
    )


def to_exclusive(interval: Interval) -> tuple[float, float]:
    """Convert an inclusive frame interval to the authors' half-open one.

    Everything else here treats ``(start, end)`` as **inclusive** of both
    endpoints, because that is what the corpora annotate: a blink over frames
    10 to 14 covers five frames. The authors' ``segment_iou`` measures
    ``end - start`` with no ``+1``, so the same pair means a length of four.

    Without this conversion a **single-frame prediction has length zero**, its
    IoU against itself is ``0/0 = 0``, and a perfect detection scores as a
    miss -- measured on the real corpus, that alone cost 2.4 points of AP.
    The annotation never triggers it (its shortest blink spans two frames, and
    none is degenerate across 4 809 events), but a thresholded signal readily
    produces one-frame intervals.

    Args:
        interval (Interval): Inclusive ``(start, end)``.

    Returns:
        tuple[float, float]: Half-open ``(start, end + 1)``, so an inclusive
        span of ``n`` frames has exclusive length ``n``.
    """
    return (float(interval[0]), float(interval[1]) + 1.0)


def blink_ap(
    annotated: dict[str, list[Interval]],
    predicted: dict[str, list[tuple[int, int, float]]],
    tiou_thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """MPEblink's Blink-AP: average precision over a sweep of tIoU thresholds.

    A faithful reimplementation of the authors' ``compute_average_precision_detection``,
    so a number produced here can be placed beside their published 8.58
    (@0.5:0.95) and 25.13 (@0.5).

    The algorithm, and why each step matters:

    1. **Rank every prediction across the whole corpus by score**, not per
       recording. AP integrates one global precision-recall curve, so a
       confident detection in one clip outranks a doubtful one in another.
    2. For each prediction, compute tIoU against the annotations **of its own
       instance**, and try them from highest overlap down.
    3. At each tIoU threshold independently: the first annotation clearing the
       threshold *and not already claimed* makes this a true positive and locks
       it. One already claimed is skipped in favour of the next. A prediction
       that clears the threshold against nothing unclaimed is a false positive.
    4. Integrate precision against recall with
       :func:`interpolated_average_precision`, once per threshold.

    Because matching is one-to-one *per threshold*, a model firing twice on one
    blink earns one true positive and one false positive -- the same discipline
    :func:`match` applies at a single operating point.

    **This is oracle-instance Blink-AP.** The authors score only the instances
    their detector already matched; here the instances come from the annotation,
    so instance detection is free and the result is an upper bound on their
    joint figure. It must be reported as such -- see
    :mod:`blinklinmult.preprocess.mpeblink`.

    Args:
        annotated (dict): Instance id to its annotated blink intervals.
        predicted (dict): Instance id to ``(start, end, score)`` predictions.
            Ids absent from ``annotated`` contribute only false positives.
        tiou_thresholds (np.ndarray | None): The sweep; defaults to
            :data:`BLINK_AP_TIOU`.

    Returns:
        np.ndarray: ``(len(tiou_thresholds),)`` average precision per threshold.
        All zeros when nothing is annotated or nothing is predicted.
    """
    thresholds = BLINK_AP_TIOU if tiou_thresholds is None else np.asarray(tiou_thresholds)
    scores = np.zeros(len(thresholds), dtype=np.float64)

    positives = float(sum(len(events) for events in annotated.values()))
    flat = [
        (instance, start, end, score)
        for instance, events in predicted.items()
        for start, end, score in events
    ]
    if positives == 0 or not flat:
        return scores

    # One global ranking: AP integrates a single corpus-wide curve.
    #
    # `argsort()[::-1]`, exactly as the authors write it. This is deliberately
    # numpy's own call rather than an equivalent Python sort: the default kind
    # is quicksort, which is **unstable**, so the order of equally-scored
    # predictions depends on numpy's partitioning and no sorted-key expression
    # reproduces it. Ties are common -- scores quantise -- and reordering two
    # tied predictions can move a false positive across a true one. Emulating
    # the sort instead of calling it left ~3% of random corpora disagreeing by
    # up to 3.3e-2 AP.
    ranking = np.asarray([row[3] for row in flat], dtype=np.float64).argsort()[::-1]
    flat = [flat[i] for i in ranking]

    # One flat ground-truth table with global row numbers, and a lock indexed by
    # those. The authors build exactly this (a DataFrame grouped by video-id,
    # whose `index` column survives the grouping) and it is not equivalent to a
    # per-instance lock: the row number is what `lock_gt` is keyed on.
    rows: list[tuple[float, float]] = []
    positions: dict[str, list[int]] = {}
    for instance, events in annotated.items():
        for event in events:
            positions.setdefault(instance, []).append(len(rows))
            rows.append(to_exclusive(event))

    truth = {
        instance: np.asarray([rows[i] for i in indices], dtype=np.float64).reshape(-1, 2)
        for instance, indices in positions.items()
    }
    claimed = np.full((len(thresholds), len(rows)), -1, dtype=np.int64)

    true_positive = np.zeros((len(thresholds), len(flat)))
    false_positive = np.zeros((len(thresholds), len(flat)))

    for index, (instance, start, end, _score) in enumerate(flat):
        candidates = truth.get(instance)
        if candidates is None or candidates.size == 0:
            # A prediction on an instance with no annotated blink is a false
            # alarm at every threshold.
            false_positive[:, index] = 1
            continue

        overlaps = segment_iou(to_exclusive((start, end)), candidates)
        order = overlaps.argsort()[::-1]
        global_rows = positions[instance]

        for level, threshold in enumerate(thresholds):
            for candidate in order:
                if overlaps[candidate] < threshold:
                    false_positive[level, index] = 1
                    break
                if claimed[level, global_rows[candidate]] >= 0:
                    continue
                true_positive[level, index] = 1
                claimed[level, global_rows[candidate]] = index
                break
            # Cleared the threshold, but every match was already taken.
            if false_positive[level, index] == 0 and true_positive[level, index] == 0:
                false_positive[level, index] = 1

    hits = np.cumsum(true_positive, axis=1)
    alarms = np.cumsum(false_positive, axis=1)
    recall = hits / positives
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(hits + alarms > 0, hits / (hits + alarms), 0.0)

    for level in range(len(thresholds)):
        scores[level] = interpolated_average_precision(precision[level], recall[level])
    return scores


def blink_ap_summary(scores: np.ndarray) -> dict[str, float]:
    """Name the Blink-AP figures the authors report.

    Args:
        scores (np.ndarray): The output of :func:`blink_ap`, length 10.

    Returns:
        dict[str, float]: ``blink_ap`` (the @0.5:0.95 mean, their headline),
        plus ``blink_ap50``, ``blink_ap75`` and ``blink_ap95``.
    """
    return {
        "blink_ap": float(scores.mean()),
        "blink_ap50": float(scores[0]),
        "blink_ap75": float(scores[5]),
        "blink_ap95": float(scores[-1]),
    }


def matches(predicted: Interval, annotated: Interval, criterion: str) -> bool:
    """Whether one prediction counts as detecting one annotated blink.

    Args:
        predicted (Interval): Predicted interval.
        annotated (Interval): Annotated blink.
        criterion (str): A key of :data:`CRITERIA`.

    Returns:
        bool: Whether the pair matches.

    Raises:
        EventError: If the criterion is unknown.
    """
    if criterion not in CRITERIA:
        raise EventError(f"unknown criterion {criterion!r}; expected {sorted(CRITERIA)}.")

    overlap = temporal_iou(predicted, annotated)
    minimum = CRITERIA[criterion]
    # `any` accepts any intersection at all; the IoU criteria are thresholds
    # the overlap must exceed, following the sources they come from.
    return overlap > 0.0 if minimum == 0.0 else overlap > minimum


def match(
    predicted: list[Interval],
    annotated: list[Interval],
    criterion: str = "any",
) -> MatchResult:
    """Count hits, misses, and false alarms under one criterion.

    An annotated blink is detected when *any* prediction matches it, and counts
    once however many do. Every prediction matching no annotation is a false
    alarm — including the surplus ones when several land on the same blink,
    which is what stops a jittery model from inflating its recall.

    Args:
        predicted (list[Interval]): Predicted intervals.
        annotated (list[Interval]): Annotated blinks.
        criterion (str): A key of :data:`CRITERIA`.

    Returns:
        MatchResult: The counts, with the annotation-to-prediction map.
    """
    hit_by: dict[int, list[int]] = {}
    # At most one prediction is credited per annotation. A second prediction on
    # the same blink is a *false alarm*, not a second hit: otherwise a model
    # that fires twice per blink scores double the recall it earned. Predictions
    # are taken in order, so the earliest wins and the result is deterministic.
    credited: set[int] = set()

    for annotation_index, annotation in enumerate(annotated):
        for prediction_index, prediction in enumerate(predicted):
            if prediction_index in credited:
                continue
            if matches(prediction, annotation, criterion):
                hit_by[annotation_index] = [prediction_index]
                credited.add(prediction_index)
                break

    true_positives = len(hit_by)
    return MatchResult(
        true_positives=true_positives,
        false_positives=len(predicted) - len(credited),
        false_negatives=len(annotated) - true_positives,
        matched=hit_by,
    )


def froc(
    signal: np.ndarray,
    mask: np.ndarray,
    annotated: list[Interval],
    minutes: float,
    criterion: str = "any",
    thresholds: np.ndarray | None = None,
    low_ratio: float | None = None,
) -> dict[str, np.ndarray]:
    """Sweep the threshold, tracing recall against false alarms per minute.

    The headline result. A single operating point is a choice that changes
    every number reported at it; the curve makes the trade-off visible instead
    of burying it in a constant.

    Args:
        signal (np.ndarray): ``(T,)`` averaged per-frame predictions.
        mask (np.ndarray): ``(T,)`` coverage mask.
        annotated (list[Interval]): Annotated blinks.
        minutes (float): Recording duration, for the false-alarm rate.
        criterion (str): A key of :data:`CRITERIA`.
        thresholds (np.ndarray | None): Operating points; defaults to
            :data:`DEFAULT_THRESHOLDS`.
        low_ratio (float | None): Hysteresis, as a fraction of the swept
            threshold. Expressed as a ratio rather than an absolute value so the
            pair stays ordered across the whole sweep -- a constant low
            threshold would exceed the high one at the bottom of the sweep and
            silently disable hysteresis. ``None`` keeps the single cut.

    Returns:
        dict[str, np.ndarray]: ``thresholds``, ``recall``, ``precision``,
        ``f1``, ``false_alarms_per_minute``, and the ``tp``/``fp``/``fn``
        counts they were derived from, aligned by index.

        The counts are returned because **curves from several recordings pool by
        summing counts at each threshold, never by averaging rates** — a
        thirty-second clip must not weigh the same as a five-minute recording.
        Without them a corpus curve could only be assembled the wrong way.
    """
    points = DEFAULT_THRESHOLDS if thresholds is None else np.asarray(thresholds, dtype=np.float64)

    recall, precision, f1, rate = [], [], [], []
    hits, alarms, misses = [], [], []
    for threshold in points:
        # The low threshold tracks the high one as a fixed ratio, so the pair
        # stays ordered across the whole sweep. A constant low threshold would
        # invert at the bottom of the sweep and silently disable hysteresis.
        point = float(threshold)
        low = point * low_ratio if low_ratio is not None else None
        result = match(to_intervals(signal, mask, point, low), annotated, criterion)
        recall.append(result.recall)
        precision.append(result.precision)
        f1.append(result.f1)
        rate.append(result.false_alarms_per_minute(minutes))
        hits.append(float(result.true_positives))
        alarms.append(float(result.false_positives))
        misses.append(float(result.false_negatives))

    return {
        "thresholds": points,
        "recall": np.asarray(recall),
        "precision": np.asarray(precision),
        "f1": np.asarray(f1),
        "false_alarms_per_minute": np.asarray(rate),
        "tp": np.asarray(hits),
        "fp": np.asarray(alarms),
        "fn": np.asarray(misses),
    }


def pool_curves(curves: Sequence[dict[str, np.ndarray]], minutes: float) -> dict[str, np.ndarray]:
    """Combine per-recording FROC curves into one corpus curve.

    **Counts are summed at each threshold and the rates re-derived**, which is
    the same rule :meth:`~blinklinmult.train.callbacks.EventReport._aggregate`
    applies to the single operating point. Averaging the per-recording rates
    instead would give a thirty-second clip the same weight as a five-minute
    recording, and a corpus is not a mean of its recordings.

    Args:
        curves (Sequence[dict[str, np.ndarray]]): One :func:`froc` result per
            recording, all swept over the same thresholds.
        minutes (float): Total duration across the recordings, for the
            false-alarm rate.

    Returns:
        dict[str, np.ndarray]: A curve in the same shape :func:`froc` returns.
        Empty when no curves are given.

    Raises:
        EventError: If the curves were swept over different thresholds, which
            would make summing them index-by-index meaningless.
    """
    usable = [c for c in curves if c.get("thresholds") is not None and len(c["thresholds"])]
    if not usable:
        return {}

    points = np.asarray(usable[0]["thresholds"], dtype=np.float64)
    for curve in usable[1:]:
        if not np.array_equal(np.asarray(curve["thresholds"], dtype=np.float64), points):
            raise EventError(
                "cannot pool FROC curves swept over different thresholds; "
                "every recording must use the same operating points."
            )

    hits = np.sum([np.asarray(c["tp"], dtype=np.float64) for c in usable], axis=0)
    alarms = np.sum([np.asarray(c["fp"], dtype=np.float64) for c in usable], axis=0)
    misses = np.sum([np.asarray(c["fn"], dtype=np.float64) for c in usable], axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(hits + misses > 0, hits / (hits + misses), 0.0)
        precision = np.where(hits + alarms > 0, hits / (hits + alarms), 0.0)
        f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0.0)
    rate = alarms / minutes if minutes > 0 else np.zeros_like(alarms)

    return {
        "thresholds": points,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "false_alarms_per_minute": rate,
        "tp": hits,
        "fp": alarms,
        "fn": misses,
    }


def average_precision(curve: dict[str, np.ndarray]) -> float:
    """Area under the precision-recall curve of a threshold sweep.

    The curve is anchored at ``recall = 0`` before integrating. Without that
    anchor a *perfect* detector scores zero: it holds recall at 1.0 across every
    threshold, the recall axis has no extent, and the trapezoid collapses —
    the best possible result reported as the worst. Anchoring is what the
    standard definition does, and it makes the integral the fraction of the
    recall range achieved at a given precision.

    Args:
        curve (dict[str, np.ndarray]): Output of :func:`froc`.

    Returns:
        float: Average precision over the swept operating points.
    """
    recall = np.asarray(curve["recall"], dtype=np.float64).reshape(-1)
    precision = np.asarray(curve["precision"], dtype=np.float64).reshape(-1)
    if recall.size == 0:
        return 0.0

    order = np.argsort(recall)
    recall, precision = recall[order], precision[order]

    # Anchor at recall 0 with the precision of the strictest operating point.
    if recall[0] > 0.0:
        recall = np.concatenate([[0.0], recall])
        precision = np.concatenate([[precision[0]], precision])

    if recall.size < 2 or recall[-1] <= 0.0:
        return 0.0
    return float(np.trapezoid(precision, recall))


def event_metrics(
    signal: np.ndarray,
    mask: np.ndarray,
    annotated: list[Interval],
    minutes: float,
    threshold: float = 0.5,
    thresholds: np.ndarray | None = None,
    curves: dict[str, dict[str, np.ndarray]] | None = None,
    low_threshold: float | None = None,
    low_ratio: float | None = None,
) -> dict[str, float]:
    """Every event-level metric, under every criterion.

    Args:
        signal (np.ndarray): ``(T,)`` averaged per-frame predictions.
        mask (np.ndarray): ``(T,)`` coverage mask.
        annotated (list[Interval]): Annotated blinks.
        minutes (float): Recording duration.
        threshold (float): Operating point for the single-threshold counts.
        thresholds (np.ndarray | None): Sweep for the curves.
        curves (dict[str, dict[str, np.ndarray]] | None): Filled in with the
            FROC curve per criterion when given. The curve is what
            ``average_precision`` and ``best_f1`` are derived from, and callers
            that pool several recordings need the points themselves — returning
            only the two scalars discards the headline result.
        low_threshold (float | None): Absolute hysteresis threshold for the
            single-point counts. See :func:`to_intervals`.
        low_ratio (float | None): Hysteresis as a fraction of the threshold,
            used for the swept curves and as a fallback for the single point
            when ``low_threshold`` is not given.

    Returns:
        dict[str, float]: ``event/<criterion>/<metric>`` for each of
        :data:`CRITERIA`, plus descriptive counts that keep a rate readable.
    """
    low = low_threshold
    if low is None and low_ratio is not None:
        low = threshold * low_ratio
    predicted = to_intervals(signal, mask, threshold, low)

    metrics: dict[str, float] = {
        "event/n_annotated": float(len(annotated)),
        "event/n_predicted": float(len(predicted)),
        "event/minutes": float(minutes),
        "event/frames_covered": float(int(np.asarray(mask, dtype=bool).sum())),
    }
    if annotated:
        lengths = [stop - start + 1 for start, stop in annotated]
        metrics["event/mean_blink_frames"] = float(np.mean(lengths))
        metrics["event/median_blink_frames"] = float(np.median(lengths))

    for criterion in CRITERIA:
        result = match(predicted, annotated, criterion)
        curve = froc(signal, mask, annotated, minutes, criterion, thresholds, low_ratio)
        if curves is not None:
            curves[criterion] = curve
        prefix = f"event/{criterion}"
        metrics.update(
            {
                f"{prefix}/tp": float(result.true_positives),
                f"{prefix}/fp": float(result.false_positives),
                f"{prefix}/fn": float(result.false_negatives),
                f"{prefix}/recall": result.recall,
                f"{prefix}/precision": result.precision,
                f"{prefix}/f1": result.f1,
                f"{prefix}/fa_per_min": result.false_alarms_per_minute(minutes),
                f"{prefix}/average_precision": average_precision(curve),
                f"{prefix}/best_f1": float(curve["f1"].max()) if curve["f1"].size else 0.0,
            }
        )
    return metrics


def annotated_intervals(blink_ids: np.ndarray) -> list[Interval]:
    """Recover annotated blink intervals from a per-frame id array.

    Runs of *consecutive* frames sharing an id, so a reused id yields two
    events rather than one spanning the gap between them — which is what the
    annotation means, and what the corpora's own notes describe when a subject
    "blinks twice very fast".

    Args:
        blink_ids (np.ndarray): ``(T,)`` per-frame blink id; negative means no
            blink.

    Returns:
        list[Interval]: Inclusive intervals, in frame order.
    """
    ids = np.asarray(blink_ids).reshape(-1).astype(np.int64)
    intervals: list[Interval] = []

    index = 0
    while index < ids.size:
        if ids[index] < 0:
            index += 1
            continue
        start = index
        current = ids[index]
        while index < ids.size and ids[index] == current:
            index += 1
        intervals.append((start, index - 1))
    return intervals
