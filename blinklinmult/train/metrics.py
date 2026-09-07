"""Masked classification metrics for the two blink tasks.

Both tasks are binary detection under heavy class imbalance, so the metrics are
chosen accordingly and every one of them reduces over valid positions only —
exactly the mask the losses use.

``{split}/{target}/f1``
    Harmonic mean of precision and recall. **The primary metric.** Accuracy is
    not reported as a headline number: a model predicting "no blink" everywhere
    scores above 95% accuracy on a continuously-sampled recording, which says
    nothing about whether it detects blinks.
``{split}/{target}/precision``, ``{split}/{target}/recall``
    Reported alongside F1 because the two failure modes — missing blinks and
    hallucinating them — have different costs downstream, and F1 alone hides
    which one a run has.
``{split}/{target}/average_precision``
    Area under the precision-recall curve: threshold-free, so it separates a
    model that ranks well but is badly calibrated from one that genuinely
    cannot discriminate.
``{split}/{target}/accuracy``, ``{split}/{target}/balanced_accuracy``
    Kept for comparability with the published 1.x numbers, which reported
    accuracy. Balanced accuracy is the one to read.
``{split}/mean_f1``
    Mean F1 across supervised targets. **Drives model selection.**

**Why not torchmetrics' own classification metrics.** They have no notion of a
per-position validity mask, and passing them a flattened, pre-filtered tensor
would silently change what "an epoch" means under DDP, where each rank filters a
different number of positions. Accumulating raw predictions with their masks and
reducing once at epoch end keeps the gathered state exact.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING, Any

import torch
from torchmetrics import Metric

from blinklinmult.data.schema import TASK_BPD, SchemaError, parse_sample_id

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
"""Module-level logger."""

EPSILON = 1e-8
"""Guards division when a class is absent from an epoch."""

DEFAULT_THRESHOLD = 0.5
"""Probability above which a position is predicted positive."""

PRIMARY_METRIC = "mean_f1"
"""The metric that selects checkpoints and drives early stopping.

Computed over the **frame-level** scores when frame-group ids are available, so
model selection optimises the number the benchmark reports. See
:class:`FrameAggregator`.
"""

NO_GROUP = -1
"""Sentinel frame-group id, meaning "this position cannot be grouped".

Accumulated when a caller supplies no group ids, which leaves the per-eye
metrics intact and simply disables frame-level aggregation for that epoch.
"""

FRAME_REDUCERS: tuple[str, ...] = ("max", "mean")
"""Rules for combining the two eyes' predictions into one per frame.

``max`` is the primary rule: a blink is a frame-level event that *occurred*, and
the annotation marks a frame as a blink when a closure is visible. If one eye is
occluded by a head turn while the other is plainly closing, ``max`` recovers the
event where ``mean`` halves the score and can push it under the threshold.
``mean`` is logged alongside so the choice is measured rather than assumed.
"""

PRIMARY_FRAME_REDUCER = "max"
"""The frame reducer whose F1 drives model selection."""

ESR_METRIC = "mean_f1"
"""Headline eye-state-recognition score: mean frame-level F1 across targets."""

BPD_METRIC = "bpd_f1"
"""Headline blink-presence-detection score: window-level F1.

Reported only when the run supervises blink presence, since a still-image
corpus cannot witness a blink.
"""

METRIC_NAMES: tuple[str, ...] = (
    "f1",
    "precision",
    "recall",
    "average_precision",
    "accuracy",
    "balanced_accuracy",
)
"""Per-target metrics computed every epoch, in report order."""


def _rates(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Confusion-matrix counts over valid positions.

    Args:
        probability (torch.Tensor): Predicted probabilities.
        target (torch.Tensor): Binary targets.
        mask (torch.Tensor): ``True`` where the target is real supervision.
        threshold (float): Decision threshold.

    Returns:
        tuple: ``(tp, fp, tn, fn)`` as float scalars.
    """
    valid = mask.bool()
    predicted = (probability >= threshold) & valid
    positive = (target >= 0.5) & valid
    negative = (~positive) & valid

    true_positive = (predicted & positive).sum().float()
    false_positive = (predicted & negative).sum().float()
    true_negative = ((~predicted) & negative).sum().float()
    false_negative = ((~predicted) & positive).sum().float()
    return true_positive, false_positive, true_negative, false_negative


def average_precision(
    probability: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Area under the precision-recall curve over valid positions.

    Computed by the step-wise sum ``sum_k (R_k - R_{k-1}) * P_k`` over positions
    ranked by predicted probability — the standard estimator, which unlike
    trapezoidal interpolation does not reward a model for the region between two
    observed operating points.

    Args:
        probability (torch.Tensor): Predicted probabilities, any shape.
        target (torch.Tensor): Binary targets, same shape.
        mask (torch.Tensor): ``True`` where the target is real supervision.

    Returns:
        torch.Tensor: Scalar in ``[0, 1]``; ``0`` when no positive is present,
        for which average precision is undefined.
    """
    valid = mask.bool().flatten()
    scores = probability.flatten()[valid]
    labels = (target.flatten()[valid] >= 0.5).float()

    if scores.numel() == 0 or labels.sum() == 0:
        return torch.zeros((), device=probability.device)

    order = torch.argsort(scores, descending=True)
    labels = labels[order]

    true_positive = torch.cumsum(labels, dim=0)
    ranks = torch.arange(1, labels.numel() + 1, device=labels.device, dtype=labels.dtype)
    precision = true_positive / ranks

    # Each positive contributes its precision at the rank where it is retrieved;
    # dividing by the positive count is the recall increment, uniform per positive.
    return (precision * labels).sum() / labels.sum()


def binary_metrics(
    probability: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = DEFAULT_THRESHOLD,
) -> dict[str, torch.Tensor]:
    """Every per-target metric, over valid positions.

    Args:
        probability (torch.Tensor): Predicted probabilities.
        target (torch.Tensor): Binary targets.
        mask (torch.Tensor): ``True`` where the target is real supervision.
        threshold (float): Decision threshold.

    Returns:
        dict[str, torch.Tensor]: Keyed by :data:`METRIC_NAMES`. Every value is
        ``0`` when the epoch contained no valid position, rather than NaN, which
        would poison the epoch average and any checkpoint comparison.
    """
    true_positive, false_positive, true_negative, false_negative = _rates(
        probability, target, mask, threshold
    )
    total = true_positive + false_positive + true_negative + false_negative

    precision = true_positive / (true_positive + false_positive + EPSILON)
    recall = true_positive / (true_positive + false_negative + EPSILON)
    specificity = true_negative / (true_negative + false_positive + EPSILON)

    return {
        "f1": 2 * precision * recall / (precision + recall + EPSILON),
        "precision": precision,
        "recall": recall,
        "average_precision": average_precision(probability, target, mask),
        "accuracy": (true_positive + true_negative) / (total + EPSILON),
        "balanced_accuracy": (recall + specificity) / 2,
    }


def frame_group_ids(sample_ids: Sequence[str]) -> torch.Tensor | None:
    """Map each sample to an integer id shared by the eyes of the same frames.

    OmniLoader forwards only the sample id into a batch, so the recording and
    the frame group travel encoded inside it (see
    :func:`~blinklinmult.data.schema.build_sample_id`). This strips the eye side
    and interns what remains, so the two eye-wise samples cut from one window
    receive the same id and :func:`frame_level_metrics` can recombine them.

    **The ids are stable across batches, not batch-local.** They are a hash of
    the group key rather than an enumeration, because the accumulator persists
    across every batch of an epoch: ids that restarted at 0 each batch would
    make window 0 of batch 1 collide with window 0 of batch 0, and the
    aggregation would silently max together frames from different recordings.
    Measured on RN30's test split, that collapsed 329 040 positions into a
    degenerate set and reported ``frame_max/f1 = 1.0`` against a true 0.57.

    Args:
        sample_ids (Sequence[str]): One id per sample in the batch.

    Returns:
        torch.Tensor | None: ``(B,)`` int64 ids, or ``None`` when the batch
        carries no ids or any of them is unparseable — in which case the
        per-eye metrics still run and only frame-level scoring is skipped.
    """
    if not sample_ids:
        return None

    ids: list[int] = []
    for sample_id in sample_ids:
        try:
            video_id, frame_group, _ = parse_sample_id(str(sample_id))
        except SchemaError:
            logger.debug(f"Sample id {sample_id!r} is not groupable; skipping frame metrics.")
            return None
        key = f"{video_id}|{frame_group}"
        # blake2b rather than the builtin hash(): PYTHONHASHSEED randomises str
        # hashing per process, so ids would differ between a run and its resume,
        # and between DDP ranks that must agree on which frame is which.
        # Truncated to 48 bits, which keeps the later `id * window_length`
        # inside int64 while leaving collisions negligible at corpus scale.
        digest = hashlib.blake2b(key.encode(), digest_size=6).digest()
        ids.append(int.from_bytes(digest, "big"))

    return torch.tensor(ids, dtype=torch.long)


def aggregate_by_group(
    values: torch.Tensor, groups: torch.Tensor, reducer: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce per-position values to one value per frame group.

    Args:
        values (torch.Tensor): ``(N,)`` per-position values.
        groups (torch.Tensor): ``(N,)`` int64 group id per position.
        reducer (str): One of :data:`FRAME_REDUCERS`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``(reduced, unique_groups)``, both
        ``(G,)``, ordered by group id.

    Raises:
        ValueError: If the reducer is unknown or the shapes disagree.
    """
    if reducer not in FRAME_REDUCERS:
        raise ValueError(f"Unknown frame reducer {reducer!r}; expected {list(FRAME_REDUCERS)}.")
    if values.shape != groups.shape:
        raise ValueError(
            f"values {tuple(values.shape)} and groups {tuple(groups.shape)} must agree."
        )

    # Rank-compress on the CPU before reducing. `torch.unique(...,
    # return_inverse=True)` is **wrong on MPS for large int64 values**: the
    # frame ids here span ~1e12-1e16 and the MPS kernel collapsed 164 520
    # distinct groups into 9 050, silently maxing together frames from
    # unrelated recordings and inflating frame_max/f1 from 0.52 to 0.71. The
    # same tensor on the CPU returns all 164 520.
    #
    # The compressed ids are contiguous 0..G-1, which is both small enough for
    # any backend and what scatter_reduce wants as an index anyway.
    unique, inverse = torch.unique(groups.cpu(), return_inverse=True)
    unique = unique.to(groups.device)
    inverse = inverse.to(values.device)
    reduced = torch.zeros(unique.numel(), dtype=values.dtype, device=values.device)

    if reducer == "max":
        # scatter_reduce with "amax" needs an identity below every value;
        # probabilities and binary labels are non-negative, so zero serves.
        reduced = reduced.scatter_reduce(0, inverse, values, reduce="amax")
    else:
        totals = torch.zeros_like(reduced).scatter_add(0, inverse, values)
        counts = torch.zeros_like(reduced).scatter_add(0, inverse, torch.ones_like(values))
        reduced = totals / counts.clamp(min=1)

    return reduced, unique


def frame_level_metrics(metric: BlinkMetrics) -> dict[str, torch.Tensor]:
    """Score one target at frame level, aggregating the eyes of each frame.

    This is the **eye state recognition (ESR)** protocol: one decision per
    frame. Training is eye-wise because that is how the corpora annotate, but
    the benchmark reports per frame, so the two eye-wise predictions of a frame
    are recombined here. The same aggregation runs on validation as on test, so
    model selection optimises the reported number rather than a proxy.

    A position is aggregated only when it is validly supervised; a frame whose
    every eye was masked out contributes nothing rather than a default.

    Args:
        metric (BlinkMetrics): An accumulator for one target.

    Returns:
        dict[str, torch.Tensor]: ``frame_<reducer>/<metric>`` for every reducer
        in :data:`FRAME_REDUCERS`. Empty when no group ids were accumulated,
        which is what a caller checks to know whether frame-level scoring ran.
    """
    gathered = metric._gathered()
    groups = metric.groups()
    if gathered is None or groups is None:
        return {}

    probability, target, mask = gathered
    valid = mask.bool()
    if not bool(valid.any()):
        return {}

    probability, target, groups = probability[valid], target[valid], groups[valid]

    results: dict[str, torch.Tensor] = {}
    for reducer in FRAME_REDUCERS:
        frame_probability, _ = aggregate_by_group(probability, groups, reducer)
        # Targets are aggregated by max whatever the prediction reducer: a frame
        # is a blink frame if *either* eye is annotated as blinking, which is
        # what the frame-level ground truth means.
        frame_target, _ = aggregate_by_group(target, groups, "max")
        frame_mask = torch.ones_like(frame_target, dtype=torch.bool)

        for name, value in binary_metrics(
            frame_probability, frame_target, frame_mask, metric.threshold
        ).items():
            results[f"frame_{reducer}/{name}"] = value

    return results


def window_level_metrics(metric: BlinkMetrics) -> dict[str, torch.Tensor]:
    """Score one target at window level: did this window contain a blink?

    This is the **blink presence detection (BPD)** protocol, and it is a
    genuinely different task from frame-wise eye-state recognition rather than a
    coarser view of it. A window is positive when *any* of its frames is
    annotated as blinking, and a prediction is positive when the model's
    sequence output crosses the threshold at any timestep — a ``max`` over the
    window.

    Aggregating a single sequence head, rather than training a second
    clip-level head, is what keeps the two tasks consistent: a window cannot
    come out "no frame shows a closed eye" *and* "a blink occurred", which two
    independent heads would permit.

    BPD is the more error-tolerant protocol, and that is why it is reported
    alongside ESR: the corpora do not annotate blink boundaries precisely, since
    a closure is motion that begins before and ends after the frames marked as
    blinking. A frame-wise score punishes a one-frame boundary disagreement that
    a window-wise score absorbs.

    Args:
        metric (BlinkMetrics): An accumulator for one target, whose group ids
            identify the sample each position came from.

    Returns:
        dict[str, torch.Tensor]: ``window/<metric>``. Empty when no group ids
        were accumulated.
    """
    gathered = metric._gathered()
    groups = metric.groups()
    if gathered is None or groups is None:
        return {}

    probability, target, mask = gathered
    valid = mask.bool()
    if not bool(valid.any()):
        return {}

    probability, target, groups = probability[valid], target[valid], groups[valid]

    # A window is one sample, so its positions share a sample id -- which the
    # frame group encodes as `sample_index * T + step`. Dividing it back out
    # recovers the sample, collapsing both eyes and every timestep of one
    # window into a single decision.
    sample_ids = torch.div(groups, metric.window_length, rounding_mode="floor")

    window_probability, _ = aggregate_by_group(probability, sample_ids, "max")
    window_target, _ = aggregate_by_group(target, sample_ids, "max")
    window_mask = torch.ones_like(window_target, dtype=torch.bool)

    return {
        f"window/{name}": value
        for name, value in binary_metrics(
            window_probability, window_target, window_mask, metric.threshold
        ).items()
    }


def _stack_state(state: Any) -> torch.Tensor | None:
    """Concatenate one torchmetrics state into a single tensor.

    ``add_state`` declares the state as a ``Tensor`` but populates a ``list`` at
    runtime, and ``dist_reduce_fx="cat"`` swaps it back to a single tensor once
    gathered across devices — so both forms are handled.

    Args:
        state (Any): Accumulated batches, or the already-gathered tensor.

    Returns:
        torch.Tensor | None: The concatenated state, or ``None`` when nothing is
        accumulated.
    """
    if not isinstance(state, list):
        return state
    if not state:
        return None
    return torch.cat(state, dim=0)


class BlinkMetrics(Metric):
    """Accumulates an epoch of masked predictions for one target.

    A :class:`torchmetrics.Metric` rather than plain lists, so state is gathered
    across devices automatically. Appending to Python lists on the module would,
    under DDP, silently report only rank 0's shard.

    Args:
        target_name (str): The target these metrics describe.
        threshold (float): Decision threshold.

    Raises:
        ValueError: If the threshold is outside ``(0, 1)``.
    """

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, target_name: str, threshold: float = DEFAULT_THRESHOLD):
        super().__init__()
        if not 0.0 < threshold < 1.0:
            raise ValueError(f"threshold must be in (0, 1), got {threshold}.")

        self.target_name = target_name
        self.threshold = threshold
        # Set on the first update. The frame-group id folds the timestep index
        # in (`sample * T + step`), so recovering which sample a position came
        # from -- which the window-level protocol needs -- requires knowing T.
        self.window_length = 1
        self.add_state("probability", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("mask", default=[], dist_reduce_fx="cat")
        # Which frame each position describes. Two eye-wise samples of the same
        # frame share a value, which is what lets FrameAggregator recombine
        # them. Accumulated as a metric state so it is gathered across devices
        # alongside the predictions it indexes.
        self.add_state("group", default=[], dist_reduce_fx="cat")

    def update(
        self,
        probability: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one batch.

        Args:
            probability (torch.Tensor): Predicted probabilities, ``(B, T)``.
            target (torch.Tensor): Binary targets, same shape.
            mask (torch.Tensor): ``True`` where the target is real supervision.
            group (torch.Tensor | None): ``(B,)`` integer frame-group id per
                sample, broadcast across its timesteps. ``None`` accumulates a
                sentinel, which disables frame-level aggregation for this epoch
                but leaves the per-eye metrics intact.

        Raises:
            ValueError: If the shapes disagree.
        """
        if probability.shape != target.shape or mask.shape != target.shape:
            raise ValueError(
                f"{self.target_name}: shapes must agree, got probability="
                f"{tuple(probability.shape)}, target={tuple(target.shape)}, "
                f"mask={tuple(mask.shape)}."
            )
        if group is not None and group.shape[0] != probability.shape[0]:
            raise ValueError(
                f"{self.target_name}: group has {group.shape[0]} entries for "
                f"{probability.shape[0]} samples."
            )

        # A (B, T) batch is the normal case; a flat (N,) update is treated as
        # one position per sample, which is what a caller scoring pre-flattened
        # predictions means.
        self.window_length = int(probability.shape[1]) if probability.ndim > 1 else 1

        if group is None:
            groups = torch.full_like(probability, NO_GROUP, dtype=torch.long).flatten()
        else:
            # One id per sample, repeated across its timesteps, so a position's
            # frame is recoverable after flattening. Timesteps of one sample are
            # distinct frames, so the step index is folded in.
            steps = torch.arange(self.window_length, device=probability.device)
            expanded = group.to(probability.device).reshape(-1, 1) * self.window_length
            groups = (expanded + steps).flatten()

        # Flattened to (N,) before accumulating: batches differ in nothing but
        # their leading axis.
        # `add_state(default=[])` makes these lists at runtime; a static reader
        # only sees the declared Tensor default.
        # Moved to the CPU before accumulating. The reduction happens once at
        # epoch end, so nothing here needs to stay on the accelerator -- and
        # keeping it there costs far more than the bytes suggest. A frame-wise
        # test pass is 34 819 batches; at four tensors per target that is around
        # 280 000 small allocations, which fragmented the MPS pool until a
        # routine 45 MiB request failed against the 42.43 GiB ceiling -- six
        # times, at batch ~15 400 each time, while the raw data is only 0.09 GB.
        # Dropping the allocator cache more often did not help, because the
        # references are live rather than cached.
        self.probability.append(probability.detach().flatten().cpu())  # ty: ignore[unresolved-attribute, call-non-callable]
        self.target.append(target.detach().flatten().cpu())  # ty: ignore[unresolved-attribute, call-non-callable]
        self.mask.append(mask.detach().flatten().cpu())  # ty: ignore[unresolved-attribute, call-non-callable]
        self.group.append(groups.detach().cpu())  # ty: ignore[unresolved-attribute, call-non-callable]

    def restore(
        self,
        probability: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        group: torch.Tensor,
        window_length: int,
    ) -> None:
        """Re-accumulate already-flattened state from a previous pass.

        :meth:`update` derives the group ids from the shapes it is handed, which
        is right for a live batch but wrong for state read back from disk: the
        ids there already encode the frames they came from and must survive
        verbatim, or every consumer that groups by frame would regroup the
        restored positions incorrectly.

        Every metric is an order-independent reduction over per-position values,
        so state appended here scores identically to state computed in place --
        which is what makes resuming an interrupted test pass exact rather than
        approximate.

        Args:
            probability (torch.Tensor): ``(N,)`` predicted probabilities.
            target (torch.Tensor): ``(N,)`` binary targets.
            mask (torch.Tensor): ``(N,)`` validity mask.
            group (torch.Tensor): ``(N,)`` frame-group id per position.
            window_length (int): Timesteps per sample in the restored pass.

        Raises:
            ValueError: If the shapes disagree.
        """
        shapes = {
            "probability": probability.shape,
            "target": target.shape,
            "mask": mask.shape,
            "group": group.shape,
        }
        if len(set(shapes.values())) != 1:
            raise ValueError(
                f"{self.target_name}: restored state must agree in shape, got "
                + ", ".join(f"{name}={tuple(shape)}" for name, shape in shapes.items())
                + "."
            )

        self.window_length = int(window_length)
        self.probability.append(probability.flatten())  # ty: ignore[unresolved-attribute, call-non-callable]
        self.target.append(target.flatten())  # ty: ignore[unresolved-attribute, call-non-callable]
        self.mask.append(mask.flatten())  # ty: ignore[unresolved-attribute, call-non-callable]
        self.group.append(group.flatten())  # ty: ignore[unresolved-attribute, call-non-callable]

    def _state_device(self) -> torch.device:
        """Device the accumulated state lives on.

        `update` moves everything to the CPU before accumulating, so a
        supervised target's scores come back on the CPU whatever device the
        metric was moved to. An unsupervised target has to agree, because
        `MultiTargetMetrics.compute` stacks the two and `torch.stack` across
        CPU and MPS segfaults rather than raising.

        Read from the buffers rather than hard-coded, so the two branches stay
        in step if accumulation ever moves back onto the accelerator.

        Returns:
            torch.device: Where accumulated tensors live; the CPU when nothing
            has been accumulated yet.
        """
        state = self.probability
        if isinstance(state, torch.Tensor):
            return state.device
        if isinstance(state, list):
            for entry in state:
                if isinstance(entry, torch.Tensor):
                    return entry.device
        return torch.device("cpu")

    def _gathered(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Concatenate accumulated state.

        Returns:
            tuple | None: ``(probability, target, mask)``, or ``None`` when
            nothing has been accumulated.
        """
        probability = _stack_state(self.probability)
        target = _stack_state(self.target)
        mask = _stack_state(self.mask)
        if probability is None or target is None or mask is None:
            return None
        if probability.numel() == 0:
            return None
        return probability, target, mask

    def groups(self) -> torch.Tensor | None:
        """Frame-group id of every accumulated position.

        Returns:
            torch.Tensor | None: ``(N,)`` int64, or ``None`` when nothing was
            accumulated or no group ids were supplied.
        """
        gathered = _stack_state(self.group)
        if gathered is None or gathered.numel() == 0:
            return None
        if bool((gathered == NO_GROUP).all()):
            return None
        return gathered

    @property
    def valid_count(self) -> int:
        """Number of validly-supervised positions accumulated this epoch.

        Logged so a run makes visible how much supervision each target actually
        received — a joint run where one head saw almost nothing is a data
        problem, not a modelling one, and this is what reveals it.

        Returns:
            int: Valid position count.
        """
        gathered = self._gathered()
        return 0 if gathered is None else int(gathered[2].bool().sum())

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute every metric over the accumulated epoch.

        Returns:
            dict[str, torch.Tensor]: Keyed ``"<target>/<metric>"``. Zeros
            throughout when the target received no supervision this epoch, which
            is a legitimate state for a joint run whose batch mix happened to
            exclude the annotating corpus.
        """
        gathered = self._gathered()
        if gathered is None:
            # **On the CPU, matching where a supervised target's scores land.**
            # These zeros are stacked with other targets' scores in
            # `MultiTargetMetrics.compute`, and `torch.stack` across CPU and MPS
            # **segfaults** -- a hard crash with no exception to catch. A
            # single-corpus eval reaches this whenever a corpus does not
            # annotate an eval-only target, which is ordinary rather than
            # exceptional.
            #
            # This said `device=self.device` until the accumulation buffers
            # moved to the CPU to stop the MPS allocator fragmenting (see
            # `update`). That made the two branches disagree: a *supervised*
            # target now computes from CPU state and returns CPU scalars, while
            # an *unsupervised* one still returned `mps` zeros -- so a run with
            # one of each stacked across devices and crashed. Measured as a
            # segfault in the full test suite, invisible when either test module
            # ran alone, because the crash needs both branches in one process.
            #
            # Deriving it from the accumulation buffers rather than hard-coding
            # "cpu" keeps the two in step if that decision is ever revisited.
            return {
                f"{self.target_name}/{name}": torch.zeros((), device=self._state_device())
                for name in METRIC_NAMES
            }

        probability, target, mask = gathered
        return {
            f"{self.target_name}/{name}": value
            for name, value in binary_metrics(probability, target, mask, self.threshold).items()
        }

    def predictions(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the accumulated predictions, targets, and masks.

        Used to dump per-sample test predictions as a run artifact, so
        downstream threshold tuning and significance testing never need the
        model re-run.

        Returns:
            tuple: ``(probability, target, mask)``, each ``(N,)``; empty tensors
            when nothing was accumulated.
        """
        gathered = self._gathered()
        if gathered is None:
            empty = torch.zeros(0)
            return empty, empty, empty.bool()
        return gathered


class MultiTargetMetrics(torch.nn.Module):
    """Holds one :class:`BlinkMetrics` per supervised target.

    A ``Module`` rather than a plain dict so Lightning moves the child metrics'
    state to the right device along with the model.

    Args:
        target_names (list[str]): Targets to track, in head order.
        threshold (float): Decision threshold, shared by every target.
    """

    def __init__(self, target_names: list[str], threshold: float = DEFAULT_THRESHOLD):
        super().__init__()
        self.target_names = list(target_names)
        self.metrics = torch.nn.ModuleDict(
            {name: BlinkMetrics(name, threshold) for name in self.target_names}
        )

    def __contains__(self, target_name: object) -> bool:
        """Whether this accumulator holds one target.

        Args:
            target_name (object): The target to look for.

        Returns:
            bool: Whether it is supervised here. Lets a caller skip an
            eval-only target on the training split, where no accumulator for
            it exists, without catching ``KeyError``.
        """
        return target_name in self.target_names

    def __getitem__(self, target_name: str) -> BlinkMetrics:
        """Look up one target's accumulator.

        ``ModuleDict.__getitem__`` is typed as returning ``Module``, which loses
        every :class:`BlinkMetrics` method. Narrowing here once means callers —
        including the prediction-writing callbacks — get the real type instead
        of casting at each use.

        Args:
            target_name (str): Which target's accumulator to return.

        Returns:
            BlinkMetrics: That target's accumulator.

        Raises:
            KeyError: If this run does not supervise that target.
        """
        if target_name not in self.target_names:
            raise KeyError(
                f"This run does not supervise {target_name!r}. Targets: {self.target_names}."
            )
        metric = self.metrics[target_name]
        assert isinstance(metric, BlinkMetrics)  # noqa: S101 -- narrows ModuleDict's Module
        return metric

    def update(
        self,
        target_name: str,
        probability: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        group: torch.Tensor | None = None,
    ) -> None:
        """Accumulate one batch for one target.

        Args:
            target_name (str): Which target to update.
            probability (torch.Tensor): Predicted probabilities.
            target (torch.Tensor): Binary targets.
            mask (torch.Tensor): Validity mask.
            group (torch.Tensor | None): ``(B,)`` frame-group id per sample.
        """
        # torchmetrics types `update` from the abstract base, whose signature the
        # subclass legitimately overrides.
        self[target_name].update(probability, target, mask, group)  # ty: ignore[invalid-argument-type]

    def compute(self) -> dict[str, torch.Tensor]:
        """Compute every target's metrics plus the cross-target summary.

        Reports two families of scores. The **eye-level** ones describe what the
        model was trained on — one prediction per eye. The **frame-level** ones
        aggregate the two eyes of a frame into a single decision, which is how
        the corpora are actually scored, and are what drives model selection
        when they are available.

        Returns:
            dict[str, torch.Tensor]: ``<target>/<metric>`` per eye,
            ``<target>/frame_<reducer>/<metric>`` per frame, a
            ``<target>/valid_positions`` count, and :data:`PRIMARY_METRIC`.
        """
        results: dict[str, torch.Tensor] = {}
        eye_f1 = []
        frame_f1 = []

        for name in self.target_names:
            metric = self[name]
            computed = metric.compute()  # ty: ignore[missing-argument]
            results.update(computed)
            # Beside the computed scores, not on the metric's nominal device:
            # every other entry in this dict comes back from the CPU
            # accumulation buffers, and a lone accelerator tensor here makes the
            # result set inconsistent for anything that iterates it.
            results[f"{name}/valid_positions"] = torch.tensor(
                float(metric.valid_count), device=computed[f"{name}/f1"].device
            )
            eye_f1.append(computed[f"{name}/f1"])

            # ESR: one decision per frame, both eyes recombined.
            frame_scores = frame_level_metrics(metric)
            results.update({f"{name}/{key}": value for key, value in frame_scores.items()})
            primary = frame_scores.get(f"frame_{PRIMARY_FRAME_REDUCER}/f1")
            if primary is not None:
                frame_f1.append(primary)

            # BPD: one decision per window, and only for the blink target --
            # "does this window contain a blink" is meaningless for eye state,
            # which is a per-frame property by definition.
            if name == TASK_BPD:
                window_scores = window_level_metrics(metric)
                results.update({f"{name}/{key}": value for key, value in window_scores.items()})
                if "window/f1" in window_scores:
                    results[BPD_METRIC] = window_scores["window/f1"]

        # Same device discipline as above: every scalar entering `torch.stack`
        # must already agree, or the stack segfaults rather than raising.
        device = eye_f1[0].device if eye_f1 else torch.device("cpu")
        results["mean_eye_f1"] = (
            torch.stack(eye_f1).mean() if eye_f1 else torch.zeros((), device=device)
        )
        # Frame-level drives selection whenever it is available; a run whose
        # batches carried no group ids falls back to the eye-level mean rather
        # than reporting nothing.
        results[PRIMARY_METRIC] = (
            torch.stack(frame_f1).mean() if frame_f1 else results["mean_eye_f1"]
        )
        return results

    def reset(self) -> None:
        """Clear every target's accumulated state."""
        for name in self.target_names:
            self[name].reset()
