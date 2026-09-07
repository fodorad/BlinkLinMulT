"""Masked binary losses for partially-annotated blink supervision.

Every loss here takes a validity mask alongside its target and reduces over the
valid positions only. That is not an optimisation — it is what makes joint
training across the six corpora correct. A sample drawn from CEW carries an
all-``False`` ``blink_presence`` mask and a placeholder target; reducing over it
with an unmasked ``BCEWithLogitsLoss`` would train the blink head to predict the
placeholder on every still image in the corpus, which is the single most likely
way to get a plausible-looking run that has quietly learned the wrong thing.

**Reduction.** The mean is taken over valid positions, so a batch's loss does
not depend on how many of its samples happened to come from a corpus that
annotates the target. A batch with no valid position for a target contributes
exactly zero to that head — with the gradient path preserved, so DDP does not
deadlock waiting for a rank that skipped a parameter.

**Logits, not probabilities.** All losses consume raw logits and apply their own
sigmoid where needed. Applying a sigmoid in the model and a ``log`` here would
lose the numerically-stable fused implementation for no benefit.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

EPSILON = 1e-8
"""Guards division when a batch has no valid position for a target."""

DEFAULT_FOCAL_GAMMA = 2.0
"""Focusing parameter from the focal-loss paper."""

DEFAULT_FOCAL_ALPHA = 0.25
"""Positive-class weight from the focal-loss paper.

Blink frames are the minority class in a continuously-sampled recording — a
blink occupies a few frames per several seconds — so a class-balancing term is
the default rather than an option.
"""


class LossError(ValueError):
    """Raised when a loss is misconfigured or given inconsistent shapes."""


def _check_shapes(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
    """Verify a loss's three inputs describe the same positions.

    Args:
        logits (torch.Tensor): Raw predictions.
        target (torch.Tensor): Targets.
        mask (torch.Tensor): Validity mask.

    Raises:
        LossError: If the shapes disagree.
    """
    if logits.shape != target.shape:
        raise LossError(
            f"logits shape {tuple(logits.shape)} != target shape {tuple(target.shape)}."
        )
    if mask.shape != target.shape:
        raise LossError(f"mask shape {tuple(mask.shape)} != target shape {tuple(target.shape)}.")


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean of ``values`` over the positions ``mask`` marks valid.

    Multiplying by the mask and dividing by its sum, rather than indexing with
    it, keeps the result connected to every parameter in the graph. Boolean
    indexing on an all-``False`` mask yields an empty tensor whose backward pass
    reaches no parameter, which under DDP hangs the step waiting for gradients
    that never arrive.

    Args:
        values (torch.Tensor): Per-position values.
        mask (torch.Tensor): Boolean validity mask, broadcastable to ``values``.

    Returns:
        torch.Tensor: Scalar mean, or a differentiable zero when no position is
        valid.
    """
    weights = mask.to(values.dtype)
    total = weights.sum()
    return (values * weights).sum() / (total + EPSILON)


def masked_bce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy over valid positions.

    Args:
        logits (torch.Tensor): Raw predictions, any shape.
        target (torch.Tensor): Binary targets, same shape.
        mask (torch.Tensor): ``True`` where the target is real supervision.

    Returns:
        torch.Tensor: Scalar loss.

    Raises:
        LossError: If the shapes disagree.
    """
    _check_shapes(logits, target, mask)
    # The target is clamped because a masked-out position holds the placeholder
    # (-1), and BCE requires targets in [0, 1] even where they are weighted to
    # zero -- the function validates its input before the weighting applies.
    elementwise = F.binary_cross_entropy_with_logits(
        logits, target.clamp(0.0, 1.0), reduction="none"
    )
    return _masked_mean(elementwise, mask)


def masked_focal(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    alpha: float = DEFAULT_FOCAL_ALPHA,
) -> torch.Tensor:
    """Focal loss over valid positions.

    Down-weights positions the model already classifies confidently, which
    matters for blink presence: sampled continuously, the overwhelming majority
    of frames are easy negatives, and their aggregate gradient drowns out the
    few frames where the eye is actually closing.

    Args:
        logits (torch.Tensor): Raw predictions.
        target (torch.Tensor): Binary targets.
        mask (torch.Tensor): ``True`` where the target is real supervision.
        gamma (float): Focusing parameter; ``0`` reduces this to weighted BCE.
        alpha (float): Positive-class weight in ``[0, 1]``.

    Returns:
        torch.Tensor: Scalar loss.

    Raises:
        LossError: If the shapes disagree or a parameter is out of range.
    """
    _check_shapes(logits, target, mask)
    if gamma < 0:
        raise LossError(f"focal gamma must be >= 0, got {gamma}.")
    if not 0.0 <= alpha <= 1.0:
        raise LossError(f"focal alpha must be in [0, 1], got {alpha}.")

    clamped = target.clamp(0.0, 1.0)
    elementwise = F.binary_cross_entropy_with_logits(logits, clamped, reduction="none")

    probability = torch.sigmoid(logits)
    # Probability assigned to the true class at each position.
    p_t = probability * clamped + (1 - probability) * (1 - clamped)
    alpha_t = alpha * clamped + (1 - alpha) * (1 - clamped)

    return _masked_mean(alpha_t * (1 - p_t).pow(gamma) * elementwise, mask)


def masked_dice(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss over valid positions.

    Dice optimises overlap rather than per-position accuracy, so it is far less
    sensitive to the positive/negative imbalance of a blink sequence than
    cross-entropy is. Used as the second term of :func:`masked_dice_bce`.

    Args:
        logits (torch.Tensor): Raw predictions.
        target (torch.Tensor): Binary targets.
        mask (torch.Tensor): ``True`` where the target is real supervision.

    Returns:
        torch.Tensor: Scalar loss in ``[0, 1]``.

    Raises:
        LossError: If the shapes disagree.
    """
    _check_shapes(logits, target, mask)
    weights = mask.to(logits.dtype)
    probability = torch.sigmoid(logits) * weights
    clamped = target.clamp(0.0, 1.0) * weights

    intersection = (probability * clamped).sum()
    total = probability.sum() + clamped.sum()
    return 1.0 - (2.0 * intersection + EPSILON) / (total + EPSILON)


def masked_dice_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Sum of :func:`masked_bce` and a weighted :func:`masked_dice`.

    Args:
        logits (torch.Tensor): Raw predictions.
        target (torch.Tensor): Binary targets.
        mask (torch.Tensor): ``True`` where the target is real supervision.
        dice_weight (float): Weight of the Dice term.

    Returns:
        torch.Tensor: Scalar loss.

    Raises:
        LossError: If the shapes disagree or the weight is negative.
    """
    if dice_weight < 0:
        raise LossError(f"dice_weight must be >= 0, got {dice_weight}.")
    return masked_bce(logits, target, mask) + dice_weight * masked_dice(logits, target, mask)


class MaskedLoss(nn.Module):
    """A masked binary loss as a configurable module.

    Args:
        name (str): One of ``"bce"``, ``"focal"``, ``"dice_bce"``.
        **kwargs: Loss-specific arguments, forwarded to the underlying function.

    Raises:
        LossError: If the name is unknown or an argument is not accepted.
    """

    def __init__(self, name: str, **kwargs):
        super().__init__()
        functions = {
            "bce": masked_bce,
            "focal": masked_focal,
            "dice_bce": masked_dice_bce,
        }
        if name not in functions:
            raise LossError(f"Unknown loss {name!r}; expected one of {sorted(functions)}.")

        accepted = {
            "bce": set(),
            "focal": {"gamma", "alpha"},
            "dice_bce": {"dice_weight"},
        }[name]
        unknown = sorted(set(kwargs) - accepted)
        if unknown:
            raise LossError(
                f"loss {name!r} does not accept {unknown}; accepted: {sorted(accepted)}."
            )

        self.name = name
        self.kwargs = dict(kwargs)
        self._function = functions[name]

    def forward(
        self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits (torch.Tensor): Raw predictions.
            target (torch.Tensor): Binary targets.
            mask (torch.Tensor): ``True`` where the target is real supervision.

        Returns:
            torch.Tensor: Scalar loss.
        """
        return self._function(logits, target, mask, **self.kwargs)


def build_loss(name: str, **kwargs) -> MaskedLoss:
    """Construct a loss by config name.

    Args:
        name (str): Loss name.
        **kwargs: Loss-specific arguments.

    Returns:
        MaskedLoss: The loss module.

    Raises:
        LossError: If the name is unknown.
    """
    return MaskedLoss(name, **kwargs)


def temporal_smoothness(
    logits: torch.Tensor, mask: torch.Tensor, margin: float = 0.0
) -> torch.Tensor:
    """Penalise frame-to-frame jumps in the predicted eye-state signal.

    A blink is continuous motion -- the lid closes over several frames and
    reopens over several more -- so the signal that describes it should not
    chatter. Independent per-frame predictions have no reason to be smooth, and
    a jagged signal is what makes interval extraction brittle: every spurious
    crossing of the operating point becomes a spurious event boundary.

    Needs **no annotation**, so it applies to every corpus including those that
    label only blink presence, or nothing at all.

    Args:
        logits (torch.Tensor): ``(B, T)`` raw predictions.
        mask (torch.Tensor): ``(B, T)`` ``True`` where the frame is valid.
        margin (float): Jumps at or below this are free, so a genuine closure
            is not penalised for being fast. ``0.0`` penalises every jump.

    Returns:
        torch.Tensor: Scalar; zero when fewer than two valid adjacent frames
        exist, which is the still-image case.

    Raises:
        LossError: If the shapes disagree or the input is not ``(B, T)``.
    """
    if logits.shape != mask.shape:
        raise LossError(f"logits {tuple(logits.shape)} and mask {tuple(mask.shape)} must agree.")
    if logits.ndim != 2:
        raise LossError(f"expected (B, T) logits, got {tuple(logits.shape)}.")
    if logits.shape[1] < 2:
        return torch.zeros((), device=logits.device)

    probability = torch.sigmoid(logits)
    # A pair counts only when *both* of its frames are real supervision;
    # a jump across a masked gap is not evidence of anything.
    pairs = mask[:, 1:] & mask[:, :-1]
    jumps = (probability[:, 1:] - probability[:, :-1]).abs()
    return _masked_mean((jumps - margin).clamp(min=0.0), pairs)


def duration_prior(
    logits: torch.Tensor,
    mask: torch.Tensor,
    threshold: float = 0.5,
    min_frames: float = 3.0,
    max_frames: float = 12.0,
) -> torch.Tensor:
    """Penalise predicted closures far outside a plausible blink duration.

    A human blink lasts roughly 100-400 ms, which at 30 fps is about 3-12
    frames. A model firing on one isolated frame, or holding a closure for
    seconds, is producing something that is not a blink whatever the frame-wise
    loss says.

    Applied as a **soft count** of above-threshold frames per window rather than
    by extracting intervals, so it stays differentiable. That makes it a prior
    on how much of a window is closed, not a hard constraint on any one event --
    the honest reading, since a window may legitimately contain two blinks.

    Needs **no annotation**.

    Args:
        logits (torch.Tensor): ``(B, T)`` raw predictions.
        mask (torch.Tensor): ``(B, T)`` validity.
        threshold (float): Probability above which a frame counts as closed.
        min_frames (float): Below this, a window's closure is implausibly short.
        max_frames (float): Above this, implausibly long.

    Returns:
        torch.Tensor: Scalar; zero when the window holds no valid frame, and
        zero for any window whose soft count already lies in range.

    Raises:
        LossError: If the shapes disagree or the bounds are inverted.
    """
    if logits.shape != mask.shape:
        raise LossError(f"logits {tuple(logits.shape)} and mask {tuple(mask.shape)} must agree.")
    if min_frames > max_frames:
        raise LossError(f"min_frames {min_frames} exceeds max_frames {max_frames}.")

    probability = torch.sigmoid(logits) * mask.float()
    # Soft count: how many frames this window calls closed. A window with no
    # closure at all is not penalised -- most windows contain no blink, and
    # demanding one would invent events.
    counts = (probability - threshold).clamp(min=0.0).sum(dim=-1) / max(1.0 - threshold, 1e-6)

    too_long = (counts - max_frames).clamp(min=0.0)
    # Only windows that fired at all are held to the lower bound.
    fired = (counts > 0).float()
    too_short = (min_frames - counts).clamp(min=0.0) * fired
    return (too_long + too_short).mean()
