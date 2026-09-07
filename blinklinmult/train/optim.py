"""Optimizer and learning-rate schedule construction.

The one thing here that is specific to this project is the **discriminative
learning rate**: an ImageNet-pretrained eye backbone and a randomly-initialised
transformer do not want the same step size. Fine-tuning the backbone at the
transformer's rate destroys the pretrained features in the first few hundred
steps, which shows up as a run that trains stably and scores worse than the
frozen-backbone baseline. :func:`build_optimizer` therefore splits parameters
into two groups whenever ``optimizer.backbone_lr`` is set.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from torch.optim import SGD, Adam, AdamW, Optimizer, RAdam
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    OneCycleLR,
    ReduceLROnPlateau,
    SequentialLR,
)

from blinklinmult.train.metrics import PRIMARY_METRIC

if TYPE_CHECKING:
    from torch import nn

    from blinklinmult.train.config import OptimConfig

logger = logging.getLogger(__name__)
"""Module-level logger."""

BACKBONE_GROUP = "backbone"
"""Name of the parameter group holding the pretrained image backbone."""

HEAD_GROUP = "head"
"""Name of the parameter group holding everything else."""

MIN_ONECYCLE_STEPS = 3
"""Fewest optimizer steps a one-cycle schedule can be given.

Below this there is no room for a warmup step, a peak, and an annealing step,
and ``OneCycleLR`` divides by a zero-width phase.
"""


class OptimError(ValueError):
    """Raised when an optimizer or schedule cannot be built."""


def parameter_groups(model: nn.Module, config: OptimConfig) -> list[dict[str, Any]]:
    """Split a model's parameters into backbone and head groups.

    Args:
        model (nn.Module): The model.
        config (OptimConfig): Optimizer settings.

    Returns:
        list[dict]: Optimizer parameter groups. A single group when
        ``backbone_lr`` is unset, or when the backbone is frozen and therefore
        contributes no trainable parameter.
    """
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]

    if config.backbone_lr is None:
        return [{"params": [p for _, p in trainable], "lr": config.lr, "name": HEAD_GROUP}]

    backbone = [p for name, p in trainable if ".backbone." in f".{name}"]
    head = [p for name, p in trainable if ".backbone." not in f".{name}"]

    if not backbone:
        logger.info("No trainable backbone parameters; using a single parameter group.")
        return [{"params": head, "lr": config.lr, "name": HEAD_GROUP}]

    logger.info(
        f"Discriminative learning rates: backbone={config.backbone_lr:g} "
        f"({len(backbone)} tensors), rest={config.lr:g} ({len(head)} tensors)."
    )
    return [
        {"params": backbone, "lr": config.backbone_lr, "name": BACKBONE_GROUP},
        {"params": head, "lr": config.lr, "name": HEAD_GROUP},
    ]


def build_optimizer(config: OptimConfig, model: nn.Module) -> Optimizer:
    """Construct the configured optimizer over the model's parameter groups.

    Args:
        config (OptimConfig): Optimizer settings.
        model (nn.Module): The model whose parameters are optimized.

    Returns:
        Optimizer: The optimizer.

    Raises:
        OptimError: If the optimizer name is unknown, or the model has no
            trainable parameter at all.
    """
    groups = parameter_groups(model, config)
    if not any(group["params"] for group in groups):
        raise OptimError("The model has no trainable parameters. Check model.backbone_freeze.")

    if config.name == "adam":
        return Adam(groups, lr=config.lr, weight_decay=config.weight_decay)
    if config.name == "adamw":
        return AdamW(groups, lr=config.lr, weight_decay=config.weight_decay)
    if config.name == "radam":
        return RAdam(groups, lr=config.lr, weight_decay=config.weight_decay)
    if config.name == "sgd":
        return SGD(
            groups,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    raise OptimError(f"Unknown optimizer {config.name!r}.")


def build_scheduler(
    config: OptimConfig,
    optimizer: Optimizer,
    steps_per_epoch: int,
    max_epochs: int,
) -> dict[str, Any] | None:
    """Construct the configured LR schedule, in Lightning's config format.

    Args:
        config (OptimConfig): Optimizer settings.
        optimizer (Optimizer): The optimizer to schedule.
        steps_per_epoch (int): Optimizer steps in one epoch.
        max_epochs (int): Total epochs.

    Returns:
        dict | None: Lightning ``lr_scheduler_config``, or ``None`` when no
        schedule is configured.

    Raises:
        OptimError: If the scheduler name is unknown.
    """
    if config.scheduler == "none":
        return None

    total_steps = max(1, steps_per_epoch * max_epochs)

    if config.scheduler == "onecycle":
        if total_steps < MIN_ONECYCLE_STEPS:
            logger.warning(
                f"onecycle needs at least {MIN_ONECYCLE_STEPS} optimizer steps but this "
                f"run has {total_steps}; falling back to a constant learning rate."
            )
            return None

        # Each group keeps its own peak, so the backbone's lower rate survives
        # the schedule instead of being flattened to a single max_lr.
        max_lr = [group["lr"] for group in optimizer.param_groups]
        scheduler: LRScheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=_safe_pct_start(config.warmup_ratio, total_steps),
        )
        return {"scheduler": scheduler, "interval": "step", "frequency": 1}

    if config.scheduler == "cosine":
        # The cosine's period must cover only the steps that remain after
        # warmup. Giving it the full run while a warmup is chained in front makes
        # it complete its half-cycle early and sit at lr=0 for the tail.
        warmup_steps = int(total_steps * config.warmup_ratio)
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps), eta_min=config.lr * 0.01
        )
        return {
            "scheduler": _with_warmup(config, optimizer, cosine, total_steps),
            "interval": "step",
            "frequency": 1,
        }

    if config.scheduler == "plateau":
        # `scheduler_patience` when set, else a tenth of the run. The default
        # scales with `max_epochs` so a short smoke run does not wait longer to
        # cut the LR than it runs for; an explicit value is what a real run
        # wants, since the right patience depends on how noisy the validation
        # metric is rather than on the epoch budget.
        patience = (
            config.scheduler_patience
            if config.scheduler_patience is not None
            else max(1, max_epochs // 10)
        )
        return {
            "scheduler": ReduceLROnPlateau(
                optimizer, mode="max", factor=config.scheduler_factor, patience=patience
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": f"valid/{PRIMARY_METRIC}",
        }

    raise OptimError(f"Unknown scheduler {config.scheduler!r}.")


def _safe_pct_start(warmup_ratio: float, total_steps: int) -> float:
    """Nudge ``pct_start`` off the value that makes ``OneCycleLR`` divide by zero.

    ``OneCycleLR`` places its first phase boundary at ``pct_start * total_steps
    - 1`` and later divides by that phase's width. When ``pct_start *
    total_steps`` is exactly ``1`` the boundary lands on step 0, the width is
    zero, and the scheduler raises ``ZeroDivisionError`` on construction. This
    bites real configurations — the default ``warmup_ratio`` of 0.1 over exactly
    10 optimizer steps is enough to trigger it, which is a plausible smoke run.

    Args:
        warmup_ratio (float): Configured warmup fraction.
        total_steps (int): Total optimizer steps in the run.

    Returns:
        float: A ``pct_start`` that keeps both phases non-degenerate.
    """
    pct = warmup_ratio
    if abs(pct * total_steps - 1.0) < 1e-9:
        pct = 1.5 / total_steps
        logger.debug(
            f"nudged onecycle pct_start to {pct:.4f} to keep its warmup phase non-empty "
            f"at {total_steps} steps"
        )
    # Keep at least one step either side of the peak.
    return float(min(max(pct, 1.5 / total_steps), 1.0 - 1.0 / total_steps))


def _with_warmup(
    config: OptimConfig,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    total_steps: int,
) -> LRScheduler:
    """Prefix a schedule with a linear warmup, if one is configured.

    Args:
        config (OptimConfig): Optimizer settings.
        optimizer (Optimizer): The optimizer.
        scheduler (LRScheduler): The main schedule.
        total_steps (int): Total optimizer steps.

    Returns:
        LRScheduler: The main schedule, or a warmup chained into it.
    """
    warmup_steps = int(total_steps * config.warmup_ratio)
    if warmup_steps < 1:
        return scheduler

    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
    return SequentialLR(optimizer, schedulers=[warmup, scheduler], milestones=[warmup_steps])
