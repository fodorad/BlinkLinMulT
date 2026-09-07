"""Loading trained v2 models for inference, without the training stack.

``BlinkCNN``'s weights live in a Lightning checkpoint, but nothing about
*running* one needs Lightning. Importing
:mod:`blinklinmult.train.module` pulls in ``lightning``, ``torchmetrics``,
``h5py``, ``omniloader`` and ``pandas`` -- five packages an inference caller has
no use for -- so this module rebuilds the bare :class:`~torch.nn.Module` from the
checkpoint's own stored hyperparameters instead.

That is possible because the checkpoint is self-describing: ``save_hyperparameters``
stores ``model_config`` and ``train_config`` as plain dicts (deliberately, so
``weights_only=True`` loading works), and ``build_model`` can rebuild the network
from them. The weights then load with **zero missing and zero unexpected keys**.

Published weights are stripped of optimizer and loop state before upload -- 57 MB
of Lightning checkpoint is only 19.1 MB of parameters -- so
:func:`load_checkpoint` accepts both shapes: a full training checkpoint from
``results/``, and the slim published file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from blinklinmult.train.model import BlinkModel

logger = logging.getLogger(__name__)
"""Module-level logger."""

RETIRED_1X_MODELS = ("DenseNet121", "BlinkLinT", "BlinkLinMulT", "LinT", "AbstractModule")
"""Names ``blinklinmult.models`` exported in 1.x, when it was a *package*.

It is a module now, and a different one: this file loads trained v2 checkpoints.
An old script importing one of these gets a bare "cannot import name" otherwise,
which says nothing about where the models went -- so :func:`__getattr__` answers
that instead. See ``docs/migration.md``.
"""

MODEL_PREFIX = "model."
"""Prefix Lightning adds to the wrapped network's parameter names."""

DEFAULT_TARGETS = ["eye_state"]
"""Head names assumed when a checkpoint does not record its own."""


class ModelLoadError(Exception):
    """Raised when a checkpoint cannot be rebuilt into a usable model."""


def _target_names(hyper: dict[str, Any]) -> list[str]:
    """Recover which heads a checkpoint was trained with.

    Args:
        hyper (dict[str, Any]): The checkpoint's ``hyper_parameters``.

    Returns:
        list[str]: Head names, defaulting to :data:`DEFAULT_TARGETS`.
    """
    train = hyper.get("train_config") or {}
    targets = train.get("targets") if isinstance(train, dict) else getattr(train, "targets", None)
    return list(targets) if targets else list(DEFAULT_TARGETS)


def strip_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Reduce a training checkpoint to what inference needs.

    Drops optimizer state, LR schedules, loop counters and callback state, all of
    which exist to *resume* training and are dead weight in a published artifact.
    On ``fw-focal-augment`` this takes 57 MB down to 19.1 MB.

    Args:
        checkpoint (dict[str, Any]): A loaded Lightning checkpoint.

    Returns:
        dict[str, Any]: ``{"state_dict", "hyper_parameters"}`` only.

    Raises:
        ModelLoadError: If either required key is absent.
    """
    for key in ("state_dict", "hyper_parameters"):
        if key not in checkpoint:
            raise ModelLoadError(
                f"Checkpoint is missing {key!r}; it does not look like a "
                "Lightning checkpoint from this project."
            )
    return {
        "state_dict": checkpoint["state_dict"],
        "hyper_parameters": checkpoint["hyper_parameters"],
    }


def load_checkpoint(path: str | Path) -> BlinkModel:
    """Rebuild a trained v2 model from its checkpoint, ready for inference.

    Args:
        path (str | Path): A full training checkpoint or a stripped published one.

    Returns:
        BlinkModel: The network in ``eval`` mode with its weights loaded.

    Raises:
        ModelLoadError: If the file is not a usable checkpoint, or its weights do
            not fit the architecture its own hyperparameters describe.
    """
    # Imported here, not at module scope: `train.model` reaches `linmult` for the
    # sequence families, and keeping it lazy means a caller who only wants the
    # paper models never pays for it.
    from blinklinmult.train.model import build_model

    location = Path(path)
    if not location.is_file():
        raise ModelLoadError(f"No checkpoint at {location}.")

    # weights_only=True refuses to unpickle arbitrary classes. The configs are
    # stored as plain dicts precisely so this works.
    checkpoint = torch.load(location, map_location="cpu", weights_only=True)
    hyper = checkpoint.get("hyper_parameters")
    if hyper is None:
        raise ModelLoadError(f"{location} carries no hyper_parameters; cannot rebuild the model.")

    model_config = hyper.get("model_config")
    if model_config is None:
        raise ModelLoadError(f"{location} carries no model_config; cannot rebuild the model.")

    from blinklinmult.train.config import ModelConfig

    config = (
        model_config if isinstance(model_config, ModelConfig) else ModelConfig(**dict(model_config))
    )
    model = build_model(
        config,
        target_names=_target_names(hyper),
        image_size=int(hyper.get("image_size", 64)),
        eye_feature_dim=hyper.get("eye_feature_dim"),
    )

    state = checkpoint["state_dict"]
    weights = {
        key.removeprefix(MODEL_PREFIX): value
        for key, value in state.items()
        if key.startswith(MODEL_PREFIX)
    }
    if not weights:
        # A stripped checkpoint may already hold bare names.
        weights = dict(state)

    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as error:
        raise ModelLoadError(f"{location} does not fit the model it describes: {error}") from error

    model.eval()
    logger.info(f"Loaded {config.family!r} model from {location}.")
    return model


def __getattr__(name: str) -> object:
    """Explain the 1.x models' move rather than failing opaquely.

    Args:
        name (str): The attribute being looked up.

    Returns:
        object: Never returns; the lookup always fails.

    Raises:
        AttributeError: Naming the replacement when a retired 1.x symbol is
            asked for, and the plain message otherwise.
    """
    if name in RETIRED_1X_MODELS:
        raise AttributeError(
            f"{name!r} moved in 2.0. The 1.x models now ship as frozen ONNX "
            f"graphs and load by id:\n\n"
            f"    from blinklinmult import BlinkDetector\n"
            f"    model = BlinkDetector.from_pretrained('densenet121-union')\n\n"
            f"Ids: densenet121-union, blinklint-union, blinklinmult-union, "
            f"blinkcnn. See docs/migration.md."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
