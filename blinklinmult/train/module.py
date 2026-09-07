"""The Lightning module wrapping :class:`~blinklinmult.train.model.BlinkModel`.

One training step serves all three model families and both tasks. What makes
that possible is that every target is handled identically — logits, target, and
*mask* — so a joint run and a single-task run differ only in how many entries
the target loop has, and a corpus that does not annotate a target simply
contributes an all-``False`` mask that the loss and the metrics both skip.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, replace
from typing import TYPE_CHECKING, Any

import lightning as L
import torch

from blinklinmult.data.collate import apply_occlusion, target_and_mask, unpack_batch
from blinklinmult.data.schema import (
    EYE_EMBEDDING,
    EYE_FEATURE,
    EYE_IMAGE,
    SAMPLE_KEY,
    SOURCE_KEY,
)
from blinklinmult.train.config import ModelConfig, TrainConfig
from blinklinmult.train.losses import build_loss, duration_prior, temporal_smoothness
from blinklinmult.train.metrics import (
    PRIMARY_METRIC,
    MultiTargetMetrics,
    frame_group_ids,
)
from blinklinmult.train.model import build_model
from blinklinmult.train.optim import build_optimizer, build_scheduler

if TYPE_CHECKING:
    from collections.abc import Mapping

logger = logging.getLogger(__name__)
"""Module-level logger."""


def _as_model_config(config: ModelConfig | Mapping[str, Any]) -> ModelConfig:
    """Coerce a checkpoint's stored dict back into a typed config.

    Args:
        config (ModelConfig | Mapping): A config or its ``asdict`` form.

    Returns:
        ModelConfig: The typed config.
    """
    return config if isinstance(config, ModelConfig) else ModelConfig(**dict(config))


def _as_train_config(config: TrainConfig | Mapping[str, Any]) -> TrainConfig:
    """Coerce a checkpoint's stored dict back into a typed config.

    Args:
        config (TrainConfig | Mapping): A config or its ``asdict`` form.

    Returns:
        TrainConfig: The typed config, with nested sections rebuilt.
    """
    if isinstance(config, TrainConfig):
        return config
    return TrainConfig.from_dict(dict(config))


class BlinkLightningModule(L.LightningModule):
    """Trains the blink model on one or both tasks.

    The model is built here from its config rather than injected, so
    ``load_from_checkpoint(path)`` reconstructs a run without the caller having
    to rebuild and pass the architecture.

    Args:
        model_config (ModelConfig): Architecture settings.
        train_config (TrainConfig): Task, loss, and optimizer settings.
        image_size (int): Side length of a square eye crop, from the datasets.
        eye_feature_dim (int | None): Handcrafted feature width, from the
            datasets, or ``None`` when the run has none.
        steps_per_epoch (int): Optimizer steps per epoch, for step schedules.
    """

    def __init__(
        self,
        model_config: ModelConfig | Mapping[str, Any],
        train_config: TrainConfig | Mapping[str, Any],
        image_size: int,
        eye_feature_dim: int | None = None,
        steps_per_epoch: int = 1,
    ):
        super().__init__()

        model_config = _as_model_config(model_config)
        train_config = _as_train_config(train_config)

        # Configs are stored as plain dicts, not dataclass instances. Torch 2.6+
        # loads checkpoints with weights_only=True, which refuses to unpickle
        # arbitrary classes -- saving the dataclasses themselves would make every
        # checkpoint unloadable without an allowlist.
        self.save_hyperparameters(
            {
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
                "image_size": image_size,
                "eye_feature_dim": eye_feature_dim,
                "steps_per_epoch": steps_per_epoch,
            }
        )

        self.model_config = model_config
        self.train_config = train_config
        self.image_size = image_size
        self.eye_feature_dim = eye_feature_dim
        self.steps_per_epoch = steps_per_epoch
        self.target_names = list(train_config.targets)
        # Scored but never trained: the model grows no head for these and they
        # reach no loss. A frame-wise model predicts *closure* while the video
        # benchmark annotates blink *events*, so its one output is measured
        # against a second annotation. `scored_targets` is what the reporting
        # callbacks gate on -- `target_names` alone would exclude these.
        self.eval_target_names = [
            name for name in train_config.eval_targets if name not in self.target_names
        ]
        self.scored_targets = [*self.target_names, *self.eval_target_names]

        self.model = build_model(model_config, self.target_names, image_size, eye_feature_dim)
        self.loss_fn = build_loss(train_config.loss, **train_config.loss_kwargs)

        self.train_metrics = MultiTargetMetrics(self.target_names)
        # Validation and test carry the eval-only targets too: validation
        # because the operating point is fitted there, test because that is
        # where the benchmark is reported.
        self.valid_metrics = MultiTargetMetrics(self.scored_targets)
        self.test_metrics = MultiTargetMetrics(self.scored_targets)

        self.valid_sample_ids: list[str] = []
        self.valid_datasets: list[str] = []
        self.test_sample_ids: list[str] = []
        self.test_datasets: list[str] = []
        # Batches already scored by an interrupted pass whose state was restored
        # from disk. Set by `TestCheckpointer`; see its docstring for why
        # skipping them yields exactly the same metrics as recomputing them.
        self.skip_test_batches = 0

    @property
    def feature_names(self) -> list[str]:
        """Feature keys this model reads from a batch, in input order.

        Returns:
            list[str]: One or two keys, depending on the model family.
        """
        names = [EYE_IMAGE]
        if self.model.uses_eye_features:
            names.append(EYE_FEATURE)
        return names

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run the model over one batch.

        Args:
            batch (dict): A collated OmniLoader batch.

        Returns:
            dict[str, torch.Tensor]: Raw logits per target.
        """
        inputs, masks = unpack_batch(batch, self.feature_names)

        # Present only when a frozen encoder let the datamodule precompute them;
        # the model then skips its encoder pass entirely.
        embedding = batch.get(EYE_EMBEDDING)

        if self.model.uses_eye_features:
            return self.model(inputs[0], masks[0], inputs[1], masks[1], eye_embedding=embedding)
        return self.model(inputs[0], masks[0], eye_embedding=embedding)

    def _shared_step(
        self, batch: dict[str, Any], metrics: MultiTargetMetrics
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run one step, accumulate metrics, and sum the per-target losses.

        Args:
            batch (dict): A collated batch.
            metrics (MultiTargetMetrics): Accumulator for this split.

        Returns:
            tuple: ``(total_loss, per_target_losses)``.
        """
        # Applied here rather than in `forward`, because both the model input
        # *and* the targets must see it: masking a local copy inside `forward`
        # would leave the loss supervising an eye the model was never shown.
        if self.train_config.occlusion_yaw is not None:
            batch = apply_occlusion(batch, self.train_config.occlusion_yaw)

        logits = self(batch)
        losses: dict[str, torch.Tensor] = {}
        total = torch.zeros((), device=self.device)
        # Which frame each sample describes, so the two eyes of one frame can be
        # recombined into a single prediction at metric time.
        frame_groups = frame_group_ids(batch.get(SAMPLE_KEY, []))

        for name in self.target_names:
            target, mask = target_and_mask(batch, name)
            prediction = logits[name]

            # A head of width 1 emits (B, T, 1) against a (B, T) target; the
            # squeeze is what keeps blink presence and eye state on one path.
            if prediction.shape != target.shape and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)

            loss = self.loss_fn(prediction, target, mask)
            losses[name] = loss
            total = total + self.train_config.weight_for(name) * loss

            metrics.update(name, torch.sigmoid(prediction.detach()), target, mask, frame_groups)

        # Eval-only targets: score the model's existing prediction against a
        # second annotation, with no head and no loss. The frame-wise model
        # emits one logit -- closure -- and the video benchmark asks whether a
        # blink occurred; `P(blink | closed) = 1.0` on every corpus that
        # annotates both, so the closure signal is the blink signal, just
        # narrower in extent (it covers 19-34% of an event's frames).
        for name in self.eval_target_names:
            if name not in metrics:
                continue
            # A corpus that does not annotate this target contributes nothing to
            # it. That is normal rather than exceptional: an eval-only target is
            # scored opportunistically wherever it exists, and a still-image
            # corpus like CEW or MRL-Eye cannot witness a blink at all. The
            # trained targets above keep the strict lookup, where a missing
            # annotation really is a data error.
            if name not in batch or f"{name}_mask" not in batch:
                continue
            target, mask = target_and_mask(batch, name)
            prediction = next(iter(logits.values()))
            if prediction.shape != target.shape and prediction.shape[-1] == 1:
                prediction = prediction.squeeze(-1)
            metrics.update(name, torch.sigmoid(prediction.detach()), target, mask, frame_groups)

        # Annotation-free constraints on the frame-wise signal. They are
        # regularisers, not targets: they shape *how* the signal behaves over
        # time rather than what it predicts, so they apply to every corpus --
        # including those annotating only blink presence. Both default to a
        # weight of 0.0, so a run that does not ask for them is unchanged.
        total = total + self._consistency(logits, batch)

        return total, losses

    def _consistency(self, logits: dict[str, torch.Tensor], batch: dict[str, Any]) -> torch.Tensor:
        """Weighted sum of the annotation-free signal regularisers.

        Args:
            logits (dict[str, torch.Tensor]): Raw per-target predictions.
            batch (dict): The collated batch, for the validity mask.

        Returns:
            torch.Tensor: Scalar penalty; exactly zero when both weights are off.
        """
        weights = (self.train_config.smoothness_weight, self.train_config.duration_weight)
        if not any(weights):
            return torch.zeros((), device=self.device)

        prediction = next(iter(logits.values()))
        if prediction.ndim == 3 and prediction.shape[-1] == 1:
            prediction = prediction.squeeze(-1)
        if prediction.ndim != 2:
            return torch.zeros((), device=self.device)

        mask = batch.get(f"{EYE_IMAGE}_mask")
        valid = (
            mask.bool()
            if mask is not None and mask.shape == prediction.shape
            else torch.ones_like(prediction, dtype=torch.bool)
        )

        penalty = torch.zeros((), device=self.device)
        if self.train_config.smoothness_weight:
            penalty = penalty + self.train_config.smoothness_weight * temporal_smoothness(
                prediction, valid, self.train_config.smoothness_margin
            )
        if self.train_config.duration_weight:
            penalty = penalty + self.train_config.duration_weight * duration_prior(
                prediction, valid
            )
        return penalty

    def _log_losses(self, split: str, total: torch.Tensor, losses: dict[str, torch.Tensor]) -> None:
        """Log the total loss and, for a joint run, each target's share.

        Args:
            split (str): ``"train"``, ``"valid"``, or ``"test"``.
            total (torch.Tensor): The weighted total.
            losses (dict[str, torch.Tensor]): Per-target losses.
        """
        on_step = split == "train"
        self.log(f"{split}/loss", total, on_step=on_step, on_epoch=True, prog_bar=True)
        if len(losses) > 1:
            for name, value in losses.items():
                self.log(f"{split}/loss/{name}", value, on_step=False, on_epoch=True)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Run one training step.

        Args:
            batch (dict): A collated batch.
            batch_idx (int): Batch index within the epoch.

        Returns:
            torch.Tensor: The loss to backpropagate.
        """
        total, losses = self._shared_step(batch, self.train_metrics)
        self._log_losses("train", total, losses)
        return total

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:  # noqa: ARG002
        """Run one validation step.

        Args:
            batch (dict): A collated batch.
            batch_idx (int): Batch index within the epoch.

        Returns:
            torch.Tensor: The loss.
        """
        total, losses = self._shared_step(batch, self.valid_metrics)
        # Kept for the same reason the test ids are: the event-level threshold
        # is fitted on validation and applied to test, which needs the
        # validation predictions reassembled onto their own timelines.
        self.valid_sample_ids.extend(batch.get(SAMPLE_KEY, []))
        # Alongside the ids: the event report drops carrier-corpus samples, and
        # cannot tell which corpus a sample came from without this.
        self.valid_datasets.extend(batch.get(SOURCE_KEY, []))
        self._log_losses("valid", total, losses)
        return total

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Run one test step, keeping sample provenance for the artifact dump.

        Args:
            batch (dict): A collated batch.
            batch_idx (int): Batch index within the epoch.

        Returns:
            torch.Tensor: The loss.
        """
        # Already scored by a previous, interrupted pass: its predictions were
        # restored into the accumulator, so re-running the model over these
        # batches would double-count every position.
        if batch_idx < self.skip_test_batches:
            return torch.zeros((), device=self.device)

        total, losses = self._shared_step(batch, self.test_metrics)
        self.test_sample_ids.extend(batch.get(SAMPLE_KEY, []))
        self.test_datasets.extend(batch.get(SOURCE_KEY, []))
        self._log_losses("test", total, losses)
        return total

    def predict_step(self, batch: dict[str, Any], batch_idx: int) -> dict[str, Any]:  # noqa: ARG002
        """Predict one batch, keeping sample ids alongside the probabilities.

        Args:
            batch (dict): A collated batch.
            batch_idx (int): Batch index.

        Returns:
            dict: ``key``, ``dataset``, and a probability per target.
        """
        logits = self(batch)
        return {
            SAMPLE_KEY: batch.get(SAMPLE_KEY, []),
            SOURCE_KEY: batch.get(SOURCE_KEY, []),
            **{name: torch.sigmoid(value) for name, value in logits.items()},
        }

    def _log_epoch(self, split: str, metrics: MultiTargetMetrics) -> None:
        """Log and reset one split's epoch metrics.

        Args:
            split (str): ``"train"``, ``"valid"``, or ``"test"``.
            metrics (MultiTargetMetrics): The accumulator.
        """
        for name, value in metrics.compute().items():
            self.log(
                f"{split}/{name}",
                value,
                prog_bar=name == PRIMARY_METRIC and split == "valid",
            )
        metrics.reset()

    def on_train_epoch_end(self) -> None:
        """Log the training epoch's metrics."""
        self._log_epoch("train", self.train_metrics)

    def on_validation_epoch_start(self) -> None:
        """Clear the per-sample ids before a validation epoch.

        Cleared at the start rather than the end so a callback reading them
        after the epoch still sees the epoch that just ran.
        """
        self.valid_sample_ids = []
        self.valid_datasets = []

    def on_validation_epoch_end(self) -> None:
        """Log the validation epoch's metrics."""
        self._log_epoch("valid", self.valid_metrics)

    def release_validation_state(self) -> None:
        """Free everything the validation epoch accumulated.

        Validation state is deliberately kept alive past its own epoch so the
        callbacks that fit the event operating point can read it (see
        :meth:`on_validation_epoch_start`). In a fit run the next epoch clears
        it; in an **eval-only** run nothing does, so the whole validation split
        stays resident through the test pass -- measured at 4.75 GB of
        accelerator memory already held at test batch 0.

        Called between :meth:`~lightning.pytorch.Trainer.validate` and
        :meth:`~lightning.pytorch.Trainer.test` once the threshold has been
        fitted, so it frees the state without changing any reported number.
        """
        self.valid_sample_ids = []
        self.valid_datasets = []
        self.valid_metrics.reset()
        # The accelerator caches freed blocks rather than returning them, so the
        # reset above is not visible as reclaimed memory until the cache drops.
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_test_epoch_start(self) -> None:
        """Clear per-sample and metric state before a test epoch.

        Resetting here rather than at epoch end keeps the accumulated
        predictions readable by the artifact-writing callbacks, whatever order
        they run in, while still keeping repeated ``trainer.test()`` calls from
        accumulating on top of each other.
        """
        self.test_sample_ids = []
        self.test_datasets = []
        self.test_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log the test epoch's metrics.

        The accumulator is deliberately left populated: the prediction-writing
        and plotting callbacks read it after this hook.
        """
        for name, value in self.test_metrics.compute().items():
            self.log(f"test/{name}", value)

    @property
    def lr(self) -> float:
        """The optimizer's peak learning rate.

        A thin proxy onto ``train_config.optimizer.lr`` under the flat name
        Lightning's LR finder requires: it looks for a top-level ``lr`` or
        ``learning_rate`` attribute to read, and writes the swept value back
        through it. See :mod:`~blinklinmult.train.lr_find`.

        Nothing in the normal training path reads this --
        :meth:`configure_optimizers` reads ``train_config.optimizer.lr``
        directly, so the proxy cannot drift from what actually trains.

        Returns:
            float: The peak learning rate for the transformer and head group.
            ``backbone_lr`` is a separate, deliberately lower rate and is not
            touched by the finder.
        """
        return self.train_config.optimizer.lr

    @lr.setter
    def lr(self, value: float) -> None:
        # Every config in this project is a frozen dataclass, so `replace`
        # rebuilds the pair with just this field changed rather than mutating.
        new_optimizer = replace(self.train_config.optimizer, lr=value)
        self.train_config = replace(self.train_config, optimizer=new_optimizer)

    def configure_optimizers(self) -> dict[str, Any]:  # ty: ignore[invalid-method-override]
        """Build the optimizer and LR schedule.

        Returns:
            dict: Lightning's optimizer configuration.
        """
        optimizer = build_optimizer(self.train_config.optimizer, self.model)
        scheduler = build_scheduler(
            self.train_config.optimizer,
            optimizer,
            steps_per_epoch=self.steps_per_epoch,
            max_epochs=self.train_config.max_epochs,
        )
        if scheduler is None:
            return {"optimizer": optimizer}
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
