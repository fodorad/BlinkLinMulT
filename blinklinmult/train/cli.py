r"""Entry point for blink training runs.

Example:
    Train the joint model over every corpus::

        make train-joint

    Or directly, for control over the individual configs::

        uv run python -m blinklinmult.train.cli \
            --data config/data/all.yaml \
            --model config/model/blinklinmult.yaml \
            --train config/train/joint.yaml

    Override any config value without editing a file. The mixing temperature is
    a dataloader knob, so this needs no rebuild::

        make train-joint ARGS="--set data.strategy_kwargs.temperature=3.0"
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from blinklinmult import PROJECT_ROOT
from blinklinmult.data.datamodule import BlinkDataModule
from blinklinmult.data.embeddings import CacheKey, encoder_digest
from blinklinmult.data.schema import BLINK_PRESENCE, EYE_STATE
from blinklinmult.train.callbacks import (
    TUNING_CRITERION,
    AcceleratorCacheLimiter,
    EpochPropagator,
    EventReport,
    PerDatasetReport,
    PlotCallback,
    PredictionWriter,
    TestCheckpointer,
    TestStateReleaser,
    TimeTrackingCallback,
)
from blinklinmult.train.config import ConfigError, ExperimentConfig, parse_override
from blinklinmult.train.metrics import PRIMARY_METRIC

if TYPE_CHECKING:
    from blinklinmult.data.schema import DatasetSpec
from blinklinmult.train.mlflow_utils import (
    build_logger,
    environment_params,
    log_artifacts,
    remember_run_id,
    stored_run_id,
    truncate_params,
)
from blinklinmult.train.module import BlinkLightningModule

logger = logging.getLogger(__name__)
"""Module-level logger."""

MONITOR = f"valid/{PRIMARY_METRIC}"
"""Metric that selects the best checkpoint: mean F1 across supervised targets."""

DEFAULT_DATASET_CONFIG_DIR = PROJECT_ROOT / "config" / "data"
"""Directory of per-corpus declarations, resolved by name from the data config."""


def _safe_name(name: str) -> str:
    """Reduce a name to filename-safe characters.

    Args:
        name (str): An MLflow experiment or run name.

    Returns:
        str: The name with every other character folded to ``_``, so a name
        written for MLflow cannot nest directories or escape the results tree.
    """
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")


def run_output_dir(config: ExperimentConfig) -> Path:
    """Directory this run writes its outputs into.

    ``<output_dir>/<experiment>/<run_name>``, falling back to the experiment
    directory when the run is unnamed.

    **Grouped by experiment, not by task**, so that a sweep's runs sit together
    on disk exactly as they do in MLflow -- one directory per
    ``mlflow.experiment_name``, matching the experiment the same runs are logged
    under. Grouping by task instead put every eye-state run ever trained into
    one directory, which is how a backbone sweep's checkpoints ended up as
    ``best-v25.ckpt`` .. ``best-v28.ckpt`` with nothing to say which backbone
    produced which.

    **Named runs get their own subdirectory** because every run writes the same
    filenames -- ``metrics_test.json``, ``test_events.json``,
    ``test_predictions_<target>.csv``, ``checkpoints/best.ckpt`` -- so a sweep
    sharing one directory leaves only the last run's results, which reads as a
    finished sweep rather than as an error.

    Args:
        config (ExperimentConfig): The run's configuration.

    Returns:
        Path: Directory for this run's outputs.
    """
    group = _safe_name(config.train.mlflow.experiment_name) or config.train.task
    base = Path(config.train.output_dir) / group
    run_name = config.train.mlflow.run_name
    if not run_name:
        return base
    safe = _safe_name(run_name)
    return base / safe if safe else base


DEFAULT_REPORT_FPS = 30.0
"""Frame rate assumed when no corpus in the run declares one."""


def report_fps(specs: list[DatasetSpec] | None) -> float:
    """Frame rate for turning frame counts into minutes.

    Args:
        specs (list[DatasetSpec] | None): The run's corpora.

    Returns:
        float: The highest rate any of them declares, or
        :data:`DEFAULT_REPORT_FPS` when none does.
    """
    rates = [spec.fps for spec in (specs or []) if spec.fps]
    return max(rates) if rates else DEFAULT_REPORT_FPS


def build_callbacks(
    config: ExperimentConfig,
    output_dir: Path,
    specs: list[DatasetSpec] | None = None,
    eval_only: bool = False,
    resumable: bool = False,
    shared_threshold_dir: Path | None = None,
) -> list[L.Callback]:
    """Assemble a run's callbacks.

    Args:
        config (ExperimentConfig): The run's configuration.
        output_dir (Path): Directory for run outputs.
        specs (list[DatasetSpec] | None): The run's resolved corpora, for the
            frame rate the event report needs.
        shared_threshold_dir (Path | None): Read and write the event threshold
            cache here rather than in the run's own directory, so a per-corpus
            evaluation shares one universal operating point.
        resumable (bool): Periodically checkpoint the test pass so a killed run
            can resume it. Off by default: the state file is as large as the
            accumulated predictions, and only a very long pass justifies it.
        eval_only (bool): Keep only the callbacks that score and dump the test
            split. Training-only callbacks -- checkpoints, early stopping, the
            LR monitor, epoch propagation, and timing -- are omitted so that an
            ``--eval-only`` run re-fits the event threshold and writes the
            report artifacts without writing a checkpoint or recording timings.

    Returns:
        list[L.Callback]: The callbacks.
    """
    callbacks: list[L.Callback] = []

    if not eval_only:
        checkpoint_dir = output_dir / "checkpoints"

        # Keyed by purpose, not position: looking a checkpoint up by the metric
        # it monitors means adding or reordering callbacks cannot silently load
        # the wrong model.
        checkpoints = {
            "best": ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best",
                monitor=MONITOR,
                mode="max",
                save_top_k=1,
                # Full state, not just weights: a run must be resumable.
                save_weights_only=False,
                save_last=True,
            ),
            "best_loss": ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best_loss",
                monitor="valid/loss",
                mode="min",
                save_top_k=1,
                save_weights_only=False,
            ),
        }

        callbacks.extend(
            [
                *checkpoints.values(),
                LearningRateMonitor(logging_interval="step"),
                EpochPropagator(),
                TimeTrackingCallback(output_dir),
            ]
        )

    callbacks.extend(
        [
            # First: it bounds the accelerator's cached-block pool across every
            # stage. The pool grows even though nothing is retained -- measured
            # at 1118 MB of cache against 0.0 MB of live tensors -- and on the
            # 34 906-batch benchmark pass it reached the 42.4 GiB watermark and
            # failed a routine 45 MiB allocation.
            #
            # **The interval is not scaled down for sequence models**, even
            # though their batches are ~57x more expensive. Measured: the pool
            # jumps from 5.3 GB to 20.4 GB on the *first* batch and is flat
            # thereafter, so it is the working set of one forward+backward
            # rather than an accumulation -- dropping more often cannot lower a
            # peak that is reached immediately, and `empty_cache` costs 3.97 s
            # against a 5.8 s step. Batch size is the lever for that peak; see
            # `config/data/video_all.yaml`.
            AcceleratorCacheLimiter(),
            TestCheckpointer(output_dir, enabled=resumable),
            PredictionWriter(output_dir),
            PerDatasetReport(output_dir),
            # False alarms are reported per minute, so the report needs the rate
            # that turns frame counts into a duration. Corpora in a joint run
            # may differ; the fastest is used, which under-reports the rate for
            # slower ones rather than inventing a faster timeline than any of
            # them had. A frame-wise model predicts *closure* and is benchmarked
            # on blink *events*, so it scores `blink_presence` from
            # `eval_targets` rather than a trained head. Its criterion is `any`,
            # not `iou50`: a closure run covers only 19-34% of an annotated
            # event, so even a *perfect* closure detector scores 0.03 at iou50
            # (measured on TalkingFace, median IoU 0.333) against 0.98 at `any`.
            # Fitting the operating point on iou50 would maximise noise.
            EventReport(
                output_dir,
                fps=report_fps(specs),
                target=BLINK_PRESENCE,
                tuning_criterion="any" if config.model.family == "cnn" else TUNING_CRITERION,
                shared_threshold_dir=shared_threshold_dir,
                carrier_only=config.train.event_carrier_only,
            ),
            # With an event head, `blink_presence` is the *learned* route to an
            # interval and the hysteresis extractor no longer appears in the
            # report at all. Scoring the ESR signal as well keeps both visible:
            # the hand-fitted extractor against the learned one, same
            # predictions, same protocol, same test set. Without an event head
            # the two targets are one tensor and a second report would be a
            # duplicate, so it is only added when they differ.
            *(
                [
                    EventReport(
                        output_dir,
                        fps=report_fps(specs),
                        target=EYE_STATE,
                        tuning_criterion=TUNING_CRITERION,
                        shared_threshold_dir=shared_threshold_dir,
                        carrier_only=config.train.event_carrier_only,
                    )
                ]
                if config.model.event_head is not None and EYE_STATE in config.train.targets
                else []
            ),
            PlotCallback(output_dir),
            # Last: it frees the accumulated test predictions, and Lightning
            # runs a hook in callback order, so every consumer above has read
            # them by the time this fires.
            TestStateReleaser(),
        ]
    )

    if not eval_only and config.train.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=config.train.early_stopping.monitor,
                mode=config.train.early_stopping.mode,
                patience=config.train.early_stopping.patience,
                min_delta=config.train.early_stopping.min_delta,
            )
        )

    return callbacks


def _best_checkpoint(trainer: L.Trainer) -> str:
    """Return the path of the checkpoint selected by the primary metric.

    Args:
        trainer (L.Trainer): The finished trainer.

    Returns:
        str: The checkpoint path, or ``""`` if there is none.
    """
    for callback in trainer.checkpoint_callbacks:
        if isinstance(callback, ModelCheckpoint) and callback.monitor == MONITOR:
            return callback.best_model_path
    return ""


MAX_EPOCHS_FILE = "max_epochs.txt"
"""Name of the file recording a run's planned ``max_epochs``.

Cosine annealing is a fixed-length curve built at first launch for that total.
Resuming with a different ``max_epochs`` would silently change the LR plan
mid-run -- the schedule would anneal to the wrong floor, and the only trace
would be a loss curve that looks slightly off. Recorded here so the mismatch is
rejected instead.
"""


def check_max_epochs(output_dir: Path, max_epochs: int, resume: bool) -> None:
    """Reject a resume that changes the length of the LR schedule.

    Args:
        output_dir (Path): The run's output directory.
        max_epochs (int): What the config asks for now.
        resume (bool): Whether this launch is a resume.

    Raises:
        ValueError: If a resumed run's ``max_epochs`` differs from the original.
    """
    path = output_dir / MAX_EPOCHS_FILE
    if resume and path.is_file():
        planned = int(path.read_text().strip())
        if planned != max_epochs:
            raise ValueError(
                f"--resume must keep the original max_epochs={planned}, but the config "
                f"now says {max_epochs}. Cosine annealing is a fixed-length curve set "
                "at first launch, so resuming with a different total would change the "
                "LR plan mid-run. Set max_epochs high enough before the first run; "
                "early stopping ends it sooner when the model plateaus."
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(str(max_epochs))


def resume_from(resume: Path | bool | None, output_dir: Path) -> str | None:
    """The checkpoint a run should continue from.

    Args:
        resume (Path | bool | None): ``True`` to continue this run's own
            ``last.ckpt``, a path to continue a specific checkpoint, or
            ``None``/``False`` to start fresh.
        output_dir (Path): The run's output directory.

    Returns:
        str | None: The checkpoint path, or ``None`` to start at epoch 0.

    Raises:
        FileNotFoundError: If a resume was asked for and no checkpoint exists.
            Silently starting from scratch would waste the hours the run was
            meant to preserve, and the loss curve is the only place it would
            show.
    """
    if not resume:
        return None

    path = output_dir / "checkpoints" / "last.ckpt" if resume is True else Path(resume)
    if not path.is_file():
        raise FileNotFoundError(
            f"--resume was given but no checkpoint at {path}. Start a fresh run "
            "(drop --resume), or point it at the right run's output directory."
        )
    logger.info(f"Resuming from {path}.")
    return str(path)


def _clear_test_checkpoint(callbacks: list[L.Callback]) -> None:
    """Delete any partial test state once the pass has finished.

    Args:
        callbacks (list[L.Callback]): The run's callbacks.
    """
    for callback in callbacks:
        if isinstance(callback, TestCheckpointer):
            callback.clear()


def _reuse_cached_threshold(callbacks: list[L.Callback]) -> bool:
    """Restore every event report's operating point from its cache.

    Args:
        callbacks (list[L.Callback]): The run's callbacks.

    Returns:
        bool: Whether every :class:`~blinklinmult.train.callbacks.EventReport`
        found a usable cache. ``False`` when any did not, in which case the
        caller must run validation -- reporting one corpus at a fitted point and
        another at the default would make the two incomparable.
    """
    reports = [callback for callback in callbacks if isinstance(callback, EventReport)]
    if not reports:
        return False

    restored = [report.load_cached_threshold() is not None for report in reports]
    if all(restored):
        return True

    logger.info("No usable cached event threshold; fitting it on validation.")
    return False


def unfreeze_module(
    checkpoint: Path,
    config: ExperimentConfig,
    image_size: int,
    eye_feature_dim: int | None,
    steps_per_epoch: int,
) -> BlinkLightningModule:
    """Load a frozen-stage run and continue it with the encoder trainable.

    The two-stage recipe: train the transformer against a fixed encoder, which
    is 7.7x cheaper and cannot damage the pretrained features, then refine
    everything jointly at a much lower rate.

    **Neither existing path does this.** ``model.encoder_weights`` loads *only*
    the encoder, discarding the transformer that stage 1 spent its whole run
    training. ``--resume`` keeps the whole model but also restores the stored
    ``encoder_freeze``, so it continues stage 1 rather than starting stage 2.

    A **fresh optimiser** is deliberate. Stage 1's Adam moments describe a
    parameter set that excluded the encoder, and its schedule is mid-anneal for
    a different problem; carrying either forward would apply stage 1's momentum
    to weights it never saw.

    Args:
        checkpoint (Path): Stage 1's checkpoint, usually its ``best.ckpt``.
        config (ExperimentConfig): The stage 2 configuration, supplying the low
            learning rates.
        image_size (int): Crop size, from the datasets.
        eye_feature_dim (int | None): Feature width, from the datasets.
        steps_per_epoch (int): Optimizer steps per epoch.

    Returns:
        BlinkLightningModule: The stage 1 model, fully trainable.

    Raises:
        FileNotFoundError: If the checkpoint does not exist. Silently starting
            from scratch would discard the stage it was meant to continue, and
            only the loss curve would show it.
    """
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"--unfreeze-from was given but no checkpoint at {checkpoint}. Run the "
            "frozen stage first, or point it at that run's checkpoints/best.ckpt."
        )

    module = BlinkLightningModule.load_from_checkpoint(
        checkpoint,
        train_config=config.train,
        image_size=image_size,
        eye_feature_dim=eye_feature_dim,
        steps_per_epoch=steps_per_epoch,
    )

    encoder = module.model.encoder
    encoder.frozen = False
    for parameter in encoder.parameters():
        parameter.requires_grad_(True)
    # Back into train mode: `EyeEncoder.train` holds a frozen encoder in eval so
    # its BatchNorm statistics stop drifting, and that must now be undone.
    encoder.train()

    trainable = sum(p.numel() for p in module.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.model.parameters())
    logger.info(
        f"Unfroze the encoder from {checkpoint}: {trainable:,} of {total:,} parameters "
        f"now trainable, at lr={config.train.optimizer.lr:g} and "
        f"backbone_lr={config.train.optimizer.backbone_lr}."
    )
    return module


def embedding_key_for(config: ExperimentConfig) -> CacheKey:
    """Identity of the embedding cache this run may read.

    Built from a throwaway encoder loaded from the run's own
    ``model.encoder_weights``, because the cache is keyed by the weights that
    produced it and the datamodule is constructed before the model. Loading the
    encoder twice costs a few seconds once per run and removes any chance of the
    key describing different weights than the run actually uses.

    Args:
        config (ExperimentConfig): The resolved run configuration.

    Returns:
        CacheKey: What a cache shard must match to be read.
    """
    from blinklinmult.train.model import EyeEncoder

    image_size = config.data.image_size
    if image_size is None:
        raise ConfigError(
            "data.embedding_cache needs data.image_size set explicitly. The cache is "
            "keyed by the crop size the encoder was fed, so it cannot be left to the "
            "corpus to decide."
        )
    encoder = EyeEncoder(config.model, image_size)
    return CacheKey(
        encoder_digest=encoder_digest(encoder),
        backbone=config.model.backbone,
        output_dim=config.model.backbone_output_dim,
        image_size=image_size,
    )


def run(
    config: ExperimentConfig,
    root: Path = PROJECT_ROOT,
    config_dir: Path | None = None,
    fast_dev_run: bool = False,
    resume: Path | bool | None = None,
    eval_only: Path | None = None,
    unfreeze_from: Path | None = None,
    reuse_threshold: bool = False,
    resumable: bool = False,
    shared_threshold_dir: Path | None = None,
) -> dict[str, Any]:
    """Train, then test, one configured experiment.

    Args:
        config (ExperimentConfig): The run's configuration.
        root (Path): Repository root that data paths resolve against.
        config_dir (Path | None): Directory of per-corpus declarations.
            Defaults to ``config/data``.
        fast_dev_run (bool): Run a single batch through every stage, for
            smoke-testing.
        resume (Path | bool | None): ``True`` continues this run from its own
            ``last.ckpt`` and appends to its MLflow run; a path continues a
            specific checkpoint; ``None`` starts at epoch 0.
        unfreeze_from (Path | None): Stage-1 checkpoint to continue with the
            encoder trainable, or ``None`` for a single-stage run. See
            :func:`unfreeze_module` for why neither ``model.encoder_weights``
            nor ``--resume`` does this.
        eval_only (Path | None): Checkpoint to score on the test split without
            training. The module is loaded from this file, the event operating
            point is fitted on the validation split, and the test split is then
            scored -- the same passes a finished run performs, so its artifacts
            are directly comparable to one. Use it to recover the benchmark of a
            run that was interrupted before its test pass.
        reuse_threshold (bool): Skip the validation pass when a cached event
            threshold from a previous run is available. Fitting the operating
            point is the only thing validation contributes to an eval-only run,
            and it costs a full pass over the split, so a retry after an
            interrupted test pass can reuse the cached value instead. Falls
            back to fitting when no usable cache exists.
        resumable (bool): Checkpoint the test pass periodically so a killed run
            can pick it up. The partial state is deleted once the pass
            completes, so it costs nothing on disk afterwards.
        shared_threshold_dir (Path | None): Share the fitted event threshold
            through this directory. A per-corpus evaluation points every run at
            one location, so the corpora that can fit a threshold write it and
            the rest -- including those with no validation split -- read the
            same value instead of falling back to a fixed default.

    Returns:
        dict[str, float]: The test metrics.
    """
    L.seed_everything(config.train.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    specs = config.data.resolve_specs(config_dir or DEFAULT_DATASET_CONFIG_DIR)

    output_dir = run_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run outputs: {output_dir}")

    datamodule = BlinkDataModule(config.data, specs, root=root)

    # The cache is keyed by the encoder's weights, and the datamodule is built
    # before the model -- so the key comes from a throwaway encoder loaded from
    # the same checkpoint the run will use. Config has already refused this
    # combination unless the encoder is frozen and augmentation is off, which is
    # what makes the embeddings constant enough to cache at all.
    if config.data.embedding_cache is not None:
        datamodule.embedding_key = embedding_key_for(config)

    if eval_only is not None:
        # The checkpoint reconstructs the model from its own stored configs, so
        # the run's data config is the only file-level input needed. Validation
        # is fitted before test so the event threshold matches the loaded
        # weights rather than the fallback 0.5.
        datamodule.setup("validate")
        datamodule.setup("test")
        module = BlinkLightningModule.load_from_checkpoint(eval_only)
    else:
        datamodule.setup("fit")
        steps_per_epoch = max(1, len(datamodule.train_dataloader()))
        if unfreeze_from is not None:
            # Stage 2 of the two-stage recipe: continue a frozen run with the
            # encoder trainable. The whole model comes from that checkpoint, not
            # just its encoder.
            module = unfreeze_module(
                unfreeze_from,
                config,
                image_size=datamodule.image_size,
                eye_feature_dim=datamodule.feature_dim,
                steps_per_epoch=steps_per_epoch,
            )
        else:
            module = BlinkLightningModule(
                model_config=config.model,
                train_config=config.train,
                # Both come from the resolved dataset shape rather than the
                # config, so the model can never disagree with the data it is fed.
                image_size=datamodule.image_size,
                eye_feature_dim=datamodule.feature_dim,
                steps_per_epoch=steps_per_epoch,
            )

    check_max_epochs(output_dir, config.train.max_epochs, bool(resume))

    # A resumed run appends to the MLflow run it started, so the metric curve
    # reads as one training history rather than two disjoint halves.
    mlflow_logger = build_logger(config, run_id=stored_run_id(output_dir) if resume else None)
    remember_run_id(output_dir, mlflow_logger)
    # Logged from the config rather than the module's hyperparameters: the
    # module's copy exists to make checkpoints self-contained, and letting
    # Lightning log it too would duplicate every param under a second name.
    module._log_hyperparams = False
    mlflow_logger.log_hyperparams(
        truncate_params({**config.to_flat_dict(), **environment_params(config, specs, root)})
    )

    callbacks = build_callbacks(
        config,
        output_dir,
        specs,
        eval_only=eval_only is not None,
        resumable=resumable,
        shared_threshold_dir=shared_threshold_dir,
    )

    trainer = L.Trainer(
        max_epochs=config.train.max_epochs,
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        # Validated against PRECISIONS at config load; Lightning types this as a
        # Literal union that a plain str cannot satisfy statically.
        precision=cast("Any", config.train.precision),
        deterministic=config.train.deterministic,
        gradient_clip_val=config.train.optimizer.gradient_clip_val,
        default_root_dir=output_dir,
        logger=mlflow_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        fast_dev_run=fast_dev_run,
        # A real run over a few batches, unlike `fast_dev_run` -- which swaps in
        # a DummyLogger, so nothing reaches MLflow and the metrics cannot be
        # inspected at all. `--limit-batches 10` keeps the logger, the
        # callbacks, and the checkpointing, so every metric a full run produces
        # appears in seconds rather than hours.
        # `limit_fit_batches` narrows train/valid only; `limit_batches` narrows
        # every stage. Test deliberately ignores the former: a diagnostic sweep
        # must score every arm on the same full split or the arms cannot be
        # compared, which is the only thing a sweep is for.
        limit_train_batches=config.train.limit_fit_batches or config.train.limit_batches or 1.0,
        limit_val_batches=config.train.limit_fit_batches or config.train.limit_batches or 1.0,
        limit_test_batches=config.train.limit_test_batches or config.train.limit_batches or 1.0,
    )

    if eval_only is not None:
        # Validation exists here only to fit the event operating point, and that
        # pass costs ~21 minutes on the frame-wise benchmark to produce a single
        # number. When a previous run already cached it, skip straight to the
        # test split -- which is what makes retrying an interrupted test pass
        # cheap.
        cached = _reuse_cached_threshold(callbacks) if reuse_threshold else False

        if not cached:
            # Validation state must be released afterwards: `valid_sample_ids`
            # is cleared only in `on_validation_epoch_start`, which never fires
            # again in an eval-only run, so without this the whole validation
            # split stays live through testing. Measured on the frame-wise run,
            # the test loop began with 4.75 GB of accelerator memory resident.
            trainer.validate(module, datamodule=datamodule)
            module.release_validation_state()

        results = trainer.test(module, datamodule=datamodule)
        # The pass completed, so the partial state is now dead weight -- it is
        # as large as the predictions it holds, and the finished artifacts
        # supersede it.
        _clear_test_checkpoint(callbacks)
    else:
        trainer.fit(module, datamodule=datamodule, ckpt_path=resume_from(resume, output_dir))

        if fast_dev_run:
            trainer.test(module, datamodule=datamodule)
            return {}

        best_path = _best_checkpoint(trainer)
        logger.info(f"Testing best checkpoint: {best_path or '<none; using final weights>'}")
        results = trainer.test(module, datamodule=datamodule, ckpt_path=best_path or None)

    metrics: dict[str, Any] = dict(results[0]) if results else {}

    (output_dir / "metrics_test.json").write_text(json.dumps(metrics, indent=2))
    log_artifacts(mlflow_logger, output_dir)

    if config.train.mlflow.log_model and eval_only is None and best_path:
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id, best_path, artifact_path="model"
        )

    logger.info(f"test/{PRIMARY_METRIC} = {metrics.get(f'test/{PRIMARY_METRIC}')}")
    return metrics


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="Data config YAML.")
    parser.add_argument("--model", type=Path, required=True, help="Model config YAML.")
    parser.add_argument("--train", type=Path, required=True, help="Train config YAML.")
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=DEFAULT_DATASET_CONFIG_DIR,
        help="Directory of per-corpus dataset declarations.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=PROJECT_ROOT,
        help="Repository root that data paths resolve against.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set data.batch_size=8.",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=None,
        help=(
            "Continue a stopped run. Bare --resume picks up this run's own "
            "last.ckpt and appends to its MLflow run; pass a path to continue a "
            "specific checkpoint."
        ),
    )
    parser.add_argument(
        "--shared-threshold-dir",
        type=Path,
        default=None,
        help=(
            "Read and write the fitted event threshold here instead of in the "
            "run directory, so a per-corpus evaluation shares one universal "
            "operating point."
        ),
    )
    parser.add_argument(
        "--resumable",
        action="store_true",
        help=(
            "Checkpoint the test pass periodically so a killed run can resume "
            "it. The partial state is deleted once the pass completes."
        ),
    )
    parser.add_argument(
        "--reuse-threshold",
        action="store_true",
        help=(
            "With --eval-only, skip the validation pass when a previous run "
            "cached the fitted event threshold. Validation only produces that "
            "one number, so a retried test pass need not repeat it."
        ),
    )
    parser.add_argument(
        "--unfreeze-from",
        type=Path,
        default=None,
        help=(
            "Continue a frozen-encoder run with the encoder trainable, starting "
            "from that run's checkpoint. Loads the *whole* model -- encoder, "
            "transformer and heads -- unlike model.encoder_weights, which keeps "
            "only the encoder, and starts a fresh optimiser, unlike --resume, "
            "which would also restore the frozen state. Pair it with a low "
            "learning rate: config/train/video_unfreeze.yaml."
        ),
    )
    parser.add_argument(
        "--eval-only",
        type=Path,
        default=None,
        help=(
            "Score this checkpoint on the test split without training. The "
            "event operating point is fitted on validation first, so the "
            "artifacts match a finished run. Use it to recover the benchmark "
            "of a run that was interrupted before its test pass."
        ),
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Run one batch through every stage and exit. Disables logging.",
    )
    parser.add_argument(
        "--limit-batches",
        type=float,
        default=None,
        help=(
            "Batches per epoch in every stage: an integer count, or a fraction "
            "in (0, 1]. Unlike --fast-dev-run this keeps logging and "
            "checkpointing, so the metrics are written and reach MLflow."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    overrides = dict(parse_override(text) for text in args.overrides)
    if args.limit_batches is not None:
        # argparse parses it as a float; Lightning reads an int as a batch count
        # and a float as a fraction, so `--limit-batches 10` has to arrive as
        # the integer 10 rather than 10.0, which would be an invalid fraction.
        limit = args.limit_batches
        overrides["train.limit_batches"] = int(limit) if limit >= 1 else limit
    config = ExperimentConfig.from_files(args.data, args.model, args.train, overrides)
    run(
        config,
        root=args.root,
        config_dir=args.config_dir,
        fast_dev_run=args.fast_dev_run,
        resume=args.resume,
        eval_only=args.eval_only,
        unfreeze_from=args.unfreeze_from,
        reuse_threshold=args.reuse_threshold,
        resumable=args.resumable,
        shared_threshold_dir=args.shared_threshold_dir,
    )


if __name__ == "__main__":
    main()
