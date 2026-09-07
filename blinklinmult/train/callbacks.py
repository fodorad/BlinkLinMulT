"""Callbacks for blink training runs.

MLflow is the metric store, so what remains here is only what MLflow does not do
on its own: epoch timing, per-sample prediction dumps, per-corpus score
breakdowns, and the diagnostic plots that make a run readable at a glance.

The per-corpus breakdown matters more here than in a single-dataset project. A
joint run mixes six corpora of wildly different size and difficulty, and one
aggregate F1 can hide a model that has learned MRL-Eye's stills and fails on
EyeBlink8's video entirely. :class:`PerDatasetReport` is what makes that
visible.
"""

from __future__ import annotations

import json
import logging
import pickle
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as L
import matplotlib
import numpy as np
import torch

from blinklinmult.data.schema import BLINK_PRESENCE, SchemaError, parse_sample_id
from blinklinmult.train.events import (
    CRITERIA,
    average_overlapping,
    average_precision,
    blink_ap,
    blink_ap_summary,
    event_metrics,
    froc,
    pool_curves,
    to_intervals,
)
from blinklinmult.train.metrics import NO_GROUP, binary_metrics

if TYPE_CHECKING:
    from collections.abc import Sequence

    from blinklinmult.train.module import BlinkLightningModule

matplotlib.use("Agg")  # No display on a training host.
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)
"""Module-level logger."""

SIGNALS_FILENAME = "test_signals.npz"
"""Per-recording reassembled timelines, saved beside the event report.

The report keeps only aggregates, so without this the per-frame signal every
event score was derived from is discarded and can only be recovered by
re-running inference over the whole split.
"""

SIGNALS_TARGET_KEY = "_target"
"""Archive key naming the target the ``truth`` arrays hold.

Without it the archive is ambiguous, and the ambiguity is dangerous rather than
merely untidy: on the frame-wise benchmark the event report scores
``blink_presence``, so ``truth`` is a blink *interval* -- 3-4x wider than
closure, and present for corpora that annotate no closure at all. Read as
``eye_state`` it manufactures phantom false positives without erroring.
"""

TEST_STATE_FILENAME = "test_state.pt"
"""Partial test-pass state, so a killed benchmark run can resume.

See :class:`TestCheckpointer` for why resuming is exact rather than approximate.
"""

THRESHOLD_FILENAME = "event_threshold.json"
"""Cache file for the operating point fitted on validation.

Fitting it costs a full validation pass but produces a single number, so it is
persisted: an interrupted or retried test pass can reuse it instead of spending
the pass again. See :meth:`EventReport.load_cached_threshold`.
"""

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
    0.075,
)
"""Hysteresis ratios searched when fitting the operating point.

The low threshold is ``high * ratio``, so the pair stays ordered across the
whole sweep; a constant low value would exceed the high one at the bottom and
silently disable hysteresis.

``None`` is first so the single-threshold behaviour is always in the search and
wins ties -- hysteresis has to *earn* its place on validation rather than being
assumed.

**The range is set by measurement, not by the literature.** The clinical pair
(onset 0.25 against complete 0.75, a ratio of 1/3) is in the grid but is not
where this model lands. Swept over ``fw-focal``'s 35 RN test recordings at
``iou50``:

===========  ======  ==========  ======
ratio        high    F1          prec
===========  ======  ==========  ======
single       0.30    0.1972      0.152
1/3          0.54    0.3397      0.401
0.2          0.55    0.4095      0.500
**0.15**     0.58    **0.4226**  0.547
0.125        0.65    0.4091      0.572
0.1          0.65    0.3703      0.519
0.075        0.77    0.3344      0.548
===========  ======  ==========  ======

A clean interior optimum at 0.15 -- F1 falls away on both sides -- so the grid
brackets it rather than ending on it. An earlier grid stopping at 0.2 selected
its own edge, which is a boundary hit rather than a fitted optimum.

The gap from the single threshold is the point: **F1 0.197 to 0.423** and
**precision 0.152 to 0.547**. One cut has to be low enough for a shallow
closure onset and high enough to ignore noise, and no value does both.
"""

TUNING_CRITERION = "iou50"
"""Matching criterion the operating point is fitted against.

``iou50`` rather than ``any``: it is the criterion MPEblink's Blink-AP uses, so
the threshold is tuned for the same notion of "detected" the headline number
reports. Tuning on ``any`` would favour a threshold that fires early and often,
since a single overlapping frame counts as a hit there.
"""


class TimeTrackingCallback(L.Callback):
    """Records wall-clock time per epoch and for the whole run.

    Args:
        output_dir (Path): Directory to write ``time.json`` into.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.epoch_times: list[dict[str, float]] = []
        self._run_start = 0.0
        self._epoch_start = 0.0

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # noqa: ARG002
        """Start the run timer.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
        """
        self._run_start = time.perf_counter()

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # noqa: ARG002
        """Start the epoch timer.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
        """
        self._epoch_start = time.perf_counter()

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Record the epoch's duration.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
        """
        elapsed = time.perf_counter() - self._epoch_start
        self.epoch_times.append({"epoch": trainer.current_epoch, "seconds": elapsed})
        pl_module.log("train/epoch_seconds", elapsed)

    def on_fit_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # noqa: ARG002
        """Write the timing summary.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
        """
        total = time.perf_counter() - self._run_start
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "time.json"
        path.write_text(
            json.dumps(
                {
                    "total_seconds": total,
                    "epochs": self.epoch_times,
                    "mean_epoch_seconds": (
                        sum(e["seconds"] for e in self.epoch_times) / len(self.epoch_times)
                        if self.epoch_times
                        else 0.0
                    ),
                },
                indent=2,
            )
        )
        logger.info(f"Run took {total:.1f}s; timings written to {path}")


class EpochPropagator(L.Callback):
    """Advances the datamodule's epoch so mixing and augmentation move on.

    OmniLoader's mixing sampler and its seedable augmentations are both
    epoch-aware: without this the same mixed draw and the same augmentation
    parameters repeat every epoch, which silently reduces an N-epoch run to one
    epoch of data seen N times.
    """

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:  # noqa: ARG002
        """Propagate the upcoming epoch to the datamodule.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
        """
        # `Trainer.datamodule` exists at runtime but is absent from Lightning's
        # stubs, so it is fetched dynamically rather than silenced at the use.
        datamodule = getattr(trainer, "datamodule", None)
        if datamodule is not None and hasattr(datamodule, "set_epoch"):
            datamodule.set_epoch(trainer.current_epoch)


class PredictionWriter(L.Callback):
    """Writes per-position test predictions next to their targets.

    This artifact is what lets threshold tuning, per-corpus analysis, and
    significance testing run later without re-running the model.

    Args:
        output_dir (Path): Directory to write the CSVs into.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def on_test_epoch_end(  # ty: ignore[invalid-method-override]
        self,
        trainer: L.Trainer,
        pl_module: BlinkLightningModule,  # noqa: ARG002
    ) -> None:
        """Dump each target's probabilities, targets, and masks to CSV.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (BlinkLightningModule): The module under test.
        """
        import pandas as pd

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Eval-only targets too: a frame-wise model's per-corpus blink scores
        # are exactly what the benchmark reports, and they live under a target
        # it was never trained on.
        for name in pl_module.scored_targets:
            metric = pl_module.test_metrics[name]
            probability, target, mask = metric.predictions()
            if probability.numel() == 0:
                logger.warning(f"No test predictions accumulated for {name!r}.")
                continue

            frame = pd.DataFrame(
                {
                    "probability": probability.cpu().numpy(),
                    "target": target.cpu().numpy(),
                    "valid": mask.bool().cpu().numpy(),
                }
            )
            path = self.output_dir / f"test_predictions_{name}.csv"
            frame.to_csv(path, index=False)
            logger.info(f"Wrote {len(frame)} {name} predictions to {path}")


class PerDatasetReport(L.Callback):
    """Scores the test split separately for each contributing corpus.

    Args:
        output_dir (Path): Directory to write ``test_per_dataset.json`` into.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def on_test_epoch_end(  # ty: ignore[invalid-method-override]
        self,
        trainer: L.Trainer,
        pl_module: BlinkLightningModule,  # noqa: ARG002
    ) -> None:
        """Write per-corpus metrics for every supervised target.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (BlinkLightningModule): The module under test.
        """
        datasets = pl_module.test_datasets
        if not datasets:
            logger.warning("No per-sample dataset ids recorded; skipping the breakdown.")
            return

        report: dict[str, dict[str, dict[str, float]]] = {}

        for name in pl_module.target_names:
            metric = pl_module.test_metrics[name]
            probability, target, mask = metric.predictions()
            if probability.numel() == 0:
                continue

            # The accumulated state is flat over (sample, position); each sample
            # contributes the same number of positions, so the sample a position
            # belongs to is recoverable by integer division.
            per_sample = probability.numel() // len(datasets)
            if per_sample == 0:
                continue
            source = np.repeat(np.asarray(datasets), per_sample)[: probability.numel()]

            report[name] = {}
            for corpus in sorted(set(source)):
                selector = torch.from_numpy(source == corpus)
                scores = binary_metrics(probability[selector], target[selector], mask[selector])
                valid = int(mask[selector].bool().sum())
                report[name][str(corpus)] = {
                    **{k: float(v) for k, v in scores.items()},
                    "valid_positions": valid,
                }

        if not report:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / "test_per_dataset.json"
        path.write_text(json.dumps(report, indent=2))

        for name, corpora in report.items():
            for corpus, scores in corpora.items():
                if scores["valid_positions"]:
                    logger.info(
                        f"test/{name}/{corpus}: f1={scores['f1']:.3f} "
                        f"ap={scores['average_precision']:.3f} "
                        f"(n={scores['valid_positions']})"
                    )
        logger.info(f"Per-dataset breakdown written to {path}")


class PlotCallback(L.Callback):
    """Plots a precision-recall curve per target at the end of testing.

    Args:
        output_dir (Path): Directory to write plots into.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)

    def on_test_epoch_end(  # ty: ignore[invalid-method-override]
        self,
        trainer: L.Trainer,
        pl_module: BlinkLightningModule,  # noqa: ARG002
    ) -> None:
        """Write one precision-recall curve per supervised target.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (BlinkLightningModule): The module under test.
        """
        curves = []
        for name in pl_module.target_names:
            metric = pl_module.test_metrics[name]
            probability, target, mask = metric.predictions()
            valid = mask.bool()
            if not valid.any():
                continue
            curves.append((name, probability[valid], (target[valid] >= 0.5).float()))

        if not curves:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        figure, axes = plt.subplots(1, len(curves), figsize=(4.0 * len(curves), 3.6), squeeze=False)

        for index, (name, scores, labels) in enumerate(curves):
            axis = axes[0][index]
            precision, recall = _pr_curve(scores, labels)
            axis.plot(recall, precision, linewidth=1.5)
            axis.set_title(name, fontsize=10)
            axis.set_xlabel("recall")
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.3)
            if index == 0:
                axis.set_ylabel("precision")

        figure.tight_layout()
        path = self.output_dir / "test_precision_recall.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        logger.info(f"Wrote precision-recall curves to {path}")


def _pr_curve(scores: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall at every observed operating point.

    Args:
        scores (torch.Tensor): Predicted probabilities, ``(N,)``.
        labels (torch.Tensor): Binary labels, ``(N,)``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(precision, recall)``, ordered by
        descending threshold. Both are empty when no positive is present.
    """
    positives = labels.sum()
    if positives == 0:
        return np.array([]), np.array([])

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order]

    true_positive = torch.cumsum(sorted_labels, dim=0)
    # `torch.arange` defaults to CPU; on an accelerator the division would
    # otherwise fail with a device mismatch.
    ranks = torch.arange(
        1,
        sorted_labels.numel() + 1,
        dtype=true_positive.dtype,
        device=true_positive.device,
    )

    precision = (true_positive / ranks).cpu().numpy()
    recall = (true_positive / positives).cpu().numpy()
    return precision, recall


class AcceleratorCacheLimiter(L.Callback):
    """Drops cached accelerator blocks periodically during evaluation.

    PyTorch's MPS allocator keeps freed blocks in a per-shape pool instead of
    returning them to the system, and over a long evaluation pass that pool
    grows without bound even though **nothing is retained**. Measured on a bare
    conv over 300 identical batches: ``current_allocated_memory`` stayed at
    0.0 MB throughout while ``driver_allocated_memory`` reached 1118 MB, and a
    single :func:`torch.mps.empty_cache` returned it to 11 MB.

    On the frame-wise benchmark -- 34 906 test batches over seven corpora --
    that pool reached the 42.4 GiB watermark and a routine 45 MiB convolution
    failed to allocate, roughly 43% of the way through the split.

    Dropping the cache costs a re-allocation on the next batch, so it runs
    every ``every_n_batches`` rather than every batch: often enough to bound
    the pool, rarely enough that the cost stays negligible.

    **Every 100, not 500.** At 500 the pool still reached the 42.43 GiB ceiling
    partway through the frame-wise test pass, four times in a row -- both BCE
    arms during their runs and again on a fresh re-evaluation, each dying near
    batch 15 400-15 900 of 34 819. Raising the dataloader to eight workers had
    made batches arrive about four times faster, so the pool refilled between
    drops faster than the drops cleared it. The focal arms survived the
    identical split, so the margin is thin rather than absent.

    **Training is covered too, since 2026-08-30.** The callback previously ran
    only on validation and test batches, on the assumption that a training step
    frees its own activations. It does not free them from the *allocator pool*:
    on a sequence model, one batch pushes 32 windows x 45 frames = 1 440 crops
    through the CNN, and the pool grew to **20 GB of wired memory** -- wired,
    so the OS cannot page it out -- climbing about 8 GB per 20 seconds until
    the machine had 0.06 GB free. Killing the run returned 20.2 GB at once.

    Wired growth is the signature to watch for: ordinary leaks show up as
    resident memory, but the MPS pool is pinned for the GPU, so ``ps`` reports
    a process holding only 0.2 GB while the system runs out of RAM.

    Args:
        every_n_batches (int): How often to drop the cache.
    """

    def __init__(self, every_n_batches: int = 100):
        if every_n_batches < 1:
            raise ValueError(f"every_n_batches must be >= 1, got {every_n_batches}.")
        self.every_n_batches = every_n_batches

    @staticmethod
    def _drop() -> None:
        """Return cached blocks to the system, whichever accelerator is live."""
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _maybe_drop(self, batch_idx: int) -> None:
        """Drop the cache on the configured cadence.

        Args:
            batch_idx (int): Index of the batch that just finished.
        """
        if batch_idx > 0 and batch_idx % self.every_n_batches == 0:
            self._drop()

    def on_train_batch_end(
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: L.LightningModule,  # noqa: ARG002
        outputs: Any,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
    ) -> None:
        """Drop the cache periodically during training.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
            outputs (Any): The step's return value.
            batch (Any): The batch just finished.
            batch_idx (int): Its index.
        """
        self._maybe_drop(batch_idx)

    def on_test_batch_end(
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: L.LightningModule,  # noqa: ARG002
        outputs: Any,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Bound the allocator pool during the test pass.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module under test.
            outputs (Any): The step's return value.
            batch (Any): The batch just processed.
            batch_idx (int): Its index.
            dataloader_idx (int): Which dataloader it came from.
        """
        self._maybe_drop(batch_idx)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: L.LightningModule,  # noqa: ARG002
        outputs: Any,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Bound the allocator pool during validation.

        The eval-only flow validates over the full unstrided split before
        testing, which is long enough to grow the pool on its own.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (L.LightningModule): The module.
            outputs (Any): The step's return value.
            batch (Any): The batch just processed.
            batch_idx (int): Its index.
            dataloader_idx (int): Which dataloader it came from.
        """
        self._maybe_drop(batch_idx)


class TestCheckpointer(L.Callback):
    """Makes a long test pass resumable after a crash or a kill.

    The frame-wise benchmark scores 2.4M frames across seven corpora and takes
    hours; losing it at 40% to an out-of-memory kill means starting over. This
    periodically writes the accumulated predictions to disk, and on a later run
    reloads them and **skips the batches already scored**.

    **Why this is safe.** The test dataloader is built with ``shuffle=False``,
    no sampler, and ``drop_last=False``
    (:meth:`~blinklinmult.data.datamodule.BlinkDataModule.test_dataloader`), so
    batch *n* holds the same samples on every run over the same split. Every
    reported metric is an order-independent reduction over per-sample
    predictions -- counts for the rates, a sort for average precision, a
    per-recording regroup for the event scores -- so predictions restored from
    disk are indistinguishable from predictions just computed. The resumed run
    is not an approximation of the full pass; it is the full pass.

    **What invalidates a checkpoint.** The state is keyed by the split's shape
    and the targets scored. A checkpoint whose key disagrees with the current
    run is ignored rather than merged, because silently mixing predictions from
    two different models or corpora would corrupt the benchmark in a way no
    later check would catch.

    Args:
        output_dir (Path): Directory to keep the partial state in.
        every_n_batches (int): How often to persist. Each save rewrites the
            accumulated state, so this trades I/O against how much work a kill
            can destroy.
        enabled (bool): Whether to checkpoint at all.
    """

    def __init__(self, output_dir: Path, every_n_batches: int = 2000, enabled: bool = False):
        if every_n_batches < 1:
            raise ValueError(f"every_n_batches must be >= 1, got {every_n_batches}.")
        self.output_dir = Path(output_dir)
        self.every_n_batches = every_n_batches
        self.enabled = enabled
        self.skip_batches = 0
        self._restored = False

    @property
    def state_path(self) -> Path:
        """Where the partial test state lives."""
        return self.output_dir / TEST_STATE_FILENAME

    def _key(self, pl_module: BlinkLightningModule) -> str:
        """Identity of the pass a checkpoint belongs to.

        Args:
            pl_module (BlinkLightningModule): The module under test.

        Returns:
            str: A key that changes whenever resuming would be unsound.
        """
        return "|".join(pl_module.scored_targets)

    def on_test_start(  # ty: ignore[invalid-method-override]
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: BlinkLightningModule,
    ) -> None:
        """Restore a previous partial pass, if one matches this run.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (BlinkLightningModule): The module under test.
        """
        self.skip_batches = 0
        self._restored = False
        pl_module.skip_test_batches = 0
        if not self.enabled or not self.state_path.is_file():
            return

        try:
            state = torch.load(self.state_path, map_location="cpu", weights_only=False)
        except (OSError, RuntimeError, EOFError, pickle.UnpicklingError, KeyError):
            logger.warning(f"Ignoring an unreadable test checkpoint at {self.state_path}.")
            return

        if state.get("key") != self._key(pl_module):
            logger.warning(
                f"Ignoring the test checkpoint at {self.state_path}: it was written for a "
                "different set of scored targets."
            )
            return

        for name, payload in state["targets"].items():
            if name not in pl_module.test_metrics:
                continue
            # `restore` rather than `update`: the stored group ids already
            # encode the frames they came from and must survive verbatim.
            pl_module.test_metrics[name].restore(
                payload["probability"].to(pl_module.device),
                payload["target"].to(pl_module.device),
                payload["mask"].to(pl_module.device),
                payload["group"].to(pl_module.device),
                payload["window_length"],
            )

        pl_module.test_sample_ids = list(state["sample_ids"])
        pl_module.test_datasets = list(state["datasets"])
        self.skip_batches = int(state["batches"])
        pl_module.skip_test_batches = self.skip_batches
        self._restored = True
        logger.info(
            f"Resuming the test pass from {self.state_path}: "
            f"{self.skip_batches} batches ({len(pl_module.test_sample_ids)} samples) "
            "already scored."
        )

    def on_test_batch_end(  # ty: ignore[invalid-method-override]
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: BlinkLightningModule,
        outputs: Any,  # noqa: ARG002
        batch: Any,  # noqa: ARG002
        batch_idx: int,
        dataloader_idx: int = 0,  # noqa: ARG002
    ) -> None:
        """Persist the accumulated state on the configured cadence.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (BlinkLightningModule): The module under test.
            outputs (Any): The step's return value.
            batch (Any): The batch just processed.
            batch_idx (int): Its index.
            dataloader_idx (int): Which dataloader it came from.
        """
        if not self.enabled:
            return
        if batch_idx > 0 and batch_idx % self.every_n_batches == 0:
            self.save(pl_module, batch_idx + 1)

    def save(self, pl_module: BlinkLightningModule, batches: int) -> None:
        """Write the accumulated predictions to disk.

        Written to a temporary file and renamed, so a kill during the write
        cannot leave a half-flushed checkpoint that a later run would trust.

        Args:
            pl_module (BlinkLightningModule): The module under test.
            batches (int): How many batches have been scored.
        """
        targets: dict[str, dict[str, Any]] = {}
        for name in pl_module.scored_targets:
            metric = pl_module.test_metrics[name]
            probability, target, mask = metric.predictions()
            if probability.numel() == 0:
                continue
            groups = metric.groups()
            targets[name] = {
                "probability": probability.cpu(),
                "target": target.cpu(),
                "mask": mask.cpu(),
                "group": (
                    groups.cpu()
                    if groups is not None
                    else torch.full_like(probability.cpu(), NO_GROUP, dtype=torch.long)
                ),
                "window_length": metric.window_length,
            }

        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp_path = self.state_path.with_suffix(".tmp")
        torch.save(
            {
                "key": self._key(pl_module),
                "batches": batches,
                "targets": targets,
                "sample_ids": pl_module.test_sample_ids,
                "datasets": pl_module.test_datasets,
            },
            tmp_path,
        )
        tmp_path.replace(self.state_path)
        logger.info(f"Checkpointed the test pass at batch {batches} to {self.state_path}.")

    def clear(self) -> None:
        """Delete the partial state once the pass has completed."""
        self.state_path.unlink(missing_ok=True)


class TestStateReleaser(L.Callback):
    """Frees the test accumulator once every reporting callback has read it.

    :meth:`BlinkLightningModule.on_test_epoch_end` deliberately leaves the
    accumulated predictions in place, because the prediction writer, the
    per-corpus report, the event report, and the plotter all read them
    afterwards. Nothing then releases them, so a full benchmark pass holds every
    test prediction -- and the accelerator blocks backing them -- until the
    process exits.

    Released in ``on_test_end``, **not** ``on_test_epoch_end``. Lightning runs
    every callback's ``on_test_epoch_end`` *before* the LightningModule's own
    (``evaluation_loop.py``: ``_call_callback_hooks`` then
    ``_call_lightning_module_hook``), so releasing there would empty the
    accumulator before :meth:`BlinkLightningModule.on_test_epoch_end` computed
    the epoch's metrics -- which reported ``test/mean_f1 = 0.0`` against a real
    per-corpus F1 of 0.968. ``on_test_end`` runs after the whole stage, once
    every reporter and the module itself have read the state.
    """

    def on_test_end(  # ty: ignore[invalid-method-override]
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: BlinkLightningModule,
    ) -> None:
        """Release the test predictions and drop cached accelerator blocks.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (BlinkLightningModule): The module under test.
        """
        pl_module.test_metrics.reset()
        pl_module.test_sample_ids = []
        pl_module.test_datasets = []
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()


class EventReport(L.Callback):
    """Scores blink detection as events, the way the literature does.

    Frame-wise and window-wise metrics answer "was this frame/window right?".
    This answers "was this *blink* found?", which is the question the field
    reports and the only one comparable with published numbers.

    Per-frame predictions are reassembled into one signal per recording,
    averaged wherever the sweep's windows overlap, thresholded into intervals,
    and matched against the annotated blinks under all four literature criteria
    — see :mod:`blinklinmult.train.events`.

    **Grouped per recording**, because false alarms are reported per minute and
    a duration only means something for one continuous timeline. The corpus row
    aggregates by summing counts and re-deriving the rates, never by averaging
    rates: that would weigh a three-second clip the same as a five-minute one.

    **The operating point is fitted on validation, not fixed.** A threshold is
    a choice, and 0.5 is only the right one if the model happens to be
    calibrated there. Measured across the RN30 diagnostic arms, it is often
    not: focal-loss runs peaked at 0.35 and 0.30 and lost 11.5 and 16.5 points
    of event F1 to the default, while a BCE run happened to peak at 0.50 and
    lost nothing. Reporting all of them at 0.5 would have compared calibration
    rather than detection.

    Fitting on validation and applying to test keeps the test split honest --
    picking the threshold that maximises the *test* score would report a number
    no deployment could reproduce.

    Args:
        output_dir (Path): Directory to write ``test_events.json`` into.
        fps (float): Frame rate, for converting frame counts to minutes.
        threshold (float): Fallback operating point, used when validation
            produced nothing to fit on.
        tune_threshold (bool): Fit the operating point on validation. Set
            ``False`` to report at the fixed ``threshold`` instead.
        shared_threshold_dir (Path | None): Read and write the threshold cache
            here instead of in the run's own directory, so a per-corpus
            evaluation can share **one** universal operating point. ``None``
            keeps the cache private to the run.
        carrier_only (bool): Drop every corpus but the largest, on the
            assumption that the smaller ones are only carriers supplying the
            prediction head.

            **Correct for a single-corpus evaluation, wrong for a joint one.**
            An eval-only corpus needs a `datasets` entry to build the head, and
            those carrier samples would otherwise be counted as recordings --
            measured, CEW's 366 stills contributed 192 false positives and no
            true positives to every eval-only run. But a *joint* test split has
            no carrier: taking the majority there discarded RN, HUST-LEBW and
            TalkingFace because MPEblink held 51 083 of 64 027 samples, and the
            report then scored a corpus that annotates no closure at all.

            Off by default, so a joint split is never silently reduced; the
            per-corpus eval scripts turn it on.
        low_ratios (Sequence[float] | None): Hysteresis ratios to search
            alongside the threshold. The low threshold is ``high * ratio``, so
            the pair stays ordered across the whole sweep -- a constant low
            value would exceed the high one at the bottom and silently disable
            hysteresis. ``None`` searches :data:`DEFAULT_LOW_RATIOS`; pass
            ``[None]`` to keep the single-threshold behaviour.

            A single cut has to be low enough to catch a shallow closure onset
            and high enough not to fire on noise, and no value does both.
            Measured on ``fw-focal``: a single 0.37 cut gives precision 0.275
            with 15 602 false alarms, while 0.75/0.25 gives **0.660** with
            **2 133** -- 2.4x the precision for 86% fewer false alarms. The
            gain comes from rejecting flickers: 37.4% of false-positive runs
            are a single frame, peaking at a median 0.513, against a median
            0.919 for real blinks.
    """

    def __init__(
        self,
        output_dir: Path,
        fps: float = 30.0,
        threshold: float = 0.5,
        tune_threshold: bool = True,
        target: str = BLINK_PRESENCE,
        tuning_criterion: str = TUNING_CRITERION,
        shared_threshold_dir: Path | None = None,
        low_ratios: Sequence[float | None] | None = None,
        carrier_only: bool = False,
    ):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.threshold = threshold
        self.tune_threshold = tune_threshold
        self.target = target
        self.tuning_criterion = tuning_criterion
        self.shared_threshold_dir = (
            Path(shared_threshold_dir) if shared_threshold_dir is not None else None
        )
        self.low_ratios: tuple[float | None, ...] = (
            tuple(DEFAULT_LOW_RATIOS) if low_ratios is None else tuple(low_ratios)
        )
        self.carrier_only = carrier_only
        self.fitted_threshold: float | None = None
        self.fitted_low_ratio: float | None = None

    def on_validation_epoch_end(  # ty: ignore[invalid-method-override]
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: BlinkLightningModule,
    ) -> None:
        """Fit the event threshold on this validation epoch.

        Runs every epoch and keeps the latest, so the threshold matches the
        weights that testing will actually load -- an early-stopped run tests
        its best checkpoint, and a threshold fitted on some earlier epoch would
        describe a different model.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (BlinkLightningModule): The module being validated.
        """
        if not self.tune_threshold or self.target not in pl_module.scored_targets:
            return

        sample_ids = pl_module.valid_sample_ids
        if not sample_ids:
            return

        metric = pl_module.valid_metrics[self.target]
        probability, target, _ = metric.predictions()
        if probability.numel() == 0:
            return
        per_sample = probability.numel() // len(sample_ids)
        if per_sample == 0:
            return

        recordings = self._by_recording(
            sample_ids, probability, target, per_sample, pl_module.valid_datasets
        )
        if not recordings:
            return

        # One sweep per hysteresis ratio, and the pair with the best smoothed
        # validation F1 wins. `None` is searched first and wins ties, so
        # hysteresis has to earn its place rather than being assumed.
        total_minutes = sum(self._minutes(v) for v in recordings.values())
        best_pooled: dict | None = None
        best_index = 0
        best_score = -np.inf
        best_ratio: float | None = None
        for ratio in self.low_ratios:
            curves = [
                froc(
                    *self._signal_and_truth(values),
                    minutes=self._minutes(values),
                    criterion=TUNING_CRITERION,
                    low_ratio=ratio,
                )
                for values in recordings.values()
            ]
            pooled = pool_curves(curves, minutes=total_minutes)
            if not pooled or not pooled["f1"].size:
                continue
            index = self._select_threshold(pooled)
            score = float(np.asarray(pooled["f1"])[index])
            if score > best_score:
                best_pooled, best_index, best_score, best_ratio = pooled, index, score, ratio

        if best_pooled is None:
            return

        pooled, best = best_pooled, best_index
        self.fitted_threshold = float(pooled["thresholds"][best])
        self.fitted_low_ratio = best_ratio
        if best_ratio is None:
            hysteresis = "single threshold"
        else:
            low = self.fitted_threshold * best_ratio
            hysteresis = f"hysteresis low={low:.2f} (ratio {best_ratio:.2f})"
        logger.info(
            f"Event operating point fitted on validation: high={self.fitted_threshold:.2f}, "
            f"{hysteresis} (F1 {pooled['f1'][best]:.4f} at {self.tuning_criterion}); "
            f"default was {self.threshold:.2f}."
        )

        # A fit that stops at either end of the sweep has not found an optimum,
        # it has run out of range -- or the objective is flat and the search is
        # buying recall by firing at everything. Measured on RN15 the fitted
        # point was the sweep floor with a validation F1 of 0.14, and test then
        # lost 7 points of event F1 to miscalibration. Silent in the old code;
        # loud here, because the number it produces is not trustworthy.
        self._warn_on_boundary(pooled, best)
        self._save_threshold(pooled["f1"][best])

    @staticmethod
    def _select_threshold(pooled: dict, window: int = 5) -> int:
        """Pick the operating point from a **smoothed** validation F1 curve.

        A plain ``argmax`` chases a single point on a curve estimated from very
        few events -- RN15's validation split yields a few dozen after
        rasterising -- so it lands wherever the noise happens to peak. Measured
        on the frame-wise benchmark it selected 0.01 and 0.06 while the same
        corpora's test optima were 0.51 and 0.65, costing 10 and 20 points of
        event F1.

        Averaging over neighbouring thresholds first prefers a **broad**
        maximum to a tall narrow one, which is the operating point more likely
        to survive the move from validation to test. It cannot manufacture a
        good threshold where the signal has none, but it stops a single lucky
        threshold from being mistaken for one.

        Args:
            pooled (dict): The pooled FROC curve.
            window (int): Thresholds to average over; must be odd.

        Returns:
            int: Index of the selected threshold in the sweep.
        """
        f1 = np.asarray(pooled["f1"], dtype=np.float64)
        if f1.size <= window or window < 3:
            return int(f1.argmax())

        pad = window // 2
        # Edge-padded so the ends stay comparable with the middle rather than
        # being dragged toward zero, which would bias selection inward.
        smoothed = np.convolve(np.pad(f1, pad, mode="edge"), np.ones(window) / window, mode="valid")
        return int(smoothed.argmax())

    @staticmethod
    def _warn_on_boundary(pooled: dict, best: int) -> None:
        """Warn when the fit landed on either end of the sweep.

        A search that stops at a boundary has run out of range, or the objective
        is flat and it is buying recall by firing at everything -- neither is a
        fitted optimum. Measured on RN15 the fitted point was the sweep floor at
        a validation F1 of 0.14, and test then lost 7 points of event F1 to the
        miscalibration, silently.

        Args:
            pooled (dict): The pooled FROC curve.
            best (int): Index the fit selected.
        """
        thresholds = np.asarray(pooled["thresholds"], dtype=np.float64)
        if thresholds.size == 0 or best not in (0, thresholds.size - 1):
            return
        edge = "lowest" if best == 0 else "highest"
        logger.warning(
            f"The fitted event threshold {thresholds[best]:.2f} is the {edge} value in "
            f"the sweep, so it is a boundary hit rather than a fitted optimum "
            f"(validation F1 {float(np.asarray(pooled['f1'])[best]):.4f}). The operating "
            "point is unreliable and the reported event scores will understate the model."
        )

    @property
    def threshold_path(self) -> Path:
        """Where the fitted operating point is cached.

        ``shared_threshold_dir`` overrides the run's own directory, which is
        what makes one **universal** operating point possible across a
        per-corpus evaluation: the corpora that can fit a threshold write it
        there, and every corpus reads it back instead of fitting its own.
        Without that, each process fits privately and the corpora end up scored
        on different scales -- measured at 0.01, 0.06, and two silent 0.50
        fallbacks across one split run.
        """
        directory = self.shared_threshold_dir or self.output_dir
        if self.target == BLINK_PRESENCE:
            return directory / THRESHOLD_FILENAME
        # A second report fits its own operating point against a different
        # signal; sharing one cache file would have each overwrite the other.
        return directory / f"{Path(THRESHOLD_FILENAME).stem}_{self.target}.json"

    def _save_threshold(self, f1: float) -> None:
        """Persist the fitted point so a re-run need not refit it.

        Fitting costs a full validation pass -- 21 minutes on the frame-wise
        benchmark -- and yields exactly one number. Caching it means a test pass
        that dies partway can be retried without paying for validation again.

        Args:
            f1 (float): The pooled F1 at the chosen threshold, recorded so a
                cached value can be sanity-checked by eye.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.threshold_path.write_text(
            json.dumps(
                {
                    "threshold": self.fitted_threshold,
                    "low_ratio": self.fitted_low_ratio,
                    "low_threshold": (
                        None
                        if self.fitted_low_ratio is None or self.fitted_threshold is None
                        else self.fitted_threshold * self.fitted_low_ratio
                    ),
                    "criterion": self.tuning_criterion,
                    "target": self.target,
                    "valid_f1": float(f1),
                },
                indent=2,
            )
        )
        logger.info(f"Cached the fitted threshold to {self.threshold_path}.")

    def load_cached_threshold(self) -> float | None:
        """Restore a previously fitted operating point, if one matches.

        The cache is keyed by target and criterion: a threshold fitted for a
        different criterion describes a different notion of "detected" and must
        not be reused silently. A mismatch, a missing file, or malformed content
        all return ``None``, which leaves the normal fitting path to run.

        Returns:
            float | None: The cached threshold, or ``None`` to refit.
        """
        path = self.threshold_path
        if not path.is_file():
            return None
        try:
            cached = json.loads(path.read_text())
            value = float(cached["threshold"])
            ratio = cached.get("low_ratio")
            ratio = None if ratio is None else float(ratio)
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning(f"Ignoring unreadable threshold cache at {path}.")
            return None

        if cached.get("criterion") != self.tuning_criterion or cached.get("target") != self.target:
            logger.warning(
                f"Ignoring the threshold cache at {path}: it was fitted for "
                f"target={cached.get('target')!r} criterion={cached.get('criterion')!r}, "
                f"but this run wants target={self.target!r} "
                f"criterion={self.tuning_criterion!r}."
            )
            return None

        self.fitted_threshold = value
        self.fitted_low_ratio = ratio
        hysteresis = "single threshold" if ratio is None else f"hysteresis low={value * ratio:.2f}"
        logger.info(
            f"Reusing the cached event operating point high={value:.2f}, {hysteresis} "
            f"from {path} (validation F1 {cached.get('valid_f1', float('nan')):.4f}); "
            "validation was skipped."
        )
        return value

    def _minutes(self, values: dict[str, np.ndarray]) -> float:
        """Duration of one recording, for the false-alarm rate.

        Args:
            values (dict[str, np.ndarray]): Its predictions, targets, and frame ids.

        Returns:
            float: Length in minutes.
        """
        frames = np.asarray(values["f"], dtype=np.int64)
        return (int(frames.max()) + 1) / self.fps / 60.0

    def _signal_and_truth(self, values: dict[str, np.ndarray]):
        """Reassemble one recording's signal, mask, and annotated blinks.

        Args:
            values (dict[str, np.ndarray]): Its predictions, targets, and frame ids.

        Returns:
            tuple: ``(signal, mask, annotated)`` ready for :func:`froc`.
        """
        frames = np.asarray(values["f"], dtype=np.int64)
        n_frames = int(frames.max()) + 1
        signal, mask = average_overlapping(values["p"], frames, n_frames)
        truth, _ = average_overlapping(values["t"], frames, n_frames)
        return signal, mask, to_intervals(truth, mask, 0.5)

    @property
    def operating_point(self) -> float:
        """The threshold test scoring will use.

        Returns:
            float: The validation-fitted threshold when one was found,
            otherwise the fixed fallback.
        """
        return self.threshold if self.fitted_threshold is None else self.fitted_threshold

    @property
    def low_operating_point(self) -> float | None:
        """The hysteresis low threshold test scoring will use.

        Derived from the fitted ratio rather than stored directly, so it stays
        consistent with :attr:`operating_point` however that was arrived at.

        Returns:
            float | None: ``high * ratio`` when hysteresis was fitted,
            otherwise ``None`` for a single cut.
        """
        if self.fitted_low_ratio is None:
            return None
        return self.operating_point * self.fitted_low_ratio

    def on_test_epoch_end(  # ty: ignore[invalid-method-override]
        self,
        trainer: L.Trainer,  # noqa: ARG002
        pl_module: BlinkLightningModule,
    ) -> None:
        """Write the event-level report and log its corpus row.

        Args:
            trainer (L.Trainer): The trainer.
            pl_module (BlinkLightningModule): The module under test.
        """
        if self.target not in pl_module.scored_targets:
            return

        sample_ids = pl_module.test_sample_ids
        if not sample_ids:
            logger.warning("No per-sample ids recorded; skipping the event report.")
            return

        metric = pl_module.test_metrics[self.target]
        probability, target, _ = metric.predictions()
        if probability.numel() == 0:
            return

        per_sample = probability.numel() // len(sample_ids)
        if per_sample == 0:
            return

        recordings = self._by_recording(
            sample_ids, probability, target, per_sample, pl_module.test_datasets
        )
        if not recordings:
            return

        collected: dict[str, list[dict]] = {criterion: [] for criterion in CRITERIA}
        report: dict[str, dict[str, float]] = {}
        for name, values in recordings.items():
            per_recording: dict[str, dict] = {}
            report[name] = self._score(values, per_recording)
            for criterion, curve in per_recording.items():
                collected[criterion].append(curve)

        summary, pooled = self._aggregate(report, collected)
        # Corpus-wide, from every instance at once -- see `_blink_ap`.
        summary.update(self._blink_ap(recordings))

        # The same predictions scored a second time at this corpus's own best
        # threshold. The universal point above is the deployable number and the
        # only one a corpus without a validation split (TalkingFace ships 0
        # valid windows) can have; the tuned point is what the corpus could
        # reach if it were allowed its own parameter. Reporting both makes the
        # cost of tuning visible instead of hiding it in whichever was chosen.
        summary.update(self._tuned_scores(recordings, pooled))
        report["_corpus"] = summary

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Keyed by target so the two routes to an interval can be reported side
        # by side: `eye_state` is the hand-fitted hysteresis extractor, and
        # `blink_presence` is the learned event head when one is configured.
        # Comparing them is the point; overwriting one with the other is not.
        stem = "test_events" if self.target == BLINK_PRESENCE else f"test_events_{self.target}"
        path = self.output_dir / f"{stem}.json"
        payload: dict[str, Any] = dict(report)
        # The curve points, not just the two scalars derived from them: the FROC
        # curve is the headline result and it was previously computed and thrown
        # away, so a finished run could not be plotted or re-thresholded without
        # re-running inference.
        payload["_froc"] = {
            criterion: {key: value.tolist() for key, value in curve.items()}
            for criterion, curve in pooled.items()
        }
        path.write_text(json.dumps(payload, indent=2))

        self._write_signals(recordings)
        self._log_curves(pl_module, pooled)

        # Logged as metrics, not only written to the artifact: these are the
        # numbers the literature reports, so they have to be sortable and
        # plottable in MLflow beside frame F1 rather than requiring a download.
        # Per-recording rows stay in the JSON -- one run would otherwise log
        # hundreds of series and bury the corpus row.
        pl_module.log_dict(
            {f"test/{key}": float(value) for key, value in summary.items()},
            sync_dist=True,
        )

        for criterion in CRITERIA:
            logger.info(
                f"test/event/{criterion}: recall={summary[f'event/{criterion}/recall']:.3f} "
                f"precision={summary[f'event/{criterion}/precision']:.3f} "
                f"f1={summary[f'event/{criterion}/f1']:.3f} "
                f"fa/min={summary[f'event/{criterion}/fa_per_min']:.2f}"
            )
        if "event/blink_ap" in summary:
            logger.info(
                f"test/event/blink_ap: {summary['event/blink_ap']:.4f} (@0.5:0.95)  "
                f"@0.5={summary['event/blink_ap50']:.4f}  "
                f"@0.75={summary['event/blink_ap75']:.4f}  "
                "-- oracle-instance; see docs/data.md"
            )
        logger.info(f"Event-level report written to {path}")

    def _write_signals(self, recordings: dict[str, dict[str, np.ndarray]]) -> None:
        """Dump each recording's reassembled timeline.

        The event report keeps only aggregates, so the per-frame signal it was
        derived from is thrown away -- and rebuilding it means re-running
        inference over the whole split. Saved here as a compressed archive so
        curve-shape analysis, phase derivation, and extractor comparisons can
        run on a finished benchmark in seconds.

        **The ``truth`` arrays are this callback's own target**, which is
        ``blink_presence`` on the frame-wise benchmark -- *not* ``eye_state``.
        The distinction is not cosmetic: a blink *interval* covers 3-4x more
        frames than actual closure (measured P(closed | inside a blink) = 0.237
        / 0.310 / 0.344 on RN15 / RN30 / TalkingFace), and MPEblink carries no
        closure annotation at all. Reading these arrays as closure labels
        manufactures phantom false positives -- 200 045 of them on one analysis
        of this very file, 96.7% of the apparent total. ``_target`` records the
        provenance so no reader has to guess, and :func:`load_signals` refuses
        to hand back an archive whose target is not the one asked for.

        Args:
            recordings (dict[str, dict[str, np.ndarray]]): Per-recording
                predictions, as :meth:`_by_recording` returns them.
        """
        payload: dict[str, np.ndarray] = {}
        for name, values in recordings.items():
            frames = np.asarray(values["f"], dtype=np.int64)
            if frames.size == 0:
                continue
            n_frames = int(frames.max()) + 1
            signal, mask = average_overlapping(values["p"], frames, n_frames)
            truth, _ = average_overlapping(values["t"], frames, n_frames)
            payload[f"{name}/signal"] = signal.astype(np.float32)
            payload[f"{name}/truth"] = truth.astype(np.float32)
            payload[f"{name}/mask"] = mask

        # Written even when no recording produced a timeline, so the archive is
        # never silently target-less.
        payload[SIGNALS_TARGET_KEY] = np.asarray(self.target)

        if not payload:
            return
        stem = Path(SIGNALS_FILENAME).stem
        name = SIGNALS_FILENAME if self.target == BLINK_PRESENCE else f"{stem}_{self.target}.npz"
        path = self.output_dir / name
        # numpy's stub declares `**kwds: ArrayLike` alongside `allow_pickle:
        # bool`, and the checker resolves a splatted dict against the latter.
        # Passing arrays as keywords is exactly what the function is for.
        np.savez_compressed(path, **payload)  # ty: ignore[invalid-argument-type]
        logger.info(f"Per-recording signals written to {path}")

    def _tuned_scores(
        self,
        recordings: dict[str, dict[str, np.ndarray]],
        pooled: dict[str, dict],
    ) -> dict[str, float]:
        """Re-score this corpus at its own best threshold.

        The universal operating point is fitted once, on the pooled validation
        split of the corpora the model trains on, and applied everywhere. That
        is the honest deployable number, and the only option for a corpus with
        no validation split of its own. But it says nothing about what the
        corpus could reach with a threshold of its own, and the gap between the
        two is exactly the cost of tuning.

        The tuned point is read off the **test** FROC curve, so it is an oracle
        upper bound, not a reproducible operating point. It is reported under
        ``tuned/`` names and must be labelled as tuned wherever it appears --
        quoting it as the headline would be reporting a threshold fitted on the
        test split.

        Args:
            recordings (dict[str, dict[str, np.ndarray]]): Per-recording
                predictions, as :meth:`_by_recording` returns them.
            pooled (dict[str, dict]): The pooled FROC curve per criterion.

        Returns:
            dict[str, float]: ``tuned/<criterion>/{threshold,f1}`` per criterion,
            plus the universal threshold actually used, for comparison.
        """
        scores: dict[str, float] = {"event/threshold": float(self.operating_point)}
        if self.low_operating_point is not None:
            scores["event/low_threshold"] = float(self.low_operating_point)

        for criterion, curve in pooled.items():
            f1 = np.asarray(curve.get("f1", []), dtype=np.float64)
            thresholds = np.asarray(curve.get("thresholds", []), dtype=np.float64)
            if f1.size == 0 or thresholds.size != f1.size:
                continue
            best = int(f1.argmax())
            scores[f"tuned/{criterion}/threshold"] = float(thresholds[best])
            scores[f"tuned/{criterion}/f1"] = float(f1[best])

        return scores

    def _blink_ap(self, recordings: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
        """MPEblink's Blink-AP over every instance at once.

        **Not poolable from per-recording scores.** Average precision integrates
        a single ranked list, so a confident detection in one tracklet must be
        able to outrank a doubtful one in another; averaging per-recording APs
        would be a different quantity. Every instance is therefore collected
        first and scored together.

        Each predicted interval carries the **peak** of the averaged signal
        across it as its confidence. The authors rank by their detector's own
        instance score, which this project has no equivalent of -- an oracle
        instance has no score -- so the peak stands in for it. That difference
        belongs in any write-up beside the oracle-instance caveat.

        Args:
            recordings (dict[str, dict[str, np.ndarray]]): Per-instance predictions,
                targets, and frame ids, as :meth:`_by_recording` returns them.

        Returns:
            dict[str, float]: ``event/blink_ap`` and its @0.5/@0.75/@0.95
            companions, ready to log. Empty when nothing was annotated.
        """
        annotated: dict[str, list] = {}
        predicted: dict[str, list[tuple[int, int, float]]] = {}

        for name, values in recordings.items():
            frames = np.asarray(values["f"], dtype=np.int64)
            if frames.size == 0:
                continue
            n_frames = int(frames.max()) + 1

            signal, mask = average_overlapping(values["p"], frames, n_frames)
            truth, _ = average_overlapping(values["t"], frames, n_frames)

            events = to_intervals(truth, mask, 0.5)
            if events:
                annotated[name] = events

            intervals = to_intervals(signal, mask, self.operating_point, self.low_operating_point)
            if intervals:
                predicted[name] = [
                    (start, stop, float(signal[start : stop + 1].max()))
                    for start, stop in intervals
                ]

        if not annotated:
            return {}

        scores = blink_ap(annotated, predicted)
        return {f"event/{key}": value for key, value in blink_ap_summary(scores).items()}

    @staticmethod
    def _log_curves(pl_module: BlinkLightningModule, pooled: dict[str, dict]) -> None:
        """Log the pooled FROC curve as a step-series.

        MLflow plots a metric against its step, so logging each threshold as one
        step makes the curve readable in the UI without downloading the
        artifact. The series are named ``test/froc/<criterion>/<metric>``, kept
        distinct from the single-operating-point ``test/event/<criterion>/...``
        scalars so a sweep and a threshold are never confused for each other.

        Args:
            pl_module (BlinkLightningModule): The module under test.
            pooled (dict[str, dict]): Pooled curve per criterion.
        """
        logger_ = getattr(pl_module, "logger", None)
        experiment = getattr(logger_, "experiment", None)
        run_id = getattr(logger_, "run_id", None)
        if experiment is None or run_id is None or not hasattr(experiment, "log_metric"):
            return

        for criterion, curve in pooled.items():
            for index, threshold in enumerate(curve["thresholds"]):
                for key in ("recall", "precision", "f1", "false_alarms_per_minute"):
                    experiment.log_metric(
                        run_id,
                        f"test/froc/{criterion}/{key}",
                        float(curve[key][index]),
                        step=index,
                    )
                experiment.log_metric(
                    run_id, f"test/froc/{criterion}/threshold", float(threshold), step=index
                )

    def _by_recording(
        self,
        sample_ids: list[str],
        probability: torch.Tensor,
        target: torch.Tensor,
        per_sample: int,
        datasets: list[str] | None = None,
    ) -> dict[str, dict[str, np.ndarray]]:
        """Group per-frame predictions by the recording they came from.

        A sample's key encodes its recording and the frame its window starts
        at, so a window's Nth position is frame ``start + N`` — which is what
        lets overlapping windows be averaged back onto a single timeline.

        **Samples from a carrier corpus are dropped.** Evaluating a corpus that
        supplies no trained target -- TalkingFace and MPEblink on the
        frame-wise benchmark -- still needs a `datasets` entry to build the
        prediction head, and the Makefile uses CEW for that. Without this
        filter those carrier samples are counted as *recordings* in the event
        report: measured, CEW's 366 single-frame stills contributed **192 false
        positives and 0 true positives** to every eval-only run, which dragged
        TalkingFace from its true F1 of 0.9573 (precision 1.0000, zero false
        alarms) down to a reported 0.3625. A still image cannot contain a blink
        *event* at all, so its contribution is noise by construction.

        The corpus under evaluation is the one the eval-only route names, so it
        is taken as the corpus that supplied the *most* samples: a carrier only
        has to be large enough to build a head, never larger than the corpus
        actually being scored.

        Args:
            sample_ids (list[str]): Per-sample keys.
            probability (torch.Tensor): Flat per-position predictions.
            target (torch.Tensor): Flat per-position targets.
            per_sample (int): Positions per sample.
            datasets (list[str] | None): Per-sample source corpus, aligned with
                ``sample_ids``. ``None`` keeps every sample, which is correct
                when a single corpus is being scored.

        Returns:
            dict[str, dict[str, np.ndarray]]: Predictions, targets, and frame ids per
            recording.
        """
        keep: str | None = None
        if self.carrier_only and datasets and len(datasets) == len(sample_ids):
            counts: dict[str, int] = {}
            for name in datasets:
                counts[name] = counts.get(name, 0) + 1
            if len(counts) > 1:
                keep = max(counts, key=lambda name: counts[name])
                dropped = {n: c for n, c in counts.items() if n != keep}
                logger.info(
                    f"Event report scoring {keep!r} ({counts[keep]} samples); dropping "
                    f"carrier samples from {dropped}. A carrier corpus supplies the "
                    "head, not the benchmark."
                )
        probabilities = probability.detach().cpu().numpy().reshape(-1)
        targets = target.detach().cpu().numpy().reshape(-1)

        # Slices are collected per recording and concatenated once, rather than
        # `.extend(...tolist())` per sample. `tolist()` boxes each float32 into
        # a 24-byte Python float in an 8-byte list slot: measured at 98.6 bytes
        # per frame against 8 as arrays, a 12x blow-up, and this runs over every
        # frame of the test split.
        chunks: dict[str, dict[str, list[np.ndarray]]] = {}

        for index, sample_id in enumerate(sample_ids):
            if keep is not None and datasets is not None and datasets[index] != keep:
                continue
            try:
                video_id, frame_group, _ = parse_sample_id(sample_id)
                start = int(frame_group)
            except (SchemaError, ValueError):
                continue

            begin = index * per_sample
            end = begin + per_sample
            if end > probabilities.size:
                break

            entry = chunks.setdefault(video_id, {"p": [], "t": [], "f": []})
            entry["p"].append(probabilities[begin:end])
            entry["t"].append(targets[begin:end])
            entry["f"].append(np.arange(start, start + per_sample, dtype=np.int64))

        # `_score` reads these by key and passes them to `average_overlapping`
        # and `np.asarray`, both of which take an array as readily as a list, so
        # the concatenated arrays are drop-in.
        return {
            video_id: {key: np.concatenate(parts) for key, parts in entry.items()}
            for video_id, entry in chunks.items()
        }

    def _score(
        self, values: dict[str, np.ndarray], curves: dict[str, dict] | None = None
    ) -> dict[str, float]:
        """Score one recording.

        Args:
            values (dict[str, np.ndarray]): Its predictions, targets, and frame ids.
            curves (dict[str, dict] | None): Filled in with this recording's
                FROC curve per criterion, for pooling into the corpus curve.

        Returns:
            dict[str, float]: Event-level metrics.
        """
        frames = np.asarray(values["f"], dtype=np.int64)
        n_frames = int(frames.max()) + 1

        signal, mask = average_overlapping(values["p"], frames, n_frames)
        # The target is the same masked average: a frame is annotated as
        # blinking if the windows covering it say so, and they agree because
        # they read the same annotation.
        truth, _ = average_overlapping(values["t"], frames, n_frames)
        annotated = to_intervals(truth, mask, 0.5)

        return event_metrics(
            signal,
            mask,
            annotated,
            minutes=n_frames / self.fps / 60.0,
            threshold=self.operating_point,
            # Both, deliberately: `low_threshold` fixes the single operating
            # point, `low_ratio` keeps the swept FROC curves on the same
            # extractor. Passing only the first would score the headline number
            # with hysteresis and the curve beside it without.
            low_threshold=self.low_operating_point,
            low_ratio=self.fitted_low_ratio,
            curves=curves,
        )

    @staticmethod
    def _aggregate(
        report: dict[str, dict[str, float]],
        curves: dict[str, list[dict]] | None = None,
    ) -> tuple[dict[str, float], dict[str, dict]]:
        """Combine per-recording scores into one corpus row.

        Counts are summed and the rates re-derived from them. Averaging the
        rates instead would give a three-second clip the same weight as a
        five-minute recording — and the same rule applies to the FROC curve,
        which is pooled by :func:`~blinklinmult.train.events.pool_curves`
        rather than by averaging the per-recording curves.

        ``average_precision`` and ``best_f1`` are **recomputed from the pooled
        curve**, not summed: they are derived from a whole sweep, so a corpus
        value has to come from the corpus sweep. An earlier version dropped them
        here silently, which left the two headline FROC numbers out of the
        corpus row -- and therefore out of MLflow -- while they sat present in
        every per-recording row.

        Args:
            report (dict[str, dict[str, float]]): Per-recording metrics.
            curves (dict[str, list[dict]] | None): Per-criterion list of the
                recordings' FROC curves.

        Returns:
            tuple[dict[str, float], dict[str, dict]]: Corpus-level metrics, and
            the pooled curve per criterion.
        """
        total: dict[str, float] = {}
        minutes = sum(values.get("event/minutes", 0.0) for values in report.values())
        total["event/minutes"] = minutes
        total["event/n_annotated"] = sum(
            values.get("event/n_annotated", 0.0) for values in report.values()
        )

        pooled: dict[str, dict] = {}
        for criterion in CRITERIA:
            prefix = f"event/{criterion}"
            counts = {
                key: sum(values.get(f"{prefix}/{key}", 0.0) for values in report.values())
                for key in ("tp", "fp", "fn")
            }
            hits, misses, alarms = counts["tp"], counts["fn"], counts["fp"]
            recall = hits / (hits + misses) if hits + misses else 0.0
            precision = hits / (hits + alarms) if hits + alarms else 0.0
            total.update(
                {
                    f"{prefix}/tp": hits,
                    f"{prefix}/fp": alarms,
                    f"{prefix}/fn": misses,
                    f"{prefix}/recall": recall,
                    f"{prefix}/precision": precision,
                    f"{prefix}/f1": (
                        2 * precision * recall / (precision + recall) if precision + recall else 0.0
                    ),
                    f"{prefix}/fa_per_min": alarms / minutes if minutes > 0 else 0.0,
                }
            )

            curve = pool_curves((curves or {}).get(criterion, []), minutes)
            if curve:
                pooled[criterion] = curve
                total[f"{prefix}/average_precision"] = average_precision(curve)
                total[f"{prefix}/best_f1"] = float(curve["f1"].max())

        return total, pooled


def load_signals(
    path: Path | str,
    expected_target: str,
) -> dict[str, dict[str, np.ndarray]]:
    """Read a :data:`SIGNALS_FILENAME` archive, refusing a target mismatch.

    The archive holds one ``signal``/``truth``/``mask`` triple per recording,
    plus :data:`SIGNALS_TARGET_KEY` naming which target ``truth`` came from.
    That name is checked rather than trusted, because the failure it guards
    against is silent: the frame-wise benchmark's event report scores
    ``blink_presence``, so its ``truth`` is a blink *interval*, 3-4x wider than
    closure and present for corpora that annotate no closure at all. An analysis
    that read those arrays as ``eye_state`` counted 200 045 phantom false
    positives -- 96.7% of its apparent total -- and produced a plausible,
    entirely wrong number.

    Args:
        path (Path | str): The ``.npz`` archive to read.
        expected_target (str): The target the caller intends to analyse. A
            mismatch raises rather than returning data that would be
            misinterpreted.

    Returns:
        dict[str, dict[str, np.ndarray]]: ``{recording: {"signal", "truth",
        "mask"}}``.

    Raises:
        ValueError: If the archive names a different target, or predates the
            provenance key and so cannot be checked.
    """
    archive = np.load(Path(path), allow_pickle=False)
    if SIGNALS_TARGET_KEY not in archive:
        raise ValueError(
            f"{path} carries no {SIGNALS_TARGET_KEY!r} key, so the target its 'truth' "
            "arrays hold cannot be determined. It was written before the target was "
            "recorded; re-run the evaluation rather than guessing, because reading a "
            "blink_presence archive as eye_state silently fabricates false positives."
        )

    found = str(archive[SIGNALS_TARGET_KEY])
    if found != expected_target:
        raise ValueError(
            f"{path} holds predictions for target {found!r}, not {expected_target!r}. "
            "These are different quantities -- a blink interval covers 3-4x more frames "
            "than actual closure -- so scoring one against the other measures a "
            "convention gap rather than the model."
        )

    recordings: dict[str, dict[str, np.ndarray]] = {}
    for key in archive.files:
        if key == SIGNALS_TARGET_KEY:
            continue
        name, _, field = key.rpartition("/")
        recordings.setdefault(name, {})[field] = archive[key]
    return recordings
