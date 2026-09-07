r"""Find a good peak learning rate for a run, via Lightning's LR range test.

Ramps the learning rate exponentially over a few hundred steps, records the
loss at each, and suggests the point of steepest descent just before the loss
diverges -- the standard LR-range-test heuristic. It prints the suggestion and
saves a loss-vs-lr plot, because the automatic suggestion is readily fooled by
a noisy curve and is meant to be checked by eye rather than trusted.

This only probes the LR. It trains nothing and writes no checkpoint.

**Read the plot; do not take the printed suggestion.** Lightning picks the
point of steepest descent, which on this model lands on a small early bump --
the measured RN30 sweep suggested 5.25e-06 while the loss went on falling
steadily for another three decades, bottoming out near 1e-02. Taking the
suggestion literally would have trained ~200x slower than the data supports.
The suggestion is a starting point for reading the curve, not a result.

**Why this matters here.** The optimiser recipe was carried over from
PersonalityLinMulT, which found its peak LR by running exactly this test on
*its* model and data. A learning rate is not a portable constant: this project
trains a CNN encoder jointly with a transformer under discriminative rates,
which that project did not. Inheriting its 0.005 would be borrowing the answer
without the measurement.

Note the reported suggestion applies to ``optimizer.lr``, the transformer and
head group. ``optimizer.backbone_lr`` is deliberately a tenth of it -- a
pretrained backbone fine-tuned at the head's rate loses its features within a
few hundred steps -- so scale that separately rather than reading it off the
same curve.

Example:
    Probe the LR for BlinkLinT on RN30, saving under that run's results
    directory::

        uv run python -m blinklinmult.train.lr_find \\
            --data config/data/single_rn30.yaml \\
            --model config/model/blinklint_baseline.yaml \\
            --train config/train/lint_blink.yaml
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.tuner import Tuner

from blinklinmult import PROJECT_ROOT
from blinklinmult.data.datamodule import BlinkDataModule
from blinklinmult.train.cli import DEFAULT_DATASET_CONFIG_DIR, run_output_dir
from blinklinmult.train.config import ExperimentConfig, parse_override
from blinklinmult.train.module import BlinkLightningModule

logger = logging.getLogger(__name__)
"""Module-level logger."""

DEFAULT_MIN_LR = 1e-6
"""Sweep start."""

DEFAULT_MAX_LR = 1.0
"""Sweep end -- high enough that the loss diverges for any reasonable model."""

DEFAULT_NUM_TRAINING_STEPS = 200
"""Steps in the sweep.

Lightning's default of 100 is coarse enough that the suggestion jitters between
runs; 200 gives a smoother curve at no meaningful cost.
"""


def find_lr(
    config: ExperimentConfig,
    root: Path | None = None,
    config_dir: Path | None = None,
    output_path: Path | None = None,
    min_lr: float = DEFAULT_MIN_LR,
    max_lr: float = DEFAULT_MAX_LR,
    num_training: int = DEFAULT_NUM_TRAINING_STEPS,
) -> float:
    """Run the LR range test and save its plot.

    Args:
        config (ExperimentConfig): The run the LR is being chosen for.
        root (Path | None): Dataset root, as the CLI uses it.
        config_dir (Path | None): Where per-dataset config lives.
        output_path (Path | None): Plot destination. Defaults to
            ``lr_find.png`` inside the run's own results directory.
        min_lr (float): Sweep start.
        max_lr (float): Sweep end.
        num_training (int): Steps in the sweep.

    Returns:
        float: The suggested peak learning rate.

    Raises:
        RuntimeError: If the sweep produces no result or no suggestion, which
            means the loss never decreased across the swept range.
    """
    L.seed_everything(config.train.seed, workers=True)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")

    specs = config.data.resolve_specs(config_dir or DEFAULT_DATASET_CONFIG_DIR)
    datamodule = BlinkDataModule(config.data, specs, root=root or PROJECT_ROOT)
    datamodule.setup("fit")
    steps_per_epoch = max(1, len(datamodule.train_dataloader()))

    module = BlinkLightningModule(
        model_config=config.model,
        train_config=config.train,
        image_size=datamodule.image_size,
        eye_feature_dim=datamodule.feature_dim,
        steps_per_epoch=steps_per_epoch,
    )

    trainer = L.Trainer(
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        precision=config.train.precision,  # ty: ignore[invalid-argument-type]
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )
    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(
        module,
        datamodule=datamodule,
        min_lr=min_lr,
        max_lr=max_lr,
        num_training=num_training,
    )
    if lr_finder is None:
        raise RuntimeError(
            "LR finder produced no result -- the loss may never have decreased "
            "across the swept range. Check --min-lr/--max-lr."
        )
    raw_suggestion = lr_finder.suggestion()
    if raw_suggestion is None:
        raise RuntimeError(
            "LR finder produced no suggestion -- the loss may never have "
            "decreased across the swept range. Check --min-lr/--max-lr."
        )
    suggestion = float(raw_suggestion)

    if output_path is None:
        output_path = run_output_dir(config) / "lr_find.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Lightning's own plot(suggest=True) marks the suggestion with a red dot and
    # no number, which defeats the purpose of eyeballing it. Annotate the value.
    fig = lr_finder.plot(suggest=True)
    if fig is None:
        raise RuntimeError("LR finder could not produce a plot (matplotlib unavailable?).")
    axis = fig.axes[0]
    axis.set_title(f"LR range test -- suggested lr = {suggestion:.2e}")
    axis.annotate(
        f"{suggestion:.2e}",
        xy=(suggestion, lr_finder.results["loss"][lr_finder._optimal_idx]),
        xytext=(10, 10),
        textcoords="offset points",
        color="red",
        fontweight="bold",
    )
    fig.savefig(output_path)  # ty: ignore[unresolved-attribute]
    logger.info(f"Saved LR-vs-loss plot to {output_path}")
    logger.info(f"Suggested lr: {suggestion:.2e}")
    logger.info(
        f"Set it with: --set train.optimizer.lr={suggestion:.2e} (backbone_lr stays ~10x lower)"
    )
    return suggestion


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="Data config YAML.")
    parser.add_argument("--model", type=Path, required=True, help="Model config YAML.")
    parser.add_argument("--train", type=Path, required=True, help="Train config YAML.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config value, e.g. --set data.batch_size=16.",
    )
    parser.add_argument("--root", type=Path, default=None, help="Dataset root.")
    parser.add_argument("--config-dir", type=Path, default=None, help="Per-dataset config dir.")
    parser.add_argument("--min-lr", type=float, default=DEFAULT_MIN_LR)
    parser.add_argument("--max-lr", type=float, default=DEFAULT_MAX_LR)
    parser.add_argument("--num-training", type=int, default=DEFAULT_NUM_TRAINING_STEPS)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the loss-vs-lr plot. Default: the run's results dir.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    overrides = dict(parse_override(text) for text in args.overrides)
    config = ExperimentConfig.from_files(args.data, args.model, args.train, overrides)
    find_lr(
        config,
        root=args.root,
        config_dir=args.config_dir,
        output_path=args.output,
        min_lr=args.min_lr,
        max_lr=args.max_lr,
        num_training=args.num_training,
    )


if __name__ == "__main__":
    main()
