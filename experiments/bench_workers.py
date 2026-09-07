"""Measure dataloader throughput against worker count, on the real corpora.

`num_workers` was set to 0 from a measurement on CEW -- a 123 MB corpus where
worker startup cost more than the workers saved. The frame-wise run now pulls
from 54 GB across six corpora, and `datamodule.py` flags exactly this case:
"Larger corpora may invert that trade, which is why the value is named here
rather than hard-coded."

This settles it by measurement rather than argument. It times the **second**
pass over a fixed number of batches, so the figure reflects steady-state
throughput rather than the one-off index build, and reports batches per second
so the numbers compare directly.

    uv run python experiments/bench_workers.py --batches 60
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from blinklinmult.data import datamodule
from blinklinmult.data.datamodule import BlinkDataModule
from blinklinmult.train.config import ExperimentConfig

logger = logging.getLogger(__name__)
"""Module-level logger."""

DEFAULT_COUNTS: tuple[int, ...] = (0, 2, 4, 6, 8)
"""Worker counts to try, spanning in-process to more than the performance cores."""


def throughput(module: BlinkDataModule, batches: int) -> float:
    """Batches per second over a fixed number of batches.

    The loader is drained once before timing, so the figure is steady-state:
    the first pass pays for worker startup and the HDF5 handles opening, which
    is a real cost but a one-off one, and folding it into the rate would make a
    short benchmark look worse than a real epoch.

    Args:
        module (BlinkDataModule): A set-up datamodule.
        batches (int): How many batches to time.

    Returns:
        float: Batches per second.
    """
    loader = module.train_dataloader()

    warmed = 0
    for _ in loader:
        warmed += 1
        if warmed >= 5:
            break

    start = time.perf_counter()
    seen = 0
    for _ in loader:
        seen += 1
        if seen >= batches:
            break
    elapsed = time.perf_counter() - start
    return seen / elapsed if elapsed > 0 else 0.0


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", type=int, default=60, help="Batches to time per setting.")
    parser.add_argument(
        "--workers",
        type=int,
        nargs="*",
        default=list(DEFAULT_COUNTS),
        help="Worker counts to try.",
    )
    parser.add_argument("--root", type=Path, default=Path(), help="Repository root.")
    args = parser.parse_args()

    root = args.root
    results: list[tuple[int, float]] = []

    for workers in args.workers:
        config = ExperimentConfig.from_files(
            root / "config" / "data" / "stills_all.yaml",
            root / "config" / "model" / "blinkcnn.yaml",
            root / "config" / "train" / "frame_wise.yaml",
            {"data.num_workers": workers, "data.persistent_workers": workers > 0},
        )
        # `worker_count` clamps every request to MAX_WORKERS, which is what this
        # benchmark exists to re-evaluate -- so it is raised here rather than in
        # the library, where changing it before knowing the answer would be the
        # assumption this is meant to replace.
        datamodule.MAX_WORKERS = workers

        module = BlinkDataModule(
            config.data, config.data.resolve_specs(root / "config" / "data"), root=root
        )
        module.setup("fit")

        rate = throughput(module, args.batches)
        results.append((workers, rate))
        print(f"  num_workers={workers:2d}  {rate:6.2f} batches/s", flush=True)

    best, best_rate = max(results, key=lambda row: row[1])
    baseline = dict(results).get(0, best_rate)
    print()
    print(f"fastest: num_workers={best} at {best_rate:.2f} batches/s")
    if baseline > 0:
        print(f"  {best_rate / baseline:.2f}x the in-process rate")


if __name__ == "__main__":
    main()
