"""Prepare a trained v2 checkpoint for publication.

A Lightning checkpoint carries optimizer state, LR schedules, loop counters and
callback state -- everything needed to *resume* training, and nothing needed to
run the model. On the shipped frame-wise arm that is two thirds of the file: 57.4
MB in, 19.1 MB out.

It also writes the ``.json`` sidecar that travels with the weights, recording the
input contract, the normalisation the model expects, and its **validation-fitted
operating point**. That last part matters: the hysteresis pair
(``threshold=0.53``, ``low_ratio=0.25``) scores 0.52 event F1 where the
single-threshold variant of the same run scores 0.19, so publishing the weights
without it would quietly halve event quality for anyone using the defaults.

Run it through the Makefile, which supplies the shipped arm's path::

    make export-blinkcnn

or directly, naming a checkpoint::

    uv run python tools/export_blinkcnn.py --checkpoint <path.ckpt> --out artifacts/onnx
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from blinklinmult.models import load_checkpoint, strip_checkpoint
from blinklinmult.registry import spec

logger = logging.getLogger(__name__)
"""Module-level logger."""

MODEL_ID = "blinkcnn"
"""Registry id this script publishes."""


def export(checkpoint: Path, out_dir: Path, model_id: str = MODEL_ID) -> Path:
    """Strip a checkpoint and write it with its sidecar.

    Args:
        checkpoint (Path): The trained Lightning checkpoint.
        out_dir (Path): Directory to write into.
        model_id (str): Registry id supplying the published metadata.

    Returns:
        Path: The written weights file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{model_id}.pt"

    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.save(strip_checkpoint(raw), destination)

    before = checkpoint.stat().st_size / 1048576
    after = destination.stat().st_size / 1048576
    logger.info(f"{model_id}: {before:.1f} MB -> {after:.1f} MB")

    # Load the stripped file back before publishing it. A file that cannot be
    # rebuilt is worse than no file, and this is the last chance to notice.
    load_checkpoint(destination)
    logger.info(f"{model_id}: stripped checkpoint loads and rebuilds cleanly")

    entry = spec(model_id)
    sidecar = {
        "model_id": entry.model_id,
        "generation": entry.generation,
        "runtime": "pytorch",
        "image_size": entry.image_size,
        "needs_features": entry.needs_features,
        "normalisation": {"mean": list(entry.mean), "std": list(entry.std)},
        "operating_point": {
            "threshold": entry.threshold,
            "low_ratio": entry.low_ratio,
            "low_threshold": (
                None if entry.low_ratio is None else entry.threshold * entry.low_ratio
            ),
        },
        "source_checkpoint": checkpoint.name,
        "note": (
            "Stripped of optimizer and loop state; inference only. The operating "
            "point was fitted on validation, not chosen by hand."
        ),
    }
    (out_dir / f"{model_id}.pt.json").write_text(json.dumps(sidecar, indent=2) + "\n")
    logger.info(f"{model_id}: wrote sidecar")
    return destination


def main() -> None:
    """Strip and describe one checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trained .ckpt to publish.")
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/onnx"), help="Output directory."
    )
    parser.add_argument("--model-id", default=MODEL_ID, help="Registry id for the metadata.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    export(args.checkpoint, args.out, args.model_id)


if __name__ == "__main__":
    main()
