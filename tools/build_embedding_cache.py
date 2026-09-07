r"""Precompute eye embeddings for a frozen encoder.

A frozen encoder returns the identical vector for a given crop on every epoch,
so every epoch after the first recomputes a constant. This runs the encoder once
over every corpus and split and stores the result, turning the expensive part of
a training batch into a lookup: measured 0.74 s/batch with a frozen encoder
against milliseconds from cache.

**Only valid for a frozen encoder with augmentation off.** Both are refused in
config (:class:`~blinklinmult.train.config.DataConfig`); this script assumes the
caller has honoured them, and keys every shard by the encoder's own weights so a
mismatch is caught on load rather than trusted.

Run::

    uv run python tools/build_embedding_cache.py \\
      --encoder results/frame-wise/fw-focal-augment/checkpoints/best.ckpt \\
      --root cache/embeddings

Resumable per corpus and split: an interrupted build skips what it already
wrote, so a machine shut down mid-build does not start over.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import torch

from blinklinmult.data.embeddings import (
    CacheKey,
    encoder_digest,
    shard_path,
    write_shard,
)
from blinklinmult.train.config import DataConfig, ExperimentConfig
from blinklinmult.train.model import BlinkModel

logger = logging.getLogger(__name__)
"""Module-level logger."""

BATCH = 32
"""Windows per forward pass while building.

Larger than the training batch: this is inference under ``no_grad``, so there
are no stored activations and the memory ceiling that forces batch 8 in training
does not apply.
"""


def build_split(
    model: BlinkModel,
    corpus: str,
    subset: str,
    h5_path: Path,
    key: CacheKey,
    root: Path,
    device: str,
) -> None:
    """Encode one corpus-split and write its shard.

    Args:
        model (BlinkModel): Model whose encoder does the work.
        corpus (str): Corpus name.
        subset (str): Split name.
        h5_path (Path): The built corpus.
        key (CacheKey): Cache identity.
        root (Path): Cache root.
        device (str): Torch device.
    """
    destination = shard_path(root, key, corpus, subset)
    if destination.is_file():
        logger.info(f"{corpus}/{subset}: already cached, skipping.")
        return

    with h5py.File(h5_path, "r") as handle:
        if subset not in handle:
            logger.info(f"{corpus}: no {subset!r} split; skipping.")
            return
        sample_ids = sorted(handle[subset].keys())
        if not sample_ids:
            logger.info(f"{corpus}: {subset!r} split is empty; skipping.")
            return

        embeddings: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        for start in range(0, len(sample_ids), BATCH):
            chunk = sample_ids[start : start + BATCH]
            images = np.stack(
                [np.asarray(handle[subset][k]["eye_image"][:], dtype=np.float32) for k in chunk]
            )
            mask = np.stack(
                [np.asarray(handle[subset][k]["eye_image_mask"][:], dtype=bool) for k in chunk]
            )
            with torch.no_grad():
                encoded = model.encoder(torch.from_numpy(images).to(device))
            embeddings.append(encoded.cpu().numpy())
            masks.append(mask)
            if start % (BATCH * 20) == 0:
                logger.info(f"  {corpus}/{subset}: {start + len(chunk)}/{len(sample_ids)}")

    write_shard(
        destination,
        key,
        sample_ids,
        np.concatenate(embeddings),
        np.concatenate(masks),
    )


def main() -> None:
    """Build the cache for every corpus a data config names."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, type=Path, help="Frame-wise checkpoint.")
    parser.add_argument("--root", type=Path, default=Path("cache/embeddings"))
    parser.add_argument("--data", type=Path, default=Path("config/data/video_all.yaml"))
    parser.add_argument("--model", type=Path, default=Path("config/model/blinklint.yaml"))
    parser.add_argument("--train", type=Path, default=Path("config/train/video.yaml"))
    parser.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = ExperimentConfig.from_files(
        args.data,
        args.model,
        args.train,
        {
            "model.encoder_weights": str(args.encoder),
            "model.encoder_freeze": True,
            "data.augment": None,
        },
    )
    model = BlinkModel(
        config.model,
        target_names=list(config.train.targets),
        image_size=config.data.image_size,
    ).to(args.device)
    model.eval()

    key = CacheKey(
        encoder_digest=encoder_digest(model.encoder),
        backbone=config.model.backbone,
        output_dim=config.model.backbone_output_dim,
        image_size=config.data.image_size,
    )
    logger.info(f"Encoder digest {key.encoder_digest[:12]}; writing under {args.root}.")

    data = DataConfig.from_dict(
        {**config.data.__dict__, "augment": None},
    )
    corpora = sorted({*data.datasets, *data.eval_datasets})
    for corpus in corpora:
        h5_path = Path("data/processed") / corpus / f"{corpus}.h5"
        if not h5_path.is_file():
            logger.warning(f"{corpus}: not built at {h5_path}; skipping.")
            continue
        # Test is cached too: it is read once per arm, and a frozen encoder
        # produces the same embeddings there as anywhere else. Caching it does
        # not subsample or otherwise alter the split.
        for subset in ("train", "valid", "test"):
            build_split(model, corpus, subset, h5_path, key, args.root, args.device)

    logger.info("Embedding cache complete.")


if __name__ == "__main__":
    main()
