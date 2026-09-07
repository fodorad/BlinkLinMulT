"""List what an embedding cache holds.

Answers "which corpora and splits are cached, and which encoder built them?"
without reading the embedding arrays.

Run::

    uv run python tools/show_embedding_cache.py
    uv run python tools/show_embedding_cache.py --root cache/embeddings
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from blinklinmult.data.embeddings import describe

logger = logging.getLogger(__name__)
"""Module-level logger."""


def main() -> None:
    """Print one row per cached corpus-split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("cache/embeddings"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    shards = describe(args.root)
    if not shards:
        logger.info(
            f"No embedding cache under {args.root}. Build one with:\n"
            "  uv run python tools/build_embedding_cache.py "
            "--encoder results/frame-wise/<arm>/checkpoints/best.ckpt"
        )
        return

    logger.info(f"{'encoder':14s} {'corpus':14s} {'split':7s} {'windows':>9s}  built")
    total = 0
    for shard in shards:
        digest = str(shard.get("encoder_digest", ""))[:12]
        windows = int(shard.get("windows", 0))
        total += windows
        flag = "" if shard.get("present") else "  [MISSING .h5]"
        logger.info(
            f"{digest:14s} {shard.get('corpus', '?'):14s} {shard.get('subset', '?'):7s} "
            f"{windows:9,d}  {str(shard.get('built', ''))[:19]}{flag}"
        )
    logger.info(f"\n{len(shards)} shards, {total:,} windows.")


if __name__ == "__main__":
    main()
