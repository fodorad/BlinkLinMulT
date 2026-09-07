"""Precomputed eye embeddings, for frozen-encoder runs.

A **frozen** encoder computes the identical 256-d vector for a given crop on
every epoch: the weights are fixed and so is the input, so every epoch after the
first recomputes a constant. Caching that turns the expensive part of a batch
into a lookup.

Measured on this benchmark: a batch is ``batch_size x 45`` crops through
ConvNeXt, costing **0.74 s** with a frozen encoder against milliseconds from
cache, and 3.4 GB of embeddings replaces 51 GB of crops -- small enough to sit
in RAM, which removes the I/O question along with the compute one.

**Two preconditions, both refused in config rather than warned about**
(:class:`~blinklinmult.train.config.DataConfig`):

* ``model.encoder_freeze`` -- an encoder that is still learning produces
  different embeddings after every step, so a cache is stale from step one.
* ``data.augment`` must be ``None`` -- augmentation perturbs the crop *before*
  the encoder, so caching would freeze one augmented view per window for the
  whole run, keeping the distortion and losing the variety.

Either mistake trains against embeddings from a model that no longer exists and
reports a plausible, wrong number. That is the same failure class as the
carrier-corpus contamination and the write-only quality signals found earlier in
this project, so the checks here are hard errors that name the mismatched field.

**The cache is keyed by the encoder's own weights.** ``encoder_digest`` hashes
the state dict, and the digest is part of the directory path, so pointing
``model.encoder_weights`` at a different arm misses the cache and rebuilds
rather than silently reusing the wrong vectors.

Caches are **not** deleted after a run. One cache serves every frozen arm that
shares a seed checkpoint -- deleting it between arms would rebuild it each time,
at a full forward pass over the corpus per rebuild, to reclaim 3.4 GB against
the corpora's 51 GB. ``make clean-cache`` removes them when that is actually
wanted.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)
"""Module-level logger."""

MANIFEST_NAME = "manifest.json"
"""Per-shard sidecar describing what the cache holds and what built it."""

EMBEDDING_DTYPE = np.float16
"""Stored precision.

Half the size at no measurable cost: these are inputs to a ``d_model=32``
transformer, far above float16's resolution. The equivalence test in
``tests/data/test_embeddings.py`` asserts the round trip stays within tolerance
rather than assuming it.
"""


class EmbeddingCacheError(RuntimeError):
    """Raised when a cache cannot be used for the run that asked for it."""


@dataclass(frozen=True)
class CacheKey:
    """Everything that determines a cache's contents.

    Compared field by field on load, so a mismatch names the field that changed
    rather than failing obscurely later. A cache that silently disagreed with
    its run would train on the wrong vectors and report a believable number.

    Attributes:
        encoder_digest (str): Hash of the encoder's ``state_dict``.
        backbone (str): Backbone name, for a readable error.
        output_dim (int): Embedding width.
        image_size (int): Crop size the encoder was fed.
    """

    encoder_digest: str
    backbone: str
    output_dim: int
    image_size: int

    def as_dict(self) -> dict[str, object]:
        """Serialise for the manifest.

        Returns:
            dict[str, object]: JSON-safe fields.
        """
        return {
            "encoder_digest": self.encoder_digest,
            "backbone": self.backbone,
            "output_dim": self.output_dim,
            "image_size": self.image_size,
        }


def encoder_digest(encoder: torch.nn.Module) -> str:
    """Hash an encoder's parameters, to key a cache by the weights that built it.

    Parameters are hashed in sorted name order so the digest is stable across
    runs, and on the CPU in float32 so it does not depend on the device or the
    autocast state a run happened to use.

    Args:
        encoder (torch.nn.Module): The eye encoder.

    Returns:
        str: Hex SHA-256 of the parameter tensors.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(encoder.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().to("cpu", torch.float32).numpy().tobytes())
    return digest.hexdigest()


def shard_path(root: Path | str, key: CacheKey, corpus: str, subset: str) -> Path:
    """Where one corpus-split's embeddings live.

    The digest is a path component, not merely a manifest field: a different
    encoder lands in a different directory, so the wrong cache cannot be picked
    up even if a manifest check were somehow skipped.

    Args:
        root (Path | str): Cache root from ``data.embedding_cache``.
        key (CacheKey): Identity of the encoder that built it.
        corpus (str): Corpus name.
        subset (str): Split name.

    Returns:
        Path: The shard's ``.h5`` path.
    """
    return Path(root) / key.encoder_digest[:12] / corpus / f"{subset}.h5"


def write_shard(
    path: Path,
    key: CacheKey,
    keys: list[str],
    embeddings: np.ndarray,
    masks: np.ndarray,
) -> None:
    """Write one corpus-split's embeddings and its manifest.

    Written to a temporary file and renamed, so an interrupted build leaves no
    half-written shard that a later run would trust.

    Args:
        path (Path): Destination, from :func:`shard_path`.
        key (CacheKey): Identity to record.
        keys (list[str]): Sample ids, in row order.
        embeddings (np.ndarray): ``(N, T, D)``.
        masks (np.ndarray): ``(N, T)`` bool.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(".partial")
    with h5py.File(staging, "w") as handle:
        handle.create_dataset("embedding", data=embeddings.astype(EMBEDDING_DTYPE))
        handle.create_dataset("mask", data=masks.astype(bool))
        handle.create_dataset("keys", data=np.array(keys, dtype=h5py.string_dtype()))
    staging.rename(path)

    manifest = dict(key.as_dict())
    # Corpus and split are recorded as well as encoded in the path, so a shard
    # describes itself: a file copied out of its directory, or inspected without
    # the tree around it, still says what it holds.
    manifest.update(
        corpus=path.parent.name,
        subset=path.stem,
        windows=len(keys),
        built=datetime.now(UTC).isoformat(),
    )
    (path.parent / f"{path.stem}.{MANIFEST_NAME}").write_text(json.dumps(manifest, indent=2))
    logger.info(f"Cached {len(keys)} windows to {path}.")


def read_shard(path: Path, key: CacheKey) -> dict[str, np.ndarray]:
    """Load a shard, refusing one built by a different encoder or config.

    Args:
        path (Path): The shard.
        key (CacheKey): What this run requires.

    Returns:
        dict[str, np.ndarray]: ``keys``, ``embedding`` and ``mask``.

    Raises:
        EmbeddingCacheError: If the shard or its manifest is missing, or any
            identity field disagrees with ``key``.
    """
    manifest_path = path.parent / f"{path.stem}.{MANIFEST_NAME}"
    if not path.is_file() or not manifest_path.is_file():
        raise EmbeddingCacheError(f"No embedding cache at {path}. Build it first.")

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise EmbeddingCacheError(f"{manifest_path} is unreadable: {error}.") from error

    for field, expected in key.as_dict().items():
        found = manifest.get(field)
        if found != expected:
            raise EmbeddingCacheError(
                f"{path} was built with {field}={found!r}, but this run needs "
                f"{expected!r}. The cache describes a different encoder or input size; "
                "rebuild it rather than reusing embeddings from another model."
            )

    with h5py.File(path, "r") as handle:
        return {
            "keys": np.array([k.decode() for k in handle["keys"][:]]),
            "embedding": handle["embedding"][:],
            "mask": handle["mask"][:],
        }


def describe(root: Path | str) -> list[dict[str, object]]:
    """Every shard under a cache root, with what it holds.

    Answers "which corpora and splits are cached, and which encoder built
    them?" without reading the embedding arrays -- the manifests alone carry
    it. Sorted by encoder digest, then corpus, then split, so several encoders'
    caches under one root stay visually grouped.

    Args:
        root (Path | str): Cache root, as passed to ``data.embedding_cache``.

    Returns:
        list[dict[str, object]]: One manifest per shard, each with an added
        ``path``. Empty when the root does not exist or holds no shard.
    """
    base = Path(root)
    if not base.is_dir():
        return []

    found: list[dict[str, object]] = []
    for manifest_path in sorted(base.rglob(f"*.{MANIFEST_NAME}")):
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, ValueError):
            logger.warning(f"Skipping unreadable manifest {manifest_path}.")
            continue
        shard = manifest_path.parent / f"{manifest_path.name.split('.')[0]}.h5"
        manifest["path"] = str(shard)
        manifest["present"] = shard.is_file()
        found.append(manifest)

    return sorted(
        found,
        key=lambda m: (
            str(m.get("encoder_digest", "")),
            str(m.get("corpus", "")),
            str(m.get("subset", "")),
        ),
    )
