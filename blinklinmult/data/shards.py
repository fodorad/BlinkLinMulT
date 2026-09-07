"""Per-clip shards, so a long corpus build can be resumed rather than restarted.

MPEblink decodes 1 842 untrimmed videos and takes the better part of a day. A
single-file build has no notion of progress: :class:`~blinklinmult.data.writer.H5Writer`
opens its temporary file with mode ``"w"``, so an interrupted run leaves a
partial file that is indistinguishable from a complete one except by sample
count, and the next run starts from zero.

Sharding fixes that by making the **clip** the unit of durability. Each clip
writes its own small HDF5 file; a clip whose shard already exists is skipped, and
the shards are merged into the corpus file once every clip has one. A shard is
only renamed into place after it closes cleanly, so a shard that exists is a clip
that finished -- there is no partial-shard ambiguity to reason about.

The cost is one extra pass over the data at merge time, which is I/O-bound and
minutes rather than hours.
"""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING

import h5py

from blinklinmult.data.schema import SUBSETS

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

SHARD_DIRNAME = "shards"
"""Directory, beside the corpus file, holding one HDF5 per completed clip."""


def shard_dir(h5_path: Path) -> Path:
    """Where a corpus's shards live.

    Args:
        h5_path (Path): The corpus's final ``.h5``.

    Returns:
        Path: The shard directory, which may not exist yet.
    """
    return h5_path.parent / SHARD_DIRNAME


def shard_path(h5_path: Path, clip_id: str) -> Path:
    """The shard for one clip.

    Args:
        h5_path (Path): The corpus's final ``.h5``.
        clip_id (str): Identifier unique within the corpus, e.g. ``test_1``.

    Returns:
        Path: The shard's path.
    """
    return shard_dir(h5_path) / f"{clip_id}.h5"


def completed_clips(h5_path: Path) -> set[str]:
    """Which clips already have a shard.

    A shard is renamed into place only after it closes cleanly, so its presence
    means that clip finished. A run therefore resumes by skipping these rather
    than by trusting a progress counter that a crash could have left stale.

    Args:
        h5_path (Path): The corpus's final ``.h5``.

    Returns:
        set[str]: Clip identifiers already built.
    """
    directory = shard_dir(h5_path)
    if not directory.is_dir():
        return set()
    return {path.stem for path in directory.glob("*.h5")}


def merge_shards(h5_path: Path, attrs: dict, remove: bool = True) -> int:
    """Combine every shard into the corpus file.

    Samples are copied group by group, which preserves each dataset's
    compression and dtype exactly -- re-encoding here would silently change what
    a shard already got right.

    Args:
        h5_path (Path): The corpus's final ``.h5``.
        attrs (dict): Root attributes for the merged file, as the shards'
            writer would have written them.
        remove (bool): Delete the shard directory once merged. ``False`` keeps
            it, which is what a debugging run wants.

    Returns:
        int: Samples written.

    Raises:
        FileNotFoundError: If there are no shards to merge.
    """
    directory = shard_dir(h5_path)
    shards = sorted(directory.glob("*.h5")) if directory.is_dir() else []
    if not shards:
        raise FileNotFoundError(f"no shards under {directory}.")

    tmp_path = h5_path.with_suffix(h5_path.suffix + ".tmp")
    written = 0
    with h5py.File(tmp_path, "w") as target:
        for key, value in attrs.items():
            target.attrs[key] = value
        for subset in SUBSETS:
            target.create_group(subset)

        for shard in shards:
            with h5py.File(shard, "r") as source:
                for subset in SUBSETS:
                    if subset not in source:
                        continue
                    for sample_id in source[subset]:
                        source.copy(f"{subset}/{sample_id}", target[subset], name=sample_id)
                        written += 1

    tmp_path.replace(h5_path)
    logger.info(f"merged {len(shards)} shards into {h5_path} ({written} samples)")

    if remove:
        shutil.rmtree(directory)
    return written


def shard_attrs(h5_path: Path) -> dict:
    """Root attributes from any one shard, for the merged file.

    Every shard is written by the same writer from the same declaration, so they
    agree; reading the first avoids reconstructing them at the merge site and
    lets them drift.

    Args:
        h5_path (Path): The corpus's final ``.h5``.

    Returns:
        dict: The attributes.

    Raises:
        FileNotFoundError: If there are no shards.
    """
    shards = sorted(shard_dir(h5_path).glob("*.h5"))
    if not shards:
        raise FileNotFoundError(f"no shards under {shard_dir(h5_path)}.")
    with h5py.File(shards[0], "r") as handle:
        return dict(handle.attrs)


def pending(entries: Iterator, h5_path: Path, clip_id) -> list:
    """The entries whose clips have no shard yet.

    Args:
        entries (Iterator): Everything the build would process.
        h5_path (Path): The corpus's final ``.h5``.
        clip_id: Callable mapping an entry to its clip identifier.

    Returns:
        list: The entries still to do, in the original order.
    """
    done = completed_clips(h5_path)
    remaining = [entry for entry in entries if clip_id(entry) not in done]
    if done:
        logger.info(f"resuming: {len(done)} clips already built, {len(remaining)} to go")
    return remaining
