"""Read-only helpers for inspecting a built corpus.

Separated from the notebook that uses them so they can be tested: a ``# %%``
script runs its top-level cells on import, which makes it untestable as a
module. :mod:`notebooks.corpus_report` imports these and adds only plotting.

Nothing here writes to a corpus. It opens, reads, and measures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import h5py

SPLIT_ORDER: tuple[str, ...] = ("train", "valid", "test")
"""Splits in the order a report should present them."""


def populated_splits(handle: h5py.File) -> list[str]:
    """Splits that hold at least one sample, in conventional order.

    Not every corpus fills every split: TalkingFace is held out entirely as a
    test set, so anything hard-coded to ``train`` fails on it.

    Args:
        handle (h5py.File): Open corpus.

    Returns:
        list[str]: Populated split names.
    """
    return [name for name in SPLIT_ORDER if name in handle and len(handle[name])]


def sample_keys(handle: h5py.File, split: str, count: int, offset: int = 0) -> list[str]:
    """Keys spread evenly across a split, starting at an offset.

    Evenly rather than a head slice: the keys are sorted, so the first N are one
    recording and their statistics are not the corpus's. The offset is what lets
    an interactive browser show a *different* set on each re-run, wrapping around
    rather than running out.

    Args:
        handle (h5py.File): Open corpus.
        split (str): Which split.
        count (int): How many keys to return.
        offset (int): How many sets to skip.

    Returns:
        list[str]: Sample keys; empty when the split is.
    """
    keys = list(handle[split])
    if not keys or count < 1:
        return []
    picks = np.linspace(0, len(keys) - 1, count).astype(int) + offset * count
    return [keys[int(index) % len(keys)] for index in picks]


def read_field(handle: h5py.File, split: str, key: str, name: str) -> np.ndarray | None:
    """One field of one sample, float16 upcast to float32.

    Args:
        handle (h5py.File): Open corpus.
        split (str): Which split.
        key (str): Sample key.
        name (str): Field name.

    Returns:
        np.ndarray | None: The value, or ``None`` when this corpus omits the
        field -- which is normal, since corpora declare different subsets.
    """
    group = handle[split][key]
    if name not in group:
        return None
    value = group[name][()]
    if isinstance(value, np.ndarray) and value.dtype == np.float16:
        return value.astype(np.float32)
    return value


def gather_field(handle: h5py.File, split: str, name: str, limit: int = 4000) -> np.ndarray:
    """Every frame's value of one field across a split.

    Capped at ``limit`` samples so a 40 GB corpus stays interactive; the keys are
    spread evenly, so the cap costs coverage rather than representativeness.

    Args:
        handle (h5py.File): Open corpus.
        split (str): Which split.
        name (str): Field name.
        limit (int): Cap on samples read.

    Returns:
        np.ndarray: Flattened values, empty when the corpus omits the field.
    """
    keys = sample_keys(handle, split, min(limit, len(handle[split])))
    values = [read_field(handle, split, key, name) for key in keys]
    usable = [np.asarray(value).reshape(-1) for value in values if value is not None]
    return np.concatenate(usable) if usable else np.asarray([], dtype=np.float32)


def occlusion_share(yaw_degrees: np.ndarray, threshold: float) -> float:
    """Fraction of frames a yaw threshold would mask as self-occluded.

    Args:
        yaw_degrees (np.ndarray): Yaw per frame, in degrees.
        threshold (float): Absolute yaw above which the far eye is occluded.

    Returns:
        float: Share in ``[0, 1]``; ``0`` for an empty input.
    """
    values = np.asarray(yaw_degrees, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return 0.0
    return float((np.abs(values) > threshold).mean())


def count_events(handle: h5py.File, split: str, limit: int = 3000) -> int:
    """Distinct annotated blink events in a split.

    Counted as ``(recording, event id)`` pairs, because ids are numbered per
    recording and would otherwise collide across them.

    Args:
        handle (h5py.File): Open corpus.
        split (str): Which split.
        limit (int): Cap on samples read.

    Returns:
        int: Distinct event count; ``0`` when the corpus has no ``blink_ids``.
    """
    events: set[tuple[str, int]] = set()
    for key in sample_keys(handle, split, min(limit, len(handle[split]))):
        ids = read_field(handle, split, key, "blink_ids")
        if ids is None:
            continue
        video = read_field(handle, split, key, "video_id")
        name = video.decode() if isinstance(video, bytes) else str(video)
        events.update((name, int(value)) for value in np.asarray(ids).reshape(-1) if value >= 0)
    return len(events)
