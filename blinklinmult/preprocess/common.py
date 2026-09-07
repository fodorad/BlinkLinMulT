"""Shared machinery for the per-corpus preprocess scripts.

Each corpus has its own script under :mod:`blinklinmult.preprocess` that knows
how *that* corpus stores its images and labels. Everything downstream of that
knowledge is the same for all six, and lives here: cropping an eye from a frame,
writing a window's arrays to disk, assigning splits, and emitting the
``manifest.jsonl`` that :mod:`blinklinmult.data.builder` consumes.

**Split assignment is by recording, never by window.** Windows cut from one
recording overlap and share a subject; splitting them at random puts near-copies
of the same frames on both sides of the train/test boundary and inflates every
score. :func:`assign_splits` therefore hashes the *group* key — the recording or
the participant — so a group lands wholly in one split. This is the single most
important correctness property of the preprocess layer, and it is what the 1.x
code, which had no split logic at all, left to whoever ran the training script.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np

from blinklinmult import PROJECT_ROOT
from blinklinmult.data.schema import (
    IMAGE_CHANNELS,
    SUBSETS,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

DEFAULT_SPLIT_RATIOS: dict[str, float] = {"train": 0.7, "valid": 0.15, "test": 0.15}
"""Default proportion of *groups* assigned to each split."""

ARRAYS_DIRNAME = "arrays"
"""Subdirectory of ``data/processed/<name>`` holding the per-sample ``.npy`` files."""

MANIFEST_NAME = "manifest.jsonl"
"""Name of the manifest the builder reads."""


APPLEDOUBLE_PREFIX = "._"
"""Prefix macOS gives the resource-fork sidecar it writes beside a real file.

Unpacking an archive on macOS, or copying one through a non-HFS volume, leaves a
``._<name>`` file next to every real one. They are not images: they carry the
same extension and match the same glob, so an unfiltered listing of MRL-Eye
returns ~85k sidecars alongside ~85k images, and the first one reaching a parser
fails on a filename that was never meant to be read.
"""


LOG_PROGRESS_SECONDS = 30.0
"""Seconds between progress lines when stderr is not a terminal.

A build is normally started under ``nohup ... > log 2>&1``, where a bar that
redraws on every item fills the file with thousands of carriage returns. Held to
one line every half minute, the same bar stays readable in a log and still
answers "how far has it got?".
"""


class PreprocessError(RuntimeError):
    """Raised when a corpus cannot be preprocessed as configured."""


def progress(iterable, desc: str, total: int | None = None):
    """Wrap an iterable in a progress bar that also works in a log file.

    Args:
        iterable: The iterable to wrap.
        desc (str): Bar description.
        total (int | None): Item count, when the iterable has no length.

    Returns:
        The wrapped iterable.
    """
    from tqdm import tqdm

    on_terminal = sys.stderr.isatty()
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        mininterval=0.1 if on_terminal else LOG_PROGRESS_SECONDS,
        # A redrawing bar is unreadable in a file; one line per update is not.
        ascii=not on_terminal,
        dynamic_ncols=on_terminal,
    )


def subsample_evenly(items: list[Path], limit: int) -> list[Path]:
    """Take ``limit`` items spread across a list, not the first ``limit``.

    A corpus that stores its files in label order — MRL-Eye lists every
    closed-eye image before every open one — turns a head-of-list truncation
    into a single-class dataset. A smoke run then trains happily on data that
    contains one label, which is a far more expensive discovery later.

    Args:
        items (list[Path]): Ordered paths.
        limit (int): How many to keep. Non-positive keeps nothing.

    Returns:
        list[Path]: Evenly spaced items, in the original order.
    """
    if limit <= 0:
        return []
    if limit >= len(items):
        return items
    step = len(items) / limit
    return [items[int(index * step)] for index in range(limit)]


def list_files(directory: Path, pattern: str) -> list[Path]:
    """List real files matching a glob, skipping macOS sidecars.

    Args:
        directory (Path): Directory to search.
        pattern (str): Glob pattern, relative to ``directory``.

    Returns:
        list[Path]: Matching paths, sorted, without AppleDouble sidecars.
    """
    matches = list(directory.glob(pattern))
    found = sorted(p for p in matches if not p.name.startswith(APPLEDOUBLE_PREFIX))
    hidden = len(matches) - len(found)
    if hidden:
        logger.debug(f"{directory}: skipped {hidden} macOS sidecar files matching {pattern!r}.")
    return found


def crop_square(image: np.ndarray, centre_x: int, centre_y: int, size: int) -> np.ndarray:
    """Cut a square patch around a point, padding where it leaves the frame.

    The 1.x code clipped the crop box to the image bounds and returned whatever
    was left, producing crops of varying size that a later ``resize`` then
    stretched by varying amounts — an eye near the frame edge came out
    geometrically distorted relative to one in the middle. Padding instead keeps
    every crop square and every eye the same scale.

    Args:
        image (np.ndarray): Source image, ``(H, W, 3)``.
        centre_x (int): Patch centre, x.
        centre_y (int): Patch centre, y.
        size (int): Side length of the patch.

    Returns:
        np.ndarray: ``(size, size, 3)`` patch.

    Raises:
        PreprocessError: If the size is not positive or the image is not 3-channel.
    """
    if size < 1:
        raise PreprocessError(f"crop size must be >= 1, got {size}.")
    if image.ndim != 3 or image.shape[2] != IMAGE_CHANNELS:
        raise PreprocessError(
            f"expected an (H, W, {IMAGE_CHANNELS}) image, got shape {image.shape}."
        )

    half = size // 2
    x0, y0 = centre_x - half, centre_y - half
    patch = np.zeros((size, size, image.shape[2]), dtype=image.dtype)

    # Intersection of the desired box with the image, in both coordinate frames.
    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1 = min(x0 + size, image.shape[1])
    src_y1 = min(y0 + size, image.shape[0])

    if src_x1 <= src_x0 or src_y1 <= src_y0:
        # DEBUG, not WARNING: a face tracked past the frame edge makes this a
        # routine, sub-1% occurrence on MPEblink, and one line per eye per
        # frame buries anything that actually matters. Deciding whether a
        # barely-visible crop is usable belongs to the caller, which has the
        # frame size -- see `EyeBox.visible_fraction`.
        logger.debug(
            f"crop centred at ({centre_x}, {centre_y}) lies entirely outside a "
            f"{image.shape[1]}x{image.shape[0]} image; returning a blank patch."
        )
        return patch

    patch[src_y0 - y0 : src_y1 - y0, src_x0 - x0 : src_x1 - x0] = image[
        src_y0:src_y1, src_x0:src_x1
    ]
    return patch


def eye_centre(corners: Sequence[int] | np.ndarray) -> tuple[int, int]:
    """Centre point of an eye from its two annotated corners.

    Args:
        corners (Sequence[int] | np.ndarray): ``(x1, y1, x2, y2)``.

    Returns:
        tuple[int, int]: ``(x, y)`` centre.
    """
    corners = np.asarray(corners)
    return int((corners[0] + corners[2]) // 2), int((corners[1] + corners[3]) // 2)


def eye_span(corners: Sequence[int] | np.ndarray) -> float:
    """Corner-to-corner distance of an eye, used to scale its crop.

    Args:
        corners (Sequence[int] | np.ndarray): ``(x1, y1, x2, y2)``.

    Returns:
        float: Euclidean distance between the corners.
    """
    corners = np.asarray(corners, dtype=np.float64)
    return float(np.linalg.norm(corners[:2] - corners[2:]))


def normalise_image(image: np.ndarray) -> np.ndarray:
    """Scale an 8-bit image to ``[0, 1]`` floats in channel-first order.

    Args:
        image (np.ndarray): ``(H, W, 3)`` uint8 image.

    Returns:
        np.ndarray: ``(3, H, W)`` float32 in ``[0, 1]``.
    """
    return np.transpose(image.astype(np.float32) / 255.0, (2, 0, 1))


def stack_eye_window(frames: list[np.ndarray]) -> np.ndarray:
    """Assemble one eye's per-frame crops into a window array.

    Args:
        frames (list[np.ndarray]): Per-frame crops of a single eye, each
            ``(3, H, W)``.

    Returns:
        np.ndarray: ``(T, 3, H, W)`` float32.

    Raises:
        PreprocessError: If the window is empty or a frame has the wrong shape.
    """
    if not frames:
        raise PreprocessError("cannot stack an empty eye window.")

    shapes = {frame.shape for frame in frames}
    if len(shapes) != 1:
        raise PreprocessError(
            f"every frame of a window must have the same shape, got {sorted(shapes)}."
        )
    if frames[0].ndim != 3 or frames[0].shape[0] != IMAGE_CHANNELS:
        raise PreprocessError(
            f"each frame must be ({IMAGE_CHANNELS}, H, W), got {frames[0].shape}."
        )

    return np.stack(frames, axis=0).astype(np.float32)


def _check_ratios(ratios: dict[str, float]) -> None:
    """Validate split proportions.

    Args:
        ratios (dict[str, float]): Split proportions.

    Raises:
        PreprocessError: If a split is unknown, a value is negative, or the
            proportions do not sum to 1.
    """
    unknown = sorted(set(ratios) - set(SUBSETS))
    if unknown:
        raise PreprocessError(f"unknown splits in ratios: {unknown}.")
    if any(value < 0 for value in ratios.values()):
        raise PreprocessError(f"split ratios must be >= 0, got {ratios}.")
    total = sum(ratios.values())
    if abs(total - 1.0) > 1e-6:
        raise PreprocessError(f"split ratios must sum to 1.0, got {total} from {ratios}.")


def split_of(group: str, ratios: dict[str, float] | None = None, salt: str = "") -> str:
    """Deterministically assign one group to a split.

    Hashing rather than shuffling means a split assignment is reproducible
    without storing it, stable when new recordings are added to a corpus, and
    identical across machines — a seeded shuffle is none of those once the input
    list changes.

    Args:
        group (str): Recording or participant identifier.
        ratios (dict[str, float] | None): Split proportions. Defaults to
            :data:`DEFAULT_SPLIT_RATIOS`.
        salt (str): Mixed into the hash, so two corpora with coincidentally
            equal group names do not receive correlated assignments.

    Returns:
        str: One of :data:`~blinklinmult.data.schema.SUBSETS`.

    Raises:
        PreprocessError: If the ratios are not positive and do not sum to 1.
    """
    ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    _check_ratios(ratios)

    digest = hashlib.sha256(f"{salt}:{group}".encode()).digest()
    # The first eight bytes give ~19 significant digits, far more resolution
    # than any ratio needs.
    position = int.from_bytes(digest[:8], "big") / float(1 << 64)

    cumulative = 0.0
    for name in SUBSETS:
        cumulative += ratios.get(name, 0.0)
        if position < cumulative:
            return name
    return SUBSETS[-1]


def proportional_splits(
    groups: Iterable[str],
    ratios: dict[str, float] | None = None,
    seed: int = 42,
    salt: str = "",
) -> dict[str, str]:
    """Divide groups into splits by shuffling, hitting the ratios exactly.

    The counterpart to :func:`assign_splits`, and the right choice for a
    *fixed* corpus. Hashing assigns each group independently, so the resulting
    proportions are a random draw rather than a division: 8 EyeBlink8
    recordings at 70/15/15 came out 5/3/0, with no test split at all. Shuffling
    a sorted list under a fixed seed gives the requested counts exactly and is
    just as reproducible.

    What it gives up is stability under change. Add a recording and every group
    may be reassigned, where hashing would leave the existing ones alone. Use
    this where the corpus is published and closed; use :func:`assign_splits`
    where it may grow.

    Args:
        groups (Iterable[str]): Group identifiers — a recording, participant,
            or whatever unit must not span a split.
        ratios (dict[str, float] | None): Split proportions. Defaults to
            :data:`DEFAULT_SPLIT_RATIOS`.
        seed (int): Shuffle seed, so a rebuild reproduces the division.
        salt (str): Corpus name, for the log line.

    Returns:
        dict[str, str]: Group id to split name.

    Raises:
        PreprocessError: If the ratios are invalid or there are no groups.
    """
    import random

    ratios = dict(ratios or DEFAULT_SPLIT_RATIOS)
    _check_ratios(ratios)

    # Sorted before shuffling: a set's iteration order is not stable across
    # runs, which would make the "seeded" division reproducible in name only.
    unique = sorted(set(groups))
    if not unique:
        raise PreprocessError(f"{salt or 'dataset'}: no groups to split.")

    shuffled = list(unique)
    random.Random(seed).shuffle(shuffled)

    # Largest remainder, so the counts sum to the total exactly rather than
    # losing a group to rounding.
    total = len(shuffled)
    exact = {name: ratios.get(name, 0.0) * total for name in SUBSETS}
    counts = {name: int(value) for name, value in exact.items()}
    for name in sorted(SUBSETS, key=lambda n: exact[n] - counts[n], reverse=True):
        if sum(counts.values()) >= total:
            break
        counts[name] += 1

    assignment: dict[str, str] = {}
    index = 0
    for name in SUBSETS:
        for group in shuffled[index : index + counts[name]]:
            assignment[group] = name
        index += counts[name]

    logger.info(f"{salt or 'dataset'}: {total} groups split as {counts}")
    empty = [name for name, count in counts.items() if count == 0 and ratios.get(name, 0.0) > 0]
    if empty:
        logger.warning(
            f"{salt or 'dataset'}: splits {empty} received no group. With {total} groups "
            "the requested ratios cannot be met."
        )
    return assignment


def assign_splits(
    groups: Iterable[str], ratios: dict[str, float] | None = None, salt: str = ""
) -> dict[str, str]:
    """Assign every group to a split and log the resulting balance.

    Args:
        groups (Iterable[str]): Group identifiers.
        ratios (dict[str, float] | None): Split proportions.
        salt (str): Mixed into the hash, usually the corpus name.

    Returns:
        dict[str, str]: Group to split name.

    Raises:
        PreprocessError: If no group is given.
    """
    unique = sorted(set(groups))
    if not unique:
        raise PreprocessError("cannot assign splits with no groups.")

    assignment = {group: split_of(group, ratios, salt) for group in unique}

    counts = {name: sum(1 for v in assignment.values() if v == name) for name in SUBSETS}
    logger.info(f"{salt or 'dataset'}: {len(unique)} groups split as {counts}")

    empty = [name for name, count in counts.items() if count == 0]
    if empty:
        logger.warning(
            f"{salt or 'dataset'}: splits {empty} received no group. With "
            f"{len(unique)} groups this may be unavoidable; assign splits "
            "explicitly if the corpus has too few recordings to divide."
        )
    return assignment


def git_sha(root: Path = PROJECT_ROOT) -> str:
    """Return the current commit SHA, for traceability of a built file.

    A ``-dirty`` suffix marks a build made from an uncommitted tree, so such an
    artifact is never mistaken for one a clean checkout would reproduce.

    Args:
        root (Path): Repository root.

    Returns:
        str: The SHA, with a ``-dirty`` suffix where applicable, or
        ``"unknown"`` outside a git checkout.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha
