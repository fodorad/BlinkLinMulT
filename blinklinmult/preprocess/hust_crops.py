"""Locate HUST-LEBW's pre-cut eye crops inside their source frames.

The corpus ships both the full 1920x800 frames **and** per-eye crops cut from
them, in ``you`` (right) and ``zuo`` (left) directories. The crops are pixel
copies rather than resamples, so their position in the frame is recoverable
exactly by template matching -- measured at a correlation of **1.0000** on every
crop of a sample clip, with crop *N* matching frame *N* one-to-one.

That recovers what the annotation does not state: **where** each eye is. Two
things follow.

**The crop identifies the subject's face.** A film frame holds several people
and only one is annotated; the face box containing the located eye is the
subject's, and the rest are bystanders. That is a far more direct answer than
detecting every face and guessing from an annotated midpoint.

**A crop is too tight to describe.** The shipped crops frame the eye alone,
which is enough to see a lid but not enough for the eyelid-contour landmarks the
descriptor needs. Re-cutting the same centre at :data:`RECROP_SCALE` times the
size restores the surrounding context.

**A missing directory means an occluded eye.** Where an eye was not visible the
corpus ships no crop directory at all -- ``train/unblink/100`` has ``zuo`` and no
``you``. Nothing is inferred there; the eye simply has no sample.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from blinklinmult.preprocess.common import PreprocessError

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

logger = logging.getLogger(__name__)
"""Module-level logger."""

EYE_DIRECTORIES: dict[str, str] = {"left": "zuo", "right": "you"}
"""Which directory holds which eye. ``zuo`` is left, ``you`` is right."""

CROP_PATTERN = re.compile(r"^(\d+)eye_(?:you|zuo)\.bmp$")
"""Crop filenames: a 1-based index, the eye, and the extension."""

RECROP_SCALE = 2.0
"""How much wider than the shipped crop to re-cut the eye.

The shipped crops frame the eye alone -- tight enough to read a lid, too tight
for the eyelid-contour landmarks a descriptor needs, which want the brow and the
socket corner as context. Doubling the box restores that without a scale
estimate: the corpus's own crop defines the eye's size, so the multiple is
relative to the eye rather than to a face box that may not exist.
"""

MATCH_THRESHOLD = 0.9
"""Correlation below which a crop is not considered located.

The crops are pixel copies, so a genuine match scores ~1.0; anything materially
below that means the crop did not come from this frame and locating it would be
a guess.
"""


class CropError(PreprocessError):
    """Raised when a shipped crop cannot be located in its frame."""


def crop_files(directory: Path) -> list[tuple[int, Path]]:
    """The eye crops in one directory, in frame order.

    Args:
        directory (Path): A ``you`` or ``zuo`` directory.

    Returns:
        list[tuple[int, Path]]: ``(index, path)`` sorted by index, where the
        index is 1-based and matches the clip's Nth frame. Empty when the
        directory is absent, which is how the corpus records an occluded eye.
    """
    if not directory.is_dir():
        return []

    found: list[tuple[int, Path]] = []
    for path in directory.iterdir():
        match = CROP_PATTERN.match(path.name)
        if match is not None:
            found.append((int(match.group(1)), path))
    return sorted(found)


def locate(frame: np.ndarray, crop: np.ndarray) -> tuple[int, int, float]:
    """Find where a crop was cut from its frame.

    Args:
        frame (np.ndarray): ``(H, W, 3)`` full frame.
        crop (np.ndarray): ``(h, w, 3)`` crop taken from it.

    Returns:
        tuple[int, int, float]: ``(x, y)`` of the crop's top-left corner in the
        frame, and the correlation the match scored.

    Raises:
        CropError: If the crop is larger than the frame, which cannot match.
    """
    import cv2

    if crop.shape[0] > frame.shape[0] or crop.shape[1] > frame.shape[1]:
        raise CropError(f"crop {crop.shape[:2]} is larger than its frame {frame.shape[:2]}.")

    result = cv2.matchTemplate(frame, crop, cv2.TM_CCOEFF_NORMED)
    _, score, _, location = cv2.minMaxLoc(result)
    return int(location[0]), int(location[1]), float(score)


def recrop(
    frame: np.ndarray, x: int, y: int, width: int, height: int, scale: float = RECROP_SCALE
) -> np.ndarray:
    """Re-cut a located crop at a wider scale, around the same centre.

    Clamped to the frame rather than padded: real pixels beat zeros, and an eye
    near the edge is better described by an off-centre window of the face than
    by a centred window half full of black.

    Args:
        frame (np.ndarray): ``(H, W, 3)`` full frame.
        x (int): Located crop's left edge.
        y (int): Its top edge.
        width (int): Its width.
        height (int): Its height.
        scale (float): How much wider to cut.

    Returns:
        np.ndarray: The wider crop.

    Raises:
        CropError: If the scale is not greater than zero.
    """
    if scale <= 0:
        raise CropError(f"scale must be positive, got {scale}.")

    centre_x, centre_y = x + width / 2.0, y + height / 2.0
    half_width, half_height = width * scale / 2.0, height * scale / 2.0

    left = int(round(max(0.0, centre_x - half_width)))
    top = int(round(max(0.0, centre_y - half_height)))
    right = int(round(min(float(frame.shape[1]), centre_x + half_width)))
    bottom = int(round(min(float(frame.shape[0]), centre_y + half_height)))
    return frame[top:bottom, left:right]


def containing_box(
    boxes: list[tuple[int, int, int, int]], x: int, y: int, width: int, height: int
) -> tuple[int, int, int, int] | None:
    """The face box the located eye sits inside.

    A film frame holds several people and only one is annotated. The eye's
    position says which face is the subject's, so the others can be ignored
    without guessing. Where boxes nest, the **smallest** containing one is the
    tightest fit and so the face rather than a group.

    Args:
        boxes (list[tuple[int, int, int, int]]): ``(x1, y1, x2, y2)`` per face.
        x (int): Eye crop's left edge.
        y (int): Its top edge.
        width (int): Its width.
        height (int): Its height.

    Returns:
        tuple[int, int, int, int] | None: The subject's face box, or ``None``
        when the eye falls outside every detection.
    """
    centre_x, centre_y = x + width / 2.0, y + height / 2.0
    containing = [
        box for box in boxes if box[0] <= centre_x <= box[2] and box[1] <= centre_y <= box[3]
    ]
    if not containing:
        return None
    return min(containing, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
