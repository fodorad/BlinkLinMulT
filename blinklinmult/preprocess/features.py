"""Handcrafted eye descriptors and head pose, from exordium.

The cross-modal model reads a 160-d vector per eye per frame alongside the
image. This module produces it. **Every number comes from exordium** — nothing
here reimplements a landmark or a rotation; it locates the eyes, calls the
wrappers, and flattens their output in a fixed order.

**The 160 dimensions**

======================================  ====  ==========================
Source                                  Dims  From
======================================  ====  ==========================
``eye_region_landmarks``  (71, 2)        142  ``IrisWrapper``
``iris_landmarks``        (5, 2)          10  ``IrisWrapper``
``iris_diameters``        (2,)             2  ``IrisWrapper``
``eyelid_pupil_distances``(2,)            2  ``IrisWrapper``
``ear``                   scalar           1  ``IrisWrapper``
``[yaw, pitch, roll]``    (3,)             3  ``SixDRepNetWrapper``
======================================  ====  ==========================

157 + 3 = :data:`~blinklinmult.data.schema.EYE_FEATURE_DIM`. The order is fixed
by :data:`IRIS_FIELDS` and asserted on every call, because a vector whose
meaning shifts between corpora is worse than no vector at all.

**Head pose needs a face**, so only the video corpora supply this stream. CEW
and MRL-Eye train the frame-wise model, which reads crops alone — they set
``feature_dim: null`` and write no ``eye_feature``. See
:mod:`blinklinmult.preprocess.geometry` for how the eyes are located.

**Failures are masked, never faked.** A frame whose landmarker fails yields a
zero vector *and* a ``False`` mask entry; the loss skips it. Zero is a
meaningful value here — a frontal head pose, a closed eye — so writing it
without the mask would be a lie the model learns.

**What is here, and what is next door.** This module holds the parts that are
pure: the layout, the flattening, the arithmetic, the masking rules, and
:class:`FakeExtractor` — so all of it is unit-tested without downloading a
model. The exordium bindings that actually run the networks live in
:mod:`blinklinmult.preprocess.extractors`, which is imported only from the
preprocess entry points and never from the training path.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Protocol

import numpy as np

from blinklinmult.data.schema import EYE_FEATURE_DIM
from blinklinmult.preprocess.common import PreprocessError
from blinklinmult.preprocess.geometry import LEFT, RIGHT, EyeBox

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)
"""Module-level logger."""

MODEL_SPACE = 64.0
"""Side of the square space ``IrisWrapper`` reports landmarks in.

It resizes every crop to 64x64 internally and returns coordinates in *that*
space, not the source frame's — which is what makes the descriptors comparable
across corpora recorded at different resolutions.
"""

IRIS_FIELDS: tuple[tuple[str, int, float], ...] = (
    ("eye_region_landmarks", 142, MODEL_SPACE),
    ("iris_landmarks", 10, MODEL_SPACE),
    ("iris_diameters", 2, MODEL_SPACE),
    ("eyelid_pupil_distances", 2, MODEL_SPACE),
    ("ear", 1, 1.0),
)
"""``IrisWrapper.eye_to_feature``'s keys, their flattened widths, and divisors.

The concatenation order *is* the feature layout. Changing it silently
invalidates every built corpus, so it is declared once here and asserted
against the wrapper's actual output on every call.

**The divisors matter as much as the order.** Measured on TalkingFace, the raw
blocks span wildly different magnitudes:

======================  ===============
Block                   Raw range
======================  ===============
eye_region_landmarks    9.1 .. 55.9
iris_landmarks          22.0 .. 40.1
iris_diameters          12.3 .. 12.8
eyelid_pupil_distances  4.3 .. 6.8
ear                     0.36
head pose (degrees)     -8.5 .. 6.7
======================  ===============

Concatenated raw, the 152 pixel-space values would dominate the first linear
layer by two orders of magnitude over the EAR — the single most informative
number for eye closure. Every pixel-space block is therefore divided by
:data:`MODEL_SPACE`, putting it in ``[0, 1]``; the EAR is already a ratio and
passes through.
"""

IRIS_FEATURE_DIM = sum(width for _, width, _ in IRIS_FIELDS)
"""Dimensions contributed by the iris landmarker: 157."""

HEADPOSE_DIM = 3
"""Dimensions contributed by head pose: ``[yaw, pitch, roll]`` in degrees."""

POSE_SCALE = 90.0
"""Divisor bringing pose degrees into roughly ``[-1, 1]``.

A quarter turn is the practical limit of a face the detector still finds, so
90 degrees maps to 1.0 and the block sits alongside the normalised landmarks
rather than swamping them.
"""

if IRIS_FEATURE_DIM + HEADPOSE_DIM != EYE_FEATURE_DIM:  # pragma: no cover - import guard
    raise ImportError(
        f"feature layout is {IRIS_FEATURE_DIM} + {HEADPOSE_DIM} dims but the schema "
        f"declares {EYE_FEATURE_DIM}. One of them is wrong."
    )


class FeatureError(PreprocessError):
    """Raised when eye features cannot be extracted as configured."""


class EyeFeatureExtractor(Protocol):
    """What the preprocessing pipeline needs of a feature extractor.

    Declared as a protocol so the corpus scripts do not import exordium's
    multi-GB stack merely to be type-checked, and so the tests can substitute
    :class:`FakeExtractor` without patching.
    """

    def head_pose(self, frame: np.ndarray) -> np.ndarray:
        """Head rotation for one frame, in degrees.

        Args:
            frame (np.ndarray): Full frame, ``(H, W, 3)`` uint8 RGB.

        Returns:
            np.ndarray: ``(3,)`` ``[yaw, pitch, roll]`` in degrees.
        """
        ...

    def frame_features(
        self, frame: np.ndarray, boxes: dict[str, EyeBox | None]
    ) -> dict[str, tuple[np.ndarray, bool]]:
        """Describe both eyes of one frame.

        Args:
            frame (np.ndarray): Full frame, ``(H, W, 3)`` uint8 RGB.
            boxes (dict[str, EyeBox | None]): Crop boxes per eye side.

        Returns:
            dict[str, tuple[np.ndarray, bool]]: Per side, the ``(160,)``
            float32 descriptor and whether it is valid.
        """
        ...


def empty_feature() -> np.ndarray:
    """A descriptor standing in for one that could not be computed.

    Returns:
        np.ndarray: ``(160,)`` of zeros, float32. Only ever written alongside a
        ``False`` mask entry.
    """
    return np.zeros(EYE_FEATURE_DIM, dtype=np.float32)


def flatten_iris(features: dict) -> np.ndarray:
    """Concatenate ``eye_to_feature``'s dict into the fixed 157-d layout.

    Args:
        features (dict): Output of ``IrisWrapper.eye_to_feature``.

    Returns:
        np.ndarray: ``(157,)`` float32.

    Raises:
        FeatureError: If a declared key is absent or has an unexpected width,
            which means exordium's output changed and every built corpus would
            otherwise silently acquire a different feature layout.
    """
    parts = []
    for name, width, divisor in IRIS_FIELDS:
        if name not in features:
            raise FeatureError(
                f"IrisWrapper.eye_to_feature returned no {name!r}; expected keys "
                f"{[key for key, _, _ in IRIS_FIELDS]}."
            )
        value = np.asarray(_to_numpy(features[name]), dtype=np.float32).reshape(-1)
        if value.size != width:
            raise FeatureError(
                f"IrisWrapper.eye_to_feature gave {name!r} with {value.size} values, "
                f"expected {width}. The feature layout has changed upstream; every "
                "corpus would need rebuilding."
            )
        parts.append(value / divisor)
    return np.concatenate(parts)


def _to_numpy(value) -> np.ndarray:
    """Convert a torch tensor or scalar to a numpy array.

    Args:
        value: A tensor, array, or number.

    Returns:
        np.ndarray: The value as an array.
    """
    detach = getattr(value, "detach", None)
    if detach is not None:
        return detach().cpu().numpy()
    return np.asarray(value)


def assemble(iris: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """Join the iris block and the head pose into one descriptor.

    Args:
        iris (np.ndarray): ``(157,)`` from :func:`flatten_iris`.
        pose (np.ndarray): ``(3,)`` ``[yaw, pitch, roll]`` in degrees.

    Returns:
        np.ndarray: ``(160,)`` float32, with pose scaled by :data:`POSE_SCALE`.

    Raises:
        FeatureError: If either block has the wrong width.
    """
    iris = np.asarray(iris, dtype=np.float32).reshape(-1)
    pose = np.asarray(pose, dtype=np.float32).reshape(-1)
    if iris.size != IRIS_FEATURE_DIM:
        raise FeatureError(f"iris block is {iris.size}-d, expected {IRIS_FEATURE_DIM}.")
    if pose.size != HEADPOSE_DIM:
        raise FeatureError(f"pose block is {pose.size}-d, expected {HEADPOSE_DIM}.")
    return np.concatenate([iris, pose / POSE_SCALE]).astype(np.float32)


def stack_window(
    features: Sequence[tuple[np.ndarray, bool]],
) -> tuple[np.ndarray, np.ndarray]:
    """Stack a window's per-frame descriptors into the stored arrays.

    Args:
        features (Sequence[tuple[np.ndarray, bool]]): Per frame, the descriptor
            and its validity.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(T, 160)`` float32 descriptors and
        their ``(T,)`` bool mask.

    Raises:
        FeatureError: If the window is empty.
    """
    if not features:
        raise FeatureError("cannot stack an empty window of eye features.")
    values = np.stack([np.asarray(value, dtype=np.float32).reshape(-1) for value, _ in features])
    mask = np.asarray([valid for _, valid in features], dtype=bool)
    return values, mask


class FakeExtractor:
    """A deterministic stand-in for the exordium stack, for tests and smoke runs.

    Produces correctly shaped descriptors from the crop's pixels alone, so the
    whole pipeline can be exercised — shapes, masks, stacking, the h5 layout —
    without downloading model weights or requiring a GPU.

    Args:
        fail_sides (tuple[str, ...]): Eye sides to report as invalid, for
            exercising the masking path.
    """

    def __init__(self, fail_sides: tuple[str, ...] = ()):
        self.fail_sides = tuple(fail_sides)
        self.calls = 0

    def head_pose(self, frame: np.ndarray) -> np.ndarray:
        """Head rotation for one frame, in degrees.

        Derived from the frame's mean intensity so it is deterministic and
        varies between frames, which is what a test binning by yaw needs from a
        stand-in -- a constant would put every frame in one bin.

        Args:
            frame (np.ndarray): Full frame, ``(H, W, 3)`` uint8 RGB.

        Returns:
            np.ndarray: ``(3,)`` ``[yaw, pitch, roll]`` in degrees.
        """
        level = float(np.asarray(frame, dtype=np.float64).mean())
        return np.asarray([level % 90.0 - 45.0, 0.0, 0.0], dtype=np.float32)

    def frame_features(
        self, frame: np.ndarray, boxes: dict[str, EyeBox | None]
    ) -> dict[str, tuple[np.ndarray, bool]]:
        """Describe both eyes of one frame.

        Args:
            frame (np.ndarray): Full frame, ``(H, W, 3)`` uint8 RGB.
            boxes (dict[str, EyeBox | None]): Crop boxes per eye side.

        Returns:
            dict[str, tuple[np.ndarray, bool]]: Descriptor and validity per side.
        """
        self.calls += 1
        result: dict[str, tuple[np.ndarray, bool]] = {}
        for side in (LEFT, RIGHT):
            box = boxes.get(side)
            if box is None or side in self.fail_sides:
                result[side] = (empty_feature(), False)
                continue
            patch = box.crop(frame)
            value = np.full(EYE_FEATURE_DIM, float(patch.mean()) / 255.0, dtype=np.float32)
            result[side] = (value, True)
        return result
