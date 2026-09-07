"""One pass over a recording: decode, extract, crop, write.

The video corpora are processed in a single sequential sweep per recording.
Frames are decoded once, their eyes located and described once, and the results
held in a small per-recording cache; the windows are then assembled from that
cache and written straight into the corpus's HDF5 file.

**Why not a window at a time.** Sliding evaluation windows overlap: measured on
a 300-frame recording with two blinks, a 15-frame window at stride 5 references
each frame 2.9 times. Extracting per window would run the face detector and the
landmarker three times over the same pixels, and those two calls are essentially
the whole runtime.

**Why not decode to PNGs first.** The 1.x pipeline wrote every annotated frame
to disk and read it back. That is a second copy of the corpus, and the decode
step held every frame of the recording in a Python list — about 8 GB for five
minutes at 640x480. Here the decode is streamed and only the crops survive:
a 64x64 float16 crop is 24 KB, so a 9000-frame recording caches ~430 MB for
both eyes rather than gigabytes of full frames.

**What is cached.** Per frame and eye: the resized crop and the 160-d
descriptor with its validity flag. Nothing else — the full frame is released as
soon as the crops are taken.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from blinklinmult.preprocess.common import PreprocessError, normalise_image, progress
from blinklinmult.preprocess.features import empty_feature
from blinklinmult.preprocess.geometry import LEFT, RIGHT, boxes_from_annotation
from blinklinmult.preprocess.quality import (
    blur_score,
    box_jitter,
    exposure_score,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from blinklinmult.preprocess.annotation import TagFile, TagRecord
    from blinklinmult.preprocess.features import EyeFeatureExtractor

logger = logging.getLogger(__name__)
"""Module-level logger."""

PROGRESS_EVERY = 500
"""Log a line every N frames, so a multi-hour build shows progress."""


class StreamError(PreprocessError):
    """Raised when a recording cannot be streamed."""


@dataclass
class FrameCache:
    """Per-frame crops and descriptors for one recording.

    Args:
        image_size (int): Side length the crops were resized to.
        feature_dim (int | None): Descriptor width, or ``None`` when this
            corpus supplies no handcrafted stream.
    """

    image_size: int
    feature_dim: int | None = None
    crops: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    features: dict[tuple[int, str], tuple[np.ndarray, bool]] = field(default_factory=dict)
    poses: dict[int, np.ndarray] = field(default_factory=dict)
    """Head rotation per frame, in degrees. One per frame, not per eye."""
    boxes: dict[tuple[int, str], tuple[float, float, float]] = field(default_factory=dict)
    """``(centre_x, centre_y, span)`` per eye, for jitter and symmetry."""

    def add(
        self,
        frame_id: int,
        side: str,
        crop: np.ndarray,
        feature: tuple[np.ndarray, bool] | None = None,
        pose: np.ndarray | None = None,
        box: tuple[float, float, float] | None = None,
    ) -> None:
        """Record one eye of one frame.

        Args:
            frame_id (int): The frame's id.
            side (str): Eye side.
            crop (np.ndarray): ``(C, H, W)`` normalised crop.
            feature (tuple[np.ndarray, bool] | None): Descriptor and validity.
            pose (np.ndarray | None): ``(3,)`` head rotation in degrees. Stored
                once per frame -- it describes the head, not the eye.
            box (tuple[float, float, float] | None): ``(centre_x, centre_y,
                span)`` of the eye box, which jitter and symmetry read.
        """
        self.crops[(frame_id, side)] = crop
        if feature is not None:
            self.features[(frame_id, side)] = feature
        if pose is not None:
            self.poses[frame_id] = np.asarray(pose, dtype=np.float32).reshape(-1)[:3]
        if box is not None:
            self.boxes[(frame_id, side)] = box

    def window_pose(self, frame_ids: list[int]) -> np.ndarray:
        """Head rotation across a window, in degrees.

        Args:
            frame_ids (list[int]): The window's frames, in order.

        Returns:
            np.ndarray: ``(T, 3)``. A frame with no estimate reports zeros --
            a frontal reading, which is the least-wrong default; the image mask
            already records that the frame was unreadable.
        """
        return np.stack(
            [self.poses.get(frame_id, np.zeros(3, dtype=np.float32)) for frame_id in frame_ids]
        ).astype(np.float32)

    def window_quality(self, frame_ids: list[int], side: str) -> dict[str, np.ndarray]:
        """Per-frame quality signals for one eye across a window.

        Computed here rather than at crop time because two of the signals are
        *temporal* -- jitter needs the previous frame's box, and symmetry needs
        the partner eye of the same frame.

        Args:
            frame_ids (list[int]): The window's frames, in order.
            side (str): Eye side.

        Returns:
            dict[str, np.ndarray]: Each signal as ``(T,)`` in ``[0, 1]``.
        """
        blur, exposure = [], []
        centres, spans = [], []

        for frame_id in frame_ids:
            crop = self.crops.get((frame_id, side))
            blur.append(0.0 if crop is None else blur_score(crop))
            exposure.append(0.0 if crop is None else exposure_score(crop))

            box = self.boxes.get((frame_id, side))
            centres.append((0.0, 0.0) if box is None else (box[0], box[1]))
            spans.append(0.0 if box is None else box[2])

        return {
            "eye_blur": np.asarray(blur, dtype=np.float32),
            "eye_exposure": np.asarray(exposure, dtype=np.float32),
            "eye_jitter": box_jitter(
                np.asarray(centres, dtype=np.float64), np.asarray(spans, dtype=np.float64)
            ).astype(np.float32),
        }

    def blank_crop(self) -> np.ndarray:
        """A crop standing in for a frame that could not be read.

        Returns:
            np.ndarray: ``(C, H, W)`` of zeros, float32.
        """
        return np.zeros((3, self.image_size, self.image_size), dtype=np.float32)

    def window_images(self, frame_ids: list[int], side: str) -> tuple[np.ndarray, np.ndarray]:
        """Stack one eye's crops for a window.

        Args:
            frame_ids (list[int]): The window's frames, in order.
            side (str): Eye side.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(T, C, H, W)`` crops and their
            ``(T,)`` validity mask.
        """
        images, mask = [], []
        for frame_id in frame_ids:
            crop = self.crops.get((frame_id, side))
            images.append(self.blank_crop() if crop is None else crop)
            mask.append(crop is not None)
        return np.stack(images), np.asarray(mask, dtype=bool)

    def window_features(self, frame_ids: list[int], side: str) -> tuple[np.ndarray, np.ndarray]:
        """Stack one eye's descriptors for a window.

        Args:
            frame_ids (list[int]): The window's frames, in order.
            side (str): Eye side.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(T, F)`` descriptors and their
            ``(T,)`` validity mask.

        Raises:
            StreamError: If this cache holds no descriptors.
        """
        if self.feature_dim is None:
            raise StreamError("this corpus supplies no handcrafted eye features.")

        values, mask = [], []
        for frame_id in frame_ids:
            entry = self.features.get((frame_id, side))
            if entry is None:
                values.append(empty_feature())
                mask.append(False)
            else:
                values.append(entry[0])
                mask.append(entry[1])
        return np.stack(values), np.asarray(mask, dtype=bool)


def frame_count(video_path: Path) -> int:
    """Count the frames a video actually decodes to.

    The container's frame-count metadata is not trusted: it is what disagrees
    with the annotation in the first place.

    Args:
        video_path (Path): The recording.

    Returns:
        int: Frames decoded.

    Raises:
        StreamError: If the video cannot be opened.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise StreamError(f"{video_path}: cannot be opened for decoding.")

    count = 0
    try:
        while capture.grab():
            count += 1
    finally:
        capture.release()
    return count


CLOCK_TOLERANCE = 0.05
"""How far the decoder's frame rate may differ from the annotation's.

A fraction: ``0.05`` allows 5%. Beyond it the two disagree about what a frame
*is*, and matching on time silently slides every label along the recording.
"""


def _is_contiguous(frame_ids: np.ndarray) -> bool:
    """Whether the ids name an unbroken run starting at the first decoded frame.

    Positional mapping walks the decoder and the annotation in step, so it is
    only correct when the annotation covers frames ``0, 1, 2, ...`` with no
    gaps. An annotation that skips frames — HUST-LEBW's sparse clips, or a
    recording annotated from the middle — must be matched some other way.

    Args:
        frame_ids (np.ndarray): Sorted annotated frame ids.

    Returns:
        bool: ``True`` when the ids run from 0 with a step of 1.
    """
    if frame_ids.size == 0:
        return False
    return bool(
        frame_ids[0] == 0 and np.array_equal(np.diff(frame_ids), np.ones(frame_ids.size - 1))
    )


def _check_clocks(video_path: Path, timestamps: dict[int, float], decoded_total: int) -> None:
    """Refuse to match on time when the two clocks disagree.

    Nearest-timestamp matching assumes the decoder and the annotator measure the
    same seconds. When they do not, every label slides a little further along
    the recording and the corpus is quietly wrong — RN's containers declare 30
    fps for footage annotated at ~16.7, which drifted 5 seconds over 200 frames
    and put open eyes under every blink label.

    Args:
        video_path (Path): The recording.
        timestamps (dict[int, float]): Frame id to timestamp in seconds.
        decoded_total (int): Frames the decoder reports.

    Raises:
        StreamError: If the implied frame rates differ by more than
            :data:`CLOCK_TOLERANCE`.
    """
    import cv2

    ids = np.asarray(sorted(timestamps), dtype=np.float64)
    times = np.asarray([timestamps[int(i)] for i in ids], dtype=np.float64)
    span = float(times[-1] - times[0])
    if len(times) < 2 or span <= 0:
        return
    # From the *id* span, not the count: an annotation may skip frames, and
    # dividing by the number of entries would measure how densely it was
    # annotated rather than how fast the recording runs.
    annotated_fps = float(ids[-1] - ids[0]) / span

    capture = cv2.VideoCapture(str(video_path))
    declared = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 0.0
    capture.release()
    if not declared or declared <= 0:
        return

    drift = abs(declared - annotated_fps) / annotated_fps
    if drift > CLOCK_TOLERANCE:
        raise StreamError(
            f"{video_path.name}: the container declares {declared:.2f} fps but the "
            f"annotation runs at {annotated_fps:.2f} fps ({100 * drift:.0f}% apart). "
            f"Matching {decoded_total} decoded frames on time would slide every label "
            "along the recording. Fix the timestamps, or pass positional=True if the "
            "frames really do correspond one to one."
        )


def decode_annotated_frames(
    video_path: Path,
    timestamps: dict[int, float],
    positional: bool | None = None,
    annotated_ids: Sequence[int] | None = None,
) -> Iterator[tuple[int, np.ndarray]]:
    """Stream the frames an annotation refers to, in decode order.

    Two ways to attach a label to an image, and picking the wrong one corrupts
    the corpus without failing:

    **Positional.** Frame *i* of the video is frame *i* of the annotation.
    Exact — no frame dropped, duplicated, or shifted.

    **Nearest timestamp.** Each annotation takes the decoded frame closest to it
    in time. Only correct when the decoder and the annotator agree about the
    frame rate.

    **The count that decides is the annotation's, not the timestamp file's.**
    RN ships a ``.txt`` carrying exactly six timestamps more than its video has
    frames, so comparing against the ``.txt`` failed the equality test on all
    107 recordings and fell through to timestamp matching — while the ``.tag``
    count matched the video exactly in 96 of them. That mattered because RN's
    containers *declare 30 fps for footage annotated at ~16.7*, so matching on
    time drifted by 5 seconds over 200 frames and captioned every window with a
    label from up to a second earlier. Blink frames came out showing open eyes,
    and no model could learn from it.

    When the counts genuinely differ the timestamp path is still used, but the
    two clocks are checked first: see :data:`CLOCK_TOLERANCE`.

    Args:
        video_path (Path): The recording.
        timestamps (dict[int, float]): Frame id to timestamp in seconds.
        positional (bool | None): Force a strategy, or ``None`` to choose.
        annotated_ids (Sequence[int] | None): The frame ids the annotation
            actually supplies. Preferred over ``timestamps``' keys for the
            positional decision, since a timestamp file may carry a tail the
            annotation does not.

    Yields:
        tuple[int, np.ndarray]: Frame id and its ``(H, W, 3)`` uint8 RGB image.

    Raises:
        StreamError: If the video cannot be opened, decodes nothing, or its
            declared rate disagrees with the annotation's.
    """
    import cv2

    if not timestamps:
        raise StreamError(f"{video_path}: no timestamps to align against.")

    wanted_ids = np.asarray(sorted(timestamps))
    # The annotation is the authority on how many frames are labelled; the
    # timestamp file is only how they are timed.
    alignment_ids = np.asarray(sorted(annotated_ids)) if annotated_ids is not None else wanted_ids

    if positional is None:
        decoded_total = frame_count(video_path)
        contiguous = _is_contiguous(alignment_ids)
        # Positional needs the annotation to name a contiguous run of frames
        # that the decoder actually produces. Exact equality is too strict: RN's
        # `test/rn15/1` decodes 2729 frames for 2728 annotated ones, and
        # refusing the one-frame surplus would push it onto the timestamp path
        # whose clock is wrong by 80%.
        positional = contiguous and len(alignment_ids) <= decoded_total
        if not positional:
            _check_clocks(video_path, timestamps, decoded_total)
        logger.info(
            f"{video_path.name}: {decoded_total} decoded frames against "
            f"{len(alignment_ids)} annotated; "
            + ("mapping positionally." if positional else "matching on timestamps.")
        )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise StreamError(f"{video_path}: cannot be opened for decoding.")

    wanted_times = np.asarray([timestamps[int(i)] for i in wanted_ids])
    best: dict[int, tuple[float, np.ndarray]] = {}
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if positional:
                # Indexed by the annotation's own ids: decoded frame i carries
                # annotation i, which is what "positional" means.
                if decoded < len(alignment_ids):
                    yield int(alignment_ids[decoded]), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                decoded += 1
                continue

            # Which annotation a decoded frame is closest to is only known once
            # its timestamp is read, so the best frame per annotation is kept
            # as the stream runs.
            position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            decoded += 1
            index = int(np.argmin(np.abs(wanted_times - position)))
            frame_id = int(wanted_ids[index])
            distance = abs(float(wanted_times[index]) - position)

            previous = best.get(frame_id)
            if previous is None or distance < previous[0]:
                best[frame_id] = (distance, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()

    if not decoded:
        raise StreamError(f"{video_path}: decoded zero frames.")

    if positional:
        logger.info(f"{video_path.name}: {decoded} frames decoded, mapped one to one.")
        return

    logger.info(f"{video_path.name}: {decoded} frames decoded, {len(best)} annotated frames kept.")
    for frame_id in sorted(best):
        yield frame_id, best[frame_id][1]


def build_cache(
    video_path: Path,
    tag: TagFile,
    timestamps: dict[int, float],
    image_size: int,
    extractor: EyeFeatureExtractor | None = None,
) -> FrameCache:
    """Decode a recording once and cache every annotated eye.

    Args:
        video_path (Path): The recording.
        tag (TagFile): Its annotation.
        timestamps (dict[int, float]): Frame id to timestamp in seconds.
        image_size (int): Side length to resize crops to.
        extractor (EyeFeatureExtractor | None): Descriptor extractor; ``None``
            caches crops alone.

    Returns:
        FrameCache: Crops and descriptors, keyed by frame and eye.
    """
    import cv2

    from blinklinmult.data.schema import EYE_FEATURE_DIM

    by_id: dict[int, TagRecord] = {record.frame_id: record for record in tag.records}
    cache = FrameCache(
        image_size=image_size,
        feature_dim=EYE_FEATURE_DIM if extractor is not None else None,
    )

    processed = 0
    # The annotation decides which frames are wanted; the timestamp file may
    # carry a tail beyond it (RN ships six extra) and using its length would
    # send every recording down the drifting timestamp path.
    wanted = len(by_id)
    for frame_id, frame in progress(
        decode_annotated_frames(video_path, timestamps, annotated_ids=sorted(by_id)),
        video_path.name,
        wanted,
    ):
        record = by_id.get(frame_id)
        if record is None:
            continue

        boxes = boxes_from_annotation(record)
        # The descriptors are computed from the uint8 crop, before
        # normalisation: exordium's landmarker expects raw pixel values.
        described = extractor.frame_features(frame, boxes) if extractor is not None else None
        # Once per frame: pose describes the head, not the eye. Reused for both
        # sides rather than estimated twice.
        pose = extractor.head_pose(frame) if extractor is not None else None

        for side in (LEFT, RIGHT):
            box = boxes.get(side)
            if box is None:
                continue
            patch = cv2.resize(
                box.crop(frame), (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            cache.add(
                frame_id,
                side,
                normalise_image(patch),
                described[side] if described is not None else None,
                pose=pose,
                box=(float(box.centre_x), float(box.centre_y), float(box.side)),
            )

        processed += 1
        if processed % PROGRESS_EVERY == 0:
            logger.info(f"{video_path.name}: {processed} frames described.")

    logger.info(
        f"{video_path.name}: cached {len(cache.crops)} eye crops from {processed} annotated frames."
    )
    return cache
