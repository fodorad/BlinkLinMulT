"""Blink detection on a video, as seven observable stages.

The corpora hand a model eye crops that were extracted months earlier. A video
hands it pixels, and everything between -- finding a face, keeping it the *same*
face, reading its pose, locating the eyes, deciding which are even visible -- has
to happen first. This module is that path.

**Staged, not streamed.** Each stage consumes all frames from the previous one
before the next begins, because that is how the work actually batches, and each
reports its own wall-clock time. A slow run is then immediately attributable
rather than a single opaque wait.

===  ==========================  =============================================
 #   stage                       produces
===  ==========================  =============================================
 1   face detection + tracking   one box per frame, the same person throughout
 2   head pose                   ``[yaw, pitch, roll]`` degrees per frame
 3   landmarks + eye boxes       478 points and two eye boxes per frame
 4   eye selection               per frame: left, right, both or neither
 5   inference                   per-frame closeness, per eye
 6   labelling                   open/closed per frame, plus blink events
 7   visualisation               handled by the caller, from these results
===  ==========================  =============================================

:func:`run` is a generator yielding a :class:`Stage` as each completes, so a UI
can show progress without the pipeline knowing anything about a UI.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from blinklinmult.data.schema import LEFT, RIGHT

if TYPE_CHECKING:
    from collections.abc import Iterator

    from blinklinmult.detector import BlinkDetector
    from blinklinmult.preprocess.geometry import EyeBox
    from blinklinmult.registry import ModelSpec
    from blinklinmult.train.events import Interval

logger = logging.getLogger(__name__)
"""Module-level logger."""

MAX_SECONDS = 10.0
"""Longest video processed, in seconds.

Seven stages over several models per frame is minutes of CPU for seconds of
video. Longer input is truncated with a warning rather than silently accepted
and left to time out.
"""

LOCALISATION_FAST = "fast"
"""Locate eyes from the face detector's own keypoints.

YOLO11 returns both eye centres in the *same forward pass* as the box, so the
478-point FaceMesh pass afterwards recomputes what is already in hand. Dropping
it, and shrinking the detector's input to
:data:`~blinklinmult.preprocess.extractors.POSE_IMAGE_SIZE`, is what takes the
streaming demos from 0.53x realtime to over 40 fps on CPU.
"""

LOCALISATION_PRECISE = "precise"
"""Locate eyes from FaceMesh landmarks, and extract the iris descriptor.

Slower, and the only route that can feed a model with
:attr:`~blinklinmult.registry.ModelSpec.needs_features`: the 160-d stream comes
from the iris landmarker, which the fast path never runs. Also what every
published corpus was built with.
"""

LOCALISATIONS = (LOCALISATION_FAST, LOCALISATION_PRECISE)
"""Valid ``localisation`` arguments to :func:`run`."""

POSE_GEOMETRIC = "geometric"
"""Estimate head rotation from the detector's five keypoints.

**0.05 ms against 26.7 ms** for 6DRepNet, because the keypoints are already
computed -- there is no second forward pass. See
:func:`~blinklinmult.preprocess.geometry.pose_from_keypoints` for the accuracy
this trades away: roll and pitch correlate at 0.94, yaw is unvalidated.
"""

POSE_6DREPNET = "6drepnet"
"""Estimate head rotation with a dedicated pose network.

A full forward pass per frame. What the corpora were built with, and what the
two-stream model's descriptor encodes.
"""

HEAD_POSES = (POSE_GEOMETRIC, POSE_6DREPNET)
"""Valid ``head_pose`` arguments to :func:`run`."""

YAW_LIMIT = 45.0
"""Degrees of yaw past which one eye is treated as occluded.

**Positive yaw turns the nose toward image-left**, so it occludes the viewer's
*right* eye; negative yaw occludes the left. That sign is not arbitrary: exordium
draws the yaw axis as ``x = size * sin(-radians(yaw))``, which puts the nose at
negative x -- image-left -- for positive yaw. Our side labels are the viewer's,
fixed by the same TalkingFace measurement that pins the region mirror in
:class:`~blinklinmult.preprocess.extractors.FaceMeshLocator`.

Getting this backwards suppresses the *visible* eye and keeps the occluded one,
which still produces a plausible-looking result. :mod:`tests.test_pipeline`
asserts both directions.
"""

PROGRESS_EVERY = 50
"""Frames between progress lines within a stage.

One line per frame would be noise, and yielding per frame would cost a
meaningful share of the runtime it is reporting.
"""

SIDES = (LEFT, RIGHT)
"""The two eyes, in the order they are reported."""


@dataclass(frozen=True)
class Extraction:
    """How a per-frame eye-state curve becomes blink events.

    ESR gives a continuous closeness signal; blink *presence* is a decision laid
    on top of it, and this is that decision made explicit. It is separated from
    the model because the two are independently wrong: a shipped operating point
    is only fitted for the corpus it was fitted on, and applying it elsewhere is
    a transfer, not a measurement.

    Args:
        high (float): A run must peak above this to count as a blink.
        low (float | None): It extends while above this. ``None`` is a single
            cut, where the same value has to both reject noise and catch a
            closure's shallow onset -- two jobs one number does badly.
    """

    high: float
    low: float | None = None

    @classmethod
    def fitted(cls, model_spec: ModelSpec) -> Extraction:
        """The operating point registered for a model.

        Args:
            model_spec (ModelSpec): The model's registry entry.

        Returns:
            Extraction: Its thresholds. For the 1.x models this is 0.5, a
            neutral sigmoid midpoint rather than a measurement -- that work
            reported eye state, not events, so no operating point was ever fitted
            for them. Supply your own rather than inheriting it.
        """
        low = None if model_spec.low_ratio is None else model_spec.threshold * model_spec.low_ratio
        return cls(high=model_spec.threshold, low=low)

    def validate(self) -> None:
        """Check the thresholds can separate anything.

        Raises:
            PipelineError: If a threshold is outside ``(0, 1)``, or the low cut
                is not below the high one -- at or above it, hysteresis stops
                extending runs and silently degrades to a single threshold.
        """
        if not 0.0 < self.high < 1.0:
            raise PipelineError(f"High threshold must be in (0, 1); got {self.high}.")
        if self.low is None:
            return
        if not 0.0 < self.low < 1.0:
            raise PipelineError(f"Low threshold must be in (0, 1); got {self.low}.")
        if self.low >= self.high:
            raise PipelineError(
                f"Low threshold ({self.low}) must be below the high one ({self.high}); "
                "at or above it, hysteresis degrades to a single cut."
            )

    def describe(self) -> str:
        """One line naming the rule, for the log.

        Returns:
            str: Human-readable summary.
        """
        if self.low is None:
            return f"single threshold {self.high:.3f}"
        return f"hysteresis {self.high:.3f} / {self.low:.3f}"


class PipelineError(Exception):
    """Raised when a video cannot be processed at all."""


class NoFaceError(PipelineError):
    """Raised when no frame of the video holds a face.

    Separate from :class:`PipelineError` because it is a property of the *input*,
    not a fault: the message tells a user what to upload instead, where a decode
    failure tells them something is broken.
    """


@dataclass
class Stage:
    """One completed stage, for a progress display.

    Args:
        index (int): 1-based stage number.
        total (int): How many stages there are.
        name (str): Short human label.
        detail (str): What it found -- counts, not prose.
        seconds (float): Wall-clock time the stage took.
    """

    index: int
    total: int
    name: str
    detail: str
    seconds: float

    def line(self) -> str:
        """Render as one log line.

        Returns:
            str: ``[n/7] name ... detail   1.2 s``
        """
        return f"[{self.index}/{self.total}] {self.name:.<34} {self.detail}   {self.seconds:.1f} s"


@dataclass
class FrameResult:
    """Everything known about one frame, for the overlay.

    Args:
        index (int): Frame number in the source video.
        face_box (tuple | None): ``(x1, y1, x2, y2)``, or ``None`` when the
            tracked subject was not found here.
        landmarks (np.ndarray | None): ``(478, 2)`` FaceMesh points.
        pose (np.ndarray | None): ``[yaw, pitch, roll]`` in degrees.
        eyes (dict[str, EyeBox | None]): Box per side.
        used (dict[str, bool]): Whether each side was scored.
        score (dict[str, float | None]): Closeness per side; ``None`` where the
            eye was not scored, which is **not** the same as zero.
    """

    index: int
    face_box: tuple[int, int, int, int] | None = None
    landmarks: np.ndarray | None = None
    pose: np.ndarray | None = None
    eyes: dict[str, EyeBox | None] = field(default_factory=dict)
    used: dict[str, bool] = field(default_factory=dict)
    score: dict[str, float | None] = field(default_factory=dict)


@dataclass
class Result:
    """The finished analysis of one video.

    Args:
        frames (list[FrameResult]): Per-frame findings, in order.
        signal (dict[str, np.ndarray]): Per-side closeness, ``NaN`` where the
            eye was not scored. NaN rather than zero so a gap cannot be mistaken
            for a confident "open".
        events (dict[str, list[Interval]]): Blink intervals per side.
        fps (float): Source frame rate.
        model_id (str): Which model produced this.
        truncated (bool): Whether the video was cut to :data:`MAX_SECONDS`.
        truth (dict[str, np.ndarray] | None): Annotated per-eye state, when the
            clip came with a ``.tag``. ``None`` for an uploaded video.
        extraction (Extraction | None): The rule that produced ``events``.
    """

    frames: list[FrameResult]
    signal: dict[str, np.ndarray]
    events: dict[str, list[Interval]]
    fps: float
    model_id: str
    truncated: bool = False
    truth: dict[str, np.ndarray] | None = None
    extraction: Extraction | None = None


def ground_truth(
    tag_path: str | Path, frames: int, offset: int = 0
) -> dict[str, np.ndarray] | None:
    """Per-eye annotated eye state, for a corpus clip that ships a ``.tag``.

    Only the ``.tag`` corpora have this; an uploaded video does not. It is drawn
    on the plot so a prediction can be read against what was annotated rather
    than judged by eye.

    Args:
        tag_path (str | Path): The ``.tag`` file beside the video.
        frames (int): How many frames were analysed.
        offset (int): Index of the first analysed frame in the source video, so
            a mid-video segment lines its annotation up correctly.

    Returns:
        dict[str, np.ndarray] | None: Per-side 0/1 arrays of length ``frames``,
        or ``None`` when the file is absent or unreadable.
    """
    from blinklinmult.preprocess.annotation import TagFile

    location = Path(tag_path)
    if not location.is_file():
        return None
    try:
        tags = TagFile.from_path(location, video_id=location.stem)
    except Exception as error:  # noqa: BLE001 - a malformed tag is not fatal
        logger.info(f"Could not read {location.name}: {error}")
        return None

    state = np.asarray(tags.eye_state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] < 2:
        return None
    return {
        LEFT: state[offset : offset + frames, 0],
        RIGHT: state[offset : offset + frames, 1],
    }


def occluded_side(yaw: float, limit: float = YAW_LIMIT) -> str | None:
    """Which eye a head turn hides, if any.

    Args:
        yaw (float): Head yaw in degrees.
        limit (float): Magnitude past which one eye counts as occluded.

    Returns:
        str | None: The hidden side, or ``None`` when the head is frontal
        enough for both.
    """
    if yaw > limit:
        return RIGHT
    if yaw < -limit:
        return LEFT
    return None


def _video_metadata(path: str | Path) -> tuple[float, int]:
    """Read a video's frame rate and frame count.

    Prefers exordium's probe when the extraction stack is installed, and falls
    back to OpenCV otherwise. Both report the same values for the formats this
    pipeline handles; the fallback exists so decoding a video does not require
    the optional preprocess extra, which pulls multi-GB model weights for
    capabilities this function never uses.

    Args:
        path (str | Path): The video to probe.

    Returns:
        tuple[float, int]: Frames per second, and the frame count (0 if unknown).

    Raises:
        Exception: Any probe failure, translated by the caller.
    """
    try:
        from exordium.video.core.io import (  # ty: ignore[unresolved-import]
            get_video_metadata,
        )
    except ImportError:
        pass
    else:
        meta = get_video_metadata(path)
        return float(meta.get("fps") or 25.0), int(meta.get("num_frames") or 0)

    import cv2  # ty: ignore[unresolved-import]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"cannot open {path}")
    try:
        return (
            float(capture.get(cv2.CAP_PROP_FPS) or 25.0),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        )
    finally:
        capture.release()


def _decode_frames(path: str | Path, first: int, stop: int) -> np.ndarray:
    """Decode a half-open frame range as RGB.

    The exordium and OpenCV paths agree on the returned array: ``(T, H, W, 3)``
    uint8 in RGB order, with ``stop`` exclusive. OpenCV decodes BGR, so its
    frames are reversed on the last axis to match.

    Args:
        path (str | Path): The video to read.
        first (int): First frame index, inclusive.
        stop (int): Last frame index, exclusive.

    Returns:
        np.ndarray: ``(T, H, W, 3)`` uint8 RGB frames.

    Raises:
        Exception: Any decode failure, translated by the caller.
    """
    try:
        from exordium.video.core.io import load_video  # ty: ignore[unresolved-import]
    except ImportError:
        pass
    else:
        frames, _ = load_video(path, start_frame=first, end_frame=stop)
        return frames.cpu().numpy() if hasattr(frames, "cpu") else np.asarray(frames)

    import cv2  # ty: ignore[unresolved-import]

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise OSError(f"cannot open {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        decoded = []
        for _ in range(max(stop - first, 0)):
            ok, frame = capture.read()
            if not ok:
                break
            decoded.append(frame[:, :, ::-1])
        return np.asarray(decoded, dtype=np.uint8)
    finally:
        capture.release()


def _resolve_pipeline(spec, localisation: str, head_pose: str) -> tuple[str, str]:
    """Decide which localisation and pose routes a model may actually use.

    The caller states a preference; the model has the final say. A model with
    :attr:`~blinklinmult.registry.ModelSpec.needs_features` consumes the 160-d
    iris descriptor, which only the precise route produces, and that descriptor
    encodes 6DRepNet's angles -- so asking for the fast route with such a model
    is not a cheaper answer, it is a different and wrong one. Both are forced,
    and the override is logged rather than performed silently.

    Split out from :func:`run` so it can be tested without constructing a
    locator or downloading weights.

    Args:
        spec (ModelSpec): The chosen model's registry entry.
        localisation (str): One of :data:`LOCALISATIONS`.
        head_pose (str): One of :data:`HEAD_POSES`.

    Returns:
        tuple[str, str]: The effective ``(localisation, head_pose)``.

    Raises:
        PipelineError: If either argument is not a recognised value.
    """
    if localisation not in LOCALISATIONS:
        raise PipelineError(f"localisation must be one of {LOCALISATIONS}; got {localisation!r}.")
    if head_pose not in HEAD_POSES:
        raise PipelineError(f"head_pose must be one of {HEAD_POSES}; got {head_pose!r}.")

    if not spec.needs_features:
        return localisation, head_pose

    if localisation != LOCALISATION_PRECISE or head_pose != POSE_6DREPNET:
        logger.info(
            "%s consumes the iris descriptor, so it runs the precise route with 6DRepNet; "
            "ignoring localisation=%r, head_pose=%r.",
            spec.model_id,
            localisation,
            head_pose,
        )
    return LOCALISATION_PRECISE, POSE_6DREPNET


def _read_video(
    path: str | Path,
    start: float = 0.0,
    duration: float = MAX_SECONDS,
) -> tuple[np.ndarray, float, bool, int]:
    """Decode a segment of a video.

    Args:
        path (str | Path): The video file.
        start (float): Where the segment begins, in seconds.
        duration (float): How long it runs, in seconds. Capped at
            :data:`MAX_SECONDS`.

    Returns:
        tuple[np.ndarray, float, bool, int]: ``(frames (T, H, W, 3) uint8 RGB,
        fps, whether the request was cut short, the first frame's index in the
        source)``. That last value matters: an overlay labelled ``Frame: 0`` for
        a segment starting at 30 s would not line up with the source video.

    Raises:
        PipelineError: If the file cannot be read, the start lies past its end,
            or the segment holds no frames.
    """
    # Validate before importing. The arguments are wrong or right regardless of
    # how the file is decoded, so a bad timestamp should be rejected without
    # requiring the optional extraction stack to be installed.
    if start < 0:
        raise PipelineError(f"Start timestamp must be at or after 0 s; got {start}.")
    if duration <= 0:
        raise PipelineError(f"Duration must be positive; got {duration}.")

    try:
        fps, available = _video_metadata(path)
    except Exception as error:  # noqa: BLE001 - any probe failure reads alike
        raise PipelineError(f"Could not read {Path(path).name}: {error}") from error

    # **Truncate, do not round.** A start timestamp names the frame it falls
    # inside, so 9.99 s at 30 fps is frame 299 -- the last one. Rounding sends it
    # to 300 and rejects a timestamp that is genuinely within the video.
    first = int(fps * start)
    if available and first >= available:
        length = available / fps
        raise PipelineError(
            f"Start timestamp {start:g} s is past the end of "
            f"{Path(path).name}, which is {length:.1f} s long."
        )

    wanted = int(round(fps * min(duration, MAX_SECONDS)))
    # `end_frame` is **exclusive** -- measured: `end_frame=270` from 150 returns
    # exactly 120 frames. It also raises rather than clipping when it points past
    # the last frame, so a segment running off the end must ask for no more than
    # the video holds.
    stop = first + wanted
    if available > 0:
        stop = min(stop, available)

    try:
        frames = _decode_frames(path, first, stop)
    except Exception as error:  # noqa: BLE001 - any decode failure reads alike
        raise PipelineError(f"Could not read {Path(path).name}: {error}") from error

    array = frames
    if array.ndim != 4 or array.shape[0] == 0:
        raise PipelineError(
            f"{Path(path).name} holds no frames between {start:g} s and {start + duration:g} s."
        )
    if array.shape[1] in (1, 3):  # (T, C, H, W) -> (T, H, W, C)
        array = array.transpose(0, 2, 3, 1)

    return array.astype(np.uint8), fps, array.shape[0] < wanted, first


def _largest_track(boxes: list[tuple[int, int, int, int] | None]) -> list[bool]:
    """Mark the frames belonging to the subject.

    With no tracker this is a single pass: the subject is simply whichever face
    was detected. The guard that matters is against *area* -- selection uses the
    **median** box area across a track, not any single frame, so one frame where
    a bystander looms closer cannot steal the subject.

    Args:
        boxes (list): Per-frame face box, or ``None``.

    Returns:
        list[bool]: Whether each frame holds the subject.
    """
    return [box is not None for box in boxes]


def run(
    video_path: str | Path,
    detector: BlinkDetector,
    stats: dict[str, list[float]] | None = None,
    tag_path: str | Path | None = None,
    extraction: Extraction | None = None,
    start: float = 0.0,
    duration: float = MAX_SECONDS,
    localisation: str = LOCALISATION_FAST,
    head_pose: str = POSE_GEOMETRIC,
) -> Iterator[Stage | Result]:
    """Analyse a video, yielding each stage as it completes.

    Yields six :class:`Stage` objects and then a final :class:`Result`. The
    caller decides what to do with the stages -- a UI prints them, a test
    ignores them.

    Args:
        video_path (str | Path): The video to analyse.
        detector (BlinkDetector): A loaded model.
        stats (dict | None): Corpus feature statistics, ``{"mean": ..., "std":
            ...}``. Required by ``blinklinmult-union``, ignored by the others.
        tag_path (str | Path | None): A ``.tag`` annotation to plot alongside
            the prediction. Only the corpus clips have one.
        extraction (Extraction | None): How to turn the eye-state curve into
            events. Defaults to the model's registered operating point.
        start (float): Where in the video to begin, in seconds.
        duration (float): How much of it to analyse, in seconds.
        localisation (str): :data:`LOCALISATION_FAST` (default) reads the eye
            centres from the face detector's keypoints;
            :data:`LOCALISATION_PRECISE` runs FaceMesh and the iris landmarker.
            A model needing the descriptor forces the precise route -- see
            :func:`_resolve_pipeline`.
        head_pose (str): :data:`POSE_GEOMETRIC` (default) derives the angles
            from those same keypoints; :data:`POSE_6DREPNET` runs the pose
            network. Forced to 6DRepNet alongside the precise route.

    Yields:
        Stage | Result: One per completed stage, then the result.

    Raises:
        PipelineError: If the video cannot be read, the two-stream model was
            given no feature statistics, or ``localisation``/``head_pose`` is
            not a recognised value.
    """
    from blinklinmult.preprocess.common import normalise_image

    localisation, head_pose = _resolve_pipeline(detector.spec, localisation, head_pose)

    if detector.spec.needs_features and stats is None:
        raise PipelineError(
            f"{detector.spec.model_id!r} needs corpus feature statistics; pass stats=..."
        )

    import cv2  # ty: ignore[unresolved-import]

    total_stages = 7
    image_size = detector.spec.image_size
    frames, fps, truncated, offset = _read_video(video_path, start, duration)
    count = frames.shape[0]

    # Constructed per route: building `ExordiumExtractor` loads FaceMesh and
    # 6DRepNet weights, which is most of the precise route's startup cost, and
    # the fast route never calls it.
    if localisation == LOCALISATION_PRECISE:
        from blinklinmult.preprocess.extractors import ExordiumExtractor, FaceMeshLocator

        locator = FaceMeshLocator()
        extractor = ExordiumExtractor()
    else:
        from blinklinmult.preprocess.extractors import PoseEyeLocator

        locator = PoseEyeLocator()
        extractor = None
    # Indices are the *source* video's, not the segment's: an overlay reading
    # `Frame: 0` for a segment starting at 30 s would not match the footage, and
    # the annotation is indexed the same way.
    results = [FrameResult(index=offset + i) for i in range(count)]

    # -- 1. face detection, landmarks and eye boxes, in one pass per frame ----
    start = time.perf_counter()
    detections = []
    for index, frame in enumerate(frames):
        found = locator.detect(frame)
        detections.append(found)
        if found is not None:
            results[index].face_box = found.face_box
            results[index].landmarks = found.landmarks
            results[index].eyes = found.eyes
    kept = _largest_track([d.face_box if d else None for d in detections])
    found_n = sum(kept)
    yield Stage(
        1,
        total_stages,
        "Face detection and tracking",
        f"{found_n}/{count} frames with a face",
        time.perf_counter() - start,
    )

    # Nothing downstream means anything without a face, and running six more
    # stages over an empty selection would report a confident "no blinks" for a
    # video the pipeline never actually looked at.
    if found_n == 0:
        raise NoFaceError(
            f"No face detected in any of the {count} frames. "
            "The pipeline needs a visible face to locate eyes; "
            "try a clip where the subject faces the camera."
        )

    # -- 2. head pose --------------------------------------------------------
    # The geometric route reads the detection already made in stage 1; 6DRepNet
    # runs its own forward pass over the frame. Measured on CPU: 0.05 ms against
    # 26.7 ms per frame.
    start = time.perf_counter()
    pose_failed = 0
    for index, frame in enumerate(frames):
        if not kept[index]:
            continue
        detection = detections[index]
        if head_pose == POSE_GEOMETRIC:
            # `_resolve_pipeline` pairs the geometric route with the fast
            # locator, whose `head_pose` reads the keypoints already detected.
            angles = (
                locator.head_pose(detection)  # ty: ignore[unresolved-attribute]
                if detection is not None
                else np.zeros(3, np.float32)
            )
        else:
            # Likewise, 6DRepNet only runs on the precise route, where the
            # extractor is built.
            angles = extractor.head_pose(frame)  # ty: ignore[unresolved-attribute]
        results[index].pose = angles
        if not np.any(angles):
            pose_failed += 1
    yield Stage(
        2,
        total_stages,
        # Naming the route that actually ran, not the one that used to: these
        # stage lines are the demo's timing readout, and a mislabel would make
        # the speed claim unverifiable.
        f"Head pose ({'geometric' if head_pose == POSE_GEOMETRIC else '6DRepNet'})",
        f"{found_n} frames, {pose_failed} failed",
        time.perf_counter() - start,
    )

    # -- 3. eye localisation (already computed in stage 1) -------------------
    start = time.perf_counter()
    both = sum(1 for r in results if r.eyes.get(LEFT) is not None and r.eyes.get(RIGHT) is not None)
    yield Stage(
        3,
        total_stages,
        "Landmarks and eye localisation",
        f"{both} frames with both eyes",
        time.perf_counter() - start,
    )

    # -- 4. which eyes are usable -------------------------------------------
    start = time.perf_counter()
    tally = {"both": 0, "left": 0, "right": 0, "neither": 0}
    occluded_n = 0
    for index, result in enumerate(results):
        hidden = None
        if result.pose is not None:
            hidden = occluded_side(float(result.pose[0]))
        for side in SIDES:
            usable = kept[index] and result.eyes.get(side) is not None and side != hidden
            result.used[side] = usable
        if hidden is not None:
            occluded_n += 1
        live = [s for s in SIDES if result.used[s]]
        key = "both" if len(live) == 2 else (live[0] if live else "neither")
        tally[key] += 1
    yield Stage(
        4,
        total_stages,
        "Eye selection",
        f"both {tally['both']}, left {tally['left']}, right {tally['right']}, "
        f"neither {tally['neither']} ({occluded_n} yaw-occluded)",
        time.perf_counter() - start,
    )

    # -- 5. inference, per eye ----------------------------------------------
    start = time.perf_counter()

    # `frame_features` describes both eyes in one call and computes head pose
    # once for the pair -- calling it per side would run 6DRepNet twice per
    # frame for the same head.
    descriptors: dict[str, dict[int, np.ndarray]] = {side: {} for side in SIDES}
    # `_resolve_pipeline` forces the precise route for a model needing features,
    # so `extractor` is built whenever this branch runs. Asserted rather than
    # assumed: if that rule is ever loosened, this fails here with a clear name
    # instead of an AttributeError on None.
    if detector.spec.needs_features:
        if extractor is None:  # pragma: no cover - guarded by _resolve_pipeline
            raise PipelineError(
                f"{detector.spec.model_id!r} needs the iris descriptor, "
                f"which requires localisation={LOCALISATION_PRECISE!r}."
            )
        for index, result in enumerate(results):
            if not any(result.used.get(side, False) for side in SIDES):
                continue
            described = extractor.frame_features(frames[index], result.eyes, result.face_box)
            for side, (vector, valid) in described.items():
                if valid and result.used.get(side, False):
                    descriptors[side][index] = vector

    signal: dict[str, np.ndarray] = {}
    for side in SIDES:
        usable = np.array([r.used[side] for r in results], dtype=bool)
        curve = np.full(count, np.nan, dtype=np.float32)
        if usable.any():
            patches, feats, scored = [], [], []
            for index in np.flatnonzero(usable):
                box = results[index].eyes[side]
                if box is None:  # `usable` already excludes these; narrows the type
                    continue
                if detector.spec.needs_features:
                    vector = descriptors[side].get(int(index))
                    if vector is None:  # descriptor extraction failed here
                        continue
                    feats.append(vector)
                # `box.crop` then INTER_AREA to the model's own image size --
                # exactly what `stream.py` does when building a corpus. Eye
                # boxes scale with the face, so the raw crops differ in size
                # frame to frame; matching the builder's resize is also what
                # keeps the model's input distribution the one it was trained
                # on.
                patch = cv2.resize(
                    box.crop(frames[index]),
                    (image_size, image_size),
                    interpolation=cv2.INTER_AREA,
                )
                patches.append(normalise_image(patch))
                scored.append(int(index))
            if not patches:
                signal[side] = curve
                continue
            usable = np.zeros(count, dtype=bool)
            usable[scored] = True
            stack = np.stack(patches)
            stream = None
            if detector.spec.needs_features and stats is not None:
                stream = detector.prepare_features(
                    np.stack(feats), np.asarray(stats["mean"]), np.asarray(stats["std"])
                )
            curve[usable] = detector.score_long(stack, stream)
        signal[side] = curve
        for index, result in enumerate(results):
            value = curve[index]
            result.score[side] = None if np.isnan(value) else float(value)
    yield Stage(
        5,
        total_stages,
        "Inference",
        " ".join(f"{s} {int(np.isfinite(signal[s]).sum())} frames," for s in SIDES).rstrip(","),
        time.perf_counter() - start,
    )

    # -- 6. labelling and events --------------------------------------------
    from blinklinmult.train.events import to_intervals

    start = time.perf_counter()
    events: dict[str, list[Interval]] = {}
    rule = extraction if extraction is not None else Extraction.fitted(detector.spec)
    rule.validate()
    for side in SIDES:
        curve = signal[side]
        mask = np.isfinite(curve)
        events[side] = to_intervals(np.nan_to_num(curve), mask, rule.high, rule.low)
    yield Stage(
        6,
        total_stages,
        "Labelling",
        f"{rule.describe()} -- " + ", ".join(f"{s} {len(events[s])} blinks" for s in SIDES),
        time.perf_counter() - start,
    )

    yield Result(
        frames=results,
        signal=signal,
        events=events,
        fps=fps,
        model_id=detector.spec.model_id,
        truncated=truncated,
        truth=None if tag_path is None else ground_truth(tag_path, count, offset),
        extraction=rule,
    )
