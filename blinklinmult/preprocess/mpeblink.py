"""Preprocess MPEblink 2.0.

921 untrimmed film clips, 17 711 annotated blink events, up to 24 people
co-existing in a video and lengths to 5 976 frames — the only genuinely
multi-person, in-the-wild corpus in this benchmark.

**Why this corpus does not use the shared video pipeline.**
:mod:`blinklinmult.preprocess.video_corpus` assumes one annotated face per
recording, keyed by a ``.tag`` file. MPEblink annotates *many tracklets per
video* in a single JSON, so the unit of work is the tracklet rather than the
recording: one video is decoded once and its frames dealt out to every person
visible in them.

**Instances are given, not detected.** The annotated tracklets are used as
oracle instances, which is the limiting case MPEblink's own Blink-AP
approximates — that metric already scores only the true-positive instances of
Inst-AP, "as the blink accuracy of these predictions is rarely affected by the
face detection and tracking accuracy". Numbers produced this way are an upper
bound on the published joint figures and must be reported as oracle-instance.

**Eyes are derived, not annotated.** The corpus labels faces; there is no eye
box and no eye-state flag. The eye regions come from the WFLW landmarks —
indices 60-67 (left) and 68-75 (right), verified against the video — cropped
*per eye* rather than as one binocular box, so a sample is one eye as in every
other corpus and a pretrained model transfers without a shape change.

**Blinks arrive as segments.** ``blink`` holds ``[start, end, category]``
entries rather than a per-frame vector. The category is **ignored**, matching
the authors' own converter, which reads only elements ``[0]`` and ``[1]``.

**The build is resumable.** Decoding 1 842 untrimmed videos takes the better
part of a day, so each video writes its own shard under
``data/processed/mpeblink/shards/`` and the shards are merged once every video
has one. Re-running after an interruption skips the videos that already have a
shard; a shard is renamed into place only after it closes cleanly, so its
presence means that video finished. See :mod:`blinklinmult.data.shards`.

Example:
    ``uv run python -m blinklinmult.preprocess.mpeblink``
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from blinklinmult import PROJECT_ROOT
from blinklinmult.data.schema import (
    BLINK_CATEGORY,
    BOX_SHIFTED,
    CONFIDENCE,
    DEFAULT_WINDOW_SECONDS,
    EYE_ASPECT,
    EYE_FEATURE_DIM,
    EYE_ON_SCREEN,
    EYE_SPAN,
    FACE_BOX,
    HEAD_POSE_DIM,
    INVALID_BLINK,
    LANDMARK_SOURCE,
    LEFT,
    NO_BLINK,
    NO_CATEGORY,
    RIGHT,
    DatasetSpec,
)
from blinklinmult.data.shards import (
    merge_shards,
    pending,
    shard_attrs,
    shard_dir,
    shard_path,
)
from blinklinmult.data.writer import H5Writer
from blinklinmult.preprocess.common import PreprocessError, git_sha, normalise_image, progress
from blinklinmult.preprocess.features import empty_feature, stack_window
from blinklinmult.preprocess.geometry import MIN_EYE_ON_SCREEN, EyeBox
from blinklinmult.preprocess.quality import (
    blur_score,
    box_jitter,
    exposure_score,
    eye_aspect,
    on_screen_fraction,
    window_confidence,
)
from blinklinmult.preprocess.windows import window_frames

if TYPE_CHECKING:
    from collections.abc import Iterator

logger = logging.getLogger(__name__)
"""Module-level logger."""

NAME = "mpeblink"
"""Corpus name, matching its config and processed directory."""

RAW_DIRNAME = "mpeblink2.0"
"""Directory under ``data/raw`` holding the extracted corpus."""

ANNOTATION = "annotation_WFLW.json"
"""Per-video annotation file."""

VIDEO = "video.mp4"
"""Per-video media file."""

SPLIT_DIRS: dict[str, str] = {"train": "train", "val": "valid", "test": "test"}
"""The corpus's own split directories, mapped to this project's names.

Honoured as given: MPEblink 2.0 divides 921 videos into 540/169/212, and
re-deriving the split would make the numbers incomparable with the authors'.
"""

FPS = 25.0
"""Fallback rate when a video's container does not report one."""

EYE_LANDMARKS: dict[str, tuple[int, int]] = {LEFT: (60, 68), RIGHT: (68, 76)}
"""WFLW eyelid-contour ranges bounding each eye, as ``[start, stop)``.

WFLW — *Wider Facial Landmarks in the Wild* — annotates 98 points: 33 of face
contour, 9 per brow, 9 of nose, **8 per eyelid contour**, 20 of mouth and 2
pupil centres. The eyes are therefore annotated outright, which is why this
corpus needs no landmarker of its own.

**Named from the viewer's side.** Measured on the frames: points 60-67 sit on
the left of the *image*, which WFLW's own numbering calls the subject's right.
The ``.tag`` corpora name eyes the same way, and
:class:`~blinklinmult.preprocess.extractors.FaceMeshLocator` deliberately
crosses exordium's subject-view names to match, so ``LEFT`` means the same
thing across the whole benchmark.

Verified by rendering over the source video rather than taken on trust.

WFLW is preferred where present, but it is **not** present everywhere: see
:data:`IBUG_EYE_LANDMARKS` for the 30 test clips that ship only the 68-point
``landmark`` field.
"""

PUPILS: dict[str, int] = {LEFT: 96, RIGHT: 97}
"""WFLW pupil-centre index per eye.

**The crop is centred here, not on the eyelid contour's mean.** The contour
moves as the lid closes — which is exactly the moment the crop must stay
still — while the pupil marks the eye itself. Measured over 36 045 eyes, the
two centres differ by a median 3.1% of the crop width but by **38% at the
extreme**, and a crop that slides that far mid-blink shows the model motion the
eye did not make.

Reliable enough to depend on: over 104 168 pupil points, 99.9% fall inside
their own eyelid contour and none is missing. The 0.1% that do not fall back to
the contour mean.
"""

EYE_CROP_SCALE = 2.0
"""Crop side as a multiple of the eye's **corner-to-corner** distance.

The same rule and the same measure every other corpus uses (see
:data:`blinklinmult.preprocess.geometry.EYE_CROP_SCALE`, whose span comes from
the two annotated eye corners), so a crop means the same thing across the
benchmark and a model transfers between corpora unchanged.

Measured against the eyelid contour's **bounding-box diagonal** instead, the
crop would be 5.5% larger on an open eye and only 2.6% larger on a closing one:
the diagonal absorbs the lid's vertical travel, so the crop would breathe with
the very motion it exists to show. The corner distance is a property of the eye
socket, not of the eyelid.
"""

MIN_EYE_SPAN = 4.0
"""Smallest eye span, in pixels, worth cropping.

Films put faces at every scale; a distant face can leave an eye a handful of
pixels wide, where a crop would be interpolation noise rather than an eye.
Measured spans on a mid-shot run 20-50 px, so this only rejects the extremes.
"""

IBUG_EYE_LANDMARKS: dict[str, tuple[int, int]] = {LEFT: (36, 42), RIGHT: (42, 48)}
"""Eye-contour index range per eye in the **68-point** ``landmark`` field.

Thirty test clips -- ``test/183`` through ``test/212``, 14% of the test split --
ship no ``landmark_WFLW`` at all, only this 68-point iBUG scheme. They are the
corpus's long recordings (2 700-6 000 frames against a typical few hundred) and
carry thousands of annotated blink frames, so reading WFLW alone would silently
discard them and quietly shrink the benchmark.

Named from the viewer's side, matching :data:`EYE_LANDMARKS`: 36-41 sits on the
left of the *image*. Verified by rendering the groups over ``test/183`` rather
than taken from the standard's numbering.

The scheme has **no pupil point**, so these clips centre on the eye-contour mean
-- the same fallback :func:`eye_box` already uses for a bad WFLW pupil.
"""

WFLW_POINTS = 98
"""Point count identifying the WFLW landmark scheme."""

IBUG_POINTS = 68
"""Point count identifying the 68-point iBUG scheme."""


class MPEblinkError(PreprocessError):
    """Raised when the corpus cannot be read."""


@dataclass(frozen=True)
class Tracklet:
    """One person's annotation through one video.

    Args:
        video_id (str): ``<split>_<video>``, unique across the corpus.
        person (str): The annotation's key for this person, e.g. ``person0``.
        boxes (list): Per-frame face box, ``None`` where absent.
        landmarks (list): Per-frame WFLW landmarks, ``None`` where absent.
        blinks (np.ndarray): ``(T,)`` per-frame blink label.
        blink_ids (np.ndarray): ``(T,)`` which annotated event each frame
            belongs to, or :data:`~blinklinmult.data.schema.NO_BLINK` outside
            every event. See :func:`number_events`.
        length (int): Frames in the video.
        landmark_source (str): ``"wflw"`` or ``"ibug"``; see
            :data:`~blinklinmult.data.schema.LANDMARK_SOURCE`.
    """

    video_id: str
    person: str
    boxes: list
    landmarks: list
    blinks: np.ndarray
    length: int
    # Defaults to "no event anywhere" so a tracklet can be built from the blink
    # labels alone; `read_tracklets` always supplies the real numbering.
    blink_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    blink_categories: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    landmark_source: str = "wflw"

    def __post_init__(self) -> None:
        """Fill in an all-background numbering when none was supplied.

        ``object.__setattr__`` because the dataclass is frozen; this is the
        standard idiom for deriving a field in ``__post_init__`` and runs only
        at construction, so the instance stays immutable afterwards.
        """
        if self.blink_ids.size == 0 and self.length:
            object.__setattr__(self, "blink_ids", np.full(self.length, NO_BLINK, dtype=np.int32))

    @property
    def sample_prefix(self) -> str:
        """Identifier carrying the video and the person.

        Returns:
            str: ``<video_id>-<person>``, so two people in one video never
            collide and a prediction can be traced back to its instance.
        """
        return f"{self.video_id}-{self.person}"

    def visible(self) -> np.ndarray:
        """Frames where this person is annotated.

        Returns:
            np.ndarray: ``(T,)`` bool. 281 of 1 247 sampled tracklets have
            gaps — people leave shot and return — so this is not a contiguous
            run and windows must not span an absence.
        """
        return np.asarray(
            [b is not None and self.landmarks[i] is not None for i, b in enumerate(self.boxes)],
            dtype=bool,
        )


def rasterise(segments: list, length: int) -> np.ndarray:
    """Turn ``[start, end, category]`` blink segments into a per-frame vector.

    The **category is ignored**, exactly as the authors' converter does: it
    reads elements ``[0]`` and ``[1]`` only, so all 17 711 events count
    equally. Segments are inclusive of both endpoints, and the 86 overlapping
    pairs in the corpus (0.5%) merge harmlessly.

    Args:
        segments (list): Entries of the annotation's ``blink`` field.
        length (int): Frames in the video.

    Returns:
        np.ndarray: ``(length,)`` float32, 1 where a blink is annotated.
    """
    labels = np.zeros(length, dtype=np.float32)
    for entry in segments:
        if len(entry) < 2 or entry[0] is None or entry[1] is None:
            continue
        start, stop = int(entry[0]), int(entry[1])
        labels[max(0, start) : min(length, stop + 1)] = 1.0
    return labels


def number_events(segments: list, length: int) -> np.ndarray:
    """Number each annotated blink, so its frames identify *which* event.

    :func:`rasterise` answers "is frame *t* inside a blink"; this answers "which
    blink". Without it MPEblink carries no ``blink_id``, which is what every
    event-level analysis needs to tell one long blink from two adjacent ones --
    so incomplete blinks, double blinks, and any event-count supervision are
    undefinable on the corpus that is 80% of the benchmark. The other corpora
    all ship real ids; this brings MPEblink in line with them.

    Ids are **per recording and per tracklet**, assigned in annotation order
    from 0. :data:`~blinklinmult.data.schema.NO_BLINK` marks a frame outside
    every event, matching how the other corpora encode it.

    Overlapping segments -- 86 pairs in the corpus, 0.5% -- resolve to the
    **later** event, the same last-write-wins rule :func:`rasterise` applies
    when it merges them. The two functions therefore agree on which frames are
    blinking, which is what lets the id be trusted as a refinement of the label
    rather than a second opinion on it.

    Args:
        segments (list): Entries of the annotation's ``blink`` field.
        length (int): Frames in the video.

    Returns:
        np.ndarray: ``(length,)`` int32; the event index at each frame, or
        :data:`~blinklinmult.data.schema.NO_BLINK` outside every event.
    """
    ids = np.full(length, NO_BLINK, dtype=np.int32)
    index = 0
    for entry in segments:
        if len(entry) < 2 or entry[0] is None or entry[1] is None:
            continue
        start, stop = int(entry[0]), int(entry[1])
        ids[max(0, start) : min(length, stop + 1)] = index
        index += 1
    return ids


def event_categories(segments: list, length: int) -> np.ndarray:
    """The annotation's third field, rasterised per frame.

    **Its meaning is undocumented.** The corpus writes ``[start, end, category]``
    and the category takes three values, but the CVPR paper describes only the
    start and end frames and says nothing about a third field; the authors' own
    converter reads elements ``[0]`` and ``[1]`` only. Measured over 8 072
    segments the split is 83.8% / 6.5% / 9.8%, and it is **not** a duration
    distinction -- median lengths are 6, 7 and 7 frames.

    It is preserved rather than discarded so the question stays answerable from
    the built corpus rather than the raw tree. Across the whole corpus the
    values are:

    ========  ======  ===============================================
    value      count  note
    ========  ======  ===============================================
    0          15083  the common case
    1           1210
    2           1406
    10             1  ``train/184`` person1, frames 284-292 -- almost
                      certainly a typo for 1 or 0
    ``None``      11  no value given; 5 of them in ``train/67``
    ========  ======  ===============================================

    Both anomalies are written through as-is rather than corrected: a silent
    fix would hide a corpus quirk, and ``None`` becomes
    :data:`~blinklinmult.data.schema.NO_CATEGORY` so it stays distinguishable
    from a real value.

    Args:
        segments (list): Entries of the annotation's ``blink`` field.
        length (int): Frames in the video.

    Returns:
        np.ndarray: ``(length,)`` int32; the category at each blinking frame,
        :data:`~blinklinmult.data.schema.NO_CATEGORY` outside every event and
        wherever the annotation supplied none.
    """
    # NO_CATEGORY outside every event *and* where the annotation gave no value,
    # which is 11 events across the corpus.
    categories = np.full(length, NO_CATEGORY, dtype=np.int32)
    for entry in segments:
        if len(entry) < 2 or entry[0] is None or entry[1] is None:
            continue
        value = entry[2] if len(entry) > 2 and entry[2] is not None else NO_CATEGORY
        start, stop = int(entry[0]), int(entry[1])
        categories[max(0, start) : min(length, stop + 1)] = int(value)
    return categories


def _window_quality(
    images: list[np.ndarray],
    valid: np.ndarray,
    face_boxes: list[tuple[int, int, int, int]],
    spans: list[float],
) -> dict[str, np.ndarray]:
    """Per-frame quality signals for one eye across a window.

    Every signal describes this eye alone -- a sample is one eye, so a value
    read from its partner would leak across samples and be undefined exactly
    when one eye is occluded.

    Jitter is measured from the **face** box centre rather than the eye's:
    MPEblink tracks faces past the frame edge, so the eye box is sometimes slid
    to stay inside and its centre moves for a reason that is not tracking loss.

    Args:
        images (list[np.ndarray]): The window's crops, ``(C, H, W)`` each.
        valid (np.ndarray): ``(T,)`` which frames were readable.
        face_boxes (list[tuple[int, int, int, int]]): Per-frame face box.
        spans (list[float]): Per-frame eye width in pixels.

    Returns:
        dict[str, np.ndarray]: Each signal as ``(T,)`` in ``[0, 1]``.
    """
    centres = np.asarray(
        [((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0) for box in face_boxes],
        dtype=np.float64,
    )
    return {
        "eye_blur": np.asarray(
            [blur_score(image) if ok else 0.0 for image, ok in zip(images, valid, strict=True)],
            dtype=np.float32,
        ),
        "eye_exposure": np.asarray(
            [exposure_score(image) if ok else 0.0 for image, ok in zip(images, valid, strict=True)],
            dtype=np.float32,
        ),
        "eye_jitter": box_jitter(centres, np.asarray(spans, dtype=np.float64)).astype(np.float32),
    }


def window_blink_id(frame_ids: np.ndarray) -> int:
    """The event a window belongs to, from its per-frame ids.

    :data:`~blinklinmult.data.schema.BLINK_ID` is one integer per window, so a
    window spanning two events has to pick one. The **first** event present
    wins, which keeps the id stable as a window slides forward across a blink:
    picking the majority or the last would flip the id mid-blink and make the
    same event appear under two identities.

    Args:
        frame_ids (np.ndarray): ``(T,)`` per-frame event ids, with
            :data:`~blinklinmult.data.schema.NO_BLINK` outside every event.

    Returns:
        int: The window's event id, or :data:`NO_BLINK` when it contains none.
    """
    present = frame_ids[frame_ids != NO_BLINK]
    return int(present[0]) if present.size else NO_BLINK


def eye_box(
    landmarks: np.ndarray, side: str, frame_shape: tuple[int, int] | None = None
) -> EyeBox | None:
    """Square crop box around one eye, from either landmark scheme.

    The **size** comes from the eyelid contour, which is what scales with the
    face; the **centre** comes from the pupil, which does not move as the lid
    closes. See :data:`PUPILS`.

    Both schemas the corpus ships are accepted, chosen by point count: 98-point
    WFLW, and the 68-point iBUG layout that is all 30 of its long test clips
    carry (see :data:`IBUG_EYE_LANDMARKS`). The 68-point scheme has no pupil, so
    those crops centre on the contour mean.

    Pass ``frame_shape`` to handle eyes near the frame edge. A face box may hang
    off the screen while the eye itself is fully visible and annotated, so the
    box is **slid back into the frame** rather than padded with black — real
    pixels beat zeros. Only an eye whose own landmarks have mostly left the
    frame returns ``None``; see :data:`MIN_EYE_ON_SCREEN`.

    Args:
        landmarks (np.ndarray): ``(98, 2)`` WFLW or ``(68, 2+)`` iBUG points for
            one frame. A third column, where present, is ignored.
        side (str): :data:`~blinklinmult.data.schema.LEFT` or ``RIGHT``.
        frame_shape (tuple[int, int] | None): ``(height, width)`` of the frame.
            Omit to skip the edge handling entirely.

    Returns:
        EyeBox | None: The box, or ``None`` when the eye is too small to crop,
        mostly off-frame, or the point count matches neither scheme.
    """
    array = np.asarray(landmarks, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] not in (WFLW_POINTS, IBUG_POINTS):
        return None

    if array.shape[0] == WFLW_POINTS:
        start, stop = EYE_LANDMARKS[side]
        pupil = array[PUPILS[side], :2]
    else:
        start, stop = IBUG_EYE_LANDMARKS[side]
        pupil = None

    points = array[start:stop, :2]

    # Corner to corner, as every other corpus measures it: the leftmost and
    # rightmost contour points are the eye's two corners. Using the contour's
    # bounding-box diagonal instead would fold in the eyelid's vertical travel,
    # so the crop would shrink as the eye closes.
    horizontal = points[:, 0]
    corners = points[[int(horizontal.argmin()), int(horizontal.argmax())]]
    span = float(np.hypot(*(corners[1] - corners[0])))
    if span < MIN_EYE_SPAN:
        return None

    # 85 of 104 168 pupils land outside their own contour, which is a bad point
    # rather than a bad eye: the contour mean still describes where the eye is.
    # The 68-point scheme has no pupil at all and always takes that route.
    centre = points.mean(axis=0)
    if pupil is not None:
        lower, upper = points.min(axis=0), points.max(axis=0)
        if not (np.any(pupil < lower - 2) or np.any(pupil > upper + 2)):
            centre = pupil

    box = EyeBox(
        centre_x=int(round(centre[0])),
        centre_y=int(round(centre[1])),
        side=max(1, int(round(span * EYE_CROP_SCALE))),
        span=span,
        aspect=eye_aspect(points),
    )
    if frame_shape is None:
        return box

    # Gate on the eye's own landmarks, never on the box: the box carries twice
    # the eye's width in context and so overflows the frame long before the eye
    # does. An eye that is on screen keeps its crop, slid back into the frame.
    height, width = frame_shape
    visible = on_screen_fraction(points, height, width)
    if visible < MIN_EYE_ON_SCREEN:
        return None
    return replace(box, on_screen=visible).shifted_into(height, width)


def read_tracklets(annotation: Path, video_id: str) -> list[Tracklet]:
    """Read every person annotated in one video.

    Args:
        annotation (Path): The video's ``annotation_WFLW.json``.
        video_id (str): Identifier for the video.

    Returns:
        list[Tracklet]: One entry per annotated person.

    Raises:
        MPEblinkError: If the file cannot be parsed.
    """
    try:
        data = json.loads(annotation.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise MPEblinkError(f"{annotation}: cannot be read ({error}).") from error

    length = int(data.get("length", 0))
    if length < 1:
        raise MPEblinkError(f"{annotation}: declares {length} frames.")

    tracklets = []
    for person, entry in data.items():
        if not person.startswith("person") or not isinstance(entry, dict):
            continue
        tracklets.append(
            Tracklet(
                video_id=video_id,
                person=person,
                boxes=entry.get("bbox", []),
                # WFLW where it exists, else the 68-point field every clip
                # carries: 30 test clips ship no WFLW and would otherwise be
                # dropped whole. `eye_box` reads either.
                landmarks=entry.get("landmark_WFLW") or entry.get("landmark", []),
                blinks=rasterise(entry.get("blink", []), length),
                # Same segments, numbered rather than merged: this is what makes
                # MPEblink's events individuable like every other corpus's.
                blink_ids=number_events(entry.get("blink", []), length),
                # Undocumented third field, preserved so the question stays
                # answerable from the built corpus rather than the raw tree.
                blink_categories=event_categories(entry.get("blink", []), length),
                length=length,
                landmark_source="wflw" if entry.get("landmark_WFLW") else "ibug",
            )
        )
    return tracklets


def runs(mask: np.ndarray) -> Iterator[tuple[int, int]]:
    """Maximal runs of ``True`` in a boolean mask.

    A tracklet's visible frames are not one contiguous block — people leave
    shot and come back — and a window must never bridge an absence, which
    would splice together frames seconds apart.

    Args:
        mask (np.ndarray): ``(T,)`` bool.

    Yields:
        tuple[int, int]: ``(start, stop)`` half-open bounds of each run.
    """
    start = None
    for index, flag in enumerate(mask):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            yield start, index
            start = None
    if start is not None:
        yield start, len(mask)


def sweep(mask: np.ndarray, window: int, stride: int) -> list[int]:
    """Window start indices covering every visible run.

    The same annotation-blind 50%-overlapping sweep every other corpus uses:
    the window is the model's receptive field, and blinks are recovered
    afterwards from the averaged per-frame signal.

    Args:
        mask (np.ndarray): ``(T,)`` visibility.
        window (int): Window length in frames.
        stride (int): Frames between window starts.

    Returns:
        list[int]: Start indices, ascending.
    """
    starts: list[int] = []
    for begin, end in runs(mask):
        if end - begin < window:
            continue
        starts.extend(range(begin, end - window + 1, stride))
    return starts


def video_fps(video_path: Path) -> float:
    """The rate a video declares, falling back to the corpus mode.

    Args:
        video_path (Path): The recording.

    Returns:
        float: Frames per second.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    declared = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 0.0
    capture.release()
    return float(declared) if declared and declared > 0 else FPS


@dataclass(frozen=True)
class Described:
    """One cropped eye, with what the builder knows about its quality.

    Args:
        image (np.ndarray): ``(3, S, S)`` normalised crop.
        feature (tuple | None): Descriptor and its validity, when extracted.
        face_box (tuple[int, int, int, int]): The person's face box this frame,
            ``(x, y, w, h)``, verbatim -- it may hang off the frame.
        on_screen (float): Fraction of the eye's landmarks inside the frame.
        span (float): Eye width in pixels, corner to corner.
        aspect (float): Eye contour height/width; collapses in profile.
        shifted (bool): Whether the crop was slid to stay inside the frame.
        pose (np.ndarray): ``(3,)`` head rotation in degrees, from *this
            person's* face box rather than the frame -- up to 24 tracklets share
            a frame here.
    """

    image: np.ndarray
    feature: tuple[np.ndarray, bool] | None
    face_box: tuple[int, int, int, int]
    pose: np.ndarray
    on_screen: float
    span: float
    aspect: float
    shifted: bool


def describe_video(
    video_path: Path,
    tracklets: list[Tracklet],
    image_size: int,
    extractor=None,
) -> dict[tuple[str, int, str], Described]:
    """Decode one video once and crop every annotated eye in it.

    Decoding dominates the runtime and a video holds up to 24 tracklets, so the
    frames are dealt out to all of them in a single pass rather than the video
    being reopened per person.

    Args:
        video_path (Path): The recording.
        tracklets (list[Tracklet]): Every person annotated in it.
        image_size (int): Output crop side length.
        extractor: Optional descriptor extractor; ``None`` for image-only.

    Returns:
        dict: ``(sample_prefix, frame_index, side)`` to a :class:`Described`.
        A missing key means the eye was not usable in that frame, which is what
        marks it invalid in whatever window covers it.

    Raises:
        MPEblinkError: If the video cannot be opened.
    """
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise MPEblinkError(f"{video_path}: cannot be opened for decoding.")

    cache: dict[tuple[str, int, str], Described] = {}
    index = 0
    offscreen = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]

            for tracklet in tracklets:
                if index >= tracklet.length or tracklet.landmarks[index] is None:
                    continue
                points = np.asarray(tracklet.landmarks[index], dtype=np.float64)
                # `frame_shape` slides a box back into the frame when the eye is
                # visible but its context box overhangs, and returns None only
                # when the eye itself has left. A None keeps the frame out of the
                # cache, which is what marks it invalid for its window.
                boxes = {
                    side: eye_box(points, side, frame_shape=(height, width))
                    for side in (LEFT, RIGHT)
                }
                # Distinguish "left the frame" from "too small to crop", which
                # eye_box also reports as None, so the log means one thing.
                offscreen += sum(
                    1
                    for side, box in boxes.items()
                    if box is None and eye_box(points, side) is not None
                )
                raw_box = tracklet.boxes[index] if index < len(tracklet.boxes) else None
                # Verbatim, including negatives: the corpus tracks faces past
                # the frame edge and clamping here would destroy that.
                face_box = (
                    tuple(int(v) for v in raw_box[:4]) if raw_box is not None else (0, 0, 0, 0)
                )
                # Pose comes from *this person's* face, not the frame: up to 24
                # tracklets share a frame here, and a frame-wide estimate would
                # hand all of them one set of angles.
                region = face_box if raw_box is not None else None
                described = (
                    extractor.frame_features(rgb, boxes, face_box=region)
                    if extractor is not None
                    else None
                )
                pose = (
                    extractor.head_pose(extractor._face_region(rgb, region))
                    if extractor is not None
                    else np.zeros(HEAD_POSE_DIM, dtype=np.float32)
                )
                for side, box in boxes.items():
                    if box is None:
                        continue
                    patch = cv2.resize(
                        box.crop(rgb), (image_size, image_size), interpolation=cv2.INTER_AREA
                    )
                    cache[(tracklet.sample_prefix, index, side)] = Described(
                        image=normalise_image(patch),
                        feature=described[side] if described is not None else None,
                        face_box=face_box,
                        on_screen=box.on_screen,
                        span=box.span,
                        aspect=box.aspect,
                        shifted=box.shifted,
                        pose=pose,
                    )
            index += 1
    finally:
        capture.release()

    if not index:
        raise MPEblinkError(f"{video_path}: decoded zero frames.")
    if offscreen:
        logger.debug(f"{video_path.parent.name}: masked {offscreen} eyes as off-frame.")
    return cache


def video_dirs(raw_dir: Path) -> list[tuple[Path, str]]:
    """Every video directory, with the split it belongs to.

    Args:
        raw_dir (Path): The extracted corpus root.

    Returns:
        list[tuple[Path, str]]: Directory and split name, in corpus order.

    Raises:
        MPEblinkError: If the corpus is missing or holds no videos.
    """
    if not raw_dir.is_dir():
        raise MPEblinkError(
            f"{NAME}: raw data not found at {raw_dir}. Extract MPEblink 2.0 there; "
            "see docs/data.md."
        )

    found: list[tuple[Path, str]] = []
    for source, subset in SPLIT_DIRS.items():
        base = raw_dir / source
        if not base.is_dir():
            continue
        for entry in sorted(base.iterdir(), key=lambda p: p.name):
            # AppleDouble sidecars: an extracted archive is littered with
            # `._name` files that are not directories but glob as if they were.
            if entry.name.startswith("._") or not entry.is_dir():
                continue
            if (entry / ANNOTATION).is_file() and (entry / VIDEO).is_file():
                found.append((entry, subset))

    if not found:
        raise MPEblinkError(f"{NAME}: no annotated videos under {raw_dir}.")
    return found


def process(
    root: Path = PROJECT_ROOT,
    image_size: int = 64,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    with_features: bool = True,
    limit: int | None = None,
    device: int | None = None,
) -> Path:
    """Build the corpus into one HDF5 file.

    Args:
        root (Path): Repository root.
        image_size (int): Eye crop side length.
        window_seconds (float): Analysis window duration.
        with_features (bool): Extract the 160-d handcrafted descriptors.
        limit (int | None): Process only the first N videos per split.
        device (int | None): GPU index for extraction; ``None`` for CPU.

    Returns:
        Path: The written HDF5 file.
    """
    raw_dir = root / "data" / "raw" / RAW_DIRNAME
    processed_dir = root / "data" / "processed" / NAME
    h5_path = processed_dir / f"{NAME}.h5"

    entries = video_dirs(raw_dir)
    if limit is not None:
        kept: dict[str, int] = {}
        entries = [
            (path, subset)
            for path, subset in entries
            if kept.setdefault(subset, 0) < limit and not kept.update({subset: kept[subset] + 1})
        ]

    extractor = None
    if with_features:
        logger.info(f"{NAME}: loading the exordium extraction stack...")
        from blinklinmult.preprocess.extractors import ExordiumExtractor

        extractor = ExordiumExtractor(device_id=device)

    spec = DatasetSpec(
        name=NAME,
        fps=FPS,
        window_seconds=window_seconds,
        image_size=image_size,
        feature_dim=EYE_FEATURE_DIM if with_features else None,
        has_blink_presence=True,
        # The annotation marks blink events and carries no per-frame closure
        # flag, so there is no eye state to teach.
        has_eye_state=False,
        has_head_pose=with_features,
        has_blink_ids=True,
        # No symmetry: every signal must describe one eye alone, since a sample
        # is one eye and a partner-derived value would leak across samples.
        quality_signals=(("eye_blur", "eye_exposure", "eye_jitter") if with_features else ()),
    )

    config_path = root / "config" / "data" / f"{NAME}.yaml"
    config_yaml = config_path.read_text() if config_path.is_file() else ""
    sha = git_sha(root)
    extra_attrs = {"protocol": "sweep", "instances": "oracle"}
    written = 0

    # One shard per video, so an interrupted build resumes rather than restarts:
    # decoding 1 842 untrimmed videos takes the better part of a day, and a
    # single-file build has no notion of progress. A shard is renamed into place
    # only after it closes cleanly, so a shard that exists is a video that
    # finished. See :mod:`blinklinmult.data.shards`.
    shard_dir(h5_path).mkdir(parents=True, exist_ok=True)
    entries = pending(entries, h5_path, lambda item: f"{item[1]}_{item[0].name}")

    if entries:
        for video_dir, subset in progress(entries, f"{NAME} videos"):
            video_id = f"{subset}_{video_dir.name}"
            tracklets = read_tracklets(video_dir / ANNOTATION, video_id)
            if not tracklets:
                # Still shard it, empty: without this the video is retried on
                # every resume, forever.
                with H5Writer(
                    spec,
                    shard_path(h5_path, video_id),
                    config_yaml=config_yaml,
                    git_sha=sha,
                    extra_attrs=extra_attrs,
                ):
                    pass
                continue

            video_path = video_dir / VIDEO
            window = window_frames(video_fps(video_path), window_seconds)
            stride = max(1, window // 2)

            cache = describe_video(video_path, tracklets, image_size, extractor)

            with H5Writer(
                spec,
                shard_path(h5_path, video_id),
                config_yaml=config_yaml,
                git_sha=sha,
                extra_attrs=extra_attrs,
            ) as writer:
                for tracklet in tracklets:
                    mask = tracklet.visible()
                    for start in sweep(mask, window, stride):
                        frames = list(range(start, start + window))
                        for side in (LEFT, RIGHT):
                            images, valid = [], []
                            features: list[tuple[np.ndarray, bool] | None] = []
                            face_boxes, on_screen, spans, aspects, shifts = [], [], [], [], []
                            poses: list[np.ndarray] = []
                            for index in frames:
                                entry = cache.get((tracklet.sample_prefix, index, side))
                                if entry is None:
                                    images.append(
                                        np.zeros((3, image_size, image_size), dtype=np.float32)
                                    )
                                    valid.append(False)
                                    features.append(None)
                                    face_boxes.append((0, 0, 0, 0))
                                    on_screen.append(0.0)
                                    spans.append(0.0)
                                    aspects.append(0.0)
                                    shifts.append(False)
                                    poses.append(np.zeros(HEAD_POSE_DIM, dtype=np.float32))
                                else:
                                    images.append(entry.image)
                                    valid.append(True)
                                    features.append(entry.feature)
                                    face_boxes.append(entry.face_box)
                                    on_screen.append(entry.on_screen)
                                    spans.append(entry.span)
                                    aspects.append(entry.aspect)
                                    shifts.append(entry.shifted)
                                    poses.append(entry.pose)

                            if not any(valid):
                                continue

                            valid_mask = np.asarray(valid, dtype=bool)
                            # INVALID_BLINK where the tracker saw nothing, so the
                            # field distinguishes "eye was open" from "we never saw
                            # this frame" without a cross-check against the mask.
                            window_ids = tracklet.blink_ids[frames].astype(np.int32)
                            window_ids[~valid_mask] = INVALID_BLINK
                            quality = {
                                FACE_BOX: np.asarray(face_boxes, dtype=np.int32),
                                EYE_ON_SCREEN: np.asarray(on_screen, dtype=np.float32),
                                EYE_SPAN: np.asarray(spans, dtype=np.float32),
                                EYE_ASPECT: np.asarray(aspects, dtype=np.float32),
                                BOX_SHIFTED: np.asarray(shifts, dtype=bool),
                                LANDMARK_SOURCE: tracklet.landmark_source,
                                BLINK_CATEGORY: tracklet.blink_categories[frames].astype(np.int32),
                                CONFIDENCE: np.float32(
                                    window_confidence(
                                        np.asarray(on_screen),
                                        np.asarray(spans),
                                        np.asarray(aspects),
                                        valid=valid_mask,
                                    )
                                ),
                            }

                            values = mask_values = None
                            if with_features:
                                values, mask_values = stack_window(
                                    [
                                        f if f is not None else (empty_feature(), False)
                                        for f in features
                                    ]
                                )

                            writer.add(
                                subset=subset,
                                video_id=tracklet.sample_prefix,
                                frame_group=f"{start:06d}",
                                eye_side=side,
                                eye_images=np.stack(images),
                                eye_image_mask=np.asarray(valid, dtype=bool),
                                blink_presence=tracklet.blinks[frames],
                                blink_id=window_blink_id(window_ids),
                                eye_features=values,
                                eye_feature_mask=mask_values,
                                head_pose=np.stack(poses) if with_features else None,
                                blink_ids=window_ids,
                                quality_signals=(
                                    _window_quality(images, valid_mask, face_boxes, spans)
                                    if with_features
                                    else None
                                ),
                                quality=quality,
                            )
                            written += 1

    # Every video now has a shard; combining them is I/O-bound and takes minutes
    # against the day the decode takes.
    written = merge_shards(h5_path, shard_attrs(h5_path))

    logger.info(f"{NAME}: wrote {written} samples to {h5_path}")
    return h5_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Repository root.")
    parser.add_argument("--image-size", type=int, default=64, help="Eye crop side length.")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Analysis window in seconds.",
    )
    parser.add_argument("--no-features", action="store_true", help="Skip the exordium descriptors.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N videos per split."
    )
    parser.add_argument("--device", type=int, default=None, help="GPU index for extraction.")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    process(
        args.root,
        image_size=args.image_size,
        window_seconds=args.window_seconds,
        with_features=not args.no_features,
        limit=args.limit,
        device=args.device,
    )


if __name__ == "__main__":
    main()
