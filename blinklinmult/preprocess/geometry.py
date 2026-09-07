"""Locating the eye regions of a frame, from annotation or from detection.

Two routes reach the same thing — a square crop box per eye:

**From the annotation.** The ``.tag`` corpora (EyeBlink8, TalkingFace, RN)
record each eye's two corners per frame, so the box is read straight off the
label. This is what the published numbers were produced from, and it costs
nothing to compute.

**From detection.** An unseen video has no ``.tag`` file, so the eyes must be
found: face detection, then dense landmarks over the face crop, then the eye
regions read from the landmark indices. This is the path a deployed model
takes, and having both lets the cost of automatic localisation be measured
rather than assumed.

**The crop is square and scaled to the eye.** A box tight to the two corners
would clip the lid and brow, and the deformation of the surrounding skin is
much of what separates a blink from a downward glance. The side is therefore
:data:`EYE_CROP_SCALE` times the corner-to-corner distance, centred on the eye:
square, so the later resize to a fixed size never changes the aspect ratio, and
wide enough that the iris landmarker has the context it expects.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from blinklinmult.preprocess.common import PreprocessError, crop_square, eye_centre, eye_span

logger = logging.getLogger(__name__)
"""Module-level logger."""

EYE_CROP_SCALE = 2.0
"""Crop side as a multiple of the eye's corner-to-corner distance.

Set by what the *landmarker* needs, not by what looks tight. ``IrisWrapper``
runs a small FaceMesh subnet that regresses 71 points around the eye region,
and it was trained on crops carrying a margin well beyond the palpebral
fissure. Starve it of that context and the landmarks lose precision — the
descriptors degrade quietly, since nothing about the pipeline fails.

Measured on TalkingFace, whose median eye span is 50 px against a 246 px face,
``2.0`` gives a 100 px box: the eye, the full lid, the brow, and the
surrounding skin whose deformation is much of what separates a blink from a
downward glance.
"""

MIN_EYE_SPAN = 4.0
"""Smallest corner-to-corner span, in pixels, worth cropping.

Below this the annotation is degenerate — usually two coincident points — and
the crop would be interpolation noise rather than an eye.
"""

MIN_EYE_ON_SCREEN = 0.5
"""Least of an eye's own landmarks that must be on screen for it to count.

Measured on the **eye**, never on its crop box: the box carries
:data:`EYE_CROP_SCALE` times the eye's width in context, so it overflows the
frame long before the eye does. Over 244 000 MPEblink crops, 1.06% of boxes
overflow but only 0.47% hold an eye that is genuinely off-frame — gating on the
box would throw away 1 423 evaluable eyes.

Where the eye is on screen and the box is not, :meth:`EyeBox.shifted_into`
slides the box back into the frame rather than padding it with black. Only an
eye actually leaving the frame is masked, because a black patch carrying a
blink label teaches the model that black means whatever the label says.
"""

LEFT = "left"
"""Subject's left eye."""

RIGHT = "right"
"""Subject's right eye."""


class GeometryError(PreprocessError):
    """Raised when an eye region cannot be located."""


@dataclass(frozen=True)
class EyeBox:
    """A square crop region for one eye.

    Args:
        centre_x (int): Box centre, x, in frame pixels.
        centre_y (int): Box centre, y, in frame pixels.
        side (int): Side length in frame pixels.
        span (float): The eye's corner-to-corner distance the side derives
            from, kept for diagnostics.
        aspect (float): The eye contour's height/width ratio. Falls towards
            zero both in profile and mid-blink; see
            :data:`~blinklinmult.data.schema.EYE_ASPECT`.
        on_screen (float): Fraction of the eye's landmarks inside the frame.
            ``1.0`` unless the builder was given the frame size.
        shifted (bool): Whether :meth:`shifted_into` moved this box to keep it
            inside the frame.
    """

    centre_x: int
    centre_y: int
    side: int
    span: float
    aspect: float = 0.0
    on_screen: float = 1.0
    shifted: bool = False

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        """The box as ``(x1, y1, x2, y2)``, for drawing and comparison.

        Returns:
            tuple[int, int, int, int]: Corner coordinates; may fall outside the
            frame, which :func:`~blinklinmult.preprocess.common.crop_square`
            pads rather than clips.
        """
        half = self.side // 2
        return (
            self.centre_x - half,
            self.centre_y - half,
            self.centre_x - half + self.side,
            self.centre_y - half + self.side,
        )

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Cut this box out of a frame.

        Args:
            frame (np.ndarray): Source image, ``(H, W, 3)``.

        Returns:
            np.ndarray: ``(side, side, 3)`` patch, zero-padded where the box
            leaves the frame.
        """
        return crop_square(frame, self.centre_x, self.centre_y, self.side)

    def visible_fraction(self, height: int, width: int) -> float:
        """How much of this box actually falls on the frame.

        Diagnostic only. **Do not gate an eye on this**: the box carries
        :data:`EYE_CROP_SCALE` times the eye's own width in context, so an eye
        sitting comfortably inside the frame still overflows it near an edge.
        Measured over 244 000 MPEblink crops, 1.06% of boxes overflow while
        only 0.47% have an eye genuinely off-frame — gating on box area would
        discard 1 423 perfectly evaluable eyes, 55% of the overflow cases.
        Use :meth:`shifted_into` and gate on the eye landmarks instead.

        Args:
            height (int): Frame height in pixels.
            width (int): Frame width in pixels.

        Returns:
            float: Fraction of the box's area inside the frame, ``0.0`` when it
            lies entirely outside and ``1.0`` when fully contained.
        """
        x1, y1, x2, y2 = self.xyxy
        inside_w = max(0, min(x2, width) - max(x1, 0))
        inside_h = max(0, min(y2, height) - max(y1, 0))
        return (inside_w * inside_h) / float(self.side * self.side)

    def shifted_into(self, height: int, width: int) -> EyeBox:
        """The same box slid to lie inside the frame.

        A face near the frame edge has a visible, annotated, perfectly
        evaluable eye whose *context* box hangs off the screen. Padding that
        with black throws away real pixels the frame does hold, so the box is
        translated until it fits.

        **The side never changes.** Clipping the box to the bounds instead
        would make it non-square, and the later resize would then stretch it by
        a different factor than a mid-frame crop — the 1.x bug
        :func:`~blinklinmult.preprocess.common.crop_square` exists to avoid. A
        shifted crop keeps the eye at the same scale and merely moves it off
        centre, which is the cheaper distortion by far.

        A box larger than the frame cannot be made to fit; it is clamped to the
        top-left and still padded, which is the honest outcome.

        Args:
            height (int): Frame height in pixels.
            width (int): Frame width in pixels.

        Returns:
            EyeBox: A box of identical ``side``, translated to fit where it can.
        """
        half = self.side // 2
        x1, y1 = self.centre_x - half, self.centre_y - half
        moved_x = max(0, min(x1, width - self.side))
        moved_y = max(0, min(y1, height - self.side))
        return EyeBox(
            centre_x=moved_x + half,
            centre_y=moved_y + half,
            side=self.side,
            span=self.span,
            aspect=self.aspect,
            on_screen=self.on_screen,
            shifted=self.shifted or (moved_x, moved_y) != (x1, y1),
        )


def box_from_corners(corners: np.ndarray, scale: float = EYE_CROP_SCALE) -> EyeBox | None:
    """Build a crop box from an eye's two annotated corners.

    Args:
        corners (np.ndarray): ``(x1, y1, x2, y2)`` eye corners.
        scale (float): Side as a multiple of the corner distance.

    Returns:
        EyeBox | None: The box, or ``None`` when the annotation is degenerate
        or absent — the caller masks that frame rather than cropping noise.
    """
    span = eye_span(corners)
    if span < MIN_EYE_SPAN:
        return None

    centre_x, centre_y = eye_centre(corners)
    return EyeBox(
        centre_x=centre_x,
        centre_y=centre_y,
        side=int(round(span * scale)),
        span=float(span),
    )


def box_from_landmarks(points: np.ndarray, scale: float = EYE_CROP_SCALE) -> EyeBox | None:
    """Build a crop box from an eye's dense landmarks.

    The detection counterpart of :func:`box_from_corners`. FaceMesh gives 16
    points around each eye; the two most distant of them are the corners, which
    puts both routes on the same definition of "span" and makes the resulting
    crops comparable.

    Args:
        points (np.ndarray): ``(N, 2)`` landmark coordinates in frame pixels.
        scale (float): Side as a multiple of the corner distance.

    Returns:
        EyeBox | None: The box, or ``None`` when the landmarks are degenerate.

    Raises:
        GeometryError: If the array is not ``(N, 2)`` with at least two points.
    """
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or array.shape[0] < 2:
        raise GeometryError(f"expected (N, 2) landmarks with N >= 2, got shape {array.shape}.")

    # The widest pair, rather than two nominated indices: the corner indices
    # differ between landmark sets, and the extent is what the scale multiplies.
    x_min, y_min = array.min(axis=0)
    x_max, y_max = array.max(axis=0)
    corners = np.asarray([x_min, y_min, x_max, y_max], dtype=np.int32)
    return box_from_corners(corners, scale=scale)


def boxes_from_annotation(record, scale: float = EYE_CROP_SCALE) -> dict[str, EyeBox | None]:
    """Locate both eyes from one annotated frame.

    Args:
        record (TagRecord): The frame's annotation.
        scale (float): Side as a multiple of the corner distance.

    Returns:
        dict[str, EyeBox | None]: Boxes keyed by :data:`LEFT` / :data:`RIGHT`;
        ``None`` where that eye's annotation is degenerate.
    """
    return {
        LEFT: box_from_corners(record.left_eye_corners, scale=scale),
        RIGHT: box_from_corners(record.right_eye_corners, scale=scale),
    }


EYE_SPAN_RATIO = 0.412
"""Eye corner-to-corner span as a fraction of the inter-eye distance.

Lets a crop box be built from two eye *centres* alone, which is what a face
detector's coarse keypoints give -- no dense landmarks, no second model.

**Measured, not assumed.** Over 160 eyes of the bundled TalkingFace clip, each
eye's FaceMesh-derived span was divided by the distance between the two YOLO
eye keypoints: median **0.412**, standard deviation **0.016**, 5-95% range
``[0.383, 0.437]``. The spread is that tight because both quantities are fixed
by the same facial anatomy, which is why the approximation holds rather than
merely happening to work on one clip.

The resulting box is fed through the same :data:`EYE_CROP_SCALE` as every other
path, so a keypoint-derived crop and a landmark-derived one frame the eye the
same way.
"""


def box_from_eye_centres(
    left_centre: np.ndarray,
    right_centre: np.ndarray,
    which: str,
    scale: float = EYE_CROP_SCALE,
) -> EyeBox | None:
    """Build a crop box from the two eye centres a detector already gives.

    The cheap counterpart of :func:`box_from_landmarks`. A face detector with
    pose output returns one point per eye in the same forward pass as the box,
    so this needs no landmark model at all -- measured at ~0.05 ms against
    ~5 ms for FaceMesh.

    The eye's span cannot be observed from a single point, so it is estimated
    from the distance between the two centres via :data:`EYE_SPAN_RATIO`.

    **What this loses.** :attr:`EyeBox.aspect` -- the eye contour's height/width,
    which falls towards zero both in profile and mid-blink -- cannot be derived
    from a centre and is left at its ``0.0`` default. Any caller that reads
    ``aspect`` (the quality signals, the corpus builders) must keep the dense
    landmark path; this is for live inference, where the two eyes are cropped and
    scored and nothing downstream inspects the contour.

    Args:
        left_centre (np.ndarray): ``(2,)`` viewer's-left eye centre, in frame
            pixels.
        right_centre (np.ndarray): ``(2,)`` viewer's-right eye centre.
        which (str): Which eye to build, :data:`LEFT` or :data:`RIGHT`.
        scale (float): Side as a multiple of the estimated span.

    Returns:
        EyeBox | None: The box, or ``None`` when the two centres are too close
        to imply a usable span -- a detection that collapsed rather than a face.

    Raises:
        GeometryError: If ``which`` is not a known side, since silently
            cropping the wrong eye is the failure this guards against.
    """
    if which not in (LEFT, RIGHT):
        raise GeometryError(f"Unknown eye side {which!r}; expected {LEFT!r} or {RIGHT!r}.")

    left = np.asarray(left_centre, dtype=np.float64).reshape(-1)[:2]
    right = np.asarray(right_centre, dtype=np.float64).reshape(-1)[:2]
    inter_eye = float(np.linalg.norm(right - left))
    span = inter_eye * EYE_SPAN_RATIO
    if span < MIN_EYE_SPAN:
        return None

    centre = left if which == LEFT else right
    return EyeBox(
        centre_x=int(round(float(centre[0]))),
        centre_y=int(round(float(centre[1]))),
        side=int(round(span * scale)),
        span=span,
    )


POSE_KEYPOINTS = 5
"""Keypoints a face-pose detector supplies: both eyes, nose, two mouth corners."""

NOSE_DEPTH_RATIO = 0.5
"""Where the nose sits between the eye line and the mouth line at zero pitch.

Halfway, by construction of the face. Deviation from it is what
:func:`pose_from_keypoints` reads as pitch.
"""


def pose_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """Estimate head rotation from a detector's five facial keypoints.

    A geometric alternative to running a dedicated pose network. Measured on
    CPU: **0.05 ms against 26.7 ms for 6DRepNet**, a 500x saving, because the
    keypoints have already been computed by the face detector -- there is no
    second forward pass at all. That difference is what puts the whole streaming
    pipeline inside a 30 fps budget on CPU.

    Accuracy against 6DRepNet over 80 frames of the bundled clip:

    * **roll** -- correlation **+0.94**. The eye line's angle *is* roll, so this
      is close to a direct measurement.
    * **pitch** -- correlation **-0.94** before the sign convention is matched,
      i.e. the same signal. Read from where the nose falls between the eye line
      and the mouth line.
    * **yaw** -- **not validated.** The clip is near-frontal, 6DRepNet spanning
      only -3.5 to +2.4 degrees, so there was no rotation to correlate against.

    **The yaw caveat matters, because yaw is what self-occlusion gating uses**
    (:data:`~blinklinmult.pipeline.YAW_LIMIT`). The estimate here is a
    perspective approximation: as the head turns, the nose's projected offset
    from the eye midpoint grows, but the relationship is not linear and this
    does not model it. Treat it as a *usable ordering* rather than a calibrated
    angle, and use 6DRepNet where the number itself must be right.

    The sign convention follows the rest of the pipeline: **positive yaw turns
    the nose toward image-left**, occluding the viewer's right eye.

    Args:
        keypoints (np.ndarray): ``(5, 2)`` points in frame pixels, ordered
            left eye, right eye, nose, left mouth corner, right mouth corner --
            the layout an ultralytics pose model returns.

    Returns:
        np.ndarray: ``(3,)`` ``[yaw, pitch, roll]`` in degrees, matching the
        order :meth:`~blinklinmult.preprocess.extractors.ExordiumExtractor.head_pose`
        returns.

    Raises:
        GeometryError: If the array is not ``(5, 2)``, since a different
            keypoint layout would silently produce plausible wrong angles.
    """
    points = np.asarray(keypoints, dtype=np.float64)
    if points.shape != (POSE_KEYPOINTS, 2):
        raise GeometryError(
            f"Expected {POSE_KEYPOINTS} keypoints of (x, y), got shape {points.shape}."
        )

    left_eye, right_eye, nose = points[0], points[1], points[2]
    mouth_centre = (points[3] + points[4]) / 2.0
    eye_centre_point = (left_eye + right_eye) / 2.0

    eye_vector = right_eye - left_eye
    inter_eye = float(np.linalg.norm(eye_vector))
    if inter_eye < MIN_EYE_SPAN:
        # The two eyes have collapsed onto each other: no geometry to read.
        return np.zeros(3, dtype=np.float32)

    # Roll is the eye line's tilt, read directly.
    roll = float(np.degrees(np.arctan2(float(eye_vector[1]), float(eye_vector[0]))))

    # Yaw from the nose's sideways offset, normalised by half the inter-eye
    # distance so it is scale free. Negated because a nose to image-*left* --
    # smaller x than the eye midpoint -- is positive yaw here.
    offset = float(nose[0] - eye_centre_point[0]) / (inter_eye / 2.0)
    yaw = float(np.degrees(np.arcsin(np.clip(-offset, -1.0, 1.0))))

    # Pitch from how far down the eye-to-mouth axis the nose projects. At zero
    # pitch it sits halfway; looking down moves it towards the eyes.
    axis = mouth_centre - eye_centre_point
    axis_length_squared = float(axis @ axis)
    if axis_length_squared <= 0.0:
        pitch = 0.0
    else:
        along = float((nose - eye_centre_point) @ axis) / axis_length_squared
        # Negated to match 6DRepNet's sign: measured correlation over 80 frames
        # is -0.94 without it and +0.94 with it, so the two disagree on
        # direction only. A caller mixing the two sources must get the same
        # sign or an occlusion rule keyed on pitch would invert.
        pitch = float(np.degrees(np.arcsin(np.clip(-(along - NOSE_DEPTH_RATIO) * 2.0, -1.0, 1.0))))

    return np.array([yaw, pitch, roll], dtype=np.float32)
