"""The exordium-backed implementations of eye-feature extraction.

Separated from :mod:`blinklinmult.preprocess.features` because these two
classes construct exordium's models, which download multi-GB weights and want a
GPU. The pure logic they depend on — the fixed feature layout, the flattening,
the masking rules — lives in that module and is unit-tested; what is here is
the binding to the upstream library, exercised by running the pipeline.

Both classes are constructed once per corpus run and reused across every frame:
loading the weights costs far more than any single call.

**One bad frame must not end a multi-hour build.** A detector or landmarker
that fails on a frame yields a masked entry and a debug log line, not an
exception. The one exception is :class:`~blinklinmult.preprocess.features.FeatureError`
from a layout mismatch, which means exordium's output shape changed — a bug
that must surface rather than quietly mask every frame in the corpus.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from blinklinmult.preprocess.features import (
    HEADPOSE_DIM,
    FeatureError,
    _to_numpy,
    assemble,
    empty_feature,
    flatten_iris,
)
from blinklinmult.preprocess.geometry import (
    LEFT,
    RIGHT,
    EyeBox,
    box_from_eye_centres,
    box_from_landmarks,
    pose_from_keypoints,
)

logger = logging.getLogger(__name__)
"""Module-level logger."""

FACE_POSE_MARGIN = 0.25
"""Padding around a face box before head-pose estimation, as a fraction.

A box tight to the face clips the head outline -- jaw, crown, ears -- that a
pose estimator reads orientation from, so it is widened by a quarter on each
side. Clipped to the frame, since MPEblink tracks faces past the edge.
"""


class ExordiumExtractor:
    """The real extractor: exordium's iris landmarker and head-pose estimator.

    Constructing this loads model weights, so it is built once per corpus run
    and reused across every frame.

    Args:
        device_id (int | None): GPU index; ``None`` or negative selects CPU.

    Raises:
        FeatureError: If exordium is not installed, or cannot be imported.
    """

    def __init__(self, device_id: int | None = None):
        try:
            from exordium.video.face.headpose import SixDRepNetWrapper
            from exordium.video.face.landmark.iris import IrisWrapper
        except ImportError as error:
            raise FeatureError(
                "eye-feature extraction needs exordium: `uv sync --extra preprocess`. "
                f"Import failed with: {error}"
            ) from error

        self.iris = IrisWrapper(device_id=device_id)
        self.headpose = SixDRepNetWrapper(device_id=device_id)

    def frame_features(
        self,
        frame: np.ndarray,
        boxes: dict[str, EyeBox | None],
        face_box: tuple[int, int, int, int] | None = None,
    ) -> dict[str, tuple[np.ndarray, bool]]:
        """Describe both eyes of one face.

        Head pose is computed once and shared by both eyes -- it describes the
        head, not the eye.

        **Pass ``face_box`` whenever a frame can hold more than one person.**
        Without it the pose estimator sees the whole frame and returns a single
        pose, so in a 24-person MPEblink shot every tracklet would be given the
        same angles, and the 3 pose dimensions of the 160-d descriptor would be
        wrong for all but one of them.

        Args:
            frame (np.ndarray): Full frame, ``(H, W, 3)`` uint8 RGB.
            boxes (dict[str, EyeBox | None]): Crop boxes per eye side.
            face_box (tuple | None): This person's face box as ``(x, y, w, h)``.
                ``None`` estimates pose from the whole frame, which is correct
                only for single-subject footage.

        Returns:
            dict[str, tuple[np.ndarray, bool]]: Descriptor and validity per side.
        """
        pose, pose_valid = self._pose(self._face_region(frame, face_box))

        result: dict[str, tuple[np.ndarray, bool]] = {}
        for side in (LEFT, RIGHT):
            box = boxes.get(side)
            if box is None or not pose_valid:
                result[side] = (empty_feature(), False)
                continue
            result[side] = self._eye(box.crop(frame), pose)
        return result

    @staticmethod
    def _face_region(frame: np.ndarray, face_box: tuple[int, int, int, int] | None) -> np.ndarray:
        """Cut a person's face out of the frame, with margin, for pose.

        The box is padded by :data:`FACE_POSE_MARGIN` because a tight face crop
        starves the pose estimator of the head outline it reads orientation
        from. MPEblink tracks faces past the frame edge, so the padded box is
        clipped to what the frame actually holds.

        Args:
            frame (np.ndarray): Full frame, ``(H, W, 3)``.
            face_box (tuple | None): ``(x, y, w, h)``, possibly out of bounds.

        Returns:
            np.ndarray: The face region, or the whole frame when no box is
            given or the box has no overlap with it.
        """
        if face_box is None:
            return frame
        x, y, width, height = (int(v) for v in face_box)
        if width <= 0 or height <= 0:
            return frame
        margin_x, margin_y = int(width * FACE_POSE_MARGIN), int(height * FACE_POSE_MARGIN)
        frame_h, frame_w = frame.shape[:2]
        x0 = max(0, x - margin_x)
        y0 = max(0, y - margin_y)
        x1 = min(frame_w, x + width + margin_x)
        y1 = min(frame_h, y + height + margin_y)
        if x1 <= x0 or y1 <= y0:
            return frame
        return frame[y0:y1, x0:x1]

    def _pose(self, frame: np.ndarray) -> tuple[np.ndarray, bool]:
        """Estimate head pose from a frame.

        Args:
            frame (np.ndarray): ``(H, W, 3)`` uint8 RGB.

        Returns:
            tuple[np.ndarray, bool]: ``(3,)`` degrees and whether it is usable.
        """
        try:
            angles = self.headpose(frame)
        except Exception as error:  # noqa: BLE001 - one bad frame must not end a corpus
            logger.debug(f"head pose failed on a frame: {error}")
            return np.zeros(HEADPOSE_DIM, dtype=np.float32), False
        return _to_numpy(angles).reshape(-1)[:HEADPOSE_DIM].astype(np.float32), True

    def head_pose(self, frame: np.ndarray) -> np.ndarray:
        """Head rotation for one frame, in degrees.

        The public form of :meth:`_pose`, for corpora that want the angles as a
        stored field without extracting the handcrafted descriptor around them.
        A failed estimate returns zeros rather than raising -- a frontal reading
        is the least-wrong default, and the corpora that care record their own
        validity alongside.

        Args:
            frame (np.ndarray): ``(H, W, 3)`` uint8 RGB, the whole image.

        Returns:
            np.ndarray: ``(3,)`` ``[yaw, pitch, roll]`` in degrees.
        """
        angles, _ = self._pose(frame)
        return angles

    def _eye(self, patch: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, bool]:
        """Describe one eye crop.

        Args:
            patch (np.ndarray): ``(H, W, 3)`` uint8 RGB eye crop.
            pose (np.ndarray): ``(3,)`` head pose in degrees.

        Returns:
            tuple[np.ndarray, bool]: ``(160,)`` descriptor and its validity.
        """
        import torch

        # eye_to_feature wants (3, H, W) uint8, so the crop is taken before
        # normalisation -- an ordering the streaming pass must preserve.
        tensor = torch.from_numpy(np.ascontiguousarray(patch.transpose(2, 0, 1)))
        try:
            raw = self.iris.eye_to_feature(tensor)
            return assemble(flatten_iris(raw), pose), True
        except FeatureError:
            # A layout change is a bug, not a bad frame: let it surface.
            raise
        except Exception as error:  # noqa: BLE001 - a crop the landmarker cannot read
            logger.debug(f"iris landmarks failed on an eye crop: {error}")
            return empty_feature(), False


@dataclass(frozen=True)
class FaceDetection:
    """One face, everything a single detection pass yields.

    Args:
        face_box (tuple[int, int, int, int]): ``(x1, y1, x2, y2)`` in frame
            pixels.
        landmarks (np.ndarray): ``(478, 2)`` FaceMesh points, in **frame**
            coordinates -- the detection offset is already added back on.
        eyes (dict[str, EyeBox | None]): One box per side, ``None`` where the
            landmarks were degenerate. Sides are named from the **viewer's**
            point of view, matching the corpora rather than exordium.
    """

    face_box: tuple[int, int, int, int]
    landmarks: np.ndarray
    eyes: dict[str, EyeBox | None]


class FaceMeshLocator:
    """Locates eyes without annotation: detect a face, then read its landmarks.

    The route an unseen video takes. The ``.tag`` corpora carry eye corners and
    should use them (:func:`~blinklinmult.preprocess.geometry.boxes_from_annotation`);
    this exists so the same corpora can also be processed *without* their
    labels, which is what makes the cost of automatic localisation measurable
    rather than assumed.

    Args:
        device_id (int | None): GPU index; ``None`` or negative selects CPU.

    Raises:
        FeatureError: If exordium is not installed, or cannot be imported.
    """

    def __init__(self, device_id: int | None = None, detector_image_size: int | None = None):
        """Load the detector and the landmark model.

        Args:
            device_id (int | None): Passed to the detector.
            detector_image_size (int | None): Detector input side. ``None``
                keeps ultralytics' 640 default, which is what every published
                corpus was built with -- **do not change it for corpus work**,
                or the crops shift and the results stop being reproducible.
                Streaming callers should use :class:`PoseEyeLocator` instead,
                which defaults to the fast configuration.
        """
        try:
            from exordium.video.face.detector.yolo11 import YoloFace11Detector
            from exordium.video.face.landmark.constants import FaceMesh478Regions
            from exordium.video.face.landmark.facemesh import FaceMeshWrapper
        except ImportError as error:
            raise FeatureError(
                "automatic eye localisation needs exordium: "
                f"`uv sync --extra preprocess`. Import failed with: {error}"
            ) from error

        self.detector = YoloFace11Detector(device_id=device_id)
        if detector_image_size is not None:
            self.detector.model.overrides["imgsz"] = detector_image_size
        self.facemesh = FaceMeshWrapper()
        # The two conventions are mirror images, and this crossing is load
        # bearing. exordium names its regions from the *subject's* point of
        # view; the .tag files name theirs from the *viewer's*. Measured on
        # TalkingFace frame 170, the .tag "left" eye sits at x=322 and
        # FaceMesh's LEFT_EYE at x=433 -- a whole inter-eye distance apart.
        # Mapping them straight across puts every detected crop on the wrong
        # eye, which still trains and still scores, just against the other eye.
        self.regions = {
            LEFT: FaceMesh478Regions.RIGHT_EYE,
            RIGHT: FaceMesh478Regions.LEFT_EYE,
        }

    def boxes(self, frame: np.ndarray) -> dict[str, EyeBox | None]:
        """Locate both eyes in a frame.

        A thin wrapper over :meth:`detect` for callers that want only the eyes.

        Args:
            frame (np.ndarray): ``(H, W, 3)`` uint8 RGB.

        Returns:
            dict[str, EyeBox | None]: Boxes per side; ``None`` where no face or
            no usable landmarks were found.
        """
        found = self.detect(frame)
        if found is None:
            return {LEFT: None, RIGHT: None}
        return found.eyes

    def detect(self, frame: np.ndarray) -> FaceDetection | None:
        """Locate the face, its landmarks and both eyes, in one pass.

        :meth:`boxes` returns only the eyes, which is all the corpus builders
        need. Anything drawing an overlay -- or tracking a subject across frames
        -- also wants the face box and the dense landmarks, and re-detecting to
        get them would double the cost of the most expensive stage.

        Args:
            frame (np.ndarray): ``(H, W, 3)`` uint8 RGB.

        Returns:
            FaceDetection | None: The face box, its ``(478, 2)`` landmarks in
            frame coordinates, and an :class:`~blinklinmult.preprocess.geometry.EyeBox`
            per side. ``None`` when no face was found; a *side* is ``None`` when
            its landmarks were degenerate.
        """
        located = self._locate(frame)
        if located is None:
            return None
        face_box, landmarks = located

        eyes: dict[str, EyeBox | None] = {}
        for side, indices in self.regions.items():
            try:
                eyes[side] = box_from_landmarks(landmarks[list(indices)])
            except Exception as error:  # noqa: BLE001 - degenerate landmarks
                logger.debug(f"{side} eye box from landmarks failed: {error}")
                eyes[side] = None
        return FaceDetection(face_box=face_box, landmarks=landmarks, eyes=eyes)

    def _landmarks(self, frame: np.ndarray) -> np.ndarray | None:
        """Dense landmarks alone, discarding the face box.

        Args:
            frame (np.ndarray): ``(H, W, 3)`` uint8 RGB.

        Returns:
            np.ndarray | None: ``(478, 2)`` landmarks, or ``None``.
        """
        located = self._locate(frame)
        return None if located is None else located[1]

    def _locate(self, frame: np.ndarray) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """Detect a face and read its landmarks, keeping both.

        Args:
            frame (np.ndarray): ``(H, W, 3)`` uint8 RGB.

        Returns:
            tuple | None: ``((x1, y1, x2, y2), (478, 2) landmarks)`` in frame
            coordinates, or ``None`` when no face or no mesh was found.
        """
        import torch

        try:
            # detect_image, not __call__, and it wants (3, H, W) uint8 RGB.
            detections = self.detector.detect_image(
                torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
            )
        except Exception as error:  # noqa: BLE001 - one bad frame
            logger.debug(f"face detection failed on a frame: {error}")
            return None

        box = _first_face_box(detections)
        if box is None:
            return None

        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        meshes = self.facemesh([crop])
        if not meshes:
            return None

        # FaceMesh works in the crop's coordinates; the boxes must be in the
        # frame's, so the detection offset goes back on.
        landmarks = _to_numpy(meshes[0]).reshape(-1, 2) + np.asarray([x1, y1], dtype=np.float32)
        return (x1, y1, x2, y2), landmarks


def _first_face_box(detections) -> tuple[int, int, int, int] | None:
    """Read the first face box out of a detector's result.

    Detectors differ in what they return — a list of objects with a ``bb_xyxy``
    attribute, a plain array, or nothing at all — so the shape is probed rather
    than assumed.

    Args:
        detections: Whatever the detector returned.

    Returns:
        tuple[int, int, int, int] | None: ``(x1, y1, x2, y2)``, or ``None``.
    """
    if detections is None or len(detections) == 0:
        return None

    first = detections[0]
    for attribute in ("bb_xyxy", "xyxy", "bbox"):
        value = getattr(first, attribute, None)
        if value is not None:
            box = _to_numpy(value).reshape(-1)[:4]
            return tuple(int(round(float(v))) for v in box)  # type: ignore[return-value]

    box = _to_numpy(first).reshape(-1)
    if box.size < 4:
        return None
    return tuple(int(round(float(v))) for v in box[:4])  # type: ignore[return-value]


class FrameDescriber:
    """Locates and describes both eyes of a full frame, in one pass.

    Composes :class:`FaceMeshLocator` and :class:`ExordiumExtractor` so a corpus
    of full frames — HUST-LEBW — pays the detector once per frame rather than
    once per stage.

    **Which face.** A film frame often holds several people, and only one is
    annotated. When an eye midpoint is supplied, the detection whose box
    contains it is the subject's; the others are bystanders. Where boxes nest,
    the smallest is the tightest fit. Where none contains the point, the nearest
    box centre is the best guess left, and it is logged.

    Args:
        device_id (int | None): GPU index; ``None`` or negative selects CPU.

    Raises:
        FeatureError: If exordium is not installed.
    """

    def __init__(self, device_id: int | None = None):
        self.locator = FaceMeshLocator(device_id=device_id)
        self.extractor = ExordiumExtractor(device_id=device_id)

    def head_pose(self, frame: np.ndarray) -> np.ndarray:
        """Head rotation for one frame, in degrees.

        Delegated to the composed extractor, so a corpus using this class gets
        the same estimate as one calling the extractor directly.

        Args:
            frame (np.ndarray): ``(H, W, 3)`` uint8 RGB, the whole frame.

        Returns:
            np.ndarray: ``(3,)`` ``[yaw, pitch, roll]`` in degrees.
        """
        return self.extractor.head_pose(frame)

    def describe(
        self, frame: np.ndarray, eye_midpoint: np.ndarray | None = None
    ) -> tuple[dict[str, EyeBox | None], dict[str, tuple[np.ndarray, bool]], float | None]:
        """Locate both eyes and describe them.

        Args:
            frame (np.ndarray): ``(H, W, 3)`` uint8 RGB full frame.
            eye_midpoint (np.ndarray | None): ``(x, y)`` between the annotated
                eye centres, used to pick the subject's face among several.

        Returns:
            tuple: Eye boxes per side; per side the ``(160,)`` descriptor with
            its validity; and the detected face width, which a caller needs for
            scale when the annotation gives only one eye. All are empty of
            usable values when no face was found, which the caller resolves from
            the annotation instead.
        """
        import torch

        try:
            detections = self.locator.detector.detect_image(
                torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
            )
        except Exception as error:  # noqa: BLE001 - one bad frame
            logger.debug(f"face detection failed on a frame: {error}")
            detections = []

        detection = self._select(detections, eye_midpoint)
        if detection is None:
            return {LEFT: None, RIGHT: None}, {}, None

        box = [int(round(float(v))) for v in _to_numpy(detection.bb_xyxy).reshape(-1)[:4]]
        x1, y1, x2, y2 = box
        crop = frame[max(y1, 0) : y2, max(x1, 0) : x2]
        width = float(x2 - x1)
        if crop.size == 0:
            return {LEFT: None, RIGHT: None}, {}, width

        meshes = self.locator.facemesh([crop])
        if not meshes:
            return {LEFT: None, RIGHT: None}, {}, width

        landmarks = _to_numpy(meshes[0]).reshape(-1, 2) + np.asarray(
            [max(x1, 0), max(y1, 0)], dtype=np.float32
        )
        boxes: dict[str, EyeBox | None] = {}
        for side, indices in self.locator.regions.items():
            try:
                boxes[side] = box_from_landmarks(landmarks[list(indices)])
            except Exception as error:  # noqa: BLE001 - degenerate landmarks
                logger.debug(f"{side} eye box from landmarks failed: {error}")
                boxes[side] = None

        return boxes, self.extractor.frame_features(frame, boxes), width

    @staticmethod
    def _select(detections, eye_midpoint: np.ndarray | None):
        """Pick the detection belonging to the annotated subject.

        Args:
            detections: What the detector returned.
            eye_midpoint (np.ndarray | None): ``(x, y)``, or ``None`` to take
                the first detection.

        Returns:
            The chosen detection, or ``None`` when there were none.
        """
        if detections is None or len(detections) == 0:
            return None
        if eye_midpoint is None or len(detections) == 1:
            return detections[0]

        x, y = float(eye_midpoint[0]), float(eye_midpoint[1])
        boxes = [
            (item, _to_numpy(item.bb_xyxy).reshape(-1)[:4].astype(np.float64))
            for item in detections
        ]

        containing = [
            (item, box) for item, box in boxes if box[0] <= x <= box[2] and box[1] <= y <= box[3]
        ]
        if containing:
            return min(
                containing,
                key=lambda pair: (
                    max(0.0, pair[1][2] - pair[1][0]) * max(0.0, pair[1][3] - pair[1][1])
                ),
            )[0]

        logger.debug("no detection contains the annotated eye midpoint; taking the nearest.")
        return min(
            boxes,
            key=lambda pair: float(
                np.hypot((pair[1][0] + pair[1][2]) / 2 - x, (pair[1][1] + pair[1][3]) / 2 - y)
            ),
        )[0]


POSE_IMAGE_SIZE = 256
"""Detector input side for the streaming locator, in pixels.

**Not ultralytics' 640 default**, which upscales a typical webcam frame and
costs 40.3 ms per frame on CPU against 8.3 ms here -- the single change that
decides whether the pipeline keeps up with a camera.

Measured on the bundled clip against the 640 baseline: the face was found on
60/60 frames at every size down to 192, box IoU held at **0.984** and eye
centres moved **3.1 px** on a ~115 px inter-eye distance. Through the blink
model the open/closed decision agreed on **100%** of frames, so the speed is
effectively free.

Raise it if faces are small in frame -- the cost is roughly quadratic in this
number.
"""

POSE_KEYPOINT_EYES = (0, 1)
"""Indices of the left and right eye within a pose model's keypoints.

The ``yolo11n-pose_widerface`` layout is left eye, right eye, nose, left mouth
corner, right mouth corner -- and the eye order is the **viewer's**, matching
:data:`~blinklinmult.preprocess.geometry.LEFT` / :data:`RIGHT` directly rather
than through the mirror :class:`FaceMeshLocator` needs.
"""


class PoseEyeLocator:
    """Locates eyes from a face detector alone -- no landmark model.

    The streaming counterpart of :class:`FaceMeshLocator`. A face detector with
    pose output already returns both eye centres in the *same forward pass* as
    the box, so running FaceMesh afterwards recomputes information that is
    already in hand. Dropping it, and shrinking the detector's input to
    :data:`POSE_IMAGE_SIZE`, is what takes the pipeline from 0.53x realtime to
    over 30 fps **on CPU**.

    Returns the same :class:`FaceDetection` as :class:`FaceMeshLocator`, so
    overlay and pipeline code is unchanged -- but ``landmarks`` holds the
    detector's ``(5, 2)`` keypoints rather than FaceMesh's ``(478, 2)``.

    **When not to use this.** The boxes carry no
    :attr:`~blinklinmult.preprocess.geometry.EyeBox.aspect`, and their span is
    estimated rather than measured, so the corpus builders and the quality
    signals must keep :class:`FaceMeshLocator`. Validated for *scoring*: on the
    five annotated closed frames of the bundled clip, crops from this path and
    from FaceMesh produced the same open/closed verdict on every one, the
    closest being 0.745 against 0.718.

    Args:
        image_size (int): Detector input side. Defaults to
            :data:`POSE_IMAGE_SIZE`, which is the real-time configuration --
            constructing this class with no arguments is meant to be fast.
        device_id (int | None): Passed to the detector; ``None`` picks the
            default device. A GPU is a bonus here, not a requirement.

    Raises:
        FeatureError: If exordium is not installed.
    """

    def __init__(self, image_size: int = POSE_IMAGE_SIZE, device_id: int | None = None):
        """Load the detector and pin its input size."""
        try:
            from exordium.video.face.detector.yolo11 import YoloFace11Detector
        except ImportError as error:
            raise FeatureError(
                "streaming eye localisation needs exordium: "
                f"`uv sync --extra preprocess`. Import failed with: {error}"
            ) from error

        self.detector = YoloFace11Detector(device_id=device_id)
        self.image_size = image_size
        # exordium calls `model.predict(...)` with no `imgsz`, so the size is
        # set on the ultralytics model itself. `overrides` is ultralytics' own
        # mechanism for exactly this and survives every predict call.
        self.detector.model.overrides["imgsz"] = image_size

    def detect(self, frame: np.ndarray) -> FaceDetection | None:
        """Locate the face, its keypoints and both eyes, in one detector pass.

        Args:
            frame (np.ndarray): ``(H, W, 3)`` uint8 RGB.

        Returns:
            FaceDetection | None: The face box, the ``(5, 2)`` keypoints as
            ``landmarks``, and an :class:`~blinklinmult.preprocess.geometry.EyeBox`
            per side. ``None`` when no face was found, or when the detector
            returned no keypoints -- which means a plain detection model was
            loaded in place of a pose one.
        """
        import torch

        detections = self.detector.detect_image(
            torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
        )
        if not detections:
            return None

        first = detections[0]
        keypoints = _keypoints_of(first)
        if keypoints is None:
            logger.debug("detector returned no keypoints; a pose model is required")
            return None

        left_index, right_index = POSE_KEYPOINT_EYES
        eyes: dict[str, EyeBox | None] = {
            side: box_from_eye_centres(keypoints[left_index], keypoints[right_index], which=side)
            for side in (LEFT, RIGHT)
        }
        return FaceDetection(
            face_box=_box_of(first),
            landmarks=keypoints,
            eyes=eyes,
        )

    def head_pose(self, detection: FaceDetection) -> np.ndarray:
        """Head rotation from the keypoints already detected.

        Costs ~0.05 ms because nothing new is computed -- against ~26.7 ms for
        6DRepNet on CPU. See
        :func:`~blinklinmult.preprocess.geometry.pose_from_keypoints` for the
        accuracy this trades away, in particular that **yaw is unvalidated**.

        Args:
            detection (FaceDetection): A detection from :meth:`detect`.

        Returns:
            np.ndarray: ``(3,)`` ``[yaw, pitch, roll]`` in degrees.
        """
        return pose_from_keypoints(detection.landmarks)


def _keypoints_of(detection) -> np.ndarray | None:
    """Read a detection's facial keypoints, whatever shape it arrived in.

    exordium's detections carry the ultralytics result through, and the
    attribute name has moved between versions, so the lookup is tolerant rather
    than pinned to one spelling.

    Args:
        detection: One face detection.

    Returns:
        np.ndarray | None: ``(5, 2)`` points, or ``None`` when the model
        produced none.
    """
    for name in ("landmarks", "keypoints", "kps"):
        value = getattr(detection, name, None)
        if value is None:
            continue
        points = np.asarray(value, dtype=np.float32).reshape(-1, 2)
        if points.shape[0] >= 2:
            return points
    return None


def _box_of(detection) -> tuple[int, int, int, int]:
    """Read a detection's bounding box as ``(x1, y1, x2, y2)``.

    Args:
        detection: One face detection.

    Returns:
        tuple[int, int, int, int]: The box in frame pixels.
    """
    box = np.asarray(getattr(detection, "bb_xyxy", getattr(detection, "bbox", None)))
    x1, y1, x2, y2 = (int(round(float(v))) for v in box.reshape(-1)[:4])
    return (x1, y1, x2, y2)
