"""Preprocess the HUST-LEBW corpus.

Eye blinking in the wild: short clips cut from commercial films, each a
fixed-length sequence of **full frames** around a labelled event, with the
corpus's own train/test division::

    data/raw/HUST-LEBW/
        train/  blink/ 254 clips     test/  blink/ 127 clips
                unblink/ 194                unblink/ 98
            <clip>/
                13sign_13.txt      source frame ids
                13/  00002.bmp …   FULL FRAMES, 1920x800
                     land_13.txt   frame_id, x_left, y_left, x_right, y_right
                     zuo/ you/     the authors' 110x110 eye crops
                10/  the same clip cut to 10 frames

**Why this corpus matters.** Every other video corpus here is recorded footage
of a seated subject facing a camera. HUST-LEBW is unconstrained: varying pose,
illumination, scale, and occlusion — and often several people in shot. It is the
corpus that reveals whether a model has learned blinking or has learned one
recording setup.

**Full frames, not eye crops.** The corpus ships 1920x800 film frames, so the
eyes are located and described exactly as in every other video corpus: detect
the face, run FaceMesh over it, read the eye regions from the landmark indices,
crop at :data:`~blinklinmult.preprocess.geometry.EYE_CROP_SCALE`. The authors'
own 110x110 crops and their ``feature_hog``/``feature_haar`` files are not used
— they are computed *from* those tight crops, so they describe a different
framing and cannot be compared with the other corpora.

**The annotation picks the face.** ``land_*.txt`` gives both eye centres per
frame, and 9% of frames contain more than one face. The detection whose box
contains the annotated eye midpoint is the subject's; the rest are bystanders.

**Blink presence only.** The label is a per-clip event annotation — this clip
does or does not contain a blink — with no per-frame closure marking, so the
frame-level ``blink_presence`` target is filled with the clip's label and
``eye_state`` is not written at all. That label is valid for every frame however
the detector fared, so a clip always carries usable supervision.

Example:
    ``uv run python -m blinklinmult.preprocess.hust_lebw``
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blinklinmult import PROJECT_ROOT
from blinklinmult.data.schema import (
    EYE_FEATURE_DIM,
    HEAD_POSE_DIM,
    DatasetSpec,
)
from blinklinmult.data.writer import H5Writer
from blinklinmult.preprocess.common import (
    PreprocessError,
    git_sha,
    list_files,
    normalise_image,
    progress,
    stack_eye_window,
)
from blinklinmult.preprocess.features import empty_feature, stack_window
from blinklinmult.preprocess.hust_crops import (
    EYE_DIRECTORIES,
    MATCH_THRESHOLD,
    containing_box,
    crop_files,
    locate,
    recrop,
)
from blinklinmult.preprocess.quality import (
    blur_score,
    box_jitter,
    exposure_score,
)

logger = logging.getLogger(__name__)
"""Module-level logger."""

NAME = "hust_lebw"
"""Corpus name, matching its config and processed directory."""

FPS = 30.0
"""Frame rate of the source films the clips were cut from.

The corpus ships fixed-length clips rather than continuous video, so this only
sets what duration a clip's frame count corresponds to.
"""

TIME_DIM = 13
"""Frames per sample.

The length the literature reports for this corpus. Clips shorter than this are
padded and masked; longer ones are cut.

**This corpus does not follow** :data:`~blinklinmult.data.schema.DEFAULT_WINDOW_SECONDS`,
and cannot. The others are continuous recordings a window is swept over, so
asking for 1.5 s simply reads more frames. HUST-LEBW ships *fixed 13-frame
clips*: there is no further footage to read, and declaring a 45-frame window
would pad every sample to 71% padding while adding no information. A joint run
resolves to the longest frame count and masks the shortfall, which is the
correct handling -- the frames genuinely are not there.
"""

CLIP_LENGTHS = ("13", "10")
"""Which per-clip length directories to try, best first.

Every clip ships both, but not always usably: five clips have an empty or
near-empty ``land_13.txt`` and a complete ``land_10.txt``
(``train/unblink/14, 29, 44, 50, 119``). Reading only ``13/`` would drop them.
Whichever directory yields more annotated frames is the one used.
"""

LABELS: dict[str, float] = {"blink": 1.0, "unblink": 0.0}
"""Directory name to blink-presence value."""

VALID_FRACTION = 0.15
"""Share of each training class held out for validation.

Taken as the **tail of the clip id range**, not a random draw. The clips are
ordered by source film — ids 200-203 are all one film, 50-52 one scene — so a
random split would put the same actor in train and validation. A contiguous tail
keeps a film's clips together, which is the best available proxy given the
corpus ships no film id.
"""

FACE_TO_EYE_RATIO = 0.45
"""Inter-eye distance as a fraction of the detected face box width.

Used only when the annotation gives one eye and leaves the other ``NaN`` -- 21
rows do -- so there is no inter-eye distance to scale by and the face box is the
only measure of how large the subject is in frame.
"""

EYE_FALLBACK_RATIO = 0.85
"""Eye box side as a fraction of the inter-eye distance, when no face is found.

Measured, not guessed: over 66 eye boxes across 12 clips, the box the normal
FaceMesh route produces has a side of about 0.84x the annotated inter-eye
distance (mean 0.853, p10 0.737, p90 0.971). Using the inter-eye distance
directly would make the fallback crops ~20% wider than every other frame in the
corpus, putting a scale shift on exactly the hardest frames.
"""


@dataclass(frozen=True)
class Clip:
    """One annotated film clip.

    Args:
        split (str): ``train``, ``valid``, or ``test``.
        label (str): ``blink`` or ``unblink``.
        clip_id (str): The corpus's numbering, unique only within its class.
        directory (Path): The chosen length directory, holding the frames.
        landmarks (dict[int, np.ndarray]): Frame id to ``(x1, y1, x2, y2)`` eye
            centres.
        frames (dict[int, Path]): Frame id to its ``.bmp``.
    """

    split: str
    label: str
    clip_id: str
    directory: Path
    landmarks: dict[int, np.ndarray]
    frames: dict[int, Path]

    @property
    def sample_id(self) -> str:
        """Identifier unique across the corpus.

        The corpus numbers ``blink`` and ``unblink`` clips independently from 1,
        so the class and split are both needed to keep ids apart.

        Returns:
            str: ``<split>_<label>_<clip_id>``.
        """
        return f"{self.split}_{self.label}_{self.clip_id}"

    @property
    def frame_ids(self) -> list[int]:
        """Annotated frames, in order, capped at :data:`TIME_DIM`.

        Driven by the *landmark file*, never by globbing the directory: five
        test clips hold more ``.bmp`` files than annotated frames — one has 39
        for 13 rows — and a glob would build a window out of frames the
        annotation never covered.

        Returns:
            list[int]: Frame ids present in both the annotation and on disk.
        """
        return sorted(set(self.landmarks) & set(self.frames))[:TIME_DIM]


def parse_landmarks(path: Path) -> dict[int, np.ndarray]:
    """Read one ``land_*.txt``.

    Each row is ``frame_id x_left y_left x_right y_right`` — the two eye
    *centres* in full-frame coordinates.

    Args:
        path (Path): The landmark file.

    Returns:
        dict[int, np.ndarray]: Frame id to a ``(4,)`` float array. Empty when
        the file is absent or blank, which five clips are.

    A truncated or non-numeric row is **skipped, not fatal**. Five rows of
    17154 are malformed — ``train/blink/147`` has ``67 406 149`` and a bare
    ``-1 -1``, the corpus's own marker for "not annotated" — and aborting the
    corpus over 0.03% of its rows would be the wrong trade. Those frames simply
    fall through to face detection like any other.
    """
    if not path.is_file():
        return {}

    landmarks: dict[int, np.ndarray] = {}
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 5:
            logger.debug(f"{path}:{number}: skipping incomplete row {line.strip()!r}.")
            continue
        try:
            values = [float(value) for value in parts[:5]]
        except ValueError:
            logger.debug(f"{path}:{number}: skipping non-numeric row {line.strip()!r}.")
            continue
        landmarks[int(values[0])] = np.asarray(values[1:5], dtype=np.float32)
    return landmarks


def load_clip(directory: Path, split: str, label: str) -> Clip | None:
    """Read one clip, taking whichever length directory annotates more frames.

    Args:
        directory (Path): The clip directory, holding ``13/`` and ``10/``.
        split (str): Which split this clip belongs to.
        label (str): ``blink`` or ``unblink``.

    Returns:
        Clip | None: The clip, or ``None`` when neither length has a usable
        annotation.
    """
    best: Clip | None = None
    for length in CLIP_LENGTHS:
        source = directory / length
        if not source.is_dir():
            continue

        landmarks = parse_landmarks(source / f"land_{length}.txt")
        frames = {int(path.stem): path for path in list_files(source, "*.bmp")}
        candidate = Clip(
            split=split,
            label=label,
            clip_id=directory.name,
            directory=source,
            landmarks=landmarks,
            frames=frames,
        )
        if best is None or len(candidate.frame_ids) > len(best.frame_ids):
            best = candidate

    if best is None or not best.frame_ids:
        return None
    return best


def find_clips(root: Path, split_dir: str, label: str) -> list[Path]:
    """List a class's clip directories, in the corpus's own numbering.

    Args:
        root (Path): The corpus root.
        split_dir (str): ``train`` or ``test``.
        label (str): ``blink`` or ``unblink``.

    Returns:
        list[Path]: Clip directories, sorted numerically.

    Raises:
        PreprocessError: If the class directory is missing.
    """
    base = root / split_dir / label
    if not base.is_dir():
        raise PreprocessError(f"{NAME}: expected a {label!r} directory under {root / split_dir}.")

    clips = [path for path in base.iterdir() if path.is_dir() and path.name.isdigit()]
    return sorted(clips, key=lambda path: int(path.name))


def split_clips(clips: list[Path], split_dir: str) -> list[tuple[Path, str]]:
    """Assign each clip to a split.

    The corpus's ``test`` is honoured as given. ``train`` is divided into train
    and validation by taking the **tail** of the id range — see
    :data:`VALID_FRACTION` for why a contiguous tail beats a random draw here.

    Args:
        clips (list[Path]): Clip directories, in corpus order.
        split_dir (str): ``train`` or ``test``.

    Returns:
        list[tuple[Path, str]]: Each clip with its split name.
    """
    if split_dir == "test":
        return [(clip, "test") for clip in clips]

    cut = int(round(len(clips) * (1.0 - VALID_FRACTION)))
    return [(clip, "train" if index < cut else "valid") for index, clip in enumerate(clips)]


def _clip_quality(
    patches: list[np.ndarray], boxes: list[tuple[float, float, float]]
) -> dict[str, np.ndarray]:
    """Per-frame quality signals for one eye across a clip.

    Every signal describes this eye alone. A sample is one eye, because the
    model predicts eye-wise so that winks and per-eye patterns are detectable at
    all -- so a signal read from the partner eye would leak across samples and
    would be undefined exactly when one eye is occluded.

    Args:
        patches (list[np.ndarray]): The window's crops, ``(C, H, W)`` each.
        boxes (list[tuple[float, float, float]]): ``(centre_x, centre_y, span)``
            per frame.

    Returns:
        dict[str, np.ndarray]: Each signal as ``(T,)`` in ``[0, 1]``.
    """
    centres = np.asarray([(x, y) for x, y, _ in boxes], dtype=np.float64)
    spans = np.asarray([span for _, _, span in boxes], dtype=np.float64)
    return {
        "eye_blur": np.asarray([blur_score(patch) for patch in patches], dtype=np.float32),
        "eye_exposure": np.asarray([exposure_score(patch) for patch in patches], dtype=np.float32),
        "eye_jitter": box_jitter(centres, spans).astype(np.float32),
    }


def process(
    root: Path = PROJECT_ROOT,
    image_size: int = 64,
    limit: int | None = None,
    with_features: bool = True,
    device_id: int | None = None,
) -> Path:
    """Build the corpus's HDF5 file in one pass.

    Args:
        root (Path): Repository root.
        image_size (int): Output crop side length.
        limit (int | None): Process only N clips per split and label.
        with_features (bool): Extract the 160-d descriptors.
        device_id (int | None): GPU index for the extractors; ``None`` is CPU.

    Returns:
        Path: The written HDF5 file.

    Raises:
        PreprocessError: If the corpus is missing, or a clip yields no usable
            input at all.
    """
    import cv2

    raw_dir = root / "data" / "raw" / "HUST-LEBW"
    if not raw_dir.is_dir():
        raise PreprocessError(
            f"{NAME}: raw data not found at {raw_dir}. See the README's Data section."
        )

    locator = _build_locator(device_id) if with_features else None

    spec = DatasetSpec(
        name=NAME,
        fps=FPS,
        window_seconds=TIME_DIM / FPS,
        image_size=image_size,
        feature_dim=EYE_FEATURE_DIM if with_features else None,
        has_blink_presence=True,
        has_eye_state=False,
        has_head_pose=with_features,
        quality_signals=(("eye_blur", "eye_exposure", "eye_jitter") if with_features else ()),
    )

    processed_dir = root / "data" / "processed" / NAME
    config_path = root / "config" / "data" / f"{NAME}.yaml"
    counts: dict[str, int] = {}
    skipped: list[str] = []

    with H5Writer(
        spec,
        processed_dir / f"{NAME}.h5",
        config_yaml=config_path.read_text() if config_path.is_file() else "",
        git_sha=git_sha(root),
    ) as writer:
        for split_dir in ("train", "test"):
            for label, value in LABELS.items():
                clips = find_clips(raw_dir, split_dir, label)
                assigned = split_clips(clips, split_dir)
                if limit is not None:
                    assigned = assigned[:limit]

                for directory, subset in progress(assigned, f"{NAME} {split_dir}/{label}"):
                    clip = load_clip(directory, subset, label)
                    if clip is None:
                        skipped.append(f"{split_dir}/{label}/{directory.name}")
                        continue

                    written = _write_clip(writer, clip, value, image_size, locator, cv2)
                    counts[subset] = counts.get(subset, 0) + written

    logger.info(f"{NAME}: samples per split {counts}")
    if skipped:
        logger.warning(f"{NAME}: {len(skipped)} clips had no annotated frame: {skipped[:5]}")
    return processed_dir / f"{NAME}.h5"


def _write_clip(writer, clip: Clip, label: float, image_size: int, locator, cv2) -> int:
    """Locate this clip's shipped eye crops and write one sample per eye.

    **Driven by the corpus's own crops, not by detection.** HUST-LEBW ships
    per-eye crops cut from the frames it also ships, so their position is
    recoverable exactly by template matching -- measured at a correlation of
    1.0000 on every crop of a sample clip, with crop *N* matching frame *N*. That
    is a stronger signal than any detector: it says precisely where the
    annotator considered the eye to be, and the face box containing it is the
    subject's among the several a film frame holds.

    An eye whose crop directory is absent was occluded, and the corpus says so
    by shipping nothing -- ``train/unblink/100`` has ``zuo`` and no ``you``. No
    sample is written for it rather than a masked one: there is no supervision
    to attach.

    Args:
        writer (H5Writer): Destination.
        clip (Clip): The clip.
        label (float): Its blink-presence value.
        image_size (int): Output crop side length.
        locator: The exordium stack, or ``None``.
        cv2: The OpenCV module, passed in so the import stays at the top level.

    Returns:
        int: Samples written -- 0, 1 or 2, depending on how many eyes the corpus
        shipped crops for.
    """
    frames = [path for _, path in sorted(clip.frames.items())]
    written = 0

    for side, directory in EYE_DIRECTORIES.items():
        crops = crop_files(clip.directory / directory)
        if not crops:
            logger.debug(f"{clip.sample_id}: no {directory} crops; {side} eye omitted.")
            continue

        patches: list[np.ndarray] = []
        present: list[bool] = []
        poses: list[np.ndarray] = []
        boxes: list[tuple[float, float, float]] = []
        features: list[tuple[np.ndarray, bool]] = []

        for index, crop_path in crops:
            if index - 1 >= len(frames):
                continue
            frame = cv2.cvtColor(cv2.imread(str(frames[index - 1])), cv2.COLOR_BGR2RGB)
            shipped = cv2.cvtColor(cv2.imread(str(crop_path)), cv2.COLOR_BGR2RGB)
            x, y, score = locate(frame, shipped)
            height, width = shipped.shape[:2]

            if score < MATCH_THRESHOLD:
                # The crop did not come from this frame, so locating it would be
                # a guess: mask the frame rather than mis-crop it.
                logger.debug(f"{clip.sample_id}: crop {index} matched at {score:.3f}.")
                patches.append(np.zeros((3, image_size, image_size), dtype=np.float32))
                present.append(False)
                poses.append(np.zeros(HEAD_POSE_DIM, dtype=np.float32))
                boxes.append((0.0, 0.0, 0.0))
                features.append((empty_feature(), False))
                continue

            # Re-cut wider: the shipped crop frames the eye alone, too tight for
            # the eyelid-contour landmarks a descriptor reads.
            wide = recrop(frame, x, y, width, height)
            patches.append(
                normalise_image(
                    cv2.resize(wide, (image_size, image_size), interpolation=cv2.INTER_AREA)
                )
            )
            present.append(True)
            boxes.append((x + width / 2.0, y + height / 2.0, float(width)))

            if locator is None:
                poses.append(np.zeros(HEAD_POSE_DIM, dtype=np.float32))
                features.append((empty_feature(), False))
                continue

            # Pose from the face the eye sits in, not the whole frame: a film
            # frame holds bystanders whose pose is not the subject's.
            face = containing_box(_detected_faces(locator, frame), x, y, width, height)
            region = frame[face[1] : face[3], face[0] : face[2]] if face else frame
            poses.append(
                locator.head_pose(region)
                if region.size
                else np.zeros(HEAD_POSE_DIM, dtype=np.float32)
            )
            features.append(_describe_crop(locator, wide, poses[-1]))

        if not patches:
            continue

        values, mask = stack_window(features) if locator is not None else (None, None)
        writer.add(
            subset=clip.split,
            video_id=clip.sample_id,
            frame_group="000000",
            eye_side=side,
            eye_images=stack_eye_window(patches),
            eye_image_mask=np.asarray(present, dtype=bool),
            # A per-clip event label, broadcast across the window: the corpus
            # does not mark which frames the closure occupies. It stays valid
            # however the detector fared, so a clip always supervises.
            blink_presence=np.full(len(patches), label, dtype=np.float32),
            eye_features=values,
            eye_feature_mask=mask,
            blink_id=int(clip.clip_id) if label > 0.5 else -1,
            head_pose=np.stack(poses) if locator is not None else None,
            quality_signals=_clip_quality(patches, boxes) if locator is not None else None,
        )
        written += 1

    return written


def _detected_faces(locator, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Every face box in a frame.

    Args:
        locator: The exordium stack.
        frame (np.ndarray): ``(H, W, 3)`` RGB.

    Returns:
        list[tuple[int, int, int, int]]: ``(x1, y1, x2, y2)`` per face; empty
        when detection fails, since one bad frame must not end a long build.
    """
    import torch

    try:
        detections = locator.locator.detector.detect_image(
            torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
        )
    except Exception as error:  # noqa: BLE001 - one bad frame
        logger.debug(f"face detection failed: {error}")
        return []

    faces = []
    for detection in detections:
        values = np.asarray(detection.bb_xyxy).reshape(-1)[:4]
        faces.append(tuple(int(round(float(value))) for value in values))
    return faces


def _describe_crop(locator, crop: np.ndarray, pose: np.ndarray) -> tuple[np.ndarray, bool]:
    """The 160-d descriptor for one re-cut eye.

    Args:
        locator: The exordium stack.
        crop (np.ndarray): The wider eye crop, ``(H, W, 3)`` RGB.
        pose (np.ndarray): ``(3,)`` head rotation in degrees.

    Returns:
        tuple[np.ndarray, bool]: The descriptor and whether it is usable.
    """
    try:
        return locator.extractor._eye(crop, pose)
    except Exception as error:  # noqa: BLE001 - one bad crop
        logger.debug(f"eye description failed: {error}")
        return empty_feature(), False


def _build_locator(device_id: int | None):
    """Construct the exordium stack, once per corpus run.

    Args:
        device_id (int | None): GPU index, or ``None`` for CPU.

    Returns:
        The locator.
    """
    from blinklinmult.preprocess.extractors import FrameDescriber

    logger.info(f"{NAME}: loading the exordium extraction stack...")
    return FrameDescriber(device_id=device_id)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Repository root.")
    parser.add_argument("--image-size", type=int, default=64, help="Eye crop side length.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only N clips per split and label."
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip the exordium descriptors; the corpus is then image-only.",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="GPU index for extraction; omit for CPU."
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    process(
        args.root,
        image_size=args.image_size,
        limit=args.limit,
        with_features=not args.no_features,
        device_id=args.device,
    )


if __name__ == "__main__":
    main()
