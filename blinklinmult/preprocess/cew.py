"""Preprocess the Closed Eyes in the Wild (CEW) corpus.

2423 face images — 1192 with closed eyes, 1231 with open — each 100x100 with a
side-car file giving both eye centres::

    data/raw/CEW/dataset_B_FacialImages/
        ClosedFace/<name>.jpg
        OpenFace/<name>.jpg
        EyeCoordinatesInfo_ClosedFace.txt   "<name> lx ly rx ry"
        EyeCoordinatesInfo_OpenFace.txt

**Eye state only.** On a still image a blink is not a motion event but a closed
eye, so frame-wise eye-state recognition and blink presence detection coincide
here: one label answers both. CEW is written with ``time_dim = 1`` — the
degenerate sequence case the joint schema supports — and carries that single
label as ``eye_state``, with an all-``False`` ``blink_presence`` mask. The
corpus trains the frame-wise model, which has one eye-state classifier and no
blink-presence head, so nothing reads the second target; the sequence models
train on the video corpora, which annotate blinks as events.

**Label semantics.** The corpus's directory *is* the label, and it labels the
face, not the individual eye: an image in ``ClosedFace`` has both eyes closed.
Both eye columns therefore receive the same value, which is what the corpus
actually asserts.

Example:
    ``uv run python -m blinklinmult.preprocess.cew``
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np

from blinklinmult import PROJECT_ROOT
from blinklinmult.data.schema import HEAD_POSE_DIM, LEFT, RIGHT, DatasetSpec
from blinklinmult.data.writer import H5Writer
from blinklinmult.preprocess.common import (
    PreprocessError,
    crop_square,
    git_sha,
    normalise_image,
    progress,
    proportional_splits,
    stack_eye_window,
)
from blinklinmult.preprocess.extractors import ExordiumExtractor
from blinklinmult.preprocess.quality import blur_score, exposure_score

logger = logging.getLogger(__name__)
"""Module-level logger."""

NAME = "cew"
"""Corpus name, matching its config and processed directory."""

IMAGES_SUBDIR = "dataset_B_FacialImages"
"""Subdirectory holding the labelled face images and their coordinate files.

The corpus ships three variants and only this one is usable end to end.
``dataset_B_FacialImages_highResolution`` holds the closed-eye faces alone, at
sizes from 91x91 to 604x605, so it cannot supply a balanced two-class set;
``dataset_B_Eye_Images`` is pre-cropped to 24x24, below the backbone's minimum
input. The 1.x release used this same directory, so v2 crops are comparable
with the published numbers.
"""

SOURCES: dict[str, tuple[str, str, float]] = {
    "closed": ("ClosedFace", "EyeCoordinatesInfo_ClosedFace.txt", 1.0),
    "open": ("OpenFace", "EyeCoordinatesInfo_OpenFace.txt", 0.0),
}
"""Maps a label group to its image directory, coordinate file, and eye-state value."""

EYE_BOX = 40
"""Crop side length in source pixels, before resizing.

The images are 100x100 crops of a face, so an eye occupies roughly a third of
the width; 40px carries the eye and its surrounding lid without reaching the
other eye.
"""


def read_coordinates(path: Path) -> dict[str, np.ndarray]:
    """Parse one ``EyeCoordinatesInfo`` file.

    Args:
        path (Path): The coordinate file.

    Returns:
        dict[str, np.ndarray]: Image name to ``(lx, ly, rx, ry)``.

    Raises:
        PreprocessError: If the file is missing or a line is malformed.
    """
    if not path.is_file():
        raise PreprocessError(f"CEW coordinate file not found: {path}")

    coordinates: dict[str, np.ndarray] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise PreprocessError(
                f"{path}:{line_number}: expected '<name> lx ly rx ry', got {line.strip()!r}"
            )
        try:
            coordinates[parts[0]] = np.asarray([int(v) for v in parts[1:]], dtype=np.int32)
        except ValueError as error:
            raise PreprocessError(
                f"{path}:{line_number}: non-integer coordinate in {line.strip()!r}"
            ) from error

    if not coordinates:
        raise PreprocessError(f"{path}: no coordinates found.")
    return coordinates


def source_photo(sample_id: str) -> str:
    """The photograph one crop came from.

    ``closed_closed_eye_1362.BMP_face_2`` and ``..._face_3`` are two faces cut
    from a single picture. They are not independent samples, so they must land
    in the same split.

    Args:
        sample_id (str): This project's id for the crop.

    Returns:
        str: The source photograph's id.
    """
    return re.sub(r"_face_\d+$", "", sample_id)


def process(
    root: Path = PROJECT_ROOT,
    image_size: int = 64,
    split_ratios: dict[str, float] | None = None,
    limit: int | None = None,
    seed: int = 42,
) -> Path:
    """Crop both eyes from every CEW image into the corpus's HDF5 file.

    Args:
        root (Path): Repository root.
        image_size (int): Output crop side length.
        split_ratios (dict[str, float] | None): Group split proportions.
        limit (int | None): Process only the first N images per label group.
        seed (int): Shuffle seed for the split division.

    Returns:
        Path: The written HDF5 file.

    Raises:
        PreprocessError: If the corpus is missing or unreadable.
    """
    import cv2

    raw_dir = root / "data" / "raw" / "CEW" / IMAGES_SUBDIR
    if not raw_dir.is_dir():
        raise PreprocessError(
            f"{NAME}: raw data not found at {raw_dir}. See the README's Data section."
        )

    entries: list[tuple[str, Path, np.ndarray, float]] = []
    for group, (image_dir, coordinate_file, label) in SOURCES.items():
        coordinates = read_coordinates(raw_dir / coordinate_file)
        names = sorted(coordinates)[:limit] if limit else sorted(coordinates)

        for name in names:
            path = raw_dir / image_dir / name
            if not path.is_file():
                logger.warning(f"{NAME}: {path} is listed but missing; skipping.")
                continue
            entries.append((f"{group}_{Path(name).stem}", path, coordinates[name], label))

    if not entries:
        raise PreprocessError(f"{NAME}: no usable images found under {raw_dir}.")

    # Grouped by source photograph, not by crop. The closed half of the corpus
    # cuts several faces out of the same picture -- 1193 crops from 1135 photos
    # -- and two faces from one photograph share its lighting, camera, and
    # scene. Splitting per crop put 23 photographs on both sides of the
    # boundary.
    groups = {sample_id: source_photo(sample_id) for sample_id, _, _, _ in entries}
    by_group = proportional_splits(groups.values(), split_ratios, seed=seed, salt=NAME)
    splits = {sample_id: by_group[group] for sample_id, group in groups.items()}

    processed_dir = root / "data" / "processed" / NAME
    h5_path = processed_dir / f"{NAME}.h5"
    spec = DatasetSpec(
        name=NAME,
        fps=None,
        window_seconds=None,
        image_size=image_size,
        # No handcrafted stream: CEW trains BlinkCNN, which reads eye
        # crops alone. Extracting 160-d descriptors here would cost hours of
        # detector time to produce a tensor nothing ever reads.
        feature_dim=None,
        has_eye_state=True,
        # CEW ships full head photographs, so head pose is estimable even though
        # no handcrafted descriptor is extracted.
        has_head_pose=True,
        # No jitter: a still image has one frame and so no predecessor to move
        # from. Symmetry is available because the coordinate file gives both eye
        # centres of the same face.
        quality_signals=("eye_blur", "eye_exposure"),
    )

    config_path = root / "config" / "data" / f"{NAME}.yaml"
    config_text = config_path.read_text() if config_path.is_file() else ""

    # One extractor for the whole corpus: loading 6DRepNet per image would
    # dominate the runtime of a corpus that is otherwise pure image decoding.
    extractor = ExordiumExtractor()

    written = 0
    with H5Writer(
        spec,
        h5_path,
        config_yaml=config_text,
        git_sha=git_sha(root),
    ) as writer:
        for sample_id, path, coordinates, label in progress(entries, f"{NAME} images"):
            image = cv2.imread(str(path))
            if image is None:
                logger.warning(f"{NAME}: {path} could not be decoded; skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Estimated once per photograph and shared by both eyes, which is
            # what pose means -- it describes the head, not the eye.
            angles = extractor.head_pose(image).reshape(1, HEAD_POSE_DIM)

            # The coordinate file gives the two eye centres in (lx, ly, rx, ry)
            # order; each becomes its own sample.
            crops = {
                side: cv2.resize(
                    crop_square(
                        image, int(coordinates[offset]), int(coordinates[offset + 1]), EYE_BOX
                    ),
                    (image_size, image_size),
                    interpolation=cv2.INTER_AREA,
                )
                for side, offset in ((LEFT, 0), (RIGHT, 2))
            }
            for eye_side, resized in crops.items():
                writer.add(
                    subset=splits[sample_id],
                    video_id=sample_id,
                    frame_group="000000",
                    eye_side=eye_side,
                    # One frame: the degenerate sequence case the schema supports.
                    eye_images=stack_eye_window([normalise_image(resized)]),
                    # CEW labels the face, not the eye: an image in ClosedFace
                    # has both eyes closed, so both samples carry the same
                    # value. That is what the corpus asserts.
                    eye_state=np.full(1, label, dtype=np.float32),
                    head_pose=angles,
                    quality_signals={
                        "eye_blur": np.full(1, blur_score(resized), dtype=np.float32),
                        "eye_exposure": np.full(1, exposure_score(resized), dtype=np.float32),
                    },
                )
                written += 1

    logger.info(f"{NAME}: {written} samples from {len(entries)} images.")
    return h5_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Repository root.")
    parser.add_argument("--image-size", type=int, default=64, help="Eye crop side length.")
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only N images per label group."
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    process(args.root, image_size=args.image_size, limit=args.limit)


if __name__ == "__main__":
    main()
