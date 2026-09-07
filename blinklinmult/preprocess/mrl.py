"""Preprocess the MRL Eye corpus.

~85k infrared eye crops from 37 subjects, already cropped to the eye, with every
attribute encoded in the filename::

    data/raw/MRL-Eye/mrlEyes_2018_01/s0001/s0001_00001_0_0_0_0_0_01.png
                                     ^     ^     ^ ^ ^ ^ ^ ^  ^
                                     |     |     | | | | | |  sensor id
                                     |     |     | | | | | lighting
                                     |     |     | | | | reflection
                                     |     |     | | | eye state
                                     |     |     | | glasses
                                     |     |     | gender
                                     |     image number
                                     subject id

**The eye-state field is inverted relative to this project's convention.** MRL
encodes ``0 = closed, 1 = open``; the label everywhere here is "is the eye
closed", so the field is negated on read. The 1.x code did this too, with a
``int(not bool(int(...)))`` and a comment — getting it wrong silently trains the
model backwards on the single largest corpus in the benchmark, so it is asserted
here and covered by a test.

**One eye per image.** Unlike every other corpus, MRL supplies a single eye and
does not say which, so its samples are written with an ``unknown`` eye side
rather than a guessed one. A sample is one eye here as everywhere else.

**A still corpus.** A blink in a single crop is a closed eye, not a motion
event, so the eye-state label already answers both questions and this corpus
trains the frame-wise model. See :mod:`blinklinmult.preprocess.cew`.

**Split by subject.** The subject id is the group, so no subject appears in two
splits.

**The archive must be unpacked first.** The corpus ships as
``mrlEyes_2018_01.zip``, which contains its own top-level directory::

    cd data/raw/MRL-Eye && unzip mrlEyes_2018_01.zip

Example:
    ``uv run python -m blinklinmult.preprocess.mrl``
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from blinklinmult import PROJECT_ROOT
from blinklinmult.data.schema import UNKNOWN_EYE, DatasetSpec
from blinklinmult.data.writer import H5Writer
from blinklinmult.preprocess.common import (
    PreprocessError,
    git_sha,
    list_files,
    normalise_image,
    progress,
    stack_eye_window,
    subsample_evenly,
)
from blinklinmult.preprocess.mrl_names import parse_filename
from blinklinmult.preprocess.quality import blur_score, exposure_score

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = logging.getLogger(__name__)
"""Module-level logger."""

NAME = "mrl"
"""Corpus name, matching its config and processed directory."""

IMAGES_SUBDIR = "mrlEyes_2018_01"
"""Subdirectory holding the per-subject image directories."""

SPLITS: dict[str, str] = {
    "train": (
        "s0002 s0004 s0005 s0006 s0007 s0010 s0012 s0013 s0014 s0017 s0018 s0020 "
        "s0021 s0022 s0025 s0027 s0028 s0029 s0030 s0031 s0032 s0034 s0035 s0037"
    ),
    "valid": "s0003 s0009 s0011 s0015 s0016 s0024 s0036",
    "test": "s0001 s0008 s0019 s0023 s0026 s0033",
}
"""Which split each of the 37 subjects belongs to.

Assigned explicitly rather than by hashing the subject id. Hashing is
reproducible and leak-free, but it assigns each subject *independently*, so the
result is a random draw rather than a proportional division — and MRL's subjects
differ in size by an order of magnitude (2.2k to 10.3k images). Hashing gave
**55 / 3 / 42 percent by image count** against a 70/15/15 target, leaving 2816
validation images to select checkpoints on while the test split held 35k.

These groups were chosen largest-subject-first into whichever split was
furthest below its target share, which lands on **70.1 / 14.9 / 15.0** by image
count with 24 / 7 / 6 subjects. Still whole subjects, so no eye appears in two
splits.
"""

SUBJECT_SPLIT: dict[str, str] = {
    subject: split for split, names in SPLITS.items() for subject in names.split()
}
"""Subject id to split, flattened from :data:`SPLITS` for lookup."""


def subject_splits(subjects: Iterable[str]) -> dict[str, str]:
    """Look up each subject's split.

    Args:
        subjects (Iterable[str]): Subject directory names.

    Returns:
        dict[str, str]: Subject id to split name.

    Raises:
        PreprocessError: If a subject has no assignment, which means the corpus
            gained a directory the declared split does not cover.
    """
    found = list(subjects)
    missing = sorted(set(found) - set(SUBJECT_SPLIT))
    if missing:
        raise PreprocessError(
            f"{NAME}: no split declared for {missing}. SPLITS covers "
            f"{len(SUBJECT_SPLIT)} subjects; update it if the corpus changed."
        )
    return {subject: SUBJECT_SPLIT[subject] for subject in found}


def process(
    root: Path = PROJECT_ROOT,
    image_size: int = 64,
    limit_per_subject: int | None = None,
) -> Path:
    """Read every MRL image and write the manifest.

    Args:
        root (Path): Repository root.
        image_size (int): Output crop side length.
        limit_per_subject (int | None): Process only N images per subject, which
            is how a smoke build keeps this 85k-image corpus tractable.

    Returns:
        Path: The written manifest.

    Raises:
        PreprocessError: If the corpus is missing or unreadable.
    """
    import cv2

    raw_dir = root / "data" / "raw" / "MRL-Eye" / IMAGES_SUBDIR
    if not raw_dir.is_dir():
        archive = raw_dir.with_suffix(".zip")
        hint = (
            f"The archive is still packed; unpack it in place:\n"
            f"    cd {archive.parent} && unzip {archive.name}"
            if archive.is_file()
            else "See the README's Data section."
        )
        raise PreprocessError(f"{NAME}: raw data not found at {raw_dir}. {hint}")

    subject_dirs = sorted(p for p in raw_dir.iterdir() if p.is_dir())
    if not subject_dirs:
        raise PreprocessError(f"{NAME}: no subject directories under {raw_dir}.")

    # Split by subject: MRL's images are many per person, so splitting by image
    # would put the same eye in train and test.
    splits = subject_splits(p.name for p in subject_dirs)

    processed_dir = root / "data" / "processed" / NAME
    h5_path = processed_dir / f"{NAME}.h5"
    spec = DatasetSpec(
        name=NAME,
        fps=None,
        window_seconds=None,
        image_size=image_size,
        # No handcrafted stream. MRL ships pre-cropped eyes with no face, so
        # head pose is not computable -- but more simply, it trains
        # BlinkCNN, which reads eye crops alone.
        feature_dim=None,
        has_eye_state=True,
        has_eye_side=False,
        # Pixel signals only. MRL ships pre-cropped eyes with no surrounding
        # face, so there is no pose to estimate, no box to track between frames
        # (each image is independent), and no partner eye to compare against --
        # the corpus does not even say which side a crop is.
        quality_signals=("eye_blur", "eye_exposure"),
    )

    config_path = root / "config" / "data" / f"{NAME}.yaml"
    config_text = config_path.read_text() if config_path.is_file() else ""

    total = 0
    writer_context = H5Writer(
        spec,
        h5_path,
        config_yaml=config_text,
        git_sha=git_sha(root),
    )
    # Listed up front so the bar tracks images rather than subjects: 37 subjects
    # over ~85k images would advance it 37 times in an hour-long run, which says
    # nothing about how far along it is.
    work: list[tuple[Path, str]] = []
    for subject_dir in subject_dirs:
        paths = list_files(subject_dir, "*.png")
        if limit_per_subject is not None:
            paths = subsample_evenly(paths, limit_per_subject)
        work.extend((path, splits[subject_dir.name]) for path in paths)
        logger.info(
            f"{NAME}: {subject_dir.name} -> {splits[subject_dir.name]} ({len(paths)} images)"
        )

    with writer_context as writer:
        for path, subset in progress(work, f"{NAME} images"):
            sample = parse_filename(path)
            image = cv2.imread(str(path))
            if image is None:
                logger.warning(f"{NAME}: {path} could not be decoded; skipping.")
                continue

            # MRL is infrared and ships single-channel images; the pretrained
            # backbones expect three, so the channel is replicated rather than
            # colour being invented.
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            resized = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            crop = normalise_image(resized)

            writer.add(
                subset=subset,
                video_id=sample.subject,
                frame_group=f"{sample.image_number:06d}",
                # MRL ships one eye per image and does not say which; recorded
                # as unknown rather than guessed, so a per-side analysis can
                # exclude it.
                eye_side=UNKNOWN_EYE,
                eye_images=stack_eye_window([crop]),
                eye_state=np.full(1, sample.closed, dtype=np.float32),
                quality_signals={
                    "eye_blur": np.full(1, blur_score(crop), dtype=np.float32),
                    "eye_exposure": np.full(1, exposure_score(crop), dtype=np.float32),
                },
            )
            total += 1

    logger.info(f"{NAME}: {total} images across {len(subject_dirs)} subjects.")
    return h5_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Repository root.")
    parser.add_argument("--image-size", type=int, default=64, help="Eye crop side length.")
    parser.add_argument(
        "--limit-per-subject",
        type=int,
        default=None,
        help="Process only N images per subject.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    process(
        args.root,
        image_size=args.image_size,
        limit_per_subject=args.limit_per_subject,
    )


if __name__ == "__main__":
    main()
