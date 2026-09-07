"""Decoding the attributes MRL Eye encodes in its filenames.

Split out of :mod:`blinklinmult.preprocess.mrl` because it is pure string
handling with no image decoding: it is unit-tested, while the surrounding CLI
needs the raw corpus and OpenCV and is exercised by running the pipeline.

**The eye-state field is inverted relative to this project's convention.** MRL
encodes ``0 = closed, 1 = open``; the label everywhere here is "is the eye
closed", so the field is negated on read. Getting this wrong would silently
train the model backwards on the largest corpus in the benchmark, so it is
asserted here and covered by a test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from blinklinmult.preprocess.common import PreprocessError

if TYPE_CHECKING:
    from pathlib import Path

FILENAME_FIELDS = 8
"""Underscore-separated fields in an MRL filename."""

OPEN_EYE_CODE = 1
"""Filename value meaning the eye is open.

MRL's convention is the inverse of this project's, which labels closure.
"""


@dataclass(frozen=True)
class MrlSample:
    """The attributes encoded in one MRL filename.

    Args:
        subject (str): Subject identifier, e.g. ``"s0001"``. The split group.
        image_number (int): Image index within the subject.
        gender (int): ``0`` male, ``1`` female.
        glasses (int): ``0`` no, ``1`` yes.
        closed (float): ``1.0`` when the eye is closed — MRL's field, inverted.
        reflection (int): ``0`` none, ``1`` low, ``2`` high.
        lighting (int): ``0`` bad, ``1`` good.
        sensor (int): Capture device id.
    """

    subject: str
    image_number: int
    gender: int
    glasses: int
    closed: float
    reflection: int
    lighting: int
    sensor: int

    @property
    def sample_id(self) -> str:
        """Stable identifier for this image.

        Returns:
            str: e.g. ``"s0001_00001"``.
        """
        return f"{self.subject}_{self.image_number:05d}"


def parse_filename(path: Path) -> MrlSample:
    """Decode one MRL filename into its attributes.

    Args:
        path (Path): The image file.

    Returns:
        MrlSample: The decoded attributes.

    Raises:
        PreprocessError: If the name does not have MRL's eight fields.
    """
    fields = path.stem.split("_")
    if len(fields) != FILENAME_FIELDS:
        raise PreprocessError(
            f"{path}: expected {FILENAME_FIELDS} underscore-separated fields in the "
            f"filename, got {len(fields)}."
        )

    try:
        values = [int(field) for field in fields[1:]]
    except ValueError as error:
        raise PreprocessError(f"{path}: non-integer field in the filename.") from error

    return MrlSample(
        subject=fields[0],
        image_number=values[0],
        gender=values[1],
        glasses=values[2],
        # MRL: 0 = closed, 1 = open. This project labels closure, so invert.
        closed=float(values[3] != OPEN_EYE_CODE),
        reflection=values[4],
        lighting=values[5],
        sensor=values[6],
    )
