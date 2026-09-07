"""Parser for the ``.tag`` blink annotation format.

TalkingFace and the Researcher's Night (RN) corpora both ship their
blink annotation in the same ``.tag`` text format, so one parser serves all
three. This module reads that format and nothing else: no video decoding, no
image cropping, no window sampling. That separation is deliberate — the old
``preprocess/reader.py`` fused parsing, frame extraction, cropping, sampling,
and visualisation into a single 500-line class that could not be constructed
without the frames and the video already on disk, and therefore could not be
tested at all.

**The format.** A ``.tag`` file is a header, a ``#start`` line, one
colon-separated record per annotated frame, and an ``#end`` line::

    #start
    881:-1:X:X:X:X:X:277:174:124:124:310:222:334:221:369:220:392:222
    882:5:X:C:X:C:X:277:174:124:124:310:222:334:221:369:220:392:222
    #end

The 19 fields are documented on :class:`TagRecord`. The one that matters most
is the second, ``blink_id``: ``-1`` means "no blink at this frame", and any
non-negative value is an identifier shared by every frame of one blink. A blink
event is therefore a maximal run of frames carrying the same id, which is what
:meth:`TagFile.blink_events` recovers.

**Binary labels.** The two tasks read the same annotation differently:

* *blink presence* — ``blink_id != -1``, i.e. this frame belongs to a blink.
* *eye state* — the per-eye ``fully_closed`` flags.

Both are exposed as arrays rather than computed at each call site, so no caller
has to re-derive the convention. Note the authors of EyeBlink8 state that the
``NF``/``FC``/``NV`` flags are *not* consistently annotated across the corpus;
:attr:`TagFile.eye_state` is therefore reliable only where a dataset's own
documentation says it is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blinklinmult.data.schema import NO_BLINK as _NO_BLINK

logger = logging.getLogger(__name__)
"""Module-level logger."""

NO_BLINK = _NO_BLINK
"""``blink_id`` value meaning "this frame is not part of a blink".

Re-exported from :mod:`blinklinmult.data.schema`, which owns it: the builder
needs the same constant, and the data layer must not import this package.
"""

START_TOKEN = "#start"
"""Line that opens the record block."""

END_TOKEN = "#end"
"""Line that closes the record block."""

FIELD_COUNT = 19
"""Number of colon-separated fields in one record line."""

FLAG_ABSENT = "X"
"""Field value meaning "flag not set". Anything else sets the flag."""


class TagParseError(ValueError):
    """Raised when a ``.tag`` file does not conform to the format."""


@dataclass(frozen=True)
class TagRecord:
    """One annotated frame.

    Args:
        frame_id (int): Frame counter. Maps to a timestamp through the
            companion ``.txt`` file, and to an image through the extracted
            frame directory.
        blink_id (int): :data:`NO_BLINK` when the eye is open; otherwise an id
            shared by every frame of the same blink.
        non_frontal_face (bool): The subject is looking sideways.
        left_fully_closed (bool): Left eye is 90-100% closed.
        left_not_visible (bool): Left eye is occluded (hand, hair, lighting,
            fast head motion).
        right_fully_closed (bool): Right eye is 90-100% closed.
        right_not_visible (bool): Right eye is occluded.
        face_xywh (np.ndarray): Face box ``(x, y, w, h)``, ``int32``.
        left_eye_corners (np.ndarray): Left eye corners
            ``(x1, y1, x2, y2)``, ``int32``.
        right_eye_corners (np.ndarray): Right eye corners
            ``(x1, y1, x2, y2)``, ``int32``.
    """

    frame_id: int
    blink_id: int
    non_frontal_face: bool
    left_fully_closed: bool
    left_not_visible: bool
    right_fully_closed: bool
    right_not_visible: bool
    face_xywh: np.ndarray
    left_eye_corners: np.ndarray
    right_eye_corners: np.ndarray

    @property
    def is_blink(self) -> bool:
        """Whether this frame belongs to an annotated blink.

        Returns:
            bool: ``True`` when ``blink_id`` is not :data:`NO_BLINK`.
        """
        return self.blink_id != NO_BLINK

    @property
    def has_face_box(self) -> bool:
        """Whether a face box was annotated for this frame.

        An all-zero box is the format's way of saying "not annotated", and a
        crop taken from one would be an empty image.

        Returns:
            bool: ``True`` when the box is not all zeros.
        """
        return bool(np.any(self.face_xywh))


@dataclass(frozen=True)
class BlinkEvent:
    """A maximal run of consecutive frames sharing one ``blink_id``.

    Args:
        blink_id (int): The annotation's identifier for this blink.
        start_index (int): Index of the first frame **within the record list**,
            not a frame id. Windows are cut on record indices because a corpus
            may annotate a non-contiguous set of frames.
        length (int): Number of records in the run.
        first_frame_id (int): ``frame_id`` of the first frame, for logging and
            for naming extracted samples.
    """

    blink_id: int
    start_index: int
    length: int
    first_frame_id: int

    @property
    def stop_index(self) -> int:
        """One past the last record index of this event.

        Returns:
            int: Exclusive end index.
        """
        return self.start_index + self.length


def _parse_flag(value: str) -> bool:
    """Interpret one flag field.

    Args:
        value (str): Raw field text.

    Returns:
        bool: ``False`` for :data:`FLAG_ABSENT`, ``True`` otherwise.
    """
    return value.strip() != FLAG_ABSENT


def _parse_record(line: str, source: Path, line_number: int) -> TagRecord:
    """Parse one record line.

    Args:
        line (str): The raw line, without its trailing newline.
        source (Path): File the line came from, for error messages.
        line_number (int): 1-based line number, for error messages.

    Returns:
        TagRecord: The parsed record.

    Raises:
        TagParseError: If the field count or a numeric field is malformed.
    """
    fields = line.strip().split(":")
    if len(fields) != FIELD_COUNT:
        raise TagParseError(
            f"{source}:{line_number}: expected {FIELD_COUNT} colon-separated fields, "
            f"got {len(fields)}: {line.strip()!r}"
        )

    try:
        numbers = [int(value) for value in (*fields[:2], *fields[7:])]
    except ValueError as error:
        raise TagParseError(
            f"{source}:{line_number}: non-integer numeric field in {line.strip()!r}"
        ) from error

    frame_id, blink_id = numbers[0], numbers[1]
    geometry = np.asarray(numbers[2:], dtype=np.int32)

    return TagRecord(
        frame_id=frame_id,
        blink_id=blink_id,
        non_frontal_face=_parse_flag(fields[2]),
        left_fully_closed=_parse_flag(fields[3]),
        left_not_visible=_parse_flag(fields[4]),
        right_fully_closed=_parse_flag(fields[5]),
        right_not_visible=_parse_flag(fields[6]),
        face_xywh=geometry[0:4],
        left_eye_corners=geometry[4:8],
        right_eye_corners=geometry[8:12],
    )


def parse_tag_file(path: str | Path) -> list[TagRecord]:
    """Parse a ``.tag`` file into its records.

    Args:
        path (str | Path): The ``.tag`` file.

    Returns:
        list[TagRecord]: Records in file order.

    Raises:
        FileNotFoundError: If the file does not exist.
        TagParseError: If the ``#start``/``#end`` block is missing or malformed,
            or if any record line is invalid.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Tag file not found: {path}")

    lines = path.read_text(errors="replace").splitlines()

    start = next((i for i, line in enumerate(lines) if line.strip() == START_TOKEN), None)
    end = next((i for i, line in enumerate(lines) if line.strip() == END_TOKEN), None)

    if start is None or end is None:
        raise TagParseError(
            f"{path}: missing {START_TOKEN!r} and/or {END_TOKEN!r}; the record block "
            "cannot be located."
        )
    if end <= start:
        raise TagParseError(
            f"{path}: {END_TOKEN!r} (line {end + 1}) precedes {START_TOKEN!r} (line {start + 1})."
        )

    records = []
    for offset, line in enumerate(lines[start + 1 : end]):
        if not line.strip():
            continue
        records.append(_parse_record(line, path, start + 2 + offset))

    if not records:
        raise TagParseError(f"{path}: the record block is empty.")

    return records


def parse_timestamps(path: str | Path) -> dict[int, float]:
    """Parse a ``frame_id timestamp`` mapping file.

    These accompany the video corpora and are what makes the annotation
    reproducible: the ``.tag`` records index frames by counter, and only the
    timestamps tie those counters to positions in the video, whose decoded frame
    count need not match.

    Args:
        path (str | Path): The ``.txt`` timestamp file.

    Returns:
        dict[int, float]: Frame id to timestamp in seconds.

    Raises:
        FileNotFoundError: If the file does not exist.
        TagParseError: If a line is not a ``int float`` pair.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Timestamp file not found: {path}")

    mapping: dict[int, float] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 2:
            raise TagParseError(
                f"{path}:{line_number}: expected 'frame_id timestamp', got {line.strip()!r}"
            )
        try:
            mapping[int(parts[0])] = float(parts[1])
        except ValueError as error:
            raise TagParseError(
                f"{path}:{line_number}: non-numeric pair {line.strip()!r}"
            ) from error

    if not mapping:
        raise TagParseError(f"{path}: no timestamps found.")

    return mapping


class TagFile:
    """The parsed annotation of one video, with derived label arrays.

    Unlike the 1.x ``Tag`` class, constructing this requires only the ``.tag``
    file. Frames, video, and timestamps are the concern of whichever preprocess
    step needs them, which is what makes this class testable from a fixture
    alone.

    Args:
        records (list[TagRecord]): Parsed records, in file order.
        video_id (str): Identifier of the annotated video.

    Raises:
        TagParseError: If ``records`` is empty.
    """

    def __init__(self, records: list[TagRecord], video_id: str):
        if not records:
            raise TagParseError(f"{video_id}: cannot build a TagFile from zero records.")
        self.records = list(records)
        self.video_id = video_id

    @classmethod
    def from_path(cls, path: str | Path, video_id: str | None = None) -> TagFile:
        """Parse a ``.tag`` file.

        Args:
            path (str | Path): The ``.tag`` file.
            video_id (str | None): Identifier for this video. Defaults to the
                name of the file's parent directory, which is how every
                supported corpus lays its recordings out.

        Returns:
            TagFile: The parsed annotation.
        """
        path = Path(path)
        return cls(parse_tag_file(path), video_id or path.parent.name)

    def __len__(self) -> int:
        """Number of annotated frames.

        Returns:
            int: Record count.
        """
        return len(self.records)

    @property
    def frame_ids(self) -> np.ndarray:
        """Frame id of every record, in record order.

        Returns:
            np.ndarray: ``(N,)`` int64.
        """
        return np.asarray([r.frame_id for r in self.records], dtype=np.int64)

    @property
    def blink_ids(self) -> np.ndarray:
        """Blink id of every record, in record order.

        Returns:
            np.ndarray: ``(N,)`` int64, :data:`NO_BLINK` where the eye is open.
        """
        return np.asarray([r.blink_id for r in self.records], dtype=np.int64)

    @property
    def blink_presence(self) -> np.ndarray:
        """Per-frame blink-presence label.

        Returns:
            np.ndarray: ``(N,)`` float32, ``1.0`` where the frame belongs to a
            blink.
        """
        return (self.blink_ids != NO_BLINK).astype(np.float32)

    @property
    def eye_state(self) -> np.ndarray:
        """Per-frame, per-eye closed label.

        The corpora annotate the two eyes independently, and a
        head-turn or occlusion can close one while the other stays visible, so
        this is kept two-dimensional rather than collapsed to one flag.

        Returns:
            np.ndarray: ``(N, 2)`` float32, columns ``(left, right)``, ``1.0``
            where the eye is fully closed.
        """
        return np.asarray(
            [[r.left_fully_closed, r.right_fully_closed] for r in self.records],
            dtype=np.float32,
        )

    @property
    def validity(self) -> np.ndarray:
        """Per-frame, per-eye visibility.

        A frame whose eye is marked not-visible carries no usable supervision
        for that eye. Exposing this as a mask lets the loss skip those positions
        rather than training against a label the annotator could not see.

        Returns:
            np.ndarray: ``(N, 2)`` bool, columns ``(left, right)``, ``True``
            where the eye is visible.
        """
        return np.asarray(
            [[not r.left_not_visible, not r.right_not_visible] for r in self.records],
            dtype=bool,
        )

    def blink_events(self) -> list[BlinkEvent]:
        """Recover the individual blinks as runs of equal ``blink_id``.

        Grouping by id alone would merge two separate blinks that happen to
        reuse an id, and would produce a single event spanning the gap between
        them. Runs are therefore built from *consecutive* records, so a repeated
        id yields two events, which is what the annotation means.

        Returns:
            list[BlinkEvent]: Events in record order.
        """
        events: list[BlinkEvent] = []
        index = 0
        while index < len(self.records):
            record = self.records[index]
            if not record.is_blink:
                index += 1
                continue

            start = index
            while index < len(self.records) and self.records[index].blink_id == record.blink_id:
                index += 1

            events.append(
                BlinkEvent(
                    blink_id=record.blink_id,
                    start_index=start,
                    length=index - start,
                    first_frame_id=record.frame_id,
                )
            )
        return events

    def align_frame_ids(self, first_available_frame_id: int) -> None:
        """Shift record frame ids onto the extracted frames' numbering.

        Some corpora number their annotation from 0 and their extracted frames
        from 1. The 1.x code corrected this by mutating records in place with an
        unconditional ``+= 1`` guarded by a single ``if``; doing it explicitly
        here means the offset is logged and a caller can see that it happened.

        Args:
            first_available_frame_id (int): Frame id of the first extracted
                frame on disk.
        """
        offset = first_available_frame_id - self.records[0].frame_id
        if offset == 0:
            return

        logger.info(
            f"{self.video_id}: shifting {len(self.records)} annotation frame ids by "
            f"{offset:+d} to match the extracted frames."
        )
        self.records = [
            TagRecord(
                frame_id=record.frame_id + offset,
                blink_id=record.blink_id,
                non_frontal_face=record.non_frontal_face,
                left_fully_closed=record.left_fully_closed,
                left_not_visible=record.left_not_visible,
                right_fully_closed=record.right_fully_closed,
                right_not_visible=record.right_not_visible,
                face_xywh=record.face_xywh,
                left_eye_corners=record.left_eye_corners,
                right_eye_corners=record.right_eye_corners,
            )
            for record in self.records
        ]

    def restrict_to(self, available_frame_ids: set[int]) -> None:
        """Drop records whose frame was not extracted.

        Args:
            available_frame_ids (set[int]): Frame ids present on disk.

        Raises:
            TagParseError: If no record survives, which means the annotation and
                the extracted frames do not correspond at all.
        """
        kept = [r for r in self.records if r.frame_id in available_frame_ids]
        dropped = len(self.records) - len(kept)

        if not kept:
            raise TagParseError(
                f"{self.video_id}: none of the {len(self.records)} annotated frames "
                f"exist among the {len(available_frame_ids)} extracted ones. The "
                "annotation and the frames do not match."
            )
        if dropped:
            logger.warning(
                f"{self.video_id}: dropped {dropped}/{len(self.records)} annotated "
                "frames with no extracted image."
            )
        self.records = kept

    def summary(self) -> dict[str, int]:
        """Counts describing this annotation, for logging and manifests.

        Returns:
            dict[str, int]: Frame, blink-frame, and blink-event counts.
        """
        return {
            "frames": len(self.records),
            "blink_frames": int(self.blink_presence.sum()),
            "blink_events": len(self.blink_events()),
        }
