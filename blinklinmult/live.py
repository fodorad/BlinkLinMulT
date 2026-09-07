"""Rendering a live view: the frame, and a scrolling plot per eye.

What the batch pipeline draws
(:func:`~blinklinmult.overlay.draw`) needs a finished
:class:`~blinklinmult.pipeline.Result` -- a whole-recording object that does not
exist while frames are still arriving. So the live view draws its own, reusing
:mod:`blinklinmult.overlay`'s colours so the two look like the same product.

**Everything here is cv2 into a numpy canvas, not matplotlib.** Measured, a
matplotlib redraw of two small subplots costs **6.1 ms** per frame against
**0.13 ms** for a cv2 polyline -- a 47x difference, and the display loop only has
about 6 ms of slack once detection and ``imshow`` are paid. A plotting library
would spend a third of the remaining headroom drawing two lines.

The layout is the frame on top, and beneath it two subplots at half the frame
width each: the viewer's left eye on the left. A wink then spikes the plot on
the same side as the eye that closed, which is the whole point of the view.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from blinklinmult.overlay import (
    DROPPED_COLOUR,
    FACE_COLOUR,
    FONT_SCALE,
    LANDMARK_COLOUR,
    PANEL_COLOUR,
    TEXT_COLOUR,
    USED_COLOUR,
)

DEFAULT_SECONDS = 10.0
"""Seconds of score history each plot shows."""

PLOT_HEIGHT = 130
"""Height of one eye's subplot, in pixels."""

PLOT_BACKGROUND = (248, 248, 248)
"""Near-white behind a plot, matching the readout panel's family."""

GRID_COLOUR = (215, 215, 215)
"""Light grey for the frame around a plot."""

THRESHOLD_COLOUR = (120, 120, 200)
"""Muted blue for the operating-point rule.

Blue rather than grey: on a near-white panel a mid grey line is easy to miss,
and a viewer who cannot see the threshold cannot tell whether a spike counted as
a blink -- which is the whole reason the rule is drawn.
"""

AXIS_COLOUR = (205, 205, 205)
"""The zero line, so the trace has something to sit on.

Without it an open eye's near-zero score looks like an empty plot rather than a
measured one.
"""

TRACE_COLOUR = (200, 60, 60)
"""The score trace.

**One colour for the whole line.** Colouring it by whether the score is above
the threshold sounds informative and is not: a single spike at the end repaints
the entire history, so the plot's colour reports only the newest sample while
appearing to describe all ten seconds. The threshold rule already shows where
the operating point is, and the eye boxes on the frame carry the open/closed
state.
"""

TRACE_WIDTH = 2
"""Polyline thickness. Two pixels reads clearly without hiding detail."""

PLOT_MARGIN = 6
"""Padding inside a plot, so the trace never touches the border."""

LABEL_HEIGHT = 16
"""Rows reserved at the top for the label and the current reading."""


@dataclass
class ScoreHistory:
    """A bounded, per-eye history of closure scores.

    A ``deque`` with ``maxlen`` rather than a list: a demo left running for an
    hour uses exactly the same memory as one left running for a second, because
    the oldest sample falls off as each new one arrives.

    Args:
        capacity (int): Samples retained -- normally ``fps * seconds``.
    """

    capacity: int
    values: deque = field(init=False)

    def __post_init__(self) -> None:
        """Allocate the bounded queue."""
        self.values = deque(maxlen=max(int(self.capacity), 1))

    def push(self, value: float) -> None:
        """Add one score.

        Args:
            value (float): Closure score in ``[0, 1]``, or ``nan`` for a frame
                with no usable eye. **A gap, not a zero**: zero means "open",
                which is a claim the pipeline cannot make when it never saw the
                eye.
        """
        self.values.append(float(value))

    @property
    def latest(self) -> float:
        """The most recent score.

        Returns:
            float: The last value pushed, or ``nan`` when empty.
        """
        return self.values[-1] if self.values else float("nan")

    def as_array(self) -> np.ndarray:
        """The history, oldest first.

        Returns:
            np.ndarray: ``(n,)`` float32.
        """
        return np.asarray(self.values, dtype=np.float32)


def draw_plot(
    history: ScoreHistory,
    threshold: float,
    label: str,
    width: int,
    height: int = PLOT_HEIGHT,
) -> np.ndarray:
    """Draw one eye's scrolling score plot.

    The trace fills only the fraction of the width its samples cover, so a
    two-second-old session does not stretch to look like a full ten seconds of
    data. Gaps -- frames where no eye was found -- break the line rather than
    dropping it to zero.

    Args:
        history (ScoreHistory): The scores to draw.
        threshold (float): Operating point, drawn as a horizontal rule so a
            spike crossing it is visibly a blink rather than just a bump.
        label (str): Shown top-left, e.g. ``"left eye"``.
        width (int): Canvas width in pixels.
        height (int): Canvas height in pixels.

    Returns:
        np.ndarray: ``(height, width, 3)`` uint8 RGB.
    """
    import cv2  # ty: ignore[unresolved-import]

    canvas = np.full((height, width, 3), PLOT_BACKGROUND, dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), GRID_COLOUR, 1)

    # The label sits on the first row, so the plotting area starts below it --
    # otherwise a score near 1.0 draws through the text.
    top = PLOT_MARGIN + LABEL_HEIGHT
    bottom = height - PLOT_MARGIN
    span = max(bottom - top, 1)

    def row_of(value: float) -> int:
        """Map a score to a pixel row, 1.0 at the top.

        Args:
            value (float): Score in ``[0, 1]``.

        Returns:
            int: Row index.
        """
        return int(round(bottom - float(np.clip(value, 0.0, 1.0)) * span))

    cv2.line(canvas, (0, row_of(0.0)), (width - 1, row_of(0.0)), AXIS_COLOUR, 1)
    threshold_row = row_of(threshold)
    cv2.line(canvas, (0, threshold_row), (width - 1, threshold_row), THRESHOLD_COLOUR, 1)
    cv2.putText(
        canvas,
        f"{threshold:.2f}",
        (width - 40, max(threshold_row - 3, 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        THRESHOLD_COLOUR,
        1,
    )

    values = history.as_array()
    if values.size:
        # Position samples against the *capacity*, not against how many have
        # arrived, so the trace scrolls in from the right as history fills
        # rather than stretching to fit.
        columns = np.linspace(0, width - 1, history.values.maxlen or values.size)
        columns = columns[-values.size :]
        finite = np.isfinite(values)
        # A gap splits the trace into separate polylines: drawing across it
        # would invent a value for a frame the model never scored.
        breaks = np.flatnonzero(np.diff(finite.astype(np.int8)) != 0) + 1
        for chunk in np.split(np.arange(values.size), breaks):
            if chunk.size < 2 or not finite[chunk[0]]:
                continue
            points = np.stack([columns[chunk], [row_of(v) for v in values[chunk]]], axis=1).astype(
                np.int32
            )
            cv2.polylines(canvas, [points], False, TRACE_COLOUR, TRACE_WIDTH, cv2.LINE_AA)

    cv2.putText(
        canvas, label, (PLOT_MARGIN, 18), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOUR, 1
    )
    current = history.latest
    reading = "--" if not np.isfinite(current) else f"{current:.2f}"
    cv2.putText(
        canvas,
        reading,
        (width - 52, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        FONT_SCALE,
        TEXT_COLOUR,
        1,
    )
    return canvas


def compose(frame: np.ndarray, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack the frame above the two eye plots.

    Args:
        frame (np.ndarray): ``(H, W, 3)`` uint8 RGB, already annotated.
        left (np.ndarray): The viewer's-left eye plot.
        right (np.ndarray): The viewer's-right eye plot.

    Returns:
        np.ndarray: ``(H + plot_height, W, 3)`` uint8 RGB, the plots side by
        side beneath the frame at half its width each.
    """
    import cv2  # ty: ignore[unresolved-import]

    width = frame.shape[1]
    half = width // 2
    height = left.shape[0]

    strip = np.full((height, width, 3), PLOT_BACKGROUND, dtype=np.uint8)
    strip[:, :half] = cv2.resize(left, (half, height))
    # The right half absorbs the odd pixel when the width is not even, so the
    # strip always matches the frame exactly and the stack cannot fail.
    strip[:, half:] = cv2.resize(right, (width - half, height))
    return np.vstack([frame, strip])


def draw_overlay(
    frame: np.ndarray,
    detection,
    scores: dict[str, float],
    threshold: float,
    lines: list[str] | None = None,
) -> np.ndarray:
    """Annotate a live frame with the face, the eyes and a readout.

    Args:
        frame (np.ndarray): ``(H, W, 3)`` uint8 RGB. Not modified.
        detection: A ``FaceDetection`` from
            :meth:`~blinklinmult.preprocess.extractors.PoseEyeLocator.detect`,
            or ``None`` when no face was found.
        scores (dict[str, float]): Score per side; ``nan`` where unknown.
        threshold (float): Operating point, deciding an eye box's colour.
        lines (list[str] | None): Extra readout lines, e.g. fps and head pose.

    Returns:
        np.ndarray: A new annotated frame.
    """
    import cv2  # ty: ignore[unresolved-import]

    canvas = np.ascontiguousarray(frame.copy())
    if detection is not None:
        x1, y1, x2, y2 = detection.face_box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), FACE_COLOUR, 2)

        for point in np.asarray(detection.landmarks, dtype=np.int32):
            cv2.circle(canvas, (int(point[0]), int(point[1])), 2, LANDMARK_COLOUR, -1)

        for side, box in detection.eyes.items():
            if box is None:
                continue
            score = scores.get(side, float("nan"))
            # Green while the eye reads open, red once it crosses the operating
            # point -- the same colours the batch overlay uses for used/dropped,
            # so one legend covers both views.
            colour = DROPPED_COLOUR if np.isfinite(score) and score >= threshold else USED_COLOUR
            half = box.side // 2
            cv2.rectangle(
                canvas,
                (box.centre_x - half, box.centre_y - half),
                (box.centre_x + half, box.centre_y + half),
                colour,
                2,
            )

    if lines:
        _panel(canvas, lines)
    return canvas


def _panel(canvas: np.ndarray, lines: list[str]) -> None:
    """Draw a readout on a translucent panel.

    Text alone cannot be legible over unknown video -- whatever colour is
    chosen, some frame defeats it. The panel makes legibility a property of the
    drawing rather than a gamble on the footage.

    Args:
        canvas (np.ndarray): Modified in place.
        lines (list[str]): Readout lines.
    """
    import cv2  # ty: ignore[unresolved-import]

    from blinklinmult.overlay import LINE_HEIGHT, PANEL_ALPHA

    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)[0][0] for t in lines)
    box = canvas[4 : 12 + LINE_HEIGHT * len(lines), 4 : 16 + width]
    if box.size:
        box[:] = (box * (1 - PANEL_ALPHA) + np.array(PANEL_COLOUR) * PANEL_ALPHA).astype(np.uint8)
    for index, text in enumerate(lines):
        cv2.putText(
            canvas,
            text,
            (10, 22 + LINE_HEIGHT * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            TEXT_COLOUR,
            1,
        )


def eye_crops(detection, frame: np.ndarray, image_size: int = 64) -> np.ndarray | None:
    """Cut both eye crops into the layout the models take.

    Args:
        detection: A ``FaceDetection`` from
            :meth:`~blinklinmult.preprocess.extractors.PoseEyeLocator.detect`.
        frame (np.ndarray): ``(H, W, 3)`` RGB.
        image_size (int): Crop side the model expects.

    Returns:
        np.ndarray | None: ``(2, 3, size, size)`` in ``[0, 1]``, or ``None``
        unless **both** eyes were located. A one-eye batch is refused rather
        than scored: the caller reads the two scores positionally, so a missing
        side would silently shift the right eye's score onto the left.
    """
    import cv2  # ty: ignore[unresolved-import]

    from blinklinmult.preprocess.geometry import LEFT, RIGHT

    patches = []
    for side in (LEFT, RIGHT):
        box = detection.eyes.get(side)
        if box is None:
            return None
        patch = box.crop(frame)
        if patch.size == 0:
            return None
        resized = cv2.resize(patch, (image_size, image_size))
        patches.append(resized.transpose(2, 0, 1).astype("float32") / 255.0)
    return np.stack(patches)
