"""Draw what the pipeline saw onto the frames it saw it in.

A number in a log says a model fired; an overlay says *where it was looking* when
it did. That distinction is what makes the demo diagnostic rather than decorative
-- a red eye box explains a missing score far better than a gap in a plot.

The colour scheme is the whole legend:

============  =====================================================
colour        meaning
============  =====================================================
**blue**      the tracked face
**green**     an eye that was scored
**red**       an eye that was found but *suppressed* (head turned)
**magenta**   FaceMesh landmarks
**grey**      the readout, top-left
============  =====================================================

The readout sits on a translucent light panel, so it stays legible over any
footage without a drop shadow -- which was tried and reads as visual noise.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from blinklinmult.data.schema import LEFT, RIGHT

if TYPE_CHECKING:
    from pathlib import Path

    from blinklinmult.pipeline import Result

logger = logging.getLogger(__name__)
"""Module-level logger."""

FACE_COLOUR = (60, 130, 255)
"""Blue, RGB, for the tracked face box."""

USED_COLOUR = (60, 220, 90)
"""Green, RGB, for an eye that was scored."""

DROPPED_COLOUR = (240, 60, 60)
"""Red, RGB, for an eye found but suppressed."""

LANDMARK_COLOUR = (230, 60, 230)
"""Magenta, RGB, for the FaceMesh points."""

TEXT_COLOUR = (110, 110, 110)
"""Mid grey, RGB, for the readout.

Darker than it first appears it should be. The readout is drawn on a **panel**
(:data:`PANEL_COLOUR`) rather than straight onto the frame, so it needs contrast
against that panel, not against arbitrary video. A light grey chosen to survive a
dark background is what made the text vanish on a bright one.
"""

PANEL_COLOUR = (245, 245, 245)
"""Near-white, RGB, behind the readout.

Text alone cannot be legible over unknown video: whatever colour is picked, some
frame defeats it. A drop shadow is one fix and was rejected as visual noise; a
flat panel is the other, and it makes legibility a property of the panel rather
than a gamble on the footage.
"""

PANEL_ALPHA = 0.55
"""How opaque the panel is. Enough for contrast, translucent enough to see through."""

FONT_SCALE = 0.5
"""Text size. Small deliberately -- the readout must not cover the face."""

LINE_HEIGHT = 18
"""Pixels between readout lines."""

TEXT_ORIGIN = (8, 38)
"""Top-left corner of the readout.

One line lower than the frame's top edge: Gradio draws its own component label
over the top of the video, which would hide a first line placed there.
"""

LANDMARK_RADIUS = 1
"""Landmark dot radius. One pixel: 478 larger dots would hide the face."""


def _text(frame: np.ndarray, lines: list[str]) -> None:
    """Draw the readout into the top-left corner.

    Args:
        frame (np.ndarray): ``(H, W, 3)`` uint8 RGB, modified in place.
        lines (list[str]): One string per line.
    """
    import cv2

    x, y = TEXT_ORIGIN

    # One translucent panel behind the whole readout, blended rather than drawn
    # opaque so the frame stays visible underneath.
    width = max(
        (cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)[0][0] for text in lines),
        default=0,
    )
    top = y - LINE_HEIGHT + 4
    bottom = y + (len(lines) - 1) * LINE_HEIGHT + 6
    region = frame[max(top, 0) : bottom, max(x - 6, 0) : x + width + 6]
    if region.size:
        panel = np.full_like(region, PANEL_COLOUR, dtype=np.uint8)
        cv2.addWeighted(panel, PANEL_ALPHA, region, 1.0 - PANEL_ALPHA, 0.0, region)

    for offset, line in enumerate(lines):
        position = (x, y + offset * LINE_HEIGHT)
        cv2.putText(
            frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, TEXT_COLOUR, 1, cv2.LINE_AA
        )


def _readout(result: Result, index: int) -> list[str]:
    """Compose the per-frame readout.

    Angles are signed integers -- a tenth of a degree of yaw is noise, and the
    sign is the part that matters. Scores are two decimals. A suppressed eye
    reads ``--`` rather than a stale number, so the overlay never implies a
    measurement that was not taken.

    Args:
        result (Result): The finished analysis.
        index (int): Which frame.

    Returns:
        list[str]: Lines to draw.
    """
    frame_result = result.frames[index]
    pose = frame_result.pose
    if pose is None:
        lines = [f"Frame: {index}", "yaw: --", "pitch: --", "roll: --"]
    else:
        lines = [
            f"Frame: {index}",
            f"yaw: {int(round(float(pose[0])))}",
            f"pitch: {int(round(float(pose[1])))}",
            f"roll: {int(round(float(pose[2])))}",
        ]

    for side, label in ((LEFT, "left"), (RIGHT, "right")):
        score = frame_result.score.get(side)
        shown = "--" if score is None else f"{score:.2f}"
        # "state", not "blink": the number is per-frame eye *closeness*, and a
        # blink is an event derived from a run of them. Calling it a blink would
        # promise something the value does not carry.
        lines.append(f"{label} state: {shown}")
    return lines


def draw(frame: np.ndarray, result: Result, index: int) -> np.ndarray:
    """Annotate one frame.

    Args:
        frame (np.ndarray): ``(H, W, 3)`` uint8 RGB. Not modified.
        result (Result): The finished analysis.
        index (int): Which frame this is.

    Returns:
        np.ndarray: A new annotated frame.
    """
    import cv2

    canvas = np.ascontiguousarray(frame.copy())
    frame_result = result.frames[index]

    # The readout is drawn **first**, so its panel sits underneath the boxes and
    # landmarks rather than over them. Drawn last, the panel blanks whatever
    # shares its corner -- on a small frame or a face near the top-left, that is
    # the eye box whose colour the reader needs most.
    _text(canvas, _readout(result, index))

    if frame_result.landmarks is not None:
        for point in frame_result.landmarks:
            cv2.circle(
                canvas,
                (int(point[0]), int(point[1])),
                LANDMARK_RADIUS,
                LANDMARK_COLOUR,
                -1,
                cv2.LINE_AA,
            )

    if frame_result.face_box is not None:
        x1, y1, x2, y2 = frame_result.face_box
        cv2.rectangle(canvas, (x1, y1), (x2, y2), FACE_COLOUR, 2)

    for side in (LEFT, RIGHT):
        box = frame_result.eyes.get(side)
        if box is None:
            continue
        colour = USED_COLOUR if frame_result.used.get(side, False) else DROPPED_COLOUR
        x1, y1, x2, y2 = box.xyxy
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)

    return canvas


def render(frames: np.ndarray, result: Result, output_path: str | Path) -> Path:
    """Write the annotated video.

    Args:
        frames (np.ndarray): ``(T, H, W, 3)`` uint8 RGB, the source frames.
        result (Result): The finished analysis, same length.
        output_path (str | Path): Where to write the mp4.

    Returns:
        Path: The written file.
    """
    from pathlib import Path as _Path

    from exordium.video.core.io import save_video

    annotated = np.stack([draw(frames[i], result, i) for i in range(len(result.frames))])
    destination = _Path(output_path)
    save_video(annotated, destination, fps=result.fps)
    logger.info(f"Wrote {destination} ({annotated.shape[0]} frames).")
    return destination


def plot(result: Result):
    """Plot both eyes: closeness, thresholds and the binary prediction.

    Two stacked panels sharing the frame axis, left eye above right, so the two
    can be read against each other -- eyes blink together, and a disagreement is
    usually one eye being suppressed rather than a real difference.

    Suppressed spans are **shaded and the line breaks**, because a gap means the
    eye was not looked at. Drawing through it would assert a measurement that was
    never taken.

    Args:
        result (Result): The finished analysis.

    Returns:
        matplotlib.figure.Figure: The figure. The caller owns it and should
        close it; leaving figures open leaks memory across repeated runs.
    """
    import matplotlib.pyplot as plt

    # The rule that actually produced the events, not the registry default --
    # the caller may have overridden it, and drawing the registered line under
    # differently-extracted events would be a lie.
    if result.extraction is not None:
        high, low = result.extraction.high, result.extraction.low
    else:
        from blinklinmult.registry import spec

        entry = spec(result.model_id)
        high = entry.threshold
        low = None if entry.low_ratio is None else high * entry.low_ratio

    figure, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    frames = np.arange(len(result.frames))

    for axis, side, label in zip(axes, (LEFT, RIGHT), ("Left eye", "Right eye"), strict=True):
        curve = result.signal[side]
        missing = ~np.isfinite(curve)

        # Shade every run of suppressed frames.
        if missing.any():
            edges = np.diff(np.concatenate(([0], missing.view(np.int8), [0])))
            for start, stop in zip(
                np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True
            ):
                axis.axvspan(start - 0.5, stop - 0.5, color="grey", alpha=0.18)

        for start, stop in result.events[side]:
            axis.axvspan(start - 0.5, stop + 0.5, color="crimson", alpha=0.12)

        # The annotation, where the clip came with one. Drawn as a step rather
        # than a span: it is a per-frame label, and overlaying it as shading
        # would be indistinguishable from the prediction's own spans.
        if result.truth is not None and side in result.truth:
            axis.step(
                np.arange(result.truth[side].shape[0]),
                result.truth[side],
                where="mid",
                color="green",
                linewidth=1.4,
                alpha=0.85,
                label="ground truth",
            )

        axis.plot(frames, curve, color="#1f77b4", linewidth=1.8, label="eye closeness")
        axis.axhline(high, color="darkorange", linestyle="--", linewidth=1.1, label=f"high {high}")
        if low is not None:
            axis.axhline(
                low, color="seagreen", linestyle=":", linewidth=1.1, label=f"low {low:.3f}"
            )
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("eye closeness\n[0 open, 1 closed]")
        # Each panel names its own eye and what was found there, so a reader can
        # tell the two apart without counting rows against a shared legend.
        blinks = len(result.events[side])
        missing_n = int(missing.sum())
        suffix = f", {missing_n} frames suppressed" if missing_n else ""
        axis.set_title(
            f"{label} -- {blinks} blink{'' if blinks == 1 else 's'}{suffix}",
            fontsize=10,
            loc="left",
        )
        axis.grid(alpha=0.3)
        axis.legend(loc="upper right", fontsize=7, ncol=3)

    axes[-1].set_xlabel("frame")
    figure.suptitle(
        f"Eye state recognition -- {result.model_id}   "
        f"(red = detected blink, grey = eye suppressed)",
        fontsize=11,
    )
    figure.tight_layout()
    return figure
