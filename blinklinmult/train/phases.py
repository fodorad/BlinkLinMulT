"""Derive blink phase and event shape from a predicted eye-state signal.

The model emits one continuous value per frame. That value alone is ambiguous --
0.5 could be an eye halfway shut or halfway open -- but its **trajectory** is
not: falling means closing, rising means opening, flat-and-low means closed.
This module reads phase off the signal's derivative rather than off the
annotation's geometry, which matters because no corpus annotates phase and
inventing it from interval position would assert a closure shape rather than
measure one.

The same trajectory separates event *kinds*. A complete blink descends, holds
briefly, and ascends; an incomplete one reverses before it bottoms out; a double
blink shows two minima inside one elevated span -- which no single-threshold
extractor can split, because thresholding sees one run either way.

**These categories are derived, not annotated.** They are measurements of the
predicted signal and carry no ground truth, so they belong in analysis and in
downstream state estimation, never in a benchmark column.
"""

from __future__ import annotations

import numpy as np

OPEN = 0
"""Phase: the eye is open and not moving."""

CLOSING = 1
"""Phase: the lid is descending."""

CLOSED = 2
"""Phase: the lid is down and still."""

OPENING = 3
"""Phase: the lid is rising."""

PHASE_NAMES: dict[int, str] = {
    OPEN: "open",
    CLOSING: "closing",
    CLOSED: "closed",
    OPENING: "opening",
}
"""Human-readable phase names, for reports and plots."""

MOTION_EPSILON = 0.02
"""Per-frame change below which the lid counts as still.

Separates a plateau from a ramp. Small enough that a genuine closure -- which
moves through most of its range in three or four frames -- is never called
still, large enough that prediction noise on an open eye is not called motion.
"""

CLOSED_LEVEL = 0.5
"""Signal value above which a still frame is closed rather than open."""


class PhaseError(ValueError):
    """Raised when a signal cannot be analysed."""


def smooth(signal: np.ndarray, window: int = 3) -> np.ndarray:
    """Moving average, for a derivative that is not dominated by noise.

    Args:
        signal (np.ndarray): ``(T,)`` per-frame values.
        window (int): Frames to average over; ``1`` disables smoothing.

    Returns:
        np.ndarray: ``(T,)`` smoothed, same length as the input.

    Raises:
        PhaseError: If the window is not a positive odd integer.
    """
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if window < 1 or window % 2 == 0:
        raise PhaseError(f"window must be a positive odd integer, got {window}.")
    if window == 1 or values.size == 0:
        return values

    pad = window // 2
    # Edge-padded so the result keeps its length and the first and last frames
    # are not pulled toward zero by implicit zero padding.
    padded = np.pad(values, pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / window
    return np.convolve(padded, kernel, mode="valid")


def phases(
    signal: np.ndarray,
    window: int = 3,
    motion_epsilon: float = MOTION_EPSILON,
    closed_level: float = CLOSED_LEVEL,
) -> np.ndarray:
    """Label every frame open, closing, closed, or opening.

    Phase comes from the sign of the smoothed derivative, with the value used
    only to tell a still-and-low frame from a still-and-high one. That ordering
    is deliberate: the value is ambiguous mid-blink and the direction is not.

    Args:
        signal (np.ndarray): ``(T,)`` predicted eye-state values, higher meaning
            more closed.
        window (int): Smoothing window before differentiating.
        motion_epsilon (float): Per-frame change below which the lid is still.
        closed_level (float): Value above which a still frame is closed.

    Returns:
        np.ndarray: ``(T,)`` int8 of :data:`OPEN`, :data:`CLOSING`,
        :data:`CLOSED`, :data:`OPENING`.
    """
    values = smooth(signal, window)
    if values.size == 0:
        return np.zeros(0, dtype=np.int8)

    # Central difference, so a frame's phase reflects motion *through* it rather
    # than motion since the previous frame.
    derivative = np.gradient(values) if values.size > 1 else np.zeros(1)

    labels = np.full(values.size, OPEN, dtype=np.int8)
    moving = np.abs(derivative) > motion_epsilon
    labels[moving & (derivative > 0)] = CLOSING
    labels[moving & (derivative < 0)] = OPENING
    labels[~moving & (values >= closed_level)] = CLOSED
    return labels


def describe_event(
    signal: np.ndarray,
    start: int,
    stop: int,
) -> dict[str, float]:
    """Measure the shape of one predicted event.

    What distinguishes a complete blink from an incomplete or a double one is
    the curve, not the extent: peak depth says whether the lid ever really shut,
    the minima count says whether it shut twice, and the asymmetry says whether
    it reopened as fast as it closed (real blinks close faster than they open).

    Args:
        signal (np.ndarray): ``(T,)`` values for the whole recording.
        start (int): First frame of the event, inclusive.
        stop (int): Last frame, inclusive.

    Returns:
        dict[str, float]: ``frames``, ``peak``, ``mean``, ``n_peaks``,
        ``rise_frames``, ``fall_frames``, and ``asymmetry`` -- the fall minus
        the rise, in frames, positive when reopening takes longer.

    Raises:
        PhaseError: If the span is empty or outside the signal.
    """
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if not 0 <= start <= stop < values.size:
        raise PhaseError(f"span ({start}, {stop}) is outside a {values.size}-frame signal.")

    span = values[start : stop + 1]
    apex = int(span.argmax())

    # Local maxima, which is what a double blink shows: two closures inside one
    # elevated run, indistinguishable to any single threshold.
    #
    # The span is padded with a value below its minimum before the search, so a
    # peak sitting on the first or last frame still counts. An extracted event
    # is cut at its own boundaries, so its closures very often *are* its
    # endpoints -- an interior-only search reports one peak for a textbook
    # double blink.
    floor = span.min() - 1.0
    padded = np.concatenate([[floor], span, [floor]])
    higher_than_neighbours = (padded[1:-1] > padded[:-2]) & (padded[1:-1] > padded[2:])
    n_peaks = max(1, int(higher_than_neighbours.sum()))

    return {
        "frames": float(span.size),
        "peak": float(span.max()),
        "mean": float(span.mean()),
        "n_peaks": float(n_peaks),
        "fall_frames": float(apex),
        "rise_frames": float(span.size - 1 - apex),
        "asymmetry": float((span.size - 1 - apex) - apex),
    }


def classify_event(
    shape: dict[str, float],
    complete_peak: float = 0.7,
    min_frames: float = 2.0,
) -> str:
    """Name an event from its measured shape.

    Args:
        shape (dict[str, float]): As :func:`describe_event` returns.
        complete_peak (float): Peak below which the lid never fully shut.
        min_frames (float): Spans shorter than this are too brief to judge.

    Returns:
        str: ``"double"``, ``"incomplete"``, ``"long"``, ``"brief"``, or
        ``"complete"``. Derived from the prediction, never from an annotation.
    """
    if shape["n_peaks"] >= 2:
        return "double"
    if shape["peak"] < complete_peak:
        return "incomplete"
    if shape["frames"] < min_frames:
        return "brief"
    if shape["frames"] > 12:
        return "long"
    return "complete"
