"""Tests for the live view's rendering.

Everything here is numpy in, numpy out, so none of it needs a camera or a
display. What is *not* tested is the capture loop and the window -- those need
hardware, and the pieces they would exercise are the functions below.

The assertions concentrate on the ways a scrolling plot misleads: an inverted
y-axis, a partial history stretched to look complete, and a gap drawn as zero
when the truth is "the model never saw this frame".
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.live import (
    LABEL_HEIGHT,
    PLOT_BACKGROUND,
    PLOT_HEIGHT,
    PLOT_MARGIN,
    TRACE_COLOUR,
    ScoreHistory,
    compose,
    draw_overlay,
    draw_plot,
    eye_crops,
)
from blinklinmult.preprocess.geometry import LEFT, RIGHT

WIDTH = 480
"""Half of the 960 px display width, matching one subplot."""


def _filled(capacity: int, value: float = 0.2) -> ScoreHistory:
    """A history filled to capacity with one value.

    Args:
        capacity (int): Samples retained.
        value (float): The score to repeat.

    Returns:
        ScoreHistory: The filled history.
    """
    history = ScoreHistory(capacity=capacity)
    for _ in range(capacity):
        history.push(value)
    return history


class TestScoreHistory(unittest.TestCase):
    """The bounded per-eye buffer."""

    def test_memory_does_not_grow_with_the_session(self) -> None:
        """An hour-long demo must use the same memory as a one-second one."""
        history = ScoreHistory(capacity=300)
        for _ in range(10000):
            history.push(0.5)
        self.assertEqual(len(history.values), 300)

    def test_the_oldest_sample_falls_off(self) -> None:
        """Scrolling means the window moves, not that it grows."""
        history = ScoreHistory(capacity=3)
        for value in (0.1, 0.2, 0.3, 0.4):
            history.push(value)
        np.testing.assert_allclose(history.as_array(), [0.2, 0.3, 0.4], atol=1e-6)

    def test_latest_is_the_last_pushed(self) -> None:
        """The readout beside the plot shows this."""
        history = ScoreHistory(capacity=10)
        history.push(0.1)
        history.push(0.9)
        self.assertAlmostEqual(history.latest, 0.9, places=6)

    def test_an_empty_history_has_no_latest_value(self) -> None:
        """Before the first frame there is nothing to report, not a zero."""
        self.assertTrue(np.isnan(ScoreHistory(capacity=10).latest))

    def test_a_capacity_of_zero_still_holds_one_sample(self) -> None:
        """A degenerate configuration must not make the deque unusable."""
        history = ScoreHistory(capacity=0)
        history.push(0.5)
        self.assertEqual(len(history.values), 1)


class TestDrawPlot(unittest.TestCase):
    """One eye's subplot."""

    def test_the_canvas_has_the_requested_shape(self) -> None:
        """The composer stacks these, so a wrong shape breaks the layout."""
        plot = draw_plot(_filled(300), 0.5, "left eye", WIDTH, PLOT_HEIGHT)
        self.assertEqual(plot.shape, (PLOT_HEIGHT, WIDTH, 3))
        self.assertEqual(plot.dtype, np.uint8)

    def test_an_empty_history_draws_without_raising(self) -> None:
        """The first frames arrive before any score does."""
        plot = draw_plot(ScoreHistory(capacity=300), 0.5, "left eye", WIDTH)
        self.assertEqual(plot.shape[1], WIDTH)

    def test_the_threshold_line_is_the_right_way_up(self) -> None:
        """A high threshold must sit *above* a low one on screen.

        An inverted y-axis is the obvious bug here and would be invisible in a
        screenshot: the plot would still look like a plausible signal.
        """
        high = draw_plot(ScoreHistory(capacity=10), 0.9, "eye", WIDTH)
        low = draw_plot(ScoreHistory(capacity=10), 0.1, "eye", WIDTH)
        background = np.array(PLOT_BACKGROUND, dtype=np.uint8)

        def rule_row(canvas: np.ndarray) -> int:
            """Find the drawn horizontal rule, ignoring the border."""
            inner = canvas[1:-1, WIDTH // 2]
            changed = np.flatnonzero(np.any(inner != background, axis=1))
            return int(changed[0]) + 1

        self.assertLess(rule_row(high), rule_row(low))

    def test_the_threshold_line_lands_where_arithmetic_says(self) -> None:
        """0.5 on a 130 px canvas sits mid-way between the margins."""
        canvas = draw_plot(ScoreHistory(capacity=10), 0.5, "eye", WIDTH, PLOT_HEIGHT)
        top = PLOT_MARGIN + LABEL_HEIGHT
        bottom = PLOT_HEIGHT - PLOT_MARGIN
        expected = round(bottom - 0.5 * (bottom - top))
        background = np.array(PLOT_BACKGROUND, dtype=np.uint8)
        inner = canvas[1:-1, WIDTH // 2]
        drawn = np.flatnonzero(np.any(inner != background, axis=1)) + 1
        self.assertTrue(np.any(np.abs(drawn - expected) <= 1), f"rule not near row {expected}")

    def test_a_partial_history_does_not_fill_the_width(self) -> None:
        """One second of data must not look like ten.

        The trace scrolls in from the right; stretching it to fit would show a
        short session as a complete one. Checked by looking for the *trace
        colour* rather than for any drawn pixel -- the border, the label and the
        threshold rule all legitimately span the full width.
        """
        history = ScoreHistory(capacity=300)
        for _ in range(30):
            history.push(0.8)
        canvas = draw_plot(history, 0.5, "eye", WIDTH)
        # 0.8 is above the 0.5 threshold, so the trace is drawn in the
        # above-threshold colour and nothing else on the canvas uses it.
        trace = np.all(np.abs(canvas.astype(int) - np.array(TRACE_COLOUR)) < 60, axis=2)
        columns = np.flatnonzero(trace.any(axis=0))
        self.assertTrue(columns.size, "no trace was drawn at all")
        self.assertGreater(
            int(columns.min()), WIDTH // 2, "a tenth of the history spanned half the plot"
        )

    def test_a_gap_is_not_drawn_as_zero(self) -> None:
        """A frame with no face is unknown, not "eyes open".

        Drawing across the gap would put the line at the bottom of the plot,
        which reads as a confident open-eye measurement.
        """
        history = ScoreHistory(capacity=20)
        for _ in range(10):
            history.push(0.9)
        for _ in range(10):
            history.push(float("nan"))
        canvas = draw_plot(history, 0.5, "eye", WIDTH)
        background = np.array(PLOT_BACKGROUND, dtype=np.uint8)
        bottom_right = canvas[PLOT_HEIGHT - PLOT_MARGIN - 3, WIDTH - 10]
        self.assertTrue(np.array_equal(bottom_right, background))

    def test_the_current_value_is_shown(self) -> None:
        """The number beside the plot is what a viewer reads off it."""
        with_value = draw_plot(_filled(10, 0.42), 0.5, "eye", WIDTH)
        empty = draw_plot(ScoreHistory(capacity=10), 0.5, "eye", WIDTH)
        self.assertFalse(np.array_equal(with_value, empty))


class TestCompose(unittest.TestCase):
    """Stacking the frame above the two plots."""

    def test_the_layout_matches_the_frame_width(self) -> None:
        """Frame on top, two half-width plots beneath."""
        frame = np.zeros((540, 960, 3), dtype=np.uint8)
        plot = draw_plot(_filled(300), 0.5, "eye", 480)
        view = compose(frame, plot, plot)
        self.assertEqual(view.shape, (540 + PLOT_HEIGHT, 960, 3))

    def test_an_odd_width_still_composes(self) -> None:
        """The right half absorbs the spare pixel, so the stack cannot fail."""
        frame = np.zeros((100, 961, 3), dtype=np.uint8)
        plot = draw_plot(_filled(30), 0.5, "eye", 480)
        self.assertEqual(compose(frame, plot, plot).shape[1], 961)

    def test_the_two_plots_land_on_their_own_sides(self) -> None:
        """Left plot left, right plot right.

        The demo's whole purpose is that winking one eye moves the plot on that
        side; swapping them here would invert it silently.
        """
        frame = np.zeros((60, 400, 3), dtype=np.uint8)
        left = np.full((PLOT_HEIGHT, 200, 3), (255, 0, 0), dtype=np.uint8)
        right = np.full((PLOT_HEIGHT, 200, 3), (0, 0, 255), dtype=np.uint8)
        view = compose(frame, left, right)
        strip = view[60:]
        self.assertGreater(int(strip[PLOT_HEIGHT // 2, 50][0]), 200)
        self.assertGreater(int(strip[PLOT_HEIGHT // 2, 350][2]), 200)


class _Box:
    """A minimal stand-in for an ``EyeBox``, built by hand rather than detected."""

    def __init__(self, centre_x: int, centre_y: int, side: int) -> None:
        """Record the geometry the renderer reads."""
        self.centre_x = centre_x
        self.centre_y = centre_y
        self.side = side

    def crop(self, frame: np.ndarray) -> np.ndarray:
        """Cut this box out of a frame.

        Args:
            frame (np.ndarray): The source frame.

        Returns:
            np.ndarray: The patch.
        """
        half = self.side // 2
        return frame[
            self.centre_y - half : self.centre_y + half,
            self.centre_x - half : self.centre_x + half,
        ]


class _Detection:
    """A stand-in for ``FaceDetection`` carrying only what the renderer uses."""

    def __init__(self, eyes: dict) -> None:
        """Build a detection around a fixed face box and keypoints."""
        self.face_box = (40, 30, 200, 190)
        self.landmarks = np.array(
            [[70.0, 80.0], [150.0, 80.0], [110.0, 120.0], [85.0, 160.0], [135.0, 160.0]]
        )
        self.eyes = eyes


def _frame(height: int = 240, width: int = 320) -> np.ndarray:
    """A mid-grey frame to draw on.

    Args:
        height (int): Frame height.
        width (int): Frame width.

    Returns:
        np.ndarray: ``(height, width, 3)`` uint8 RGB.
    """
    return np.full((height, width, 3), 128, dtype=np.uint8)


class TestDrawOverlay(unittest.TestCase):
    """Annotating a live frame."""

    def _detection(self) -> _Detection:
        """A detection with both eyes located."""
        return _Detection({LEFT: _Box(70, 80, 40), RIGHT: _Box(150, 80, 40)})

    def test_the_source_frame_is_not_modified(self) -> None:
        """The caller may still want the clean frame, e.g. to record it."""
        frame = _frame()
        original = frame.copy()
        draw_overlay(frame, self._detection(), {LEFT: 0.1, RIGHT: 0.1}, 0.5)
        np.testing.assert_array_equal(frame, original)

    def test_a_missing_face_still_returns_a_frame(self) -> None:
        """No face is a normal frame in a live view, not an error."""
        out = draw_overlay(_frame(), None, {LEFT: float("nan")}, 0.5)
        self.assertEqual(out.shape, _frame().shape)

    def test_a_closed_eye_is_drawn_differently_from_an_open_one(self) -> None:
        """The box colour is what a viewer reads at a glance."""
        open_eyes = draw_overlay(_frame(), self._detection(), {LEFT: 0.1, RIGHT: 0.1}, 0.5)
        shut_eyes = draw_overlay(_frame(), self._detection(), {LEFT: 0.9, RIGHT: 0.9}, 0.5)
        self.assertFalse(np.array_equal(open_eyes, shut_eyes))

    def test_an_unknown_score_is_not_treated_as_closed(self) -> None:
        """``nan`` means the eye was never scored, which is not a blink."""
        unknown = draw_overlay(_frame(), self._detection(), {LEFT: float("nan")}, 0.5)
        shut = draw_overlay(_frame(), self._detection(), {LEFT: 0.9, RIGHT: 0.9}, 0.5)
        self.assertFalse(np.array_equal(unknown, shut))

    def test_a_missing_eye_box_is_skipped(self) -> None:
        """One eye off-frame must not stop the other being drawn."""
        detection = _Detection({LEFT: None, RIGHT: _Box(150, 80, 40)})
        out = draw_overlay(_frame(), detection, {RIGHT: 0.1}, 0.5)
        self.assertEqual(out.shape, _frame().shape)

    def test_readout_lines_are_drawn(self) -> None:
        """The panel carries fps and head pose."""
        without = draw_overlay(_frame(), self._detection(), {LEFT: 0.1}, 0.5)
        with_lines = draw_overlay(_frame(), self._detection(), {LEFT: 0.1}, 0.5, ["30 ms/frame"])
        self.assertFalse(np.array_equal(without, with_lines))


class TestEyeCrops(unittest.TestCase):
    """Cutting the two crops the model scores."""

    def test_both_eyes_give_a_batch_of_two(self) -> None:
        """Shape and range are the model's input contract."""
        detection = _Detection({LEFT: _Box(70, 80, 40), RIGHT: _Box(150, 80, 40)})
        crops = eye_crops(detection, _frame(), image_size=64)
        self.assertEqual(crops.shape, (2, 3, 64, 64))
        self.assertTrue(float(crops.max()) <= 1.0)

    def test_a_missing_eye_refuses_the_whole_batch(self) -> None:
        """Scores are read positionally, so a one-eye batch would swap sides.

        Returning a single crop would put the right eye's score under the left
        eye's label -- wrong in a way the picture would not reveal.
        """
        detection = _Detection({LEFT: None, RIGHT: _Box(150, 80, 40)})
        self.assertIsNone(eye_crops(detection, _frame()))

    def test_an_off_frame_box_refuses_the_batch(self) -> None:
        """A box outside the frame crops to nothing rather than to noise."""
        detection = _Detection({LEFT: _Box(5, 5, 40), RIGHT: _Box(150, 80, 40)})
        self.assertIsNone(eye_crops(detection, _frame()))


if __name__ == "__main__":
    unittest.main()
