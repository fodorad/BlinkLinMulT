"""Tests for the frame overlay and the two-panel plot.

The overlay is how a viewer reads what the pipeline decided, so its failures are
failures of *communication* rather than of computation: a suppressed eye showing
a stale score, or a gap drawn as a continuous line, both look like working output
while asserting something untrue.

Rendering is checked by drawing onto synthetic frames and reading the result
back, rather than by eye. Colour is asserted where it carries meaning -- green
for a scored eye, red for a suppressed one -- because that mapping is the
overlay's whole legend.
"""

from __future__ import annotations

import unittest

import matplotlib
import numpy as np

matplotlib.use("Agg")  # noqa: E402 - must precede pyplot, and no display exists in CI

import matplotlib.pyplot as plt  # noqa: E402

from blinklinmult import overlay  # noqa: E402
from blinklinmult.data.schema import LEFT, RIGHT  # noqa: E402
from blinklinmult.pipeline import Extraction, FrameResult, Result  # noqa: E402
from blinklinmult.preprocess.geometry import EyeBox  # noqa: E402

FRAME_SIZE = 480
"""Side of the synthetic frames drawn on.

Realistic rather than minimal: the readout panel occupies a fixed pixel height,
so on a tiny frame it covers the whole image and every colour assertion fails
for the wrong reason.
"""


def _box(centre_x: int) -> EyeBox:
    """Build an eye box at a given horizontal position.

    Args:
        centre_x (int): Box centre, x.

    Returns:
        EyeBox: A small square box.
    """
    return EyeBox(centre_x=centre_x, centre_y=260, side=60, span=10.0)


def _result(
    *,
    used_right: bool = True,
    score_right: float | None = 0.9,
    pose: np.ndarray | None = None,
    frames: int = 1,
) -> Result:
    """Build a one-frame analysis to render.

    Args:
        used_right (bool): Whether the right eye was scored.
        score_right (float | None): Its score, or ``None`` when suppressed.
        pose (np.ndarray | None): Head angles, or ``None`` when unavailable.
        frames (int): How many frames the result covers.

    Returns:
        Result: A minimal but complete analysis.
    """
    angles = np.array([-12.0, 3.0, -1.0]) if pose is None else pose
    per_frame = [
        FrameResult(
            index=index,
            face_box=(120, 140, 360, 400),
            landmarks=np.array([[180.0, 260.0], [300.0, 260.0]]),
            pose=angles,
            eyes={LEFT: _box(180), RIGHT: _box(300)},
            used={LEFT: True, RIGHT: used_right},
            score={LEFT: 0.04, RIGHT: score_right},
        )
        for index in range(frames)
    ]
    return Result(
        frames=per_frame,
        signal={
            LEFT: np.full(frames, 0.04, dtype=np.float32),
            RIGHT: np.full(frames, score_right if score_right is not None else np.nan, np.float32),
        },
        events={LEFT: [], RIGHT: []},
        fps=30.0,
        model_id="blinkcnn",
        extraction=Extraction(high=0.53, low=0.1325),
    )


class TestReadout(unittest.TestCase):
    """The text drawn in the corner."""

    def test_frame_id_comes_first_after_nothing(self) -> None:
        """The frame number leads, so a viewer can locate the moment."""
        lines = overlay._readout(_result(), 0)
        self.assertTrue(lines[0].startswith("Frame: 0"))

    def test_angles_are_signed_integers(self) -> None:
        """A tenth of a degree is noise; the sign is what matters."""
        lines = overlay._readout(_result(), 0)
        self.assertIn("yaw: -12", lines)
        self.assertIn("pitch: 3", lines)
        self.assertIn("roll: -1", lines)

    def test_scores_carry_two_decimals(self) -> None:
        """Enough precision to see a trend, not so much it is unreadable."""
        lines = overlay._readout(_result(), 0)
        self.assertIn("left state: 0.04", lines)
        self.assertIn("right state: 0.90", lines)

    def test_a_suppressed_eye_reads_as_unknown(self) -> None:
        """``--``, never a stale number.

        Showing the last score would assert a measurement that was not taken --
        the exact confusion the red box exists to prevent.
        """
        lines = overlay._readout(_result(used_right=False, score_right=None), 0)
        self.assertIn("right state: --", lines)
        self.assertNotIn("right state: 0.90", lines)

    def test_a_missing_pose_reads_as_unknown(self) -> None:
        """No face means no angles, and zeros would look like a frontal head."""
        result = _result()
        result.frames[0].pose = None
        lines = overlay._readout(result, 0)
        self.assertIn("yaw: --", lines)


class TestDraw(unittest.TestCase):
    """Annotating one frame."""

    def setUp(self) -> None:
        """A mid-grey frame, so any drawn colour stands out."""
        self.frame = np.full((FRAME_SIZE, FRAME_SIZE, 3), 128, dtype=np.uint8)

    def test_returns_a_new_frame(self) -> None:
        """The source is not modified, so callers can reuse it."""
        before = self.frame.copy()
        overlay.draw(self.frame, _result(), 0)
        np.testing.assert_array_equal(self.frame, before)

    def test_shape_and_dtype_survive(self) -> None:
        """The annotated frame is still a writable RGB image."""
        drawn = overlay.draw(self.frame, _result(), 0)
        self.assertEqual(drawn.shape, self.frame.shape)
        self.assertEqual(drawn.dtype, np.uint8)

    def test_something_is_actually_drawn(self) -> None:
        """A frame that comes back unchanged would pass every other test here."""
        drawn = overlay.draw(self.frame, _result(), 0)
        self.assertFalse(np.array_equal(drawn, self.frame))

    def test_a_scored_eye_is_green(self) -> None:
        """Green marks an eye whose score is real."""
        drawn = overlay.draw(self.frame, _result(), 0)
        self.assertTrue(
            np.any(np.all(drawn == np.array(overlay.USED_COLOUR, np.uint8), axis=-1)),
            "no green pixels: a used eye box was not drawn",
        )

    def test_a_suppressed_eye_is_red(self) -> None:
        """Red marks an eye that was found but not scored."""
        drawn = overlay.draw(self.frame, _result(used_right=False, score_right=None), 0)
        self.assertTrue(
            np.any(np.all(drawn == np.array(overlay.DROPPED_COLOUR, np.uint8), axis=-1)),
            "no red pixels: a suppressed eye box was not drawn",
        )

    def test_landmarks_are_magenta(self) -> None:
        """The dots are distinguishable from every box colour."""
        drawn = overlay.draw(self.frame, _result(), 0)
        self.assertTrue(
            np.any(np.all(drawn == np.array(overlay.LANDMARK_COLOUR, np.uint8), axis=-1))
        )

    def test_a_frame_without_a_face_still_renders(self) -> None:
        """No boxes, no landmarks, but the readout still says so."""
        result = _result()
        result.frames[0].face_box = None
        result.frames[0].landmarks = None
        result.frames[0].eyes = {}
        drawn = overlay.draw(self.frame, result, 0)
        self.assertEqual(drawn.shape, self.frame.shape)


class TestPlot(unittest.TestCase):
    """The two-panel figure."""

    def test_one_panel_per_eye(self) -> None:
        """Left above right, sharing the frame axis."""
        figure = overlay.plot(_result(frames=40))
        self.addCleanup(plt.close, figure)
        self.assertEqual(len(figure.axes), 2)

    def test_panels_name_their_eye(self) -> None:
        """So a reader can tell them apart without counting rows."""
        figure = overlay.plot(_result(frames=40))
        self.addCleanup(plt.close, figure)
        titles = [axis.get_title(loc="left") for axis in figure.axes]
        self.assertTrue(any("Left eye" in title for title in titles))
        self.assertTrue(any("Right eye" in title for title in titles))

    def test_draws_the_applied_thresholds(self) -> None:
        """The rule that produced the events, not the registry default.

        Drawing the registered line under differently-extracted events would
        misrepresent what happened.
        """
        result = _result(frames=40)
        result.extraction = Extraction(high=0.31, low=None)
        figure = overlay.plot(result)
        self.addCleanup(plt.close, figure)
        drawn = [
            round(float(line.get_ydata()[0]), 3)
            for axis in figure.axes
            for line in axis.get_lines()
            if line.get_label().startswith(("high", "low"))
        ]
        self.assertIn(0.31, drawn)
        self.assertNotIn(0.53, drawn)

    def test_ground_truth_is_drawn_when_present(self) -> None:
        """The annotation appears as its own trace, per eye."""
        result = _result(frames=40)
        result.truth = {
            LEFT: np.zeros(40, dtype=np.float32),
            RIGHT: np.zeros(40, dtype=np.float32),
        }
        figure = overlay.plot(result)
        self.addCleanup(plt.close, figure)
        traces = [
            line
            for axis in figure.axes
            for line in axis.get_lines()
            if line.get_label() == "ground truth"
        ]
        self.assertEqual(len(traces), 2)

    def test_no_ground_truth_trace_without_an_annotation(self) -> None:
        """An uploaded video has none, and inventing one would be a lie."""
        figure = overlay.plot(_result(frames=40))
        self.addCleanup(plt.close, figure)
        traces = [
            line
            for axis in figure.axes
            for line in axis.get_lines()
            if line.get_label() == "ground truth"
        ]
        self.assertEqual(traces, [])

    def test_suppressed_frames_are_reported_in_the_title(self) -> None:
        """A gap is explained where the blink count is read, not only in a legend."""
        result = _result(frames=40, used_right=False, score_right=None)
        figure = overlay.plot(result)
        self.addCleanup(plt.close, figure)
        titles = " ".join(axis.get_title(loc="left") for axis in figure.axes)
        self.assertIn("suppressed", titles)


if __name__ == "__main__":
    unittest.main()
