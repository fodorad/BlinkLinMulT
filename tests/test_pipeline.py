"""Tests for the seven-stage video pipeline.

The stages themselves need real models and real video, so what is tested here is
everything *around* them -- the decisions that are cheap to get wrong and
expensive to notice:

* which eye a head turn hides, and on which side of the boundary;
* how a segment request becomes a frame range, including its edges;
* whether an extraction rule can actually separate anything;
* whether a mid-video segment lines its annotation up correctly.

Each of these has a failure mode that produces plausible output rather than an
error. A yaw sign flipped the wrong way suppresses the *visible* eye and keeps
the occluded one; an annotation offset by a segment start shifts every
comparison silently.
"""

from __future__ import annotations

import importlib.util
import os
import unittest
from pathlib import Path

import numpy as np

from blinklinmult.data.schema import LEFT, RIGHT
from blinklinmult.pipeline import (
    MAX_SECONDS,
    YAW_LIMIT,
    Extraction,
    FrameResult,
    PipelineError,
    Result,
    Stage,
    ground_truth,
    occluded_side,
    run,
)
from blinklinmult.registry import spec

EXAMPLE_VIDEO = Path("data/raw/TalkingFace/talking.avi")
"""A real clip, present only in a full checkout."""

EXAMPLE_TAG = Path("data/raw/TalkingFace/talking.tag")
"""Its annotation."""

HAVE_EXAMPLE = EXAMPLE_VIDEO.is_file() and EXAMPLE_TAG.is_file()
"""Whether the corpus clip is available to read."""

HAVE_EXORDIUM = importlib.util.find_spec("exordium") is not None
"""Whether the optional extraction stack is installed.

``_read_video`` decodes through ``exordium.video.core.io``, which ships in the
``preprocess`` extra along with multi-GB model weights. CI installs a narrower
set, so these tests are skipped there rather than pulling the weights into every
run; a full local checkout has it and runs them.
"""


class TestOccludedSide(unittest.TestCase):
    """Which eye a head turn hides.

    The sign convention is derived, not guessed: exordium draws the yaw axis as
    ``x = size * sin(-radians(yaw))``, so positive yaw points the nose toward
    image-left and hides the viewer's *right* eye. Reversing it would suppress
    the visible eye and keep the occluded one -- a demo that looks correct and is
    exactly backwards, which is why both directions are asserted.
    """

    def test_positive_yaw_hides_the_right_eye(self) -> None:
        """Nose toward image-left, so the right eye turns away."""
        self.assertEqual(occluded_side(YAW_LIMIT + 1.0), RIGHT)
        self.assertEqual(occluded_side(75.0), RIGHT)

    def test_negative_yaw_hides_the_left_eye(self) -> None:
        """The mirror case."""
        self.assertEqual(occluded_side(-YAW_LIMIT - 1.0), LEFT)
        self.assertEqual(occluded_side(-75.0), LEFT)

    def test_frontal_hides_neither(self) -> None:
        """Inside the limit both eyes are visible."""
        for yaw in (-YAW_LIMIT, -10.0, 0.0, 10.0, YAW_LIMIT):
            with self.subTest(yaw=yaw):
                self.assertIsNone(occluded_side(yaw))

    def test_the_boundary_is_exclusive(self) -> None:
        """Exactly at the limit is still frontal; one degree past is not."""
        self.assertIsNone(occluded_side(YAW_LIMIT))
        self.assertIsNotNone(occluded_side(YAW_LIMIT + 0.001))

    def test_limit_is_configurable(self) -> None:
        """A caller can tighten or loosen the gate."""
        self.assertEqual(occluded_side(30.0, limit=20.0), RIGHT)
        self.assertIsNone(occluded_side(30.0, limit=40.0))


class TestExtraction(unittest.TestCase):
    """The rule turning an eye-state curve into blink events."""

    def test_fitted_reads_the_registry(self) -> None:
        """``blinkcnn`` ships a hysteresis pair, so both thresholds appear."""
        rule = Extraction.fitted(spec("blinkcnn"))
        self.assertAlmostEqual(rule.high, 0.53)
        self.assertAlmostEqual(rule.low or 0.0, 0.53 * 0.25)

    def test_fitted_handles_a_single_cut(self) -> None:
        """The 1.x models carry no low threshold, so ``low`` stays ``None``."""
        self.assertIsNone(Extraction.fitted(spec("blinklint-union")).low)

    def test_describe_names_the_rule(self) -> None:
        """The log line distinguishes the two shapes."""
        self.assertIn("single threshold", Extraction(high=0.4).describe())
        self.assertIn("hysteresis", Extraction(high=0.6, low=0.2).describe())

    def test_valid_rules_pass(self) -> None:
        """A sane single cut and a sane pair both validate."""
        Extraction(high=0.5).validate()
        Extraction(high=0.6, low=0.2).validate()

    def test_high_threshold_must_be_a_probability(self) -> None:
        """Outside ``(0, 1)`` nothing can be separated."""
        for high in (-0.1, 0.0, 1.0, 1.7):
            with self.subTest(high=high), self.assertRaises(PipelineError):
                Extraction(high=high).validate()

    def test_low_must_sit_below_high(self) -> None:
        """At or above the high cut, hysteresis degrades to a single threshold.

        Silently, which is the problem: the run would still produce events, just
        not the ones the caller asked for.
        """
        for low in (0.5, 0.9):
            with self.subTest(low=low), self.assertRaises(PipelineError):
                Extraction(high=0.5, low=low).validate()

    def test_low_threshold_must_be_a_probability(self) -> None:
        """The low cut is a probability too."""
        with self.assertRaises(PipelineError):
            Extraction(high=0.5, low=-0.2).validate()


class TestStage(unittest.TestCase):
    """The progress record each stage yields."""

    def test_line_carries_number_name_and_timing(self) -> None:
        """A reader can tell which stage ran, what it found, and how long it took."""
        line = Stage(3, 7, "Landmarks", "299 frames", 12.4).line()
        self.assertIn("[3/7]", line)
        self.assertIn("Landmarks", line)
        self.assertIn("299 frames", line)
        self.assertIn("12.4 s", line)


class TestFrameResult(unittest.TestCase):
    """Per-frame findings."""

    def test_defaults_are_empty_not_absent(self) -> None:
        """A frame with no face still has usable dicts, so callers need no guards."""
        result = FrameResult(index=7)
        self.assertEqual(result.index, 7)
        self.assertIsNone(result.face_box)
        self.assertEqual(result.eyes, {})
        self.assertEqual(result.score, {})


class TestResult(unittest.TestCase):
    """The finished analysis."""

    def test_carries_its_extraction_rule(self) -> None:
        """The plot draws the rule that produced the events, not the default."""
        rule = Extraction(high=0.3)
        result = Result(
            frames=[],
            signal={LEFT: np.array([]), RIGHT: np.array([])},
            events={LEFT: [], RIGHT: []},
            fps=30.0,
            model_id="blinkcnn",
            extraction=rule,
        )
        self.assertIs(result.extraction, rule)

    def test_truth_is_optional(self) -> None:
        """An uploaded video has no annotation, and that is not an error."""
        result = Result(
            frames=[],
            signal={},
            events={},
            fps=25.0,
            model_id="blinkcnn",
        )
        self.assertIsNone(result.truth)


@unittest.skipUnless(HAVE_EXAMPLE, "needs data/raw/TalkingFace")
class TestGroundTruth(unittest.TestCase):
    """Reading a ``.tag`` annotation for the analysed segment."""

    def test_returns_one_array_per_eye(self) -> None:
        """Both eyes are annotated separately."""
        truth = ground_truth(EXAMPLE_TAG, 299)
        self.assertIsNotNone(truth)
        assert truth is not None
        self.assertEqual(truth[LEFT].shape, (299,))
        self.assertEqual(truth[RIGHT].shape, (299,))

    def test_marks_the_annotated_blinks(self) -> None:
        """TalkingFace's first 299 frames carry three annotated blinks."""
        truth = ground_truth(EXAMPLE_TAG, 299)
        assert truth is not None
        closed = np.flatnonzero(truth[LEFT] > 0.5).tolist()
        self.assertEqual(closed, [170, 171, 227, 275, 276])

    def test_offset_shifts_the_window(self) -> None:
        """A mid-video segment reads its own slice of the annotation.

        Without this, a segment starting at frame 150 would be compared against
        the annotation for frame 0 -- every comparison wrong, and nothing to say
        so.
        """
        truth = ground_truth(EXAMPLE_TAG, 120, offset=150)
        assert truth is not None
        closed = [index + 150 for index in np.flatnonzero(truth[LEFT] > 0.5)]
        self.assertEqual(closed, [170, 171, 227])

    def test_missing_file_returns_none(self) -> None:
        """An uploaded video has no annotation; that is not an error."""
        self.assertIsNone(ground_truth(Path("no-such-file.tag"), 100))


@unittest.skipUnless(
    HAVE_EXAMPLE and HAVE_EXORDIUM, "needs data/raw/TalkingFace and the preprocess extra"
)
class TestReadVideo(unittest.TestCase):
    """Turning a seconds request into a frame range."""

    def test_default_reads_from_the_start(self) -> None:
        """10 s of 30 fps footage is 300 frames, starting at 0."""
        from blinklinmult.pipeline import _read_video

        frames, fps, _truncated, offset = _read_video(EXAMPLE_VIDEO, 0.0, 10.0)
        self.assertEqual(frames.shape[0], 300)
        self.assertAlmostEqual(fps, 30.0)
        self.assertEqual(offset, 0)

    def test_segment_uses_the_source_fps(self) -> None:
        """5 s in at 30 fps is frame 150, and 4 s is 120 frames.

        The frame range is derived from the video's own metadata, never assumed:
        the same request against 25 fps footage would be frame 125 and 100
        frames.
        """
        from blinklinmult.pipeline import _read_video

        frames, _fps, _truncated, offset = _read_video(EXAMPLE_VIDEO, 5.0, 4.0)
        self.assertEqual(frames.shape[0], 120)
        self.assertEqual(offset, 150)

    def test_frames_are_rgb_uint8(self) -> None:
        """Downstream crops assume this layout."""
        from blinklinmult.pipeline import _read_video

        frames, _fps, _truncated, _offset = _read_video(EXAMPLE_VIDEO, 0.0, 0.5)
        self.assertEqual(frames.ndim, 4)
        self.assertEqual(frames.shape[3], 3)
        self.assertEqual(frames.dtype, np.uint8)

    def test_running_past_the_end_truncates(self) -> None:
        """A segment overrunning the video returns what exists and says so."""
        from blinklinmult.pipeline import _read_video

        frames, _fps, truncated, _offset = _read_video(EXAMPLE_VIDEO, 166.0, 10.0)
        self.assertTrue(truncated)
        self.assertGreater(frames.shape[0], 0)

    def test_start_past_the_end_is_rejected(self) -> None:
        """With the video's real length in the message, so it can be corrected."""
        from blinklinmult.pipeline import _read_video

        with self.assertRaises(PipelineError) as caught:
            _read_video(EXAMPLE_VIDEO, 200.0, 5.0)
        self.assertIn("166", str(caught.exception))

    def test_negative_start_is_rejected(self) -> None:
        """There is no footage before zero."""
        from blinklinmult.pipeline import _read_video

        with self.assertRaises(PipelineError):
            _read_video(EXAMPLE_VIDEO, -1.0, 5.0)

    def test_non_positive_duration_is_rejected(self) -> None:
        """A zero-length segment holds nothing to analyse."""
        from blinklinmult.pipeline import _read_video

        for duration in (0.0, -3.0):
            with self.subTest(duration=duration), self.assertRaises(PipelineError):
                _read_video(EXAMPLE_VIDEO, 0.0, duration)

    def test_duration_is_capped(self) -> None:
        """Longer than the cap yields the cap, not the request."""
        from blinklinmult.pipeline import _read_video

        frames, fps, _truncated, _offset = _read_video(EXAMPLE_VIDEO, 0.0, MAX_SECONDS * 3)
        self.assertLessEqual(frames.shape[0], int(round(fps * MAX_SECONDS)))

    def test_missing_file_is_rejected(self) -> None:
        """A path that is not a video fails with the filename in the message."""
        from blinklinmult.pipeline import _read_video

        with self.assertRaises(PipelineError):
            _read_video(Path("no-such-video.mp4"), 0.0, 5.0)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(
    HAVE_EXAMPLE and os.environ.get("RUN_PIPELINE_TESTS") == "1",
    "set RUN_PIPELINE_TESTS=1 (needs the corpus clip and the published models)",
)
class TestRunEndToEnd(unittest.TestCase):
    """The seven stages against real video.

    Opt-in: this downloads published weights and runs YOLO11, FaceMesh, 6DRepNet
    and a blink model over ~60 frames, which is far too slow for the default
    suite. It is the only test that exercises `run` itself, so the fast suite
    covers the decisions around it and this covers the assembly.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Analyse two seconds of the example clip, once."""
        from blinklinmult import BlinkDetector

        detector = BlinkDetector.from_pretrained("blinkcnn")
        cls.stages = []
        cls.result = None
        for item in run(
            EXAMPLE_VIDEO,
            detector,
            tag_path=EXAMPLE_TAG,
            start=5.0,
            duration=2.0,
        ):
            if isinstance(item, Stage):
                cls.stages.append(item)
            else:
                cls.result = item

    def test_yields_every_stage_then_a_result(self) -> None:
        """Six stages report, and the seventh is the caller's own rendering."""
        self.assertEqual(len(self.stages), 6)
        self.assertIsNotNone(self.result)

    def test_stages_are_numbered_in_order(self) -> None:
        """A progress display can rely on the ordering."""
        self.assertEqual([stage.index for stage in self.stages], [1, 2, 3, 4, 5, 6])

    def test_frame_indices_follow_the_source_video(self) -> None:
        """5 s at 30 fps is frame 150, not frame 0.

        An overlay labelled from zero would not line up with the footage, and
        the annotation is indexed the same way.
        """
        assert self.result is not None
        self.assertEqual(self.result.frames[0].index, 150)

    def test_both_eyes_are_scored(self) -> None:
        """Each eye gets its own signal of the same length as the segment."""
        assert self.result is not None
        for side in (LEFT, RIGHT):
            with self.subTest(side=side):
                self.assertEqual(self.result.signal[side].shape[0], len(self.result.frames))

    def test_annotation_is_attached_and_aligned(self) -> None:
        """The segment carries the matching slice of the ``.tag``."""
        assert self.result is not None
        self.assertIsNotNone(self.result.truth)
        assert self.result.truth is not None
        self.assertEqual(self.result.truth[LEFT].shape[0], len(self.result.frames))

    def test_records_the_rule_that_produced_the_events(self) -> None:
        """So the plot can draw the thresholds actually applied."""
        assert self.result is not None
        self.assertIsNotNone(self.result.extraction)


@unittest.skipUnless(
    Path("blinklinmult/assets/talkingface_10s.mp4").is_file() and HAVE_EXORDIUM,
    "needs the asset and the preprocess extra",
)
class TestSegmentEdges(unittest.TestCase):
    """Segments at and past the end of a video.

    A request that *overruns* the end is legitimate -- "10 seconds from 2 s" of
    an 8 s clip means "from 2 s to the end" -- and must return what exists rather
    than failing. A start *past* the end is a mistake and must say so.

    The boundary between them is one frame wide and easy to get wrong by
    rounding, which is exactly what happened: ``round(9.99 * 30)`` is 300, one
    past the last index of a 300-frame video, so a timestamp inside the clip was
    rejected.
    """

    CLIP = Path("blinklinmult/assets/talkingface_10s.mp4")
    """The bundled 10-second, 300-frame example."""

    def test_a_segment_overrunning_the_end_is_truncated(self) -> None:
        """From 2 s, asking for 10 s of a 10 s clip yields 2 s to the end."""
        from blinklinmult.pipeline import _read_video

        frames, fps, truncated, offset = _read_video(self.CLIP, 2.0, 10.0)
        self.assertTrue(truncated)
        self.assertEqual(offset, int(fps * 2.0))
        self.assertEqual(frames.shape[0], 300 - offset)

    def test_the_last_frame_is_reachable(self) -> None:
        """9.99 s of a 300-frame 30 fps clip is frame 299, not a rejection.

        Truncation, not rounding: a start timestamp names the frame it falls
        inside.
        """
        from blinklinmult.pipeline import _read_video

        frames, _fps, _truncated, offset = _read_video(self.CLIP, 9.99, 5.0)
        self.assertEqual(offset, 299)
        self.assertEqual(frames.shape[0], 1)

    def test_a_start_exactly_at_the_end_is_rejected(self) -> None:
        """10.0 s of a 10 s clip is past the last frame."""
        from blinklinmult.pipeline import _read_video

        with self.assertRaises(PipelineError):
            _read_video(self.CLIP, 10.0, 5.0)

    def test_a_start_past_the_end_names_the_real_length(self) -> None:
        """So the message can be acted on without opening the file."""
        from blinklinmult.pipeline import _read_video

        with self.assertRaises(PipelineError) as caught:
            _read_video(self.CLIP, 20.0, 5.0)
        self.assertIn("10.0 s long", str(caught.exception))
