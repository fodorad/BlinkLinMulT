"""Tests for the streaming summary arithmetic.

The pipeline itself needs YOLO, FaceMesh and a video, so it is exercised by
running `make benchmark-streaming` rather than in the suite. What is tested here
is the reduction from per-frame timings to a verdict -- the deadline comparison,
the drop count and the realtime factor -- because those are what a reader acts
on and an off-by-one in them would misreport whether a system works.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "benchmark_streaming.py"
"""The driver under test. Loaded by path: `tools/` is not an importable package."""


def _load():
    """Import the driver module.

    Returns:
        module: The loaded module.
    """
    spec = importlib.util.spec_from_file_location("benchmark_streaming", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["benchmark_streaming"] = module
    spec.loader.exec_module(module)
    return module


class TestStageTimes(unittest.TestCase):
    """Reducing per-frame costs to a verdict."""

    def setUp(self) -> None:
        """Load the driver once per test."""
        self.module = _load()

    def _times(self, totals: list[float]):
        """Build a StageTimes with given frame totals.

        Args:
            totals (list[float]): Per-frame wall clock, milliseconds.

        Returns:
            StageTimes: Populated with plausible stage splits.
        """
        times = self.module.StageTimes()
        for total in totals:
            times.detect_ms.append(total * 0.8)
            times.crop_ms.append(total * 0.01)
            times.score_ms.append(total * 0.19)
            times.total_ms.append(total)
        return times

    def test_the_budget_comes_from_the_frame_rate(self) -> None:
        """30 fps allows 33.3 ms per frame."""
        summary = self._times([10.0] * 5).summary(fps=30.0)
        self.assertAlmostEqual(summary["budget_ms"], 1000.0 / 30.0, places=6)

    def test_frames_over_budget_are_dropped(self) -> None:
        """Three of five frames exceed a 33.3 ms deadline."""
        summary = self._times([10.0, 10.0, 50.0, 60.0, 70.0]).summary(fps=30.0)
        self.assertEqual(summary["frames_dropped"], 3)
        self.assertAlmostEqual(summary["drop_rate"], 0.6, places=6)

    def test_keeping_up_means_nothing_dropped(self) -> None:
        """A pipeline inside its budget on every frame keeps up."""
        summary = self._times([10.0] * 10).summary(fps=30.0)
        self.assertTrue(summary["keeps_up"])
        self.assertEqual(summary["frames_dropped"], 0)

    def test_a_slow_pipeline_does_not_keep_up(self) -> None:
        """The measured case: 62 ms per frame against a 33 ms budget."""
        summary = self._times([62.0] * 10).summary(fps=30.0)
        self.assertFalse(summary["keeps_up"])
        self.assertEqual(summary["frames_dropped"], 10)

    def test_realtime_factor_is_budget_over_cost(self) -> None:
        """Twice the budget is 0.5x realtime."""
        summary = self._times([66.6666] * 5).summary(fps=30.0)
        self.assertAlmostEqual(summary["realtime_factor"], 0.5, places=3)

    def test_achieved_fps_inverts_the_median(self) -> None:
        """50 ms per frame is 20 fps."""
        self.assertAlmostEqual(self._times([50.0] * 5).summary(fps=30.0)["achieved_fps"], 20.0)

    def test_the_tail_is_reported(self) -> None:
        """A p99 spike drops frames even when the median looks fine."""
        summary = self._times([10.0] * 99 + [500.0]).summary(fps=30.0)
        self.assertGreater(summary["total_ms_p99"], summary["total_ms_p50"])
        self.assertEqual(summary["frames_dropped"], 1)

    def test_frame_count_is_reported(self) -> None:
        """A drop count means nothing without the total."""
        self.assertEqual(self._times([10.0] * 7).summary(fps=30.0)["frames"], 7)


class TestConfiguration(unittest.TestCase):
    """What the driver compares, and what it leaves out."""

    def test_it_compares_both_model_families(self) -> None:
        """Frame-wise and windowed models, under identical capture conditions.

        The two frame-wise models isolate the network; ``blinklint-union`` is
        there because a 15-frame window is a different streaming problem, and
        the comparison is only meaningful if both run through the same detector
        and the same crops.
        """
        module = _load()
        self.assertIn("blinkcnn-onnx", module.STREAM_MODELS)
        self.assertIn("densenet121-union", module.STREAM_MODELS)
        self.assertIn("blinklint-union", module.STREAM_MODELS)

    def test_the_windowed_model_is_recognised_as_windowed(self) -> None:
        """Routing is driven by the registry, not by a hardcoded name list.

        A future sequence model therefore gets the background worker without
        anyone remembering to add it here.
        """
        from blinklinmult.registry import spec

        self.assertIsNotNone(spec("blinklint-union").window)
        self.assertIsNone(spec("blinkcnn-onnx").window)

    def test_head_pose_is_in_the_pipeline(self) -> None:
        """Head pose stays, because yaw drives self-occlusion gating.

        It was excluded when 6DRepNet was the only option at 26.7 ms/frame. The
        geometric estimator costs ~0.05 ms, so the stage is affordable and the
        pipeline keeps the signal that decides which eye is visible.
        """
        module = _load()
        self.assertIn("geometric", module.DEFAULT_HEAD_POSE)
        self.assertIsNotNone(module._build_head_pose(module.DEFAULT_HEAD_POSE))


if __name__ == "__main__":
    unittest.main()


class TestFastDefaults(unittest.TestCase):
    """The defaulted configuration must be the real-time one.

    The requirement is that ``make benchmark-streaming`` with no arguments
    keeps up with a camera, so these pin the defaults rather than trusting the
    caller to pass the right flags.
    """

    def setUp(self) -> None:
        """Load the driver."""
        self.module = _load()

    def test_the_default_locator_skips_the_landmark_model(self) -> None:
        """FaceMesh costs ~5 ms and the detector already gives eye centres."""
        self.assertEqual(self.module.DEFAULT_LOCATOR, "pose")

    def test_the_default_image_size_is_not_the_library_default(self) -> None:
        """ultralytics' 640 upscales a webcam frame: 40.3 ms against 8.3 ms."""
        self.assertEqual(self.module.DEFAULT_IMAGE_SIZE, 256)
        self.assertLess(self.module.DEFAULT_IMAGE_SIZE, 640)

    def test_the_default_head_pose_is_the_cheap_one(self) -> None:
        """6DRepNet costs 26.7 ms on CPU, which alone breaks a 30 fps budget."""
        self.assertEqual(self.module.DEFAULT_HEAD_POSE, "geometric")

    def test_head_pose_can_be_disabled(self) -> None:
        """A caller that does not gate on occlusion should not pay for pose."""
        self.assertIsNone(self.module._build_head_pose("none"))

    def test_an_unknown_head_pose_method_is_refused(self) -> None:
        """A typo must not silently disable occlusion gating."""
        with self.assertRaises(SystemExit):
            self.module._build_head_pose("nonesuch")

    def test_an_unknown_locator_is_refused(self) -> None:
        """Same reasoning: fail loudly rather than pick something."""
        with self.assertRaises(SystemExit):
            self.module._build_locator("nonesuch", 256)
