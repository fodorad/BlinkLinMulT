"""Tests for streaming a windowed model without stalling the capture loop.

The load-bearing claim is that ``push`` returns immediately while a slow model
is scoring. That is tested with a genuinely slow scoring *function* rather than a
mock: a mock would return instantly and prove nothing about the property the
design exists for.

The rest pin the ways a ring buffer over a background thread goes wrong --
padding a short window, growing without bound, a worker dying on one bad frame,
and a stale score reported as if it were current.
"""

from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from blinklinmult.stream import EMPTY_SCORE, StreamingScorer, StreamScore

WINDOW = 15
"""The 1.x sequence models' native window, matching the registry."""

EYES = 2
"""Both eyes are scored in one call."""


def _crops(size: int = 8) -> np.ndarray:
    """One frame's eye crops.

    Args:
        size (int): Crop side; small, since no model runs here.

    Returns:
        np.ndarray: ``(2, 3, size, size)``.
    """
    return np.zeros((EYES, 3, size, size), dtype=np.float32)


def _instant(frames: np.ndarray) -> np.ndarray:
    """Score a window immediately, returning per-frame values.

    Args:
        frames (np.ndarray): ``(eyes, window, 3, H, W)``.

    Returns:
        np.ndarray: ``(eyes, window)`` scores.
    """
    return np.full(frames.shape[:2], 0.7, dtype=np.float32)


class _Slow:
    """A scoring function that takes a measurable amount of time.

    Real, not mocked: the point is to occupy the worker so the test can observe
    that ``push`` still returns.
    """

    def __init__(self, seconds: float) -> None:
        """Record how long each call should take."""
        self.seconds = seconds
        self.calls = 0
        self.started = threading.Event()

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """Sleep, then score.

        Args:
            frames (np.ndarray): The window.

        Returns:
            np.ndarray: Per-frame scores.
        """
        self.started.set()
        self.calls += 1
        time.sleep(self.seconds)
        return np.full(frames.shape[:2], 0.5, dtype=np.float32)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Poll until a predicate holds.

    Args:
        predicate (Callable[[], bool]): What to wait for.
        timeout (float): Seconds before giving up.

    Returns:
        bool: Whether it became true.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


class TestWarmUp(unittest.TestCase):
    """Behaviour before the first window is complete."""

    def test_nothing_is_scored_before_the_window_fills(self) -> None:
        """A short window must not be padded to length.

        Padding would feed the model an input distribution it never saw, and it
        would score plausibly and wrongly rather than failing.
        """
        with StreamingScorer(_instant, window=WINDOW) as scorer:
            for _ in range(WINDOW - 1):
                scorer.push(_crops())
            time.sleep(0.05)
            self.assertTrue(scorer.latest().warming_up)

    def test_the_empty_score_carries_no_values(self) -> None:
        """A caller must be able to show "waiting" rather than a fake zero."""
        self.assertTrue(EMPTY_SCORE.warming_up)
        self.assertEqual(EMPTY_SCORE.scores.size, 0)

    def test_a_full_window_produces_a_score(self) -> None:
        """Once the buffer fills, the worker publishes."""
        with StreamingScorer(_instant, window=WINDOW) as scorer:
            for _ in range(WINDOW):
                scorer.push(_crops())
            self.assertTrue(_wait_for(lambda: not scorer.latest().warming_up))
            self.assertEqual(scorer.latest().scores.shape, (EYES,))


class TestNonBlocking(unittest.TestCase):
    """The property the whole design exists for."""

    def test_push_returns_while_the_model_is_busy(self) -> None:
        """The capture loop must not wait for a 77 ms inference.

        A slow scorer holds the worker; pushing 30 more frames must still take
        far less time than a single call would.
        """
        slow = _Slow(seconds=0.25)
        with StreamingScorer(slow, window=WINDOW) as scorer:
            for _ in range(WINDOW):
                scorer.push(_crops())
            self.assertTrue(slow.started.wait(timeout=5.0))

            started = time.perf_counter()
            for _ in range(30):
                scorer.push(_crops())
            elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 0.1, "push blocked while the model was scoring")

    def test_latest_returns_while_the_model_is_busy(self) -> None:
        """Reading the score must not wait either."""
        slow = _Slow(seconds=0.25)
        with StreamingScorer(slow, window=WINDOW) as scorer:
            for _ in range(WINDOW):
                scorer.push(_crops())
            self.assertTrue(slow.started.wait(timeout=5.0))
            started = time.perf_counter()
            scorer.latest()
            self.assertLess(time.perf_counter() - started, 0.05)


class TestBuffer(unittest.TestCase):
    """The ring buffer's bounds."""

    def test_memory_does_not_grow_with_the_stream(self) -> None:
        """A long stream must not accumulate frames.

        1000 frames into a 15-frame window leaves 15, so a demo left running
        for an hour uses the same memory as one left running for a second.
        """
        scorer = StreamingScorer(_instant, window=WINDOW)
        for _ in range(1000):
            scorer.push(_crops())
        self.assertEqual(len(scorer._buffer.frames), WINDOW)

    def test_the_frame_index_tracks_the_stream(self) -> None:
        """The published index names the newest frame the window covered."""
        with StreamingScorer(_instant, window=WINDOW) as scorer:
            for _ in range(WINDOW):
                scorer.push(_crops())
            self.assertTrue(_wait_for(lambda: not scorer.latest().warming_up))
            self.assertGreaterEqual(scorer.latest().frame_index, WINDOW - 1)


class TestAge(unittest.TestCase):
    """Reporting the lag rather than hiding it."""

    def test_age_grows_as_newer_frames_arrive(self) -> None:
        """Frames past the last completed window make the score older.

        A live readout that showed half-second-old state as current would be
        misleading, which is why the age travels with the score.
        """
        slow = _Slow(seconds=0.3)
        with StreamingScorer(slow, window=WINDOW) as scorer:
            for _ in range(WINDOW):
                scorer.push(_crops())
            self.assertTrue(_wait_for(lambda: not scorer.latest().warming_up, timeout=6.0))
            first = scorer.latest().age_frames
            for _ in range(10):
                scorer.push(_crops())
            self.assertGreater(scorer.latest().age_frames, first)

    def test_age_in_milliseconds_uses_the_frame_rate(self) -> None:
        """Ten frames at 30 fps is a third of a second."""
        score = StreamScore(scores=np.zeros(2), frame_index=100, age_frames=10)
        self.assertAlmostEqual(score.age_ms(30.0), 1000.0 * 10 / 30, places=6)

    def test_a_zero_frame_rate_does_not_divide_by_zero(self) -> None:
        """An unknown source rate must not crash the readout."""
        self.assertEqual(StreamScore(np.zeros(2), 1, 5).age_ms(0.0), 0.0)


class TestFailures(unittest.TestCase):
    """One bad frame must not end the stream."""

    def test_the_worker_survives_a_raising_scorer(self) -> None:
        """A model that throws is counted, not fatal.

        The last good score keeps standing, matching how the rest of the
        pipeline treats a single unreadable frame.
        """
        state = {"calls": 0}

        def flaky(frames: np.ndarray) -> np.ndarray:
            """Fail once, then succeed."""
            state["calls"] += 1
            if state["calls"] == 1:
                raise ValueError("a bad window")
            return np.full(frames.shape[:2], 0.9, dtype=np.float32)

        with StreamingScorer(flaky, window=WINDOW) as scorer:
            for _ in range(WINDOW):
                scorer.push(_crops())
            self.assertTrue(_wait_for(lambda: not scorer.latest().warming_up, timeout=6.0))
            self.assertGreaterEqual(scorer.latest().failures, 1)
            # `places=4`, not 5: the published value is a running mean over
            # every window that covered this frame, so it carries accumulation
            # error a single score would not.
            self.assertAlmostEqual(float(scorer.latest().scores[0]), 0.9, places=4)


class TestFrameWisePath(unittest.TestCase):
    """A model with no window needs no thread."""

    def test_a_frame_wise_model_scores_inline(self) -> None:
        """Handing a 15 ms call to a worker would only add latency."""
        scorer = StreamingScorer(lambda c: np.full(c.shape[0], 0.4, dtype=np.float32), window=None)
        self.assertFalse(scorer.threaded)
        scorer.push(_crops())
        self.assertFalse(scorer.latest().warming_up)
        self.assertAlmostEqual(float(scorer.latest().scores[0]), 0.4, places=5)

    def test_an_inline_score_is_never_stale(self) -> None:
        """Scored on the calling thread, so it describes the current frame."""
        scorer = StreamingScorer(lambda c: np.zeros(c.shape[0], dtype=np.float32), window=None)
        scorer.push(_crops())
        self.assertEqual(scorer.latest().age_frames, 0)

    def test_starting_a_frame_wise_scorer_is_a_no_op(self) -> None:
        """No worker to start, and calling it must not raise."""
        scorer = StreamingScorer(lambda c: np.zeros(c.shape[0]), window=None)
        scorer.start()
        scorer.stop()
        self.assertFalse(scorer.threaded)


class TestLifecycle(unittest.TestCase):
    """Starting and stopping the worker."""

    def test_stop_joins_the_worker(self) -> None:
        """A demo that exits must not leave a thread scoring."""
        scorer = StreamingScorer(_instant, window=WINDOW)
        scorer.start()
        scorer.stop()
        self.assertIsNone(scorer._worker)

    def test_stopping_twice_is_safe(self) -> None:
        """Shutdown paths overlap; the second call must be a no-op."""
        scorer = StreamingScorer(_instant, window=WINDOW)
        scorer.start()
        scorer.stop()
        scorer.stop()

    def test_starting_twice_does_not_spawn_a_second_worker(self) -> None:
        """Two workers would double the load and interleave their scores."""
        scorer = StreamingScorer(_instant, window=WINDOW)
        scorer.start()
        first = scorer._worker
        scorer.start()
        self.assertIs(scorer._worker, first)
        scorer.stop()

    def test_the_context_manager_stops_on_an_exception(self) -> None:
        """A failure inside the block must still shut the worker down."""
        scorer = StreamingScorer(_instant, window=WINDOW)
        with self.assertRaises(RuntimeError), scorer:
            raise RuntimeError("boom")
        self.assertIsNone(scorer._worker)


if __name__ == "__main__":
    unittest.main()
