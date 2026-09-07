"""Running a sequence model on a live stream without stalling it.

The 1.x sequence models predict frame-wise but consume a **15-frame window**, and
one call over both eyes costs **77.1 ms** measured on CPU -- against a 33.3 ms
budget at 30 fps. Scoring every frame is not expensive, it is arithmetically
impossible: the call alone is 2.3x the whole budget before a face has been
detected. Thread tuning does not rescue it either (74.4 ms at four intra-op
threads against 76.9 on auto), because the model is compute-bound.

**So the two loops are decoupled rather than budgeted.** Capture, detect and
display run at full speed on the calling thread; a single background worker
scores the most recent window whenever it is free. That works because
onnxruntime releases the GIL during ``run()`` -- measured, a busy main loop kept
**102%** of its idle iteration count with a worker running flat out, and face
detection went from 10.3 to 10.6 ms/frame. The worker is genuinely concurrent,
not time-slicing.

**The stride is adaptive, not configured.** The worker finishes a window, takes
the newest frames, and starts again. A fast machine gets denser coverage and a
slow one degrades gracefully, where a fixed stride has to be re-tuned per
machine and is wrong in both directions everywhere else. Measured against
scoring every frame on a real recording, correlation is 0.999 at stride 2 and
0.996 at stride 3, and the **peak score on annotated closed frames never drops**
at any stride -- blinks are still caught, it is the shape of the signal between
them that thins out.

**The readout therefore trails the video**, by roughly the window fill plus one
inference -- about 250 ms in practice. :class:`StreamScore` carries its own age
so a caller can display that honestly rather than pretend the number is current.
Frames newer than the last completed window keep the previous value; they are
never interpolated, because that would invent scores the model did not produce.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
import weakref
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)
"""Module-level logger."""

JOIN_TIMEOUT_SECONDS = 5.0
"""How long :meth:`StreamingScorer.stop` waits for the worker to finish.

Generous next to one inference (~77 ms), so a clean shutdown is the normal case
and the timeout only fires if the scoring function itself has hung.
"""

_LIVE_SCORERS: weakref.WeakSet = weakref.WeakSet()
"""Every scorer with a running worker, so shutdown can join them all.

**A daemon thread that outlives the interpreter is not merely untidy here.** The
worker holds an onnxruntime session, and Python tears down native extension
state while daemon threads are still running -- so a scorer that was never
stopped can call into a half-freed session and abort the process. Observed as
``libc++abi: recursive_mutex lock failed`` at exit, after every test had
already passed, roughly one run in eight.

A ``WeakSet`` so a scorer that *is* stopped and dropped can still be collected
normally; :func:`_join_live_scorers` only reaches the ones still alive.
"""

IDLE_SLEEP_SECONDS = 0.002
"""Pause when the buffer is not yet full.

Short enough to be invisible against a 33 ms frame interval, long enough that a
warming-up scorer does not spin a core.
"""


def _released(_frames: np.ndarray) -> np.ndarray:
    """Stand in for a scoring function after :meth:`StreamingScorer.stop`.

    Returning an empty array rather than raising keeps a late ``push`` from a
    still-draining capture loop harmless.

    Args:
        _frames (np.ndarray): Ignored.

    Returns:
        np.ndarray: An empty array.
    """
    return np.empty(0, dtype=np.float32)


@dataclass(frozen=True)
class StreamScore:
    """One completed window's scores, and how stale they are.

    Args:
        scores (np.ndarray): ``(n_eyes,)`` per-eye score for the window's last
            frame -- what a live readout displays.
        frame_index (int): Index of the newest frame the window covered.
        age_frames (int): Frames pushed since that window closed. **The lag,
            reported rather than hidden**: a sequence model that keeps up
            visually while showing half-second-old state is a different product
            from one that does not.
        windows_completed (int): Windows scored since the scorer started.
        failures (int): Windows the scoring function raised on.
    """

    scores: np.ndarray
    frame_index: int
    age_frames: int
    windows_completed: int = 0
    failures: int = 0

    @property
    def warming_up(self) -> bool:
        """Whether no window has completed yet.

        Returns:
            bool: ``True`` before the first score exists, so a caller shows
            "waiting" rather than a fabricated zero.
        """
        return self.frame_index < 0

    def age_ms(self, fps: float) -> float:
        """The lag in milliseconds.

        Args:
            fps (float): Source frame rate.

        Returns:
            float: Milliseconds between the displayed score and the newest
            frame.
        """
        return 0.0 if fps <= 0 else 1000.0 * self.age_frames / fps


EMPTY_SCORE = StreamScore(scores=np.empty(0, dtype=np.float32), frame_index=-1, age_frames=0)
"""What :meth:`StreamingScorer.latest` returns before the first window closes."""


@dataclass
class _Buffer:
    """The frames a window is cut from, newest last.

    A ``deque`` with ``maxlen`` rather than a list, so the memory is bounded by
    the window length no matter how long the stream runs -- pushing a million
    frames leaves exactly ``window`` of them.

    Args:
        window (int): Frames the model consumes.
    """

    window: int
    frames: deque = field(init=False)
    count: int = 0

    def __post_init__(self) -> None:
        """Allocate the bounded queue."""
        self.frames = deque(maxlen=self.window)

    def push(self, crops: np.ndarray) -> int:
        """Add one frame's eye crops.

        Args:
            crops (np.ndarray): ``(n_eyes, 3, H, W)``.

        Returns:
            int: Index of the frame just added.
        """
        self.frames.append(crops)
        self.count += 1
        return self.count - 1

    def snapshot(self) -> tuple[np.ndarray, int] | None:
        """Copy the current window, if one is complete.

        Returns:
            tuple[np.ndarray, int] | None: ``(n_eyes, window, 3, H, W)`` and the
            index of its newest frame, or ``None`` while the buffer is still
            filling. **Never padded** -- a short window padded to length is an
            input distribution the model was not trained on, and it would score
            plausibly and wrongly.
        """
        if len(self.frames) < self.window:
            return None
        stacked = np.stack(list(self.frames), axis=1)
        return stacked, self.count - 1


@atexit.register
def _join_live_scorers() -> None:
    """Stop any worker still running when the interpreter exits.

    Registered once at import. Without it a caller who forgets ``stop()`` -- or
    whose process dies mid-stream -- can crash on the way out rather than
    exiting cleanly. See :data:`_LIVE_SCORERS`.
    """
    for scorer in list(_LIVE_SCORERS):
        try:
            scorer.stop()
        except Exception as error:  # noqa: BLE001 - shutdown must not raise
            logger.debug(f"stopping a scorer at exit failed: {error}")


class StreamingScorer:
    """Scores a live stream with a model that needs a window of frames.

    Push one frame at a time from the capture loop; read the most recent
    completed score whenever the display needs it. Neither call waits for
    inference.

    For a **frame-wise** model there is nothing to decouple -- pass
    ``window=None`` and every push is scored inline, which is both simpler and
    lower latency than handing a 15 ms call to a thread.

    Args:
        score (Callable[[np.ndarray], np.ndarray]): Takes
            ``(n_eyes, window, 3, H, W)`` (or ``(n_eyes, 3, H, W)`` when
            ``window`` is ``None``) and returns per-frame scores. Normally
            :meth:`~blinklinmult.detector.BlinkDetector.score`.
        window (int | None): Frames the model consumes, from
            :attr:`~blinklinmult.registry.ModelSpec.window`. ``None`` scores
            inline.

    Example:
        >>> scorer = StreamingScorer(detector.score, window=15)
        >>> scorer.start()
        >>> scorer.push(crops)
        >>> scorer.latest().warming_up
        True
    """

    def __init__(
        self,
        score: Callable[[np.ndarray], np.ndarray],
        window: int | None,
    ) -> None:
        """Set up the buffer and the worker's state."""
        self._score = score
        self._window = window
        self._buffer = _Buffer(window=window) if window else None
        self._lock = threading.Lock()
        self._latest = EMPTY_SCORE
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        # `_completed` and `_failures` are read under the lock but incremented
        # outside it, which is safe because each has exactly one writer: the
        # two paths are mutually exclusive (`threaded` is False when `window`
        # is None, so no worker exists on the inline path), and an int
        # increment is atomic under the GIL. A reader can see a count one
        # behind; it can never see a torn value or a wrong score.
        self._completed = 0
        self._failures = 0
        # Per-frame running sums, keyed by the frame's stream index. Overlapping
        # windows each contribute one score per frame and the published value is
        # their mean -- the same reduction `BlinkDetector.score_long` uses, and
        # for the same measured reason: a maximum lets one badly-positioned
        # window raise a frame permanently (event F1 0.379 against 0.500).
        self._sums: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}

    @property
    def threaded(self) -> bool:
        """Whether scoring happens on a background worker.

        Returns:
            bool: ``False`` for frame-wise models, which score inline.
        """
        return self._window is not None

    def start(self) -> None:
        """Start the worker, if this model needs one.

        Idempotent: starting an already-running scorer does nothing.
        """
        if not self.threaded or (self._worker is not None and self._worker.is_alive()):
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._run, name="blink-scorer", daemon=True)
        self._worker.start()
        _LIVE_SCORERS.add(self)

    def stop(self) -> None:
        """Ask the worker to finish and wait for it.

        Safe to call twice, and safe to call on a scorer that never started.
        """
        self._stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=JOIN_TIMEOUT_SECONDS)
            if worker.is_alive():
                logger.warning("scoring worker did not stop within the timeout")
        self._worker = None
        _LIVE_SCORERS.discard(self)
        # Drop the reference to the scoring function once the worker is joined.
        # It usually closes over a native inference session, and holding it past
        # shutdown lets that session be finalised while another model's session
        # is being created -- which surfaces as a native crash at interpreter
        # exit rather than a Python traceback.
        self._score = _released

    def push(self, crops: np.ndarray) -> None:
        """Hand one frame's eye crops to the scorer.

        Returns immediately in the threaded case -- the frame is buffered and
        the worker picks it up when free. **This is the property the whole
        design rests on**: the capture loop must never wait for a 77 ms
        inference.

        Args:
            crops (np.ndarray): ``(n_eyes, 3, H, W)`` for one frame.
        """
        if self._buffer is None:
            # Frame-wise: score now. The call is short and a thread would only
            # add latency.
            index = self._completed
            self._completed += 1
            try:
                scores = np.asarray(self._score(crops)).reshape(-1)
            except Exception as error:  # noqa: BLE001 - one bad frame must not end a stream
                logger.debug(f"inline scoring failed: {error}")
                self._failures += 1
                return
            with self._lock:
                self._latest = StreamScore(
                    scores=scores,
                    frame_index=index,
                    age_frames=0,
                    windows_completed=self._completed,
                    failures=self._failures,
                )
            return

        with self._lock:
            self._buffer.push(crops)

    def latest(self) -> StreamScore:
        """The most recent completed score.

        Args:
            None

        Returns:
            StreamScore: The last window's result with its current age, or
            :data:`EMPTY_SCORE` while warming up. Frames newer than that window
            carry this same value rather than an interpolated one.
        """
        with self._lock:
            latest = self._latest
            if self._buffer is None or latest.frame_index < 0:
                return latest
            age = self._buffer.count - 1 - latest.frame_index
        return StreamScore(
            scores=latest.scores,
            frame_index=latest.frame_index,
            age_frames=max(age, 0),
            windows_completed=latest.windows_completed,
            failures=latest.failures,
        )

    def _run(self) -> None:
        """Score the newest window, publish, repeat.

        The adaptive stride lives here: nothing schedules the next window, the
        worker simply takes whatever is current once it is free. That is what
        makes the rate follow the machine instead of a constant someone tuned on
        different hardware.
        """
        while not self._stop.is_set():
            with self._lock:
                snapshot = None if self._buffer is None else self._buffer.snapshot()
            if snapshot is None:
                time.sleep(IDLE_SLEEP_SECONDS)
                continue

            frames, index = snapshot
            try:
                scored = np.asarray(self._score(frames))
                per_frame = scored.reshape(scored.shape[0], -1)
            except Exception as error:  # noqa: BLE001 - one bad window must not end the stream
                logger.debug(f"windowed scoring failed: {error}")
                self._failures += 1
                continue

            # Re-check before publishing: a stop() that arrived mid-inference
            # means the caller is shutting down, and touching shared state now
            # races the interpreter tearing down the native session behind
            # `self._score`. Dropping this window costs one stale reading.
            if self._stop.is_set():
                break

            scores = self._accumulate(per_frame, index)
            self._completed += 1
            with self._lock:
                self._latest = StreamScore(
                    scores=scores,
                    frame_index=index,
                    age_frames=0,
                    windows_completed=self._completed,
                    failures=self._failures,
                )

    def _accumulate(self, per_frame: np.ndarray, newest: int) -> np.ndarray:
        """Fold a window's scores into the running per-frame means.

        The window covers ``newest - window + 1 .. newest``, and consecutive
        windows overlap, so most frames are scored several times. Averaging
        those rather than taking the newest one is what makes the live signal
        the same quantity the offline benchmark reports.

        Older entries are dropped as they leave every future window's reach, so
        the two dicts stay bounded by the window length however long the stream
        runs.

        Args:
            per_frame (np.ndarray): ``(n_eyes, window)`` scores.
            newest (int): Stream index of the window's last frame.

        Returns:
            np.ndarray: ``(n_eyes,)`` mean score for the newest frame.
        """
        window = per_frame.shape[1]
        oldest = newest - window + 1
        for offset in range(window):
            frame = oldest + offset
            column = per_frame[:, offset]
            if frame in self._sums:
                self._sums[frame] = self._sums[frame] + column
                self._counts[frame] += 1
            else:
                # float64 for the running sum: a float32 accumulator drifts by
                # ~1e-6 over a few hundred windows, which is invisible in a
                # score but enough to fail an exact-value assertion.
                self._sums[frame] = column.astype(np.float64, copy=True)
                self._counts[frame] = 1

        # A frame older than the newest window can never be scored again.
        for frame in [key for key in self._sums if key < oldest]:
            del self._sums[frame]
            del self._counts[frame]

        return (self._sums[newest] / self._counts[newest]).astype(np.float32)

    def __enter__(self) -> StreamingScorer:
        """Start the worker on entry.

        Returns:
            StreamingScorer: This scorer.
        """
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        """Stop the worker on exit, however the block ended."""
        self.stop()
