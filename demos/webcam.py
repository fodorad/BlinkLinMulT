"""Live blink detection from a webcam, in one window.

The frame on top; beneath it two subplots at half the width each, showing the
last ten seconds of closure score per eye. Wink one eye and only that side
spikes -- which is also the check that the left/right labelling is not inverted.

**Three threads, because two of the three costs release the GIL.** Measured on
this machine: a busy main loop kept **95%** of its iterations with a
``cap.read()`` thread running and **102%** with an onnxruntime worker, so both
genuinely overlap rather than time-slicing.

    grabber   cap.read() -> newest-frame slot     16 ms, GIL released
    scorer    only for a windowed model           77 ms, GIL released
    display   detect, crop, score, draw, imshow   ~28 ms

The grabber keeps **only the newest frame**. A queue would grow lag the instant
the display fell behind, and for a live view a dropped frame is better than a
stale one.

**The camera is opened at its native resolution on purpose.** Asking for a
smaller frame is slower, not faster: measured 16.0 ms per read at 1920x1080
against 32.9 ms at 1280x720, because the driver rescales off its native mode.
The frame is downscaled in numpy afterwards, which is what the detector wants.

Run::

    uv run --extra preprocess --extra onnx python demos/webcam.py
    uv run --extra preprocess --extra onnx python demos/webcam.py --model blinklint-union
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from pathlib import Path

import numpy as np

from blinklinmult.live import (
    DEFAULT_SECONDS,
    PLOT_HEIGHT,
    ScoreHistory,
    compose,
    draw_overlay,
    draw_plot,
    eye_crops,
)
from blinklinmult.preprocess.geometry import LEFT, RIGHT

logger = logging.getLogger(__name__)
"""Module-level logger."""

DEFAULT_MODEL = "blinkcnn-onnx"
"""Scored inline with no lag, so a wink spikes the plot on the same frame.

A windowed model such as ``blinklint-union`` runs on a background worker and its
readout trails by ~233 ms, which reads as lag in a live view even though the
video itself stays smooth.
"""

DISPLAY_WIDTH = 960
"""Width the captured frame is scaled to before detection and display.

Halving 1080p cuts detection from 16.1 to 12.4 ms and costs nothing visible at
this window size.
"""

WINDOW_TITLE = "BlinkLinMulT - live"
"""Window name. Also the handle OpenCV uses, so it must stay stable."""

CAMERA_WARMUP_SECONDS = 10.0
"""How long to wait for the first frame.

Generous: on macOS the first ``read`` blocks while the OS asks for camera
permission, and a demo that exits immediately looks broken rather than blocked.
"""


class _Grabber:
    """Reads frames on its own thread, keeping only the newest.

    Args:
        camera (int): Device index.
    """

    def __init__(self, camera: int) -> None:
        """Open the camera at its native resolution."""
        import cv2

        self.capture = cv2.VideoCapture(camera)
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def opened(self) -> bool:
        """Whether the device is available.

        Returns:
            bool: ``True`` when the camera opened.
        """
        return bool(self.capture.isOpened())

    def start(self) -> None:
        """Begin grabbing."""
        self._thread = threading.Thread(target=self._run, name="grabber", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Replace the stored frame as fast as the device supplies one."""
        while not self._stop.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame

    def latest(self) -> np.ndarray | None:
        """The most recent frame.

        Returns:
            np.ndarray | None: ``(H, W, 3)`` BGR, or ``None`` before the first
            frame arrives.
        """
        with self._lock:
            return self._frame

    def stop(self) -> None:
        """Stop grabbing and release the device."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.capture.release()


def _prepare(frame: np.ndarray, mirror: bool) -> np.ndarray:
    """Scale and optionally mirror a captured frame.

    Args:
        frame (np.ndarray): ``(H, W, 3)`` BGR from the camera.
        mirror (bool): Whether to flip horizontally. **Mirrored by default**: an
            unmirrored self-view moves the wrong way and is disorienting.

    Returns:
        np.ndarray: ``(h, DISPLAY_WIDTH, 3)`` RGB.
    """
    import cv2

    scale = DISPLAY_WIDTH / frame.shape[1]
    small = cv2.resize(frame, (DISPLAY_WIDTH, int(round(frame.shape[0] * scale))))
    if mirror:
        small = cv2.flip(small, 1)
    return cv2.cvtColor(small, cv2.COLOR_BGR2RGB)


def main() -> None:
    """Run the live demo until the user quits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Registry id of the blink model.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--imgsz", type=int, default=256, help="Detector input side.")
    parser.add_argument(
        "--seconds", type=float, default=DEFAULT_SECONDS, help="Seconds of history plotted."
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override the model's.")
    parser.add_argument("--no-mirror", action="store_true", help="Do not flip the view.")
    parser.add_argument("--weights-dir", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--record", type=Path, default=None, help="Write the view to a video.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import cv2

    from blinklinmult import BlinkDetector
    from blinklinmult.preprocess.extractors import PoseEyeLocator
    from blinklinmult.preprocess.geometry import pose_from_keypoints
    from blinklinmult.registry import spec
    from blinklinmult.stream import StreamingScorer

    model_spec = spec(args.model)
    threshold = args.threshold if args.threshold is not None else model_spec.threshold
    detector = BlinkDetector.from_pretrained(
        args.model, weights=args.weights_dir / model_spec.filename
    )
    locator = PoseEyeLocator(image_size=args.imgsz)

    grabber = _Grabber(args.camera)
    if not grabber.opened:
        raise SystemExit(
            f"cannot open camera {args.camera}. On macOS the terminal needs camera "
            "permission in System Settings > Privacy & Security > Camera; otherwise "
            "try a different --camera index."
        )
    grabber.start()

    logger.info("waiting for the first frame ...")
    deadline = time.monotonic() + CAMERA_WARMUP_SECONDS
    while grabber.latest() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if grabber.latest() is None:
        grabber.stop()
        raise SystemExit("the camera opened but produced no frames within the warm-up window")

    fps_guess = 30.0
    capacity = max(int(args.seconds * fps_guess), 2)
    histories = {side: ScoreHistory(capacity=capacity) for side in (LEFT, RIGHT)}
    writer = None
    durations: list[float] = []

    logger.info(f"{args.model} at threshold {threshold:.2f} -- press q or Esc to quit")
    with StreamingScorer(detector.score, window=model_spec.window) as scorer:
        try:
            while True:
                started = time.perf_counter_ns()
                captured = grabber.latest()
                if captured is None:
                    break

                frame = _prepare(captured, mirror=not args.no_mirror)
                detection = locator.detect(frame)

                scores = {LEFT: float("nan"), RIGHT: float("nan")}
                if detection is not None:
                    crops = eye_crops(detection, frame, model_spec.image_size)
                    if crops is not None:
                        scorer.push(crops)
                        latest = scorer.latest()
                        if not latest.warming_up and latest.scores.size >= 2:
                            scores[LEFT] = float(latest.scores[0])
                            scores[RIGHT] = float(latest.scores[1])

                for side in (LEFT, RIGHT):
                    histories[side].push(scores[side])

                lines = [f"{np.median(durations[-30:]):.0f} ms/frame" if durations else "..."]
                if detection is not None:
                    yaw, pitch, roll = pose_from_keypoints(detection.landmarks)
                    lines.append(f"yaw {yaw:+.0f}  pitch {pitch:+.0f}  roll {roll:+.0f}")
                if model_spec.window:
                    # A windowed model's readout is older than the picture, and
                    # saying so is better than letting it look like a slow model.
                    lines.append(f"lag {scorer.latest().age_ms(fps_guess):.0f} ms")

                annotated = draw_overlay(frame, detection, scores, threshold, lines)
                view = compose(
                    annotated,
                    draw_plot(
                        histories[LEFT], threshold, "left eye", DISPLAY_WIDTH // 2, PLOT_HEIGHT
                    ),
                    draw_plot(
                        histories[RIGHT], threshold, "right eye", DISPLAY_WIDTH // 2, PLOT_HEIGHT
                    ),
                )

                if args.record is not None and writer is None:
                    writer = cv2.VideoWriter(
                        str(args.record),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps_guess,
                        (view.shape[1], view.shape[0]),
                    )
                if writer is not None:
                    writer.write(cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

                cv2.imshow(WINDOW_TITLE, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                durations.append((time.perf_counter_ns() - started) / 1e6)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            grabber.stop()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    if durations:
        median = float(np.median(durations))
        logger.info(f"display loop: {median:.1f} ms/frame -> {1000 / median:.1f} fps")


if __name__ == "__main__":
    main()
