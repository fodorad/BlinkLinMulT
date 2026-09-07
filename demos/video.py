"""The live view, driven by a video file instead of a camera.

Same rendering as ``webcam_demo.py`` -- the frame above, a scrolling per-eye
score plot beneath -- but reading a file. Two reasons it exists:

* **It can be checked.** The bundled clip carries five annotated closed frames
  (170, 171, 227, 275, 276), so the plots should spike there and nowhere else.
  A camera gives no ground truth to check against.
* **It runs without hardware.** No camera, no permissions dialog, and it can be
  recorded to a file and shown later.

Playback is paced to the file's own frame rate, so what is on screen matches
what a viewer would see live. ``--fast`` drops the pacing when the goal is a
recording rather than a demonstration.

Run::

    uv run --extra preprocess --extra onnx python demos/video.py
    uv run --extra preprocess --extra onnx python demos/video.py --record demo.mp4 --fast
"""

from __future__ import annotations

import argparse
import logging
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

DEFAULT_VIDEO = Path("blinklinmult/assets/talkingface_10s.mp4")
"""The bundled clip. Its five annotated blinks make the output checkable."""

DEFAULT_MODEL = "blinkcnn-onnx"
"""Frame-wise and lag-free, matching the webcam demo's default."""

DISPLAY_WIDTH = 960
"""Width frames are scaled to, matching the webcam demo so both look alike."""

WINDOW_TITLE = "BlinkLinMulT - video"
"""Window name."""


def main() -> None:
    """Play a video through the live view."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Registry id of the blink model.")
    parser.add_argument("--imgsz", type=int, default=256, help="Detector input side.")
    parser.add_argument(
        "--seconds", type=float, default=DEFAULT_SECONDS, help="Seconds of history plotted."
    )
    parser.add_argument("--threshold", type=float, default=None, help="Override the model's.")
    parser.add_argument("--weights-dir", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--record", type=Path, default=None, help="Write the view to a video.")
    parser.add_argument("--fast", action="store_true", help="Do not pace playback to the source.")
    parser.add_argument("--headless", action="store_true", help="Render without a window.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    import cv2

    from blinklinmult import BlinkDetector
    from blinklinmult.preprocess.extractors import PoseEyeLocator
    from blinklinmult.preprocess.geometry import pose_from_keypoints
    from blinklinmult.registry import spec
    from blinklinmult.stream import StreamingScorer

    if not args.video.is_file():
        raise SystemExit(f"no video at {args.video}")

    model_spec = spec(args.model)
    threshold = args.threshold if args.threshold is not None else model_spec.threshold
    detector = BlinkDetector.from_pretrained(
        args.model, weights=args.weights_dir / model_spec.filename
    )
    locator = PoseEyeLocator(image_size=args.imgsz)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frame_budget = 1.0 / fps

    capacity = max(int(args.seconds * fps), 2)
    histories = {side: ScoreHistory(capacity=capacity) for side in (LEFT, RIGHT)}
    writer = None
    durations: list[float] = []
    spikes: list[int] = []
    index = -1

    logger.info(
        f"{args.video.name} at {fps:.0f} fps, model {args.model}, threshold {threshold:.2f}"
    )
    with StreamingScorer(detector.score, window=model_spec.window) as scorer:
        try:
            while True:
                ok, captured = capture.read()
                if not ok:
                    break
                index += 1
                started = time.perf_counter()

                scale = DISPLAY_WIDTH / captured.shape[1]
                small = cv2.resize(captured, (DISPLAY_WIDTH, int(round(captured.shape[0] * scale))))
                frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
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
                if any(np.isfinite(v) and v >= threshold for v in scores.values()):
                    spikes.append(index)

                lines = [f"frame {index}"]
                if detection is not None:
                    yaw, pitch, roll = pose_from_keypoints(detection.landmarks)
                    lines.append(f"yaw {yaw:+.0f}  pitch {pitch:+.0f}  roll {roll:+.0f}")
                if model_spec.window:
                    lines.append(f"lag {scorer.latest().age_ms(fps):.0f} ms")

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
                        fps,
                        (view.shape[1], view.shape[0]),
                    )
                if writer is not None:
                    writer.write(cv2.cvtColor(view, cv2.COLOR_RGB2BGR))

                if not args.headless:
                    cv2.imshow(WINDOW_TITLE, cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break

                elapsed = time.perf_counter() - started
                durations.append(elapsed * 1000.0)
                if not args.fast and elapsed < frame_budget:
                    # Pace to the source so the view matches what a live camera
                    # would show, rather than racing through the file.
                    time.sleep(frame_budget - elapsed)
        finally:
            capture.release()
            if writer is not None:
                writer.release()
            cv2.destroyAllWindows()

    if durations:
        median = float(np.median(durations))
        logger.info(f"processing: {median:.1f} ms/frame -> {1000 / median:.1f} fps")
    if spikes:
        # Runs of consecutive frames above threshold, which is what a blink is.
        runs = np.split(np.asarray(spikes), np.flatnonzero(np.diff(spikes) != 1) + 1)
        logger.info(f"frames above threshold: {[f'{r[0]}-{r[-1]}' for r in runs]}")


if __name__ == "__main__":
    main()
