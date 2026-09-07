"""End-to-end streaming: can this pipeline keep up with a camera?

A model-only throughput figure answers the wrong question for a live workload,
so this replays a clip at its own frame rate through every stage a deployment
would run -- face detection, eye localisation, head pose, crop, blink model --
and reports **frames dropped against the deadline**, not just a mean.

The defaults are the real-time configuration, deliberately: running this with no
arguments is meant to keep up. Three choices get it there, each measured on CPU:

* **Detector input at 256, not ultralytics' 640** -- 8.3 ms against 40.3 ms.
  The face was still found on 60/60 frames and the blink decision was unchanged
  on every one.
* **Eye centres from the detector's own keypoints** rather than a second
  landmark model. ``yolo11n-pose`` returns both eyes in the same forward pass,
  so FaceMesh's ~5 ms buys nothing the blink model can use -- validated on the
  five annotated closed frames of the bundled clip, which score the same either
  way.
* **Geometric head pose** from those same keypoints, ~0.05 ms against 26.7 ms
  for 6DRepNet. Pose is kept rather than dropped because yaw is what decides
  self-occlusion; ``--head-pose 6drepnet`` restores the accurate path.

Two numbers, never merged: **model-only throughput** (crops in, scores out --
honest for the REST service) and **end-to-end streaming**. The comparison holds
every stage fixed and swaps only the blink model, so the delta is the model.

Run::

    uv run --extra preprocess --extra onnx python tools/benchmark_streaming.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from blinklinmult.bench.systems import MachineSpec, pin_threads

logger = logging.getLogger(__name__)
"""Module-level logger."""

SCHEMA_VERSION = 1
"""Output schema version."""

DEFAULT_VIDEO = Path("blinklinmult/assets/talkingface_10s.mp4")
"""The bundled clip, replayed as if it arrived from a webcam."""

STREAM_MODELS = ("blinkcnn-onnx", "densenet121-union", "blinklint-union")
"""The models compared under identical capture conditions.

The first two are frame-wise; ``blinklint-union`` consumes a 15-frame window and
runs on a background worker (see :mod:`blinklinmult.stream`). Including it is the
point: a sequence model that costs 77 ms per call can still feel responsive, and
the ``windows/s`` and ``lag`` columns are what show at what price.
"""

CROP_SIZE = 64
"""Crop side the models expect."""

DEFAULT_LOCATOR = "pose"
"""Eye localisation path, defaulting to the fast one.

``pose`` reads both eye centres from the face detector's own keypoints, which
are computed in the same forward pass as the box. ``facemesh`` runs the dense
landmark model afterwards -- more precise, ~5 ms more per frame, and the path
the corpus builders use.
"""

DEFAULT_IMAGE_SIZE = 256
"""Detector input side.

Not ultralytics' 640 default, which upscales a webcam frame and costs 40.3 ms
per frame on CPU against 8.3 ms here.
"""

ACCEPTABLE_DROP_RATE = 0.05
"""Share of frames that may miss the deadline and still count as real time.

Not zero. A capture loop at 42 fps against a 30 fps source has 40% headroom, and
still misses one frame in 120 when the OS schedules something else -- calling
that "cannot keep up" would report the machine's background load as a property
of the pipeline. Five percent is loose enough to absorb scheduling noise and
tight enough that a genuinely slow pipeline still fails.
"""

DEFAULT_HEAD_POSE = "geometric"
"""How head rotation is estimated.

``geometric`` derives it from the detector's five keypoints at ~0.05 ms;
``6drepnet`` runs the dedicated network at ~26.7 ms on CPU, which alone puts the
pipeline outside a 30 fps budget. The default is the one that keeps up.
"""


def _build_head_pose(method: str):
    """Build the head-pose estimator, or ``None`` when it is not wanted.

    Head pose is kept in the streaming path because yaw drives self-occlusion
    gating -- deciding which eye is visible enough to score.

    Args:
        method (str): ``"geometric"``, ``"6drepnet"`` or ``"none"``.

    Returns:
        Callable | None: Takes ``(detection, frame)`` and returns angles, or
        ``None`` when disabled.

    Raises:
        SystemExit: If the method is unknown.
    """
    if method == "none":
        return None
    if method == "geometric":
        from blinklinmult.preprocess.geometry import pose_from_keypoints

        return lambda detection, _frame: pose_from_keypoints(detection.landmarks)
    if method == "6drepnet":
        import cv2

        from blinklinmult.preprocess.extractors import FACE_POSE_MARGIN, ExordiumExtractor

        extractor = ExordiumExtractor()

        def _accurate(detection, frame):
            """Crop the face box and run the pose network on it."""
            x1, y1, x2, y2 = detection.face_box
            margin_x = int((x2 - x1) * FACE_POSE_MARGIN)
            margin_y = int((y2 - y1) * FACE_POSE_MARGIN)
            patch = frame[
                max(0, y1 - margin_y) : y2 + margin_y, max(0, x1 - margin_x) : x2 + margin_x
            ]
            return extractor.head_pose(cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))

        return _accurate
    raise SystemExit(f"unknown head-pose method {method!r}")


@dataclass
class StageTimes:
    """Per-frame costs of each stage, in milliseconds.

    Args:
        detect_ms (list[float]): YOLO11 face detection plus FaceMesh landmarks
            and eye localisation, which one ``detect`` call performs together.
        crop_ms (list[float]): Cutting and resizing both eye crops.
        score_ms (list[float]): The blink model.
        total_ms (list[float]): Wall clock for the whole frame, which is what
            the deadline is measured against.
        age_frames (list[int]): How stale the displayed score was, per frame.
            Zero throughout for a frame-wise model; for a windowed one this is
            the lag a viewer actually sees.
        windows_completed (int): Windows the background worker finished. Divided
            by the frame count it gives the stride the worker settled on, which
            is chosen by the machine rather than configured.
    """

    detect_ms: list[float] = field(default_factory=list)
    crop_ms: list[float] = field(default_factory=list)
    score_ms: list[float] = field(default_factory=list)
    total_ms: list[float] = field(default_factory=list)
    age_frames: list[int] = field(default_factory=list)
    windows_completed: int = 0

    def summary(self, fps: float) -> dict:
        """Reduce to the figures a reader needs.

        Args:
            fps (float): Source frame rate, giving the per-frame deadline.

        Returns:
            dict: Stage medians, tail latency, throughput and drop rate.
        """
        budget_ms = 1000.0 / fps
        total = np.asarray(self.total_ms)
        dropped = int((total > budget_ms).sum())
        return {
            "frames": int(total.size),
            "budget_ms": budget_ms,
            "detect_ms_p50": float(np.median(self.detect_ms)),
            "crop_ms_p50": float(np.median(self.crop_ms)),
            "score_ms_p50": float(np.median(self.score_ms)),
            "total_ms_p50": float(np.median(total)),
            "total_ms_p95": float(np.percentile(total, 95)),
            "total_ms_p99": float(np.percentile(total, 99)),
            "achieved_fps": 1000.0 / float(np.median(total)),
            "realtime_factor": budget_ms / float(np.median(total)),
            "frames_dropped": dropped,
            "drop_rate": dropped / max(int(total.size), 1),
            "keeps_up": dropped == 0,
            "lag_ms": float(np.median(self.age_frames)) * 1000.0 / fps if self.age_frames else 0.0,
            "windows_per_second": self.windows_completed * fps / max(int(total.size), 1),
            "effective_stride": (
                int(total.size) / self.windows_completed if self.windows_completed else 0.0
            ),
        }


def _read_frames(video: Path, limit: int) -> tuple[list[np.ndarray], float]:
    """Decode frames up front, so decoding is not timed as pipeline cost.

    A real camera hands frames over as they arrive; decoding a file is an
    artifact of the harness, not of the workload.

    Args:
        video (Path): The clip.
        limit (int): Most frames to read.

    Returns:
        tuple[list[np.ndarray], float]: Frames in BGR, and the source frame rate.

    Raises:
        SystemExit: If the video cannot be opened or holds no frames.
    """
    import cv2

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0

    frames: list[np.ndarray] = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        raise SystemExit(f"{video} yielded no frames")
    return frames, fps


def _crops_from(detection, frame: np.ndarray) -> np.ndarray | None:
    """Cut both eye crops into the layout the models take.

    Args:
        detection: A ``FaceDetection`` from ``FaceMeshLocator.detect``.
        frame (np.ndarray): The source frame, BGR.

    Returns:
        np.ndarray | None: ``(n_eyes, 3, 64, 64)`` in ``[0, 1]``, or ``None``
        when neither eye was localised.
    """
    import cv2

    patches = []
    for box in detection.eyes.values():
        if box is None:
            continue
        patch = box.crop(frame)
        if patch.size == 0:
            continue
        resized = cv2.resize(patch, (CROP_SIZE, CROP_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        patches.append(rgb.transpose(2, 0, 1).astype("float32") / 255.0)
    if not patches:
        return None
    return np.stack(patches)


def _build_locator(locator: str, image_size: int):
    """Construct the requested eye locator.

    Args:
        locator (str): ``"pose"`` for the detector-only path, ``"facemesh"``
            for the dense-landmark one.
        image_size (int): Detector input side.

    Returns:
        object: Something with a ``detect(frame)`` method.

    Raises:
        SystemExit: If the name is unknown.
    """
    from blinklinmult.preprocess.extractors import FaceMeshLocator, PoseEyeLocator

    if locator == "pose":
        return PoseEyeLocator(image_size=image_size)
    if locator == "facemesh":
        return FaceMeshLocator(detector_image_size=image_size)
    raise SystemExit(f"unknown locator {locator!r}; expected 'pose' or 'facemesh'")


def run_stream(
    model_id: str,
    frames: list[np.ndarray],
    weights_dir: Path,
    locator_name: str = DEFAULT_LOCATOR,
    image_size: int = DEFAULT_IMAGE_SIZE,
    head_pose: str = DEFAULT_HEAD_POSE,
) -> StageTimes:
    """Run the reduced pipeline over every frame, timing each stage.

    Args:
        model_id (str): Registry id of the scoring model.
        frames (list[np.ndarray]): Decoded frames.
        weights_dir (Path): Where local weights live.
        locator_name (str): ``"pose"`` or ``"facemesh"``.
        image_size (int): Detector input side.
        head_pose (str): ``"geometric"``, ``"6drepnet"`` or ``"none"``.

    Returns:
        StageTimes: Per-frame costs.
    """
    from blinklinmult import BlinkDetector
    from blinklinmult.registry import spec
    from blinklinmult.stream import StreamingScorer

    locator = _build_locator(locator_name, image_size)
    pose = _build_head_pose(head_pose)
    model_spec = spec(model_id)
    weights = weights_dir / model_spec.filename
    detector = BlinkDetector.from_pretrained(model_id, weights=weights)

    # Warm up on the first frame: the first YOLO and onnxruntime calls allocate
    # arenas and pick kernels, which is startup cost, not steady state.
    warm = locator.detect(frames[0])
    if warm is not None:
        crops = _crops_from(warm, frames[0])
        if crops is not None:
            detector.score(crops)

    times = StageTimes()
    # A windowed model gets a background worker; a frame-wise one is scored
    # inline, because handing a 15 ms call to a thread only adds latency.
    with StreamingScorer(detector.score, window=model_spec.window) as scorer:
        for frame in frames:
            frame_started = time.perf_counter_ns()

            started = time.perf_counter_ns()
            detection = locator.detect(frame)
            if detection is not None and pose is not None:
                # Head pose is timed inside the detection stage: it is a property
                # of the face, not of the eye crops, and a deployment that needs
                # it cannot skip it.
                pose(detection, frame)
            detect_ms = (time.perf_counter_ns() - started) / 1e6

            crop_ms = 0.0
            score_ms = 0.0
            if detection is not None:
                started = time.perf_counter_ns()
                crops = _crops_from(detection, frame)
                crop_ms = (time.perf_counter_ns() - started) / 1e6

                if crops is not None:
                    started = time.perf_counter_ns()
                    scorer.push(crops)
                    latest = scorer.latest()
                    score_ms = (time.perf_counter_ns() - started) / 1e6
                    if not latest.warming_up:
                        times.age_frames.append(latest.age_frames)

            times.detect_ms.append(detect_ms)
            times.crop_ms.append(crop_ms)
            times.score_ms.append(score_ms)
            times.total_ms.append((time.perf_counter_ns() - frame_started) / 1e6)

        times.windows_completed = scorer.latest().windows_completed

    return times


def _print_report(results: dict[str, dict], fps: float) -> None:
    """Log the stage budget and the verdict.

    Args:
        results (dict[str, dict]): Per-model summaries.
        fps (float): Source frame rate.
    """
    logger.info(f"\n  Source: {fps:.1f} fps, budget {1000.0 / fps:.1f} ms per frame\n")
    header = f"    {'model':22s}{'detect':>9s}{'crop':>8s}{'score':>8s}{'total':>9s}"
    logger.info(f"{header}{'fps':>7s}{'dropped':>9s}{'lag':>9s}{'stride':>8s}")
    for model_id, summary in results.items():
        lag = f"{summary['lag_ms']:.0f}ms" if summary["lag_ms"] else "-"
        stride = f"{summary['effective_stride']:.1f}" if summary["effective_stride"] > 1.5 else "-"
        logger.info(
            f"    {model_id:22s}{summary['detect_ms_p50']:>9.1f}{summary['crop_ms_p50']:>8.1f}"
            f"{summary['score_ms_p50']:>8.1f}{summary['total_ms_p50']:>9.1f}"
            f"{summary['achieved_fps']:>7.1f}"
            f"{summary['frames_dropped']:>6d}/{summary['frames']:<3d}"
            f"{lag:>9s}{stride:>8s}"
        )

    for model_id, summary in results.items():
        # "Keeps up" means the capture loop met its deadline on most frames, not
        # on every one: a single scheduling hiccup in 120 frames is not a
        # pipeline that cannot run live, and reporting it as one would make the
        # verdict useless on any shared machine.
        verdict = "keeps up" if summary["drop_rate"] <= ACCEPTABLE_DROP_RATE else "cannot keep up"
        if summary["lag_ms"]:
            # A windowed model: the capture loop and the readout are different
            # questions, and reporting only throughput would answer neither.
            logger.info(
                f"\n  {model_id}: capture at {summary['achieved_fps']:.1f} fps, {verdict}. "
                f"The window model runs on a worker at every {summary['effective_stride']:.1f}th "
                f"frame, so the readout trails by {summary['lag_ms']:.0f} ms."
            )
        else:
            share = 100.0 * summary["score_ms_p50"] / summary["total_ms_p50"]
            logger.info(
                f"\n  {model_id}: {summary['realtime_factor']:.2f}x realtime, {verdict}. "
                f"The model is {share:.0f}% of the per-frame budget; detection is "
                f"{100.0 * summary['detect_ms_p50'] / summary['total_ms_p50']:.0f}%."
            )

    frame_wise = {k: v for k, v in results.items() if not v["lag_ms"]}
    if len(frame_wise) == 2:
        first, second = frame_wise.values()
        model_delta = first["score_ms_p50"] - second["score_ms_p50"]
        total_delta = first["total_ms_p50"] - second["total_ms_p50"]
        share = 100.0 * first["score_ms_p50"] / first["total_ms_p50"]
        logger.info(
            f"\n  Swapping the frame-wise model changes inference by "
            f"{abs(model_delta):.1f} ms/frame and end-to-end by {abs(total_delta):.1f} "
            f"ms/frame. The model is {share:.0f}% of the budget, so Amdahl's law caps "
            "what optimising it alone can achieve at roughly that share."
        )


def main() -> None:
    """Replay a clip through the reduced pipeline for each model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--locator",
        default=DEFAULT_LOCATOR,
        choices=("pose", "facemesh"),
        help="Eye localisation path; 'pose' skips the landmark model.",
    )
    parser.add_argument(
        "--imgsz", type=int, default=DEFAULT_IMAGE_SIZE, help="Detector input side."
    )
    parser.add_argument(
        "--head-pose",
        default=DEFAULT_HEAD_POSE,
        choices=("geometric", "6drepnet", "none"),
        help="How to estimate head rotation.",
    )
    parser.add_argument("--frames", type=int, default=100, help="Frames to replay.")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--weights-dir", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pin_threads(args.threads)
    machine = MachineSpec.capture(args.threads)
    if machine.contended:
        logger.warning(
            f"  load average {machine.load_average[0]:.1f} on {machine.cpu_count} cores; "
            "timings will be pessimistic"
        )

    frames, fps = _read_frames(args.video, args.frames)
    logger.info(f"{machine.describe()}")
    logger.info(f"  {len(frames)} frames from {args.video.name}")

    results: dict[str, dict] = {}
    for model_id in STREAM_MODELS:
        weights = (
            args.weights_dir
            / __import__("blinklinmult.registry", fromlist=["spec"]).spec(model_id).filename
        )
        if not weights.is_file():
            logger.info(f"  {model_id}: no local weights, skipping")
            continue
        results[model_id] = run_stream(
            model_id,
            frames,
            args.weights_dir,
            locator_name=args.locator,
            image_size=args.imgsz,
            head_pose=args.head_pose,
        ).summary(fps)

    _print_report(results, fps)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "generated_utc": datetime.now(UTC).isoformat(),
            "video": str(args.video),
            "source_fps": fps,
            "machine": machine.__dict__,
            "pipeline": [
                f"yolo11_face(imgsz={args.imgsz})",
                "facemesh" if args.locator == "facemesh" else "detector_keypoints",
                f"head_pose({args.head_pose})",
                "eye_crop",
                "blink_model",
            ],
            "excluded": ["tracking", "descriptors"],
            "models": results,
        }
        args.out.write_text(json.dumps(document, indent=2) + "\n")
        logger.info(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
