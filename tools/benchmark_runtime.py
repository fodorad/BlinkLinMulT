"""Latency, memory, size and cold-start for every shipped model.

The result that motivated this script: ``blinkcnn`` ran **4.7x slower** than the
1.x models despite being a *smaller* network. ConvNeXt-Femto has 5.0 M
parameters against DenseNet121's ~7 M, yet measured 194 ms per 15-frame window
against 42 ms. The cause was not the architecture -- it was that the 1.x models
ship as frozen ONNX graphs and ``blinkcnn`` ran eager PyTorch. Exporting the
same weights recovered the gap, which is what makes the number a measurement
rather than a claim.

Two figures are reported and never merged:

* **Model-only throughput** -- crops in, scores out. This is what the REST
  service does, and the number is honest for that workload.
* **Cold start** -- process start to first scored frame, in a fresh subprocess
  because a warm page cache makes any in-process repeat meaningless. ``blinkcnn``
  is the *smallest* artifact on disk and the *slowest* to load, which is a real
  serving consideration and invisible in a latency table alone.

Run::

    uv run python tools/benchmark_runtime.py --threads 1,4,8 --repeats 200
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from blinklinmult.bench.systems import MachineSpec, artifact_size_bytes, peak_rss_mb, pin_threads
from blinklinmult.bench.timing import DEFAULT_REPEATS, DEFAULT_WARMUP, time_callable

logger = logging.getLogger(__name__)
"""Module-level logger."""

SCHEMA_VERSION = 1
"""Output schema version."""

WINDOW = 15
"""Frames per call.

The 1.x sequence models were trained on 15-frame windows and lose accuracy on
longer ones, so 15 is both their natural unit and a fair one for the frame-wise
models, which accept any length.
"""

COLD_START_SOURCE = """
import time, numpy as np
started = time.perf_counter_ns()
from blinklinmult import BlinkDetector
from blinklinmult.registry import spec
detector = BlinkDetector.from_pretrained({model_id!r}, weights={weights!r})
model_spec = spec({model_id!r})
frames = model_spec.window or 1
crops = np.zeros((frames, 3, model_spec.image_size, model_spec.image_size), dtype="float32")
kwargs = {{}}
if model_spec.needs_features:
    # The two-stream model refuses to score without its descriptor stream, so
    # the probe supplies one. Zeros are enough: this times loading, not accuracy.
    kwargs["features"] = detector.prepare_features(
        np.zeros((frames, 160), dtype="float32"),
        mean=np.zeros(160, dtype="float32"),
        std=np.ones(160, dtype="float32"),
    )
detector.score(crops, **kwargs)
print((time.perf_counter_ns() - started) / 1e6)
"""
"""Measured in a subprocess: import, load and first inference, from nothing.

Handles both the frame-wise models and the two-stream one, which raises rather
than scoring when given no descriptors -- a probe that ignored that reported
``nan`` for the one model whose load time is most interesting.
"""


def _cold_start_ms(model_id: str, weights: Path) -> float:
    """Time a fresh process from launch to its first scored frame.

    Run out-of-process on purpose. Within one process the imports are cached,
    the weights are in the page cache and the allocator is warm, so a second
    measurement reports almost nothing. A serving cold start pays all three.

    Args:
        model_id (str): Registry id.
        weights (Path): Local weights, so the timing excludes any download.

    Returns:
        float: Milliseconds, or ``nan`` if the subprocess failed.
    """
    source = COLD_START_SOURCE.format(model_id=model_id, weights=str(weights))
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning(f"  cold start failed for {model_id}: {result.stderr.strip()[:200]}")
        return float("nan")
    return float(result.stdout.strip().splitlines()[-1])


def _feature_stats(directory: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Corpus feature statistics, needed by the two-stream model.

    Args:
        directory (Path): Where the artifacts live.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Mean and standard deviation, or
        ``None`` when the file is absent.
    """
    path = directory / "rn30_feature_stats.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return (
        np.asarray(payload["mean"], dtype="float32"),
        np.asarray(payload["std"], dtype="float32"),
    )


def benchmark_model(
    model_id: str,
    weights: Path,
    stats: tuple[np.ndarray, np.ndarray] | None,
    repeats: int,
    warmup: int,
) -> dict:
    """Measure one model at the current thread setting.

    Args:
        model_id (str): Registry id.
        weights (Path): Local weights file.
        stats (tuple[np.ndarray, np.ndarray] | None): Feature statistics for the
            two-stream model.
        repeats (int): Timed calls.
        warmup (int): Discarded calls.

    Returns:
        dict: Latency, memory, size and cold start.
    """
    from blinklinmult import BlinkDetector
    from blinklinmult.registry import spec

    model_spec = spec(model_id)
    detector = BlinkDetector.from_pretrained(model_id, weights=weights)

    rng = np.random.default_rng(0)
    crops = rng.random((WINDOW, 3, model_spec.image_size, model_spec.image_size)).astype("float32")
    kwargs: dict = {}
    if model_spec.needs_features:
        if stats is None:
            logger.warning(f"  {model_id}: no feature statistics, skipping")
            return {}
        features = rng.random((WINDOW, 160)).astype("float32")
        kwargs["features"] = detector.prepare_features(features, mean=stats[0], std=stats[1])

    before = peak_rss_mb()
    latency = time_callable(
        lambda: detector.score(crops, **kwargs),
        frames=WINDOW,
        warmup=warmup,
        repeats=repeats,
    )
    after = peak_rss_mb()

    return {
        "runtime": model_spec.runtime,
        "latency_p50_ms": latency.p50_ms,
        "latency_p95_ms": latency.p95_ms,
        "latency_p99_ms": latency.p99_ms if latency.tail_is_trustworthy else None,
        "ms_per_frame": latency.ms_per_frame,
        "throughput_fps": latency.fps,
        "peak_rss_mb": after,
        "rss_delta_mb": after - before,
        "artifact_bytes": artifact_size_bytes(weights),
        "cold_start_ms": _cold_start_ms(model_id, weights),
        "repeats": repeats,
        "warmup": warmup,
    }


def _print_table(results: dict[int, dict[str, dict]]) -> None:
    """Log the per-thread tables and the ONNX comparison.

    Args:
        results (dict[int, dict[str, dict]]): Threads to model to metrics.
    """
    for threads, models in results.items():
        logger.info(f"\n  {threads} thread(s)")
        header = f"    {'model':22s}{'runtime':>9s}{'p50 ms':>9s}{'ms/frame':>10s}{'fps':>8s}"
        logger.info(f"{header}{'MB':>7s}{'cold ms':>9s}")
        for model_id, row in models.items():
            if not row:
                continue
            logger.info(
                f"    {model_id:22s}{row['runtime']:>9s}{row['latency_p50_ms']:>9.1f}"
                f"{row['ms_per_frame']:>10.2f}{row['throughput_fps']:>8.0f}"
                f"{row['artifact_bytes'] / 1048576:>7.1f}{row['cold_start_ms']:>9.0f}"
            )

    for threads, models in results.items():
        torch_row = models.get("blinkcnn")
        onnx_row = models.get("blinkcnn-onnx")
        if not torch_row or not onnx_row:
            continue
        speedup = torch_row["latency_p50_ms"] / onnx_row["latency_p50_ms"]
        logger.info(
            f"\n  blinkcnn runtime migration at {threads} thread(s): "
            f"{torch_row['ms_per_frame']:.2f} -> {onnx_row['ms_per_frame']:.2f} ms/frame "
            f"({speedup:.2f}x), cold start {torch_row['cold_start_ms']:.0f} -> "
            f"{onnx_row['cold_start_ms']:.0f} ms"
        )


def main() -> None:
    """Benchmark every model that has local weights."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--threads", default="1,4", help="Comma-separated thread counts.")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from blinklinmult.registry import MODELS

    stats = _feature_stats(args.weights_dir)
    thread_counts = [int(value) for value in args.threads.split(",")]
    results: dict[int, dict[str, dict]] = {}
    machines: dict[int, MachineSpec] = {}

    for threads in thread_counts:
        pin_threads(threads)
        machines[threads] = MachineSpec.capture(threads)
        if machines[threads].contended:
            logger.warning(
                f"  load average {machines[threads].load_average[0]:.1f} exceeds "
                f"{machines[threads].cpu_count} cores; timings will be pessimistic"
            )
        results[threads] = {}
        for model_id, model_spec in MODELS.items():
            weights = args.weights_dir / model_spec.filename
            if not weights.is_file():
                logger.info(f"  {model_id}: no local weights at {weights}, skipping")
                continue
            results[threads][model_id] = benchmark_model(
                model_id, weights, stats, args.repeats, args.warmup
            )

    logger.info(f"\n{machines[thread_counts[0]].describe()}")
    _print_table(results)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": SCHEMA_VERSION,
            "generated_utc": datetime.now(UTC).isoformat(),
            "window_frames": WINDOW,
            "machines": {str(threads): machine.__dict__ for threads, machine in machines.items()},
            "models": {str(threads): models for threads, models in results.items()},
        }
        args.out.write_text(json.dumps(document, indent=2) + "\n")
        logger.info(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
