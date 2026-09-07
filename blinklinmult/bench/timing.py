"""Timing a callable honestly.

Three decisions, each guarding against a way a benchmark lies.

**Warm-up runs are discarded.** The first ONNX call allocates its arenas and the
first torch call picks kernels; folding that into the average reports startup
cost as though it were steady-state throughput. ``bench_workers.py`` already
established this convention for the dataloader -- "times the *second* pass ...
so the figure reflects steady-state throughput".

**Percentiles, not just a mean.** A model whose p99 is four times its median
drops frames under a deadline even when its average looks fine. For a streaming
workload the tail is the number that decides whether it works.

**A monotonic clock.** :func:`time.perf_counter_ns` cannot go backwards or be
adjusted mid-run, unlike :func:`time.time`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

DEFAULT_WARMUP = 20
"""Calls discarded before timing begins."""

DEFAULT_REPEATS = 200
"""Timed calls.

Enough that p95 rests on ten samples and p99 on two. Reporting p99 from fewer
would be reporting the maximum under another name, which is why
:meth:`Latency.tail_is_trustworthy` exists.
"""


class BenchError(ValueError):
    """Raised when a measurement is asked for on impossible parameters."""


@dataclass(frozen=True)
class Latency:
    """Timing of one configuration.

    Args:
        samples_ms (np.ndarray): Every timed call, in milliseconds. Kept whole
            rather than pre-summarised so a caller can re-derive any statistic.
        warmup (int): Calls discarded before timing.
        frames (int): Frames per call, so per-frame cost is comparable between a
            frame-wise model and one scoring 15-frame windows.
    """

    samples_ms: np.ndarray
    warmup: int
    frames: int

    @property
    def p50_ms(self) -> float:
        """Median call time.

        Returns:
            float: Milliseconds.
        """
        return float(np.percentile(self.samples_ms, 50))

    @property
    def p95_ms(self) -> float:
        """95th percentile call time.

        Returns:
            float: Milliseconds.
        """
        return float(np.percentile(self.samples_ms, 95))

    @property
    def p99_ms(self) -> float:
        """99th percentile call time.

        Returns:
            float: Milliseconds.
        """
        return float(np.percentile(self.samples_ms, 99))

    @property
    def ms_per_frame(self) -> float:
        """Median cost of one frame.

        The only figure comparable across models with different window lengths:
        a 15-frame sequence model and a frame-wise one do different amounts of
        work per call.

        Returns:
            float: Milliseconds per frame.
        """
        return self.p50_ms / self.frames

    @property
    def fps(self) -> float:
        """Frames per second at the median.

        Returns:
            float: Frames per second.
        """
        return 1000.0 * self.frames / self.p50_ms

    @property
    def tail_is_trustworthy(self) -> bool:
        """Whether enough samples were taken for p99 to mean anything.

        With fewer than 100 repeats the 99th percentile is interpolated from the
        largest one or two samples, which is the maximum wearing a percentile's
        name.

        Returns:
            bool: Whether p99 rests on at least a few samples.
        """
        return self.samples_ms.size >= 100

    def describe(self) -> str:
        """One line for a table.

        Returns:
            str: Median, tail and throughput.
        """
        tail = f"p95={self.p95_ms:.1f}"
        if self.tail_is_trustworthy:
            tail += f" p99={self.p99_ms:.1f}"
        return (
            f"p50={self.p50_ms:.1f} ms  {tail}  "
            f"{self.ms_per_frame:.2f} ms/frame  {self.fps:.0f} fps"
        )


def time_callable(
    call: Callable[[], object],
    frames: int,
    warmup: int = DEFAULT_WARMUP,
    repeats: int = DEFAULT_REPEATS,
) -> Latency:
    """Time a callable at steady state.

    Args:
        call (Callable[[], object]): The work to time. Should allocate nothing
            it can avoid: build inputs before calling, so allocation does not
            land inside the timed region.
        frames (int): Frames processed per call, for the per-frame figure.
        warmup (int): Calls to discard first.
        repeats (int): Calls to time.

    Returns:
        Latency: The samples and their summary.

    Raises:
        BenchError: If ``frames`` or ``repeats`` is not positive, or ``warmup``
            is negative.
    """
    import time

    if frames < 1:
        raise BenchError(f"A call must process at least one frame, got {frames}.")
    if repeats < 1:
        raise BenchError(f"Need at least one timed call, got {repeats}.")
    if warmup < 0:
        raise BenchError(f"Warm-up cannot be negative, got {warmup}.")

    for _ in range(warmup):
        call()

    samples = np.empty(repeats, dtype=float)
    for index in range(repeats):
        started = time.perf_counter_ns()
        call()
        samples[index] = (time.perf_counter_ns() - started) / 1e6

    return Latency(samples_ms=samples, warmup=warmup, frames=frames)
