"""Latency, memory and size measurement for the shipped models.

A model's accuracy decides whether it is worth deploying; these numbers decide
whether it *can* be. The repo measured neither until now:
``scripts/benchmark_table.py`` is accuracy-only despite its name, and
``scripts/bench_workers.py`` times the dataloader rather than the model.

The convention throughout is the one ``bench_workers.py`` already established --
warm up, then time steady state, and report a median over many repeats rather
than a single run. Two additions matter for reproducibility:

* **Thread counts are pinned and recorded.** ``blinkcnn`` measured 469 / 199 /
  229 ms at 1 / 4 / 8 threads on a 10-core machine, so an unpinned number is
  not a measurement of the model.
* **The machine is recorded next to the numbers.** A latency without its
  hardware, library versions and thread settings cannot be compared with
  anything.

See :mod:`blinklinmult.bench.timing` for the timer and
:mod:`blinklinmult.bench.systems` for the environment block.
"""

from blinklinmult.bench.systems import MachineSpec, artifact_size_bytes, peak_rss_mb
from blinklinmult.bench.timing import BenchError, Latency, time_callable

__all__ = [
    "BenchError",
    "Latency",
    "MachineSpec",
    "artifact_size_bytes",
    "peak_rss_mb",
    "time_callable",
]
