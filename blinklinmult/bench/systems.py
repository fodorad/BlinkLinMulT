"""The environment a measurement was taken in, and the resources it used.

A latency figure without its machine is not a measurement, so every benchmark
result carries a :class:`MachineSpec`: hardware, library versions, thread
settings and the commit that produced it. Without those a reader cannot tell
whether a number disagrees with theirs because the code changed or because the
laptop did.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

CONTENTION_FRACTION = 0.5
"""Share of the cores that, once busy, makes a timing untrustworthy.

Half rather than all: measured on this machine, a one-minute load of 7.0 against
10 cores inflated every model's latency by ~1.7x. A threshold at saturation
would have called that idle.
"""

RSS_DIVISOR = 1048576 if sys.platform == "darwin" else 1024
"""Divisor turning ``ru_maxrss`` into megabytes.

The units differ by platform and nothing warns about it: macOS reports bytes,
Linux reports kibibytes. Hard-coding either gives numbers 1024x wrong on the
other, in a field a reader has no way to sanity-check.
"""


@dataclass(frozen=True)
class MachineSpec:
    """Where and with what a measurement was taken.

    Args:
        platform (str): OS and architecture.
        processor (str): CPU, where the platform reports one.
        cpu_count (int): Logical cores visible.
        python (str): Interpreter version.
        torch (str): Torch version, or empty when absent.
        onnxruntime (str): onnxruntime version, or empty when absent.
        numpy (str): numpy version.
        threads (int): Threads the run was pinned to.
        git_sha (str): Commit the code was at.
        load_average (tuple[float, float, float]): System load at capture, so a
            run taken on a busy machine can be recognised as such rather than
            quietly reported as the model being slow.
    """

    platform: str
    processor: str
    cpu_count: int
    python: str
    torch: str
    onnxruntime: str
    numpy: str
    threads: int
    git_sha: str
    load_average: tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))

    @classmethod
    def capture(cls, threads: int) -> MachineSpec:
        """Record the current environment.

        Args:
            threads (int): Threads the benchmark was pinned to.

        Returns:
            MachineSpec: The environment block.
        """
        return cls(
            platform=platform.platform(),
            processor=platform.processor() or "unknown",
            cpu_count=os.cpu_count() or 0,
            python=platform.python_version(),
            torch=_version("torch"),
            onnxruntime=_version("onnxruntime"),
            numpy=_version("numpy"),
            threads=threads,
            git_sha=_git_sha(),
            load_average=_load_average(),
        )

    @property
    def contended(self) -> bool:
        """Whether the machine looked busy enough to distort timings.

        The bar is :data:`CONTENTION_FRACTION` of the core count, not full
        saturation. Measured here: at load 7.0 on 10 cores -- comfortably below
        saturation -- the same models timed **1.7x slower** than on an idle
        machine. Waiting for load to exceed the core count would let that pass
        unremarked, and a benchmark that silently reports contention as model
        latency is worse than no benchmark.

        Returns:
            bool: Whether one-minute load crossed the fraction.
        """
        if self.cpu_count <= 0:
            return False
        return self.load_average[0] > CONTENTION_FRACTION * self.cpu_count

    def describe(self) -> str:
        """One line naming the machine.

        Returns:
            str: Platform, cores, thread pinning.
        """
        return (
            f"{self.platform}, {self.cpu_count} cores, pinned to {self.threads} "
            f"thread(s), torch {self.torch or 'n/a'}, ort {self.onnxruntime or 'n/a'}"
        )


def _version(module_name: str) -> str:
    """Version of an installed module, or empty when it is absent.

    Args:
        module_name (str): Importable name.

    Returns:
        str: Version string, or ``""``.
    """
    try:
        module = __import__(module_name)
    except ImportError:
        return ""
    return str(getattr(module, "__version__", ""))


def _git_sha() -> str:
    """Short commit hash, so a number traces back to its code.

    Returns:
        str: The SHA, or ``"unknown"`` outside a repository.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _load_average() -> tuple[float, float, float]:
    """System load over 1, 5 and 15 minutes.

    Returns:
        tuple[float, float, float]: Load averages, zeros where unavailable.
    """
    try:
        one, five, fifteen = os.getloadavg()
    except (OSError, AttributeError):
        return (0.0, 0.0, 0.0)
    return (one, five, fifteen)


def peak_rss_mb() -> float:
    """Peak resident memory this process has reached, in megabytes.

    A high-water mark, not a current reading, so it never misses a transient
    spike between samples -- and never falls, which is why callers take a
    difference around the region they care about.

    Returns:
        float: Megabytes.
    """
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / RSS_DIVISOR


def artifact_size_bytes(path: Path | str) -> int:
    """Size of a weights file on disk.

    Args:
        path (Path | str): The artifact.

    Returns:
        int: Bytes.

    Raises:
        FileNotFoundError: If the file is absent, rather than reporting zero and
            letting a missing artifact look like a very small one.
    """
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"No artifact at {resolved}.")
    return resolved.stat().st_size


def pin_threads(threads: int) -> None:
    """Pin every framework's thread pool to one count.

    Sets the environment variables **and** the runtime knobs. The environment
    ones only take effect before the libraries load, so this is best called
    early; setting both means a late call still pins torch, which is the one
    that measurably moves.

    Args:
        threads (int): Threads to allow.

    Raises:
        ValueError: If ``threads`` is not positive.
    """
    if threads < 1:
        raise ValueError(f"Need at least one thread, got {threads}.")

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)

    try:
        import torch
    except ImportError:
        return
    torch.set_num_threads(threads)
