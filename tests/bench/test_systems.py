"""Tests for the environment block.

The load-average and RSS readings depend on the host, so these check the parts
that must hold anywhere: that the platform-dependent memory unit is handled,
that a missing artifact raises rather than reporting zero bytes, and that the
contention flag actually fires below saturation -- which is the setting that
caught a 1.7x distortion during development.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from blinklinmult.bench.systems import (
    CONTENTION_FRACTION,
    RSS_DIVISOR,
    MachineSpec,
    artifact_size_bytes,
    peak_rss_mb,
    pin_threads,
)


def _spec(load: float, cores: int = 10) -> MachineSpec:
    """Build a spec with a chosen load.

    Args:
        load (float): One-minute load average.
        cores (int): Logical cores.

    Returns:
        MachineSpec: A spec for testing the contention rule.
    """
    return MachineSpec(
        platform="test",
        processor="test",
        cpu_count=cores,
        python="3.13",
        torch="",
        onnxruntime="",
        numpy="2.0",
        threads=4,
        git_sha="abc1234",
        load_average=(load, load, load),
    )


class TestPeakRss(unittest.TestCase):
    """Reading resident memory."""

    def test_it_reports_a_positive_figure(self) -> None:
        """A running process occupies memory."""
        self.assertGreater(peak_rss_mb(), 0.0)

    def test_it_never_decreases(self) -> None:
        """It is a high-water mark, so a later reading cannot be lower."""
        first = peak_rss_mb()
        _ = [0] * 100000
        self.assertGreaterEqual(peak_rss_mb(), first)

    def test_the_divisor_matches_the_platform(self) -> None:
        """macOS reports bytes and Linux kibibytes, with nothing warning.

        Getting this wrong yields a figure 1024x off in either direction, in a
        field a reader cannot sanity-check.
        """
        import sys

        self.assertEqual(RSS_DIVISOR, 1048576 if sys.platform == "darwin" else 1024)

    def test_the_reading_is_plausible(self) -> None:
        """A Python process with torch loaded is megabytes, not gigabytes."""
        self.assertLess(peak_rss_mb(), 100000.0)


class TestArtifactSize(unittest.TestCase):
    """Measuring a weights file."""

    def test_it_reports_the_byte_count(self) -> None:
        """The number that goes in the size column."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights.bin"
            path.write_bytes(b"x" * 4096)
            self.assertEqual(artifact_size_bytes(path), 4096)

    def test_a_missing_file_raises(self) -> None:
        """Reporting zero would make an absent artifact look very small."""
        with self.assertRaises(FileNotFoundError):
            artifact_size_bytes("/nonexistent/weights.onnx")

    def test_a_directory_is_not_an_artifact(self) -> None:
        """A path that exists but is not a file must still raise."""
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            artifact_size_bytes(directory)


class TestContention(unittest.TestCase):
    """Recognising a busy machine."""

    def test_an_idle_machine_is_not_contended(self) -> None:
        """Nothing to warn about."""
        self.assertFalse(_spec(0.4).contended)

    def test_load_below_saturation_still_counts(self) -> None:
        """Load 7 on 10 cores inflated measured latency ~1.7x.

        A threshold at full saturation would have called that idle, which is
        the mistake this fraction exists to prevent.
        """
        self.assertTrue(_spec(7.0, cores=10).contended)

    def test_the_boundary_is_the_fraction(self) -> None:
        """Just under passes, just over flags."""
        cores = 10
        self.assertFalse(_spec(CONTENTION_FRACTION * cores - 0.1, cores).contended)
        self.assertTrue(_spec(CONTENTION_FRACTION * cores + 0.1, cores).contended)

    def test_an_unknown_core_count_never_flags(self) -> None:
        """Without a core count there is nothing to compare load against."""
        self.assertFalse(_spec(99.0, cores=0).contended)


class TestMachineSpec(unittest.TestCase):
    """Capturing the environment."""

    def test_capture_records_the_thread_pinning(self) -> None:
        """A latency without its thread count is not reproducible."""
        self.assertEqual(MachineSpec.capture(threads=3).threads, 3)

    def test_capture_records_versions(self) -> None:
        """numpy is always present, so its version must be recorded."""
        self.assertTrue(MachineSpec.capture(threads=1).numpy)

    def test_describe_names_the_machine(self) -> None:
        """The line printed above every benchmark table."""
        text = MachineSpec.capture(threads=2).describe()
        self.assertIn("cores", text)
        self.assertIn("thread", text)


class TestPinThreads(unittest.TestCase):
    """Fixing the thread pool."""

    def test_it_sets_the_environment(self) -> None:
        """Frameworks read these, though only before they load."""
        import os

        pin_threads(2)
        self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")

    def test_it_sets_the_torch_pool(self) -> None:
        """The knob that measurably moved blinkcnn: 469/199/229 ms at 1/4/8."""
        import torch

        pin_threads(3)
        self.assertEqual(torch.get_num_threads(), 3)

    def test_a_non_positive_count_is_refused(self) -> None:
        """Zero threads is not a configuration."""
        with self.assertRaises(ValueError):
            pin_threads(0)


class TestFallbacks(unittest.TestCase):
    """The paths taken when the environment cannot answer.

    Exercised with real inputs rather than patches: an absent module really is
    absent, and a load average really is unavailable on some platforms. These
    exist because a benchmark that crashes while *describing itself* is worse
    than one that reports an unknown.
    """

    def test_an_absent_module_reports_no_version(self) -> None:
        """torch and onnxruntime are optional in a minimal install."""
        from blinklinmult.bench.systems import _version

        self.assertEqual(_version("definitely_not_an_installed_module"), "")

    def test_an_installed_module_reports_its_version(self) -> None:
        """The path that normally runs."""
        from blinklinmult.bench.systems import _version

        self.assertTrue(_version("numpy"))

    def test_load_average_returns_three_numbers(self) -> None:
        """Zeros where the platform does not offer it, never an exception."""
        from blinklinmult.bench.systems import _load_average

        load = _load_average()
        self.assertEqual(len(load), 3)
        self.assertTrue(all(value >= 0.0 for value in load))

    def test_git_sha_is_always_a_string(self) -> None:
        """ "unknown" outside a repository, never a crash."""
        from blinklinmult.bench.systems import _git_sha

        self.assertIsInstance(_git_sha(), str)
        self.assertTrue(_git_sha())


if __name__ == "__main__":
    unittest.main()
