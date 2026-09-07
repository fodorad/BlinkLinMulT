"""Tests for the ONNX export gate.

The export itself needs torch, a checkpoint and onnxruntime, so it stays opt-in.
What is tested here is the **gate** -- the probes and the thresholds that decide
whether a produced graph may ship -- because that logic is pure, and because it
has already been wrong once.

That failure is worth recording. The parity check compares the graph against the
checkpoint it was traced from, so when a *checkpoint* loses its input
normalisation the graph faithfully reproduces the loss and every probe passes.
Deliberately removing the transform and re-exporting produced a green run. The
contrast check exists for exactly that case, and these tests pin its threshold.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "export_blinkcnn_onnx.py"
"""The driver under test. Loaded by path: `tools/` is not an importable package."""

HAVE_TORCH = importlib.util.find_spec("torch") is not None
"""The module builds probes with torch, so it cannot import without it."""


def _load():
    """Import the driver module.

    Returns:
        module: The loaded module.
    """
    spec = importlib.util.spec_from_file_location("export_blinkcnn_onnx", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["export_blinkcnn_onnx"] = module
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(HAVE_TORCH, "the export driver needs torch")
class TestProbes(unittest.TestCase):
    """The inputs a candidate graph is checked against."""

    def setUp(self) -> None:
        """Load the driver once per test."""
        self.module = _load()

    def test_every_named_probe_can_be_built(self) -> None:
        """A name in the list that the builder rejects would skip a check."""
        import torch

        generator = torch.Generator().manual_seed(0)
        for kind in self.module.PARITY_PROBES:
            with self.subTest(probe=kind):
                probe = self.module._probe(kind, 15, generator)
                self.assertEqual(tuple(probe.shape), (1, 15, 3, 64, 64))

    def test_the_uniform_probe_covers_the_real_input_domain(self) -> None:
        """Crops arrive as ``[0, 1]``; a graph checked only on Gaussian noise
        is checked on inputs it will never see."""
        import torch

        probe = self.module._probe("uniform", 8, torch.Generator().manual_seed(0))
        self.assertGreaterEqual(float(probe.min()), 0.0)
        self.assertLessEqual(float(probe.max()), 1.0)

    def test_the_constant_probe_is_flat(self) -> None:
        """It exists to make a lost normalisation visible, which needs no spread."""
        import torch

        probe = self.module._probe("constant", 4, torch.Generator().manual_seed(0))
        self.assertAlmostEqual(float(probe.std()), 0.0, places=6)

    def test_an_unknown_probe_is_refused(self) -> None:
        """A typo must fail loudly rather than silently skipping a probe."""
        import torch

        with self.assertRaises(self.module.ExportError):
            self.module._probe("nonesuch", 4, torch.Generator().manual_seed(0))

    def test_probe_lengths_include_one_and_a_non_traced_length(self) -> None:
        """A graph with ``T`` accidentally unrolled still passes at the length
        it was traced with, and fails at any other."""
        self.assertIn(1, self.module.PARITY_LENGTHS)
        self.assertGreater(len(set(self.module.PARITY_LENGTHS)), 1)


@unittest.skipUnless(HAVE_TORCH, "the export driver needs torch")
class TestThresholds(unittest.TestCase):
    """The numbers that decide whether a graph ships."""

    def setUp(self) -> None:
        """Load the driver once per test."""
        self.module = _load()

    def test_the_contrast_floor_separates_the_measured_cases(self) -> None:
        """1.280 for the shipped graph, 0.117 with normalisation removed.

        The floor must sit strictly between them, or the check either passes a
        broken graph or fails a correct one.
        """
        self.assertLess(0.117, self.module.MIN_CONTRAST_SPREAD)
        self.assertLess(self.module.MIN_CONTRAST_SPREAD, 1.280)

    def test_the_parity_tolerances_are_tight(self) -> None:
        """Loose tolerances would pass a graph that is merely close.

        The observed worst deviation is ~2.6e-06, so 1e-3 relative leaves
        headroom without admitting a materially different model.
        """
        self.assertLessEqual(self.module.RTOL, 1e-3)
        self.assertLessEqual(self.module.ATOL, 1e-5)

    def test_the_opset_matches_the_published_graphs(self) -> None:
        """One runtime floor across every shipped model, not two."""
        self.assertEqual(self.module.OPSET, 17)


class TestSidecar(unittest.TestCase):
    """The provenance written beside a produced graph."""

    def test_the_shipped_sidecar_records_its_source_and_parity(self) -> None:
        """A graph that cannot be traced to a checkpoint is not reproducible."""
        import json

        sidecar = Path("artifacts/onnx/blinkcnn.onnx.json")
        if not sidecar.is_file():
            self.skipTest("run `make export-blinkcnn-onnx` first")
        payload = json.loads(sidecar.read_text())
        self.assertEqual(payload["runtime"], "onnx")
        self.assertTrue(payload["source_sha256"])
        self.assertIn("max_deviation", payload["parity"])
        self.assertLess(max(payload["parity"]["max_deviation"].values()), 1e-4)

    def test_the_sidecar_declares_unit_normalisation(self) -> None:
        """The graph standardises internally, so the outer contract is raw crops.

        Declaring ImageNet here would normalise twice: the model would still run
        and still look plausible, which is why it is asserted rather than
        assumed.
        """
        import json

        sidecar = Path("artifacts/onnx/blinkcnn.onnx.json")
        if not sidecar.is_file():
            self.skipTest("run `make export-blinkcnn-onnx` first")
        normalisation = json.loads(sidecar.read_text())["normalisation"]
        self.assertEqual(normalisation["mean"], [0.0, 0.0, 0.0])
        self.assertEqual(normalisation["std"], [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
