"""Tests for the ONNX session wrapper.

These graphs are what the 1.x models *are* now -- there is no PyTorch fallback --
so the wrapper's contract matters: which inputs a graph declares, what it returns,
and that a missing required input fails loudly rather than at inference time.

The graphs are build artefacts (``tools/export_paper_onnx.py``), not committed
files, so every case skips when ``artifacts/onnx`` is absent.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from blinklinmult.paper import FEATURE_DIM, GRAPH_FILES, SessionError, load_onnx

GRAPHS = Path("artifacts/onnx")
"""Where the exported graphs live in a checkout."""

HAVE_GRAPHS = GRAPHS.is_dir() and any(GRAPHS.glob("*.onnx"))
"""Whether the graphs are present."""

IMAGE_SIZE = 64
"""Crop side the 1.x models were trained on."""


def _window(time: int = 8, batch: int = 1) -> np.ndarray:
    """Build a random normalised window.

    Args:
        time (int): Frames.
        batch (int): Windows.

    Returns:
        np.ndarray: ``(B, T, 3, 64, 64)`` float32.
    """
    rng = np.random.default_rng(0)
    return rng.random((batch, time, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestSingleStream(unittest.TestCase):
    """The two graphs taking crops alone."""

    def test_scores_every_frame(self) -> None:
        """Output carries one logit per input frame."""
        model = load_onnx(GRAPHS / "densenet121-union.onnx")
        out = model(_window(6))
        self.assertEqual(out["sequence"].shape, (1, 6, 1))

    def test_time_axis_is_dynamic(self) -> None:
        """One graph serves any window length.

        The 1.x forward pass looped over ``T`` in Python, which would have
        unrolled at export and pinned the length. Several lengths -- including
        ones the export never traced -- are the check that it did not.
        """
        model = load_onnx(GRAPHS / "blinklint-union.onnx")
        for time in (1, 7, 23, 45):
            with self.subTest(time=time):
                self.assertEqual(model(_window(time))["sequence"].shape, (1, time, 1))

    def test_batch_axis_is_dynamic(self) -> None:
        """Several windows can be scored in one call."""
        model = load_onnx(GRAPHS / "densenet121-union.onnx")
        self.assertEqual(model(_window(4, batch=3))["sequence"].shape, (3, 4, 1))

    def test_does_not_declare_a_feature_input(self) -> None:
        """Only the two-stream graph takes descriptors."""
        self.assertFalse(load_onnx(GRAPHS / "densenet121-union.onnx").needs_features)


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestTwoStream(unittest.TestCase):
    """The graph taking crops plus the 160-d descriptors."""

    def test_declares_a_feature_input(self) -> None:
        """The wrapper reports the second input from the graph itself."""
        self.assertTrue(load_onnx(GRAPHS / "blinklinmult-union.onnx").needs_features)

    def test_returns_both_heads(self) -> None:
        """A pooled clip logit comes back alongside the per-frame sequence.

        1.x built this model with ``aggregation='meanpooling'``, so the clip head
        exists beside the sequence rather than instead of it. Exporting only one
        would silently narrow what the published model can answer.
        """
        model = load_onnx(GRAPHS / "blinklinmult-union.onnx")
        rng = np.random.default_rng(1)
        out = model(_window(9), rng.random((1, 9, FEATURE_DIM), dtype=np.float32))
        self.assertEqual(out["sequence"].shape, (1, 9, 1))
        self.assertEqual(out["clip"].shape, (1, 1))

    def test_missing_features_are_rejected(self) -> None:
        """Omitting the required input fails with a usable message."""
        model = load_onnx(GRAPHS / "blinklinmult-union.onnx")
        with self.assertRaises(SessionError):
            model(_window(5))


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestMetadata(unittest.TestCase):
    """The sidecar written beside each graph."""

    def test_records_the_normalisation(self) -> None:
        """The constants travel with the graph, not with the caller.

        v1 expects ImageNet standardisation and v2 plain ``/255``; applying the
        wrong one costs accuracy and raises nothing, so it must be recorded.
        """
        model = load_onnx(GRAPHS / "densenet121-union.onnx")
        self.assertIn("normalisation", model.metadata)
        self.assertEqual(len(model.metadata["normalisation"]["mean"]), 3)

    def test_records_its_source_weights(self) -> None:
        """Provenance back to the published 1.x checkpoint."""
        model = load_onnx(GRAPHS / "blinklint-union.onnx")
        self.assertTrue(model.metadata.get("source_weights"))

    def test_records_the_parity_gate(self) -> None:
        """Which lengths the graph was verified at, and to what tolerance."""
        model = load_onnx(GRAPHS / "densenet121-union.onnx")
        self.assertIn("parity", model.metadata)
        self.assertGreater(len(model.metadata["parity"]["lengths"]), 1)


class TestLoading(unittest.TestCase):
    """Opening a graph file."""

    def test_missing_file_is_rejected(self) -> None:
        """A wrong path fails immediately, naming it."""
        with self.assertRaises(SessionError):
            load_onnx(GRAPHS / "no-such-graph.onnx")

    def test_every_model_has_a_graph_filename(self) -> None:
        """The published filenames cover all three 1.x models."""
        self.assertEqual(len(GRAPH_FILES), 3)
        for name in GRAPH_FILES.values():
            self.assertTrue(name.endswith(".onnx"))


if __name__ == "__main__":
    unittest.main()
