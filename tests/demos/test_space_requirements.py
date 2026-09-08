"""Tests that the Hugging Face Space's dependency set matches what it runs.

The Space broke three times in a row, each on a package reached only by the
PyTorch checkpoint path -- exordium/timm, then ``linmult``, then ``omniloader``.
Every failure looked different in the logs and had the same shape: a module the
demo needed at runtime that its ``requirements.txt`` never named.

Nothing caught them because the Space is the one published surface with no CI
behind it. A build takes minutes and a failure only shows up after a deploy, so
these are text and registry checks instead -- they run in milliseconds and would
have caught all three.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from blinklinmult.registry import spec

REQUIREMENTS = Path("demos/gradio/requirements.txt")
"""The Space's dependency list."""

APP = Path("demos/gradio/app.py")
"""The Space's entry point, read as text rather than imported."""

TRAINING_ONLY = ("linmult", "omniloader", "lightning", "mlflow", "pandas")
"""Packages reached only by the PyTorch checkpoint path.

Naming any of these means the demo is loading a ``.pt`` checkpoint again, which
is what pulled the training stack onto a Space that only needs to run graphs.
"""


def _model_labels() -> dict[str, str]:
    """Read ``MODEL_LABELS`` from the app's source, without importing it.

    Parsed rather than imported because importing ``app.py`` pulls gradio, which
    lives in the ``demo`` extra that CI does not install -- these tests exist to
    guard the Space's *dependency set*, so needing a dependency to run them
    would defeat the point. The mapping is a plain literal, so ``ast`` reads it
    exactly.

    Returns:
        dict[str, str]: Display name to registry id.

    Raises:
        AssertionError: If the literal is missing or is no longer a plain dict.
    """
    tree = ast.parse(APP.read_text())
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "MODEL_LABELS" in names:
            return ast.literal_eval(node.value)
    raise AssertionError(f"MODEL_LABELS not found as a literal in {APP}")


def _requirement_names() -> set[str]:
    """The package names the Space installs, ignoring comments and extras.

    Returns:
        set[str]: Lowercased distribution names.
    """
    names = set()
    for raw in REQUIREMENTS.read_text().splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        name = line.split("[")[0].split("=")[0].split(">")[0].split("<")[0].split("@")[0]
        names.add(name.strip().lower())
    return names


class TestSpaceModels(unittest.TestCase):
    """Every model the Space offers must be runnable with what it installs."""

    def test_every_offered_model_is_a_real_registry_id(self) -> None:
        """A typo in the label map surfaces on a user's click, not at import."""
        for label, model_id in _model_labels().items():
            with self.subTest(label=label):
                self.assertEqual(spec(model_id).model_id, model_id)

    def test_every_offered_model_is_an_onnx_graph(self) -> None:
        """The Space installs no training stack, so a checkpoint cannot load.

        This is the assertion that would have caught all three build failures:
        each began with a ``.pt`` model whose rebuild path reached a package the
        Space had never installed.
        """
        for label, model_id in _model_labels().items():
            with self.subTest(label=label):
                self.assertEqual(spec(model_id).runtime, "onnx")

    def test_the_labels_are_the_published_names(self) -> None:
        """The dropdown shows model names, not storage ids.

        ``-onnx`` and ``-union`` are deployment detail; a reader looking for the
        paper's models should find the paper's names.
        """
        self.assertEqual(
            list(_model_labels()),
            ["BlinkCNN", "BlinkDenseNet121", "BlinkLinT", "BlinkLinMulT"],
        )


class TestSpaceRequirements(unittest.TestCase):
    """What the Space installs, and what it must not."""

    def test_the_file_exists(self) -> None:
        """A missing list means the Space builds from whatever pip guesses."""
        self.assertTrue(REQUIREMENTS.is_file(), f"{REQUIREMENTS} is missing")

    def test_no_training_only_package_is_named(self) -> None:
        """Their presence means the demo went back to a checkpoint model."""
        named = _requirement_names()
        for package in TRAINING_ONLY:
            with self.subTest(package=package):
                self.assertNotIn(package, named)

    def test_the_library_is_pinned_to_an_exact_version(self) -> None:
        """Hugging Face decides when to rebuild, so an unpinned install ships
        whatever ``main`` holds at that moment -- which is how the Space first
        broke.
        """
        line = next(
            raw
            for raw in REQUIREMENTS.read_text().splitlines()
            if raw.strip().startswith("blinklinmult")
        )
        self.assertIn("==", line, "the Space must pin an exact released version")

    def test_onnxruntime_is_installed(self) -> None:
        """Every offered model is a graph, so the runtime is not optional."""
        self.assertIn("blinklinmult", _requirement_names())
        line = next(
            raw
            for raw in REQUIREMENTS.read_text().splitlines()
            if raw.strip().startswith("blinklinmult")
        )
        self.assertIn("onnx", line)


if __name__ == "__main__":
    unittest.main()
