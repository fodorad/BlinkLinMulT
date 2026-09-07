"""Tests for loading trained v2 models without the training stack.

Two claims are worth pinning here.

The first is **isolation**: importing this module must not drag in ``lightning``,
``torchmetrics``, ``h5py``, ``omniloader`` or ``pandas``. Those five come along
with :mod:`blinklinmult.train.module`, which is why the rebuild exists at all, and
an innocent-looking import added later would quietly undo it.

The second is that a checkpoint must load **strictly** or not at all. A model
rebuilt from the wrong hyperparameters still runs; it just returns nonsense. So
the failure paths are tested as carefully as the success one.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from blinklinmult.models import DEFAULT_TARGETS, ModelLoadError, load_checkpoint, strip_checkpoint

HEAVY_PACKAGES = ("lightning", "torchmetrics", "h5py", "omniloader", "pandas")
"""Packages the inference path must not import."""


class TestStripCheckpoint(unittest.TestCase):
    """Reducing a training checkpoint to its inference payload."""

    def test_keeps_only_what_inference_needs(self) -> None:
        """Optimizer, loop and callback state are dropped."""
        stripped = strip_checkpoint(
            {
                "state_dict": {"a": torch.zeros(2)},
                "hyper_parameters": {"image_size": 64},
                "optimizer_states": [{"heavy": torch.zeros(1000)}],
                "loops": {"fit": "state"},
                "callbacks": {"ckpt": "state"},
            }
        )
        self.assertEqual(set(stripped), {"state_dict", "hyper_parameters"})

    def test_missing_state_dict_is_rejected(self) -> None:
        """A file without weights is not a checkpoint."""
        with self.assertRaises(ModelLoadError):
            strip_checkpoint({"hyper_parameters": {}})

    def test_missing_hyper_parameters_is_rejected(self) -> None:
        """Without them the architecture cannot be rebuilt."""
        with self.assertRaises(ModelLoadError):
            strip_checkpoint({"state_dict": {}})


class TestLoadCheckpoint(unittest.TestCase):
    """Rebuilding a model from a checkpoint file."""

    def test_missing_file_is_reported(self) -> None:
        """A wrong path fails immediately, naming the path."""
        with self.assertRaises(ModelLoadError):
            load_checkpoint("/nonexistent/checkpoint.ckpt")

    def test_checkpoint_without_hyper_parameters_is_rejected(self) -> None:
        """A bare ``state_dict`` carries no architecture to rebuild."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bare.ckpt"
            torch.save({"state_dict": {"weight": torch.zeros(2)}}, path)
            with self.assertRaises(ModelLoadError):
                load_checkpoint(path)

    def test_checkpoint_without_model_config_is_rejected(self) -> None:
        """Hyperparameters lacking ``model_config`` cannot describe a model."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.ckpt"
            torch.save({"state_dict": {}, "hyper_parameters": {"image_size": 64}}, path)
            with self.assertRaises(ModelLoadError):
                load_checkpoint(path)

    def test_default_targets_are_eye_state(self) -> None:
        """A checkpoint that records no heads is assumed frame-wise."""
        self.assertEqual(DEFAULT_TARGETS, ["eye_state"])

    def test_round_trips_a_real_checkpoint(self) -> None:
        """A model saved the way training saves one loads back and runs.

        Built here rather than read from ``results/``: the fixture has to work on
        a fresh clone with no trained artifacts. A tiny untrained ``cnn`` model is
        enough, since what is under test is the rebuild-and-load path, not the
        weights.
        """
        from blinklinmult.train.config import ModelConfig
        from blinklinmult.train.model import build_model

        config = ModelConfig(family="cnn", backbone="convnext_femto", backbone_pretrained=False)
        model = build_model(config, target_names=["eye_state"], image_size=64)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "round_trip.ckpt"
            torch.save(
                {
                    # The "model." prefix is what Lightning adds around the
                    # wrapped network, so the loader has to strip it.
                    "state_dict": {f"model.{k}": v for k, v in model.state_dict().items()},
                    "hyper_parameters": {
                        "model_config": config.__dict__,
                        "image_size": 64,
                        "eye_feature_dim": None,
                    },
                },
                path,
            )
            loaded = load_checkpoint(path)

        self.assertFalse(loaded.training, "a loaded model must be in eval mode")
        with torch.no_grad():
            out = loaded(torch.randn(1, 3, 3, 64, 64), torch.ones(1, 3, dtype=torch.bool))
        self.assertEqual(out["eye_state"].shape, (1, 3, 1))

    def test_mismatched_weights_are_rejected(self) -> None:
        """Weights that do not fit the described architecture must raise.

        Loading loosely would return a model that runs and reports nonsense,
        which is worse than failing.
        """
        from blinklinmult.train.config import ModelConfig

        config = ModelConfig(family="cnn", backbone="convnext_femto", backbone_pretrained=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mismatch.ckpt"
            torch.save(
                {
                    "state_dict": {"model.not_a_real_layer.weight": torch.zeros(3)},
                    "hyper_parameters": {"model_config": config.__dict__, "image_size": 64},
                },
                path,
            )
            with self.assertRaises(ModelLoadError):
                load_checkpoint(path)


class TestImportIsolation(unittest.TestCase):
    """The inference path must stay free of the training stack."""

    def test_inference_modules_avoid_heavy_imports(self) -> None:
        """Importing the detector must not pull the training dependencies.

        Run in a subprocess: this test process has already imported half the
        project, so checking ``sys.modules`` in-process would prove nothing.
        """
        code = (
            "import sys; import blinklinmult.detector; "
            f"print([p for p in {HEAVY_PACKAGES!r} if p in sys.modules])"
        )
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]", "detector pulled a training dependency")

    def test_paper_models_avoid_heavy_imports(self) -> None:
        """The 1.x models must not need the training stack either."""
        code = (
            "import sys; import blinklinmult.paper; "
            f"print([p for p in {HEAVY_PACKAGES!r} if p in sys.modules])"
        )
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "[]", "paper models pulled a training dependency")

    def test_paper_models_avoid_installed_linmult(self) -> None:
        """The vendored copy must be used, never the installed ``linmult`` 2.x.

        Loading a 1.x checkpoint into a 2.x architecture would fail outright, so
        this guards the whole reason for vendoring.
        """
        code = "import sys; import blinklinmult.paper; print('linmult' in sys.modules)"
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()


class TestRetiredOneXNames(unittest.TestCase):
    """``blinklinmult.models`` changed meaning between 1.x and 2.0.

    It was a *package* exporting the three paper models; it is now a *module*
    that loads v2 checkpoints. Same import path, unrelated contents -- so an old
    script deserves to be told where the models went rather than getting a bare
    "cannot import name".
    """

    def test_a_retired_name_names_its_replacement(self) -> None:
        """The error carries the new API and the model id."""
        import blinklinmult.models as module

        with self.assertRaises(AttributeError) as caught:
            _ = module.DenseNet121
        message = str(caught.exception)
        self.assertIn("BlinkDetector", message)
        self.assertIn("densenet121-union", message)

    def test_every_retired_name_is_covered(self) -> None:
        """All five 1.x exports, not just the one that happened to be tested."""
        import blinklinmult.models as module
        from blinklinmult.models import RETIRED_1X_MODELS

        for name in RETIRED_1X_MODELS:
            with self.subTest(name=name), self.assertRaises(AttributeError) as caught:
                getattr(module, name)
            self.assertIn("BlinkDetector", str(caught.exception))

    def test_an_unknown_name_stays_generic(self) -> None:
        """A typo should not be told about ONNX graphs it never asked for."""
        import blinklinmult.models as module

        with self.assertRaises(AttributeError) as caught:
            _ = module.definitely_not_a_real_attribute
        self.assertNotIn("BlinkDetector", str(caught.exception))
