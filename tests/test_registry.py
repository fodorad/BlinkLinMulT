"""Tests for the published-model registry.

The registry's job is to keep each model's *input contract* attached to its id.
The normalisation entries carry the most risk: the 1.x models were trained on
ImageNet-standardised crops and ``BlinkCNN`` on plain ``/255``, and applying the
wrong one costs accuracy **without raising anything**. A silent failure needs a
test, so the constants are asserted per generation rather than trusted.
"""

from __future__ import annotations

import unittest

from blinklinmult.registry import (
    HF_MODEL_REPO,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MODELS,
    UNIT_MEAN,
    UNIT_STD,
    ModelSpec,
    spec,
)


class TestLookup(unittest.TestCase):
    """Resolving an id to its specification."""

    def test_returns_the_matching_spec(self) -> None:
        """A known id gives back its own entry."""
        entry = spec("blinkcnn")
        self.assertIsInstance(entry, ModelSpec)
        self.assertEqual(entry.model_id, "blinkcnn")

    def test_unknown_id_raises(self) -> None:
        """An unknown id fails loudly, naming the valid options."""
        with self.assertRaises(KeyError) as caught:
            spec("no-such-model")
        self.assertIn("blinkcnn", str(caught.exception))

    def test_ids_match_their_keys(self) -> None:
        """Each entry's ``model_id`` agrees with the key it is stored under.

        A mismatch would make lookups and round-trips disagree.
        """
        for key, entry in MODELS.items():
            with self.subTest(model=key):
                self.assertEqual(key, entry.model_id)


class TestNormalisation(unittest.TestCase):
    """Each generation must carry its own preprocessing constants."""

    def test_paper_models_use_imagenet(self) -> None:
        """The 1.x models were trained on ImageNet-standardised crops."""
        for key, entry in MODELS.items():
            if entry.generation != "paper":
                continue
            with self.subTest(model=key):
                self.assertEqual(entry.mean, IMAGENET_MEAN)
                self.assertEqual(entry.std, IMAGENET_STD)

    def test_v2_models_use_unit_scaling(self) -> None:
        """``BlinkCNN`` takes crops already scaled to ``[0, 1]``, unshifted."""
        for key, entry in MODELS.items():
            if entry.generation != "v2":
                continue
            with self.subTest(model=key):
                self.assertEqual(entry.mean, UNIT_MEAN)
                self.assertEqual(entry.std, UNIT_STD)

    def test_the_two_conventions_differ(self) -> None:
        """The whole point of storing normalisation per model.

        If these ever became equal the registry would be redundant -- and, worse,
        a swap would stop being detectable.
        """
        self.assertNotEqual(IMAGENET_MEAN, UNIT_MEAN)
        self.assertNotEqual(IMAGENET_STD, UNIT_STD)


class TestOperatingPoints(unittest.TestCase):
    """The fitted thresholds that turn scores into events."""

    def test_blinkcnn_ships_its_hysteresis_fit(self) -> None:
        """``BlinkCNN`` must carry both thresholds, not just the high one.

        Its validation-fitted hysteresis pair scores 0.52 event F1 where the
        single-cut variant of the same run scores 0.19, so shipping the high
        threshold alone would quietly halve event quality.
        """
        entry = spec("blinkcnn")
        self.assertAlmostEqual(entry.threshold, 0.53)
        self.assertIsNotNone(entry.low_ratio)
        self.assertAlmostEqual(entry.low_ratio or 0.0, 0.25)

    def test_thresholds_are_probabilities(self) -> None:
        """Every threshold sits inside ``(0, 1)``, as a sigmoid output must."""
        for key, entry in MODELS.items():
            with self.subTest(model=key):
                self.assertGreater(entry.threshold, 0.0)
                self.assertLess(entry.threshold, 1.0)

    def test_low_ratio_is_a_fraction(self) -> None:
        """A hysteresis low threshold must sit below the high one.

        ``low_ratio`` multiplies ``threshold``; at 1.0 or above the low cut stops
        extending runs and hysteresis silently degrades to a single threshold.
        """
        for key, entry in MODELS.items():
            if entry.low_ratio is None:
                continue
            with self.subTest(model=key):
                self.assertGreater(entry.low_ratio, 0.0)
                self.assertLess(entry.low_ratio, 1.0)


class TestContract(unittest.TestCase):
    """Structural facts the loading code relies on."""

    def test_only_the_two_stream_model_needs_features(self) -> None:
        """The 160-d descriptor stream belongs to ``BlinkLinMulT`` alone.

        It is the only model whose use requires the exordium extraction path, so
        this flag decides whether that heavy optional dependency is needed.
        """
        needing = {key for key, entry in MODELS.items() if entry.needs_features}
        self.assertEqual(needing, {"blinklinmult-union"})

    def test_filenames_are_unique(self) -> None:
        """Two models sharing a filename would overwrite each other's cache."""
        names = [entry.filename for entry in MODELS.values()]
        self.assertEqual(len(names), len(set(names)))

    def test_generations_are_known(self) -> None:
        """Every entry declares which generation it belongs to."""
        for key, entry in MODELS.items():
            with self.subTest(model=key):
                self.assertIn(entry.generation, {"paper", "v2"})

    def test_every_model_is_described(self) -> None:
        """Descriptions surface in the model card and the demo's dropdown."""
        for key, entry in MODELS.items():
            with self.subTest(model=key):
                self.assertTrue(entry.description)

    def test_repo_is_the_public_one(self) -> None:
        """Weights resolve from the public model repo, never the private data one."""
        self.assertEqual(HF_MODEL_REPO, "fodorad/blink_detection")

    def test_specs_are_immutable(self) -> None:
        """A frozen dataclass keeps a caller from mutating the shared table."""
        entry = spec("blinkcnn")
        with self.assertRaises(AttributeError):
            entry.threshold = 0.9  # type: ignore[misc]


class TestRuntime(unittest.TestCase):
    """``runtime`` decides which loader runs, so it must not drift.

    It is a separate field from ``generation`` because ``blinkcnn`` ships in
    both runtimes: the checkpoint is the trainable artifact, the graph the
    deployable one. These tests hold the two consistent.
    """

    SUFFIXES = {"onnx": ".onnx", "pytorch": ".pt"}
    """The file extension each runtime loads."""

    def test_every_runtime_is_known(self) -> None:
        """A typo would fall through to the checkpoint loader and confuse."""
        for key, entry in MODELS.items():
            with self.subTest(model=key):
                self.assertIn(entry.runtime, self.SUFFIXES)

    def test_runtime_matches_the_filename(self) -> None:
        """A ``.onnx`` file cannot be loaded as a checkpoint, or the reverse.

        The failure this prevents is quiet at import and loud at first use, so
        it is worth catching in the table rather than in a user's session.
        """
        for key, entry in MODELS.items():
            with self.subTest(model=key):
                self.assertTrue(entry.filename.endswith(self.SUFFIXES[entry.runtime]))

    def test_generation_no_longer_implies_runtime(self) -> None:
        """The two v2 entries share a generation and differ in runtime.

        This is the case that forced the fields apart; if a refactor ever
        collapses them again, this is what fails.
        """
        self.assertEqual(spec("blinkcnn").generation, spec("blinkcnn-onnx").generation)
        self.assertNotEqual(spec("blinkcnn").runtime, spec("blinkcnn-onnx").runtime)

    def test_the_onnx_twin_keeps_unit_normalisation(self) -> None:
        """The graph normalises internally, so the outer contract is raw crops.

        Switching this to ImageNet statistics would normalise twice. The model
        would still run and still look plausible, which is exactly why it is
        pinned here.
        """
        self.assertEqual(spec("blinkcnn-onnx").mean, spec("blinkcnn").mean)
        self.assertEqual(spec("blinkcnn-onnx").std, spec("blinkcnn").std)

    def test_the_twins_share_an_operating_point(self) -> None:
        """Same weights, so a threshold fitted for one holds for the other."""
        for field in ("threshold", "low_ratio", "window", "image_size"):
            with self.subTest(field=field):
                self.assertEqual(
                    getattr(spec("blinkcnn"), field), getattr(spec("blinkcnn-onnx"), field)
                )


if __name__ == "__main__":
    unittest.main()


class TestNativeWindow(unittest.TestCase):
    """The window length each model was trained on.

    Getting this wrong is silent and expensive: the 1.x sequence models accept
    any length but lose a lot of accuracy off their native one -- measured on
    RN30, 72% -> 92% and 52% -> 84% localisation when the window is corrected.
    """

    def test_sequence_models_declare_fifteen_frames(self) -> None:
        """1.x trained its sequence models on ~0.5 s at 25 fps."""
        for model_id in ("blinklint-union", "blinklinmult-union"):
            with self.subTest(model=model_id):
                self.assertEqual(spec(model_id).window, 15)

    def test_frame_wise_models_declare_no_window(self) -> None:
        """A frame-wise model scores each frame alone, so any length is fine."""
        for model_id in ("densenet121-union", "blinkcnn"):
            with self.subTest(model=model_id):
                self.assertIsNone(spec(model_id).window)

    def test_declared_windows_are_positive(self) -> None:
        """A window of zero or fewer frames is meaningless."""
        for key, entry in MODELS.items():
            if entry.window is None:
                continue
            with self.subTest(model=key):
                self.assertGreater(entry.window, 0)
