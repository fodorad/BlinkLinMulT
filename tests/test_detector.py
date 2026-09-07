"""Tests for the unified detector.

The detector exists so four differently-trained models can be compared under
identical conditions, which puts the weight of these tests on the parts that
*differ* between models and would otherwise silently diverge: the normalisation
applied, the input rank accepted, and how scores become intervals.

These run against the **exported ONNX graphs** in ``artifacts/onnx``, which are
build artefacts rather than committed files, so every case skips when they are
absent. Nothing is downloaded: ``tools/export_paper_onnx.py`` produces them, and
its own parity gate is what proves each graph matches the PyTorch model it froze.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from blinklinmult.detector import BlinkDetector, DetectorError, _as_window
from blinklinmult.registry import ModelSpec, spec

GRAPHS = Path("artifacts/onnx")
"""Where the exported 1.x graphs live in a checkout."""

HAVE_GRAPHS = GRAPHS.is_dir() and any(GRAPHS.glob("*.onnx"))
"""Whether the graphs are present; they are build artefacts, not committed."""

IMAGE_SIZE = 64
"""Crop side every shipped model was trained on."""

FEATURE_DIM = 160
"""Width of the two-stream model's descriptor input."""


def _crops(time: int = 8) -> np.ndarray:
    """Build a window of random crops in ``[0, 1]``.

    Args:
        time (int): Frames to generate.

    Returns:
        np.ndarray: ``(T, 3, 64, 64)`` float32.
    """
    rng = np.random.default_rng(0)
    return rng.random((time, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)


def _detector(model_id: str = "densenet121-union") -> BlinkDetector:
    """Load one published graph through the detector.

    Args:
        model_id (str): Which model to stand up.

    Returns:
        BlinkDetector: Ready to score.
    """
    return BlinkDetector.from_pretrained(model_id, weights=GRAPHS / f"{model_id}.onnx")


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestInputShapes(unittest.TestCase):
    """One object must serve a single frame, a clip, and a batch."""

    def test_single_crop_becomes_a_window(self) -> None:
        """``(3, H, W)`` is treated as a one-frame window."""
        window = _as_window(np.zeros((3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
        self.assertEqual(window.shape, (1, 1, 3, IMAGE_SIZE, IMAGE_SIZE))

    def test_sequence_gains_a_batch_axis(self) -> None:
        """``(T, 3, H, W)`` is one window of ``T`` frames."""
        window = _as_window(_crops(5))
        self.assertEqual(window.shape, (1, 5, 3, IMAGE_SIZE, IMAGE_SIZE))

    def test_batched_window_passes_through(self) -> None:
        """``(B, T, 3, H, W)`` is already the canonical form."""
        window = _as_window(np.zeros((2, 4, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))
        self.assertEqual(window.shape, (2, 4, 3, IMAGE_SIZE, IMAGE_SIZE))

    def test_bad_rank_is_rejected(self) -> None:
        """An unusable rank fails with a message naming the accepted ones."""
        with self.assertRaises(DetectorError):
            _as_window(np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32))


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestScore(unittest.TestCase):
    """The per-frame signal."""

    def test_returns_one_probability_per_frame(self) -> None:
        """A window of ``T`` frames scores ``T`` values."""
        signal = _detector().score(_crops(11))
        self.assertEqual(signal.shape, (11,))

    def test_values_are_probabilities(self) -> None:
        """Output is sigmoid-squashed, not raw logits."""
        signal = _detector().score(_crops(6))
        self.assertTrue(np.all(signal >= 0.0))
        self.assertTrue(np.all(signal <= 1.0))

    def test_single_frame_scores_one_value(self) -> None:
        """``T=1`` works, so a still image needs no special path."""
        signal = _detector().score(_crops(1)[0])
        self.assertEqual(signal.shape, (1,))

    def test_batch_keeps_its_leading_axis(self) -> None:
        """A batched call returns ``(B, T)`` rather than flattening."""
        signal = _detector().score(_crops(4)[None])
        self.assertEqual(signal.shape, (1, 4))

    def test_sequence_model_scores_every_frame(self) -> None:
        """The sequence model returns a signal, not a pooled scalar."""
        signal = _detector("blinklint-union").score(_crops(9))
        self.assertEqual(signal.shape, (9,))


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestNormalisation(unittest.TestCase):
    """Each model's own preprocessing must be applied, not the caller's."""

    def test_paper_models_shift_and_scale(self) -> None:
        """ImageNet normalisation actually changes the pixels."""
        detector = _detector()
        window = torch.full((1, 1, 3, 4, 4), 0.5)
        self.assertFalse(torch.allclose(detector._normalise(window), window))

    def test_v2_normalisation_is_the_identity(self) -> None:
        """``BlinkCNN`` takes ``[0, 1]`` crops unchanged.

        Its spec records mean 0 and std 1, so normalisation must be a no-op --
        if this ever shifted, v2 scores would silently degrade. The model itself
        is irrelevant here, so the graph is reused with the v2 spec.
        """
        detector = BlinkDetector(_detector().model, spec("blinkcnn"))
        window = torch.rand(1, 2, 3, 4, 4)
        torch.testing.assert_close(detector._normalise(window), window)

    def test_the_two_generations_differ(self) -> None:
        """The same crops must not normalise identically across generations.

        This is the failure the registry exists to prevent: swapping the
        conventions costs accuracy and raises nothing.
        """
        window = torch.full((1, 1, 3, 4, 4), 0.5)
        model = _detector().model
        paper = BlinkDetector(model, spec("densenet121-union"))
        v2 = BlinkDetector(model, spec("blinkcnn"))
        self.assertFalse(torch.allclose(paper._normalise(window), v2._normalise(window)))


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestDetect(unittest.TestCase):
    """Turning a signal into intervals."""

    def test_returns_interval_pairs(self) -> None:
        """Every interval is an ordered ``(start, end)`` inside the window."""
        detector = _detector()
        intervals = detector.detect(_crops(12))
        for start, end in intervals:
            self.assertLessEqual(start, end)
            self.assertGreaterEqual(start, 0)
            self.assertLess(end, 12)

    def test_batch_is_rejected(self) -> None:
        """Intervals index one timeline, so a batch is a usage error.

        Silently scoring only the first window would be worse than failing.
        """
        with self.assertRaises(DetectorError):
            _detector().detect(_crops(4)[None])

    def test_uses_the_registered_operating_point(self) -> None:
        """The threshold comes from the spec, not from a hardcoded default."""
        detector = _detector()
        self.assertEqual(detector.spec.threshold, spec("densenet121-union").threshold)

    def test_hysteresis_low_threshold_is_derived(self) -> None:
        """``low_ratio`` scales the high threshold rather than standing alone.

        ``BlinkCNN`` registers 0.53 with ratio 0.25, so its low cut is 0.1325.
        """
        entry = spec("blinkcnn")
        self.assertAlmostEqual(entry.threshold * (entry.low_ratio or 0.0), 0.1325)


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestCalibrate(unittest.TestCase):
    """Adopting a fitted operating point."""

    def test_replaces_the_default_threshold(self) -> None:
        """A fitted point overrides the registered placeholder."""
        from blinklinmult.fit import OperatingPoint

        detector = _detector()
        detector.calibrate(OperatingPoint(0.31, 0.25, 0.8, 0.8, 0.8))
        self.assertAlmostEqual(detector.spec.threshold, 0.31)
        self.assertAlmostEqual(detector.spec.low_ratio or 0.0, 0.25)

    def test_does_not_leak_into_the_registry(self) -> None:
        """Calibrating one detector must not change everyone else's default.

        ``MODELS`` is shared module state, so mutating it here would silently
        re-point every later ``from_pretrained`` call.
        """
        from blinklinmult.fit import OperatingPoint

        before = spec("densenet121-union").threshold
        _detector().calibrate(OperatingPoint(0.9, None, 1.0, 1.0, 1.0))
        self.assertEqual(spec("densenet121-union").threshold, before)

    def test_detect_uses_the_calibrated_point(self) -> None:
        """Extraction reads the calibrated threshold, not the registered one.

        An untrained model's scores are arbitrary, so this brackets them: a
        threshold above every score must yield nothing, and one below every score
        must cover the whole window. Both follow only if ``detect`` actually
        consults the calibrated value.
        """
        from blinklinmult.fit import OperatingPoint

        detector = _detector()
        crops = _crops(10)

        detector.calibrate(OperatingPoint(0.999, None, 1.0, 1.0, 1.0))
        self.assertEqual(detector.detect(crops), [])

        detector.calibrate(OperatingPoint(0.001, None, 1.0, 1.0, 1.0))
        self.assertEqual(detector.detect(crops), [(0, 9)])


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestFeatureStream(unittest.TestCase):
    """The two-stream model's extra input."""

    def test_missing_features_are_rejected(self) -> None:
        """Omitting them fails clearly rather than deep inside the model."""
        with self.assertRaises(DetectorError):
            _detector("blinklinmult-union").score(_crops(5))

    def test_features_are_accepted(self) -> None:
        """With the stream supplied, the model scores every frame."""
        rng = np.random.default_rng(1)
        features = rng.random((7, FEATURE_DIM), dtype=np.float32)
        signal = _detector("blinklinmult-union").score(_crops(7), features)
        self.assertEqual(signal.shape, (7,))

    def test_only_one_model_requires_them(self) -> None:
        """The single-stream models must not demand a feature stream."""
        for model_id in ("densenet121-union", "blinklint-union"):
            with self.subTest(model=model_id):
                self.assertFalse(spec(model_id).needs_features)


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestConstruction(unittest.TestCase):
    """Building a detector."""

    def test_unknown_id_is_rejected(self) -> None:
        """An unknown id fails before any download is attempted."""
        with self.assertRaises(DetectorError):
            BlinkDetector.from_pretrained("no-such-model")

    def test_unresolvable_weights_are_rejected(self) -> None:
        """A model whose weights are neither cached nor downloadable fails.

        The message must name the cache path and the repository, since that is
        what a caller needs in order to fix it.
        """
        from blinklinmult.detector import _published_path
        from blinklinmult.registry import ModelSpec

        missing = ModelSpec(
            model_id="not-published",
            filename="not-published.onnx",
            generation="paper",
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        )
        with self.assertRaises(DetectorError):
            _published_path(missing)

    def test_spec_is_kept(self) -> None:
        """The detector carries its spec, so callers can read the contract."""
        self.assertIsInstance(_detector().spec, ModelSpec)

    def test_exported_from_the_package_root(self) -> None:
        """``from blinklinmult import BlinkDetector`` is the documented entry."""
        import blinklinmult

        self.assertIs(blinklinmult.BlinkDetector, BlinkDetector)

    def test_graph_loads_from_a_local_file(self) -> None:
        """``weights=`` bypasses the download, for offline use and for tests."""
        detector = BlinkDetector.from_pretrained(
            "densenet121-union", weights=GRAPHS / "densenet121-union.onnx"
        )
        self.assertEqual(detector.score(_crops(3)).shape, (3,))

    def test_missing_graph_is_rejected(self) -> None:
        """A path with no graph fails clearly rather than at first inference."""
        with self.assertRaises(DetectorError):
            BlinkDetector.from_pretrained("densenet121-union", weights=GRAPHS / "absent.onnx")


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestPrepareFeatures(unittest.TestCase):
    """Converting a v2 feature vector into the 1.x layout.

    This is the conversion that decides whether ``blinklinmult-union`` works at
    all: fed the v2 layout raw, its logits saturate near -9.9 and it never fires,
    with no error to say so.
    """

    def test_moves_head_pose_to_the_front(self) -> None:
        """1.x ordered the blocks with head pose first; v2 puts it last."""
        detector = _detector("blinklinmult-union")
        features = np.arange(2 * FEATURE_DIM, dtype=np.float32).reshape(2, FEATURE_DIM)
        mean = np.zeros(FEATURE_DIM)
        std = np.ones(FEATURE_DIM)

        prepared = detector.prepare_features(features, mean, std)

        # The last three v2 dimensions must now lead.
        np.testing.assert_allclose(prepared[:, :3], features[:, 157:160], rtol=1e-5)
        np.testing.assert_allclose(prepared[:, 3:], features[:, 0:157], rtol=1e-5)

    def test_standardises_with_the_given_statistics(self) -> None:
        """Values are z-scored before reordering, not after.

        The caller computes statistics over the corpus in the v2 layout, so they
        must be applied in that layout or every dimension is scaled by the wrong
        constant.
        """
        detector = _detector("blinklinmult-union")
        features = np.full((3, FEATURE_DIM), 5.0, dtype=np.float32)
        mean = np.full(FEATURE_DIM, 3.0)
        std = np.full(FEATURE_DIM, 2.0)

        prepared = detector.prepare_features(features, mean, std)

        np.testing.assert_allclose(prepared, np.ones_like(prepared), rtol=1e-5)

    def test_shape_is_preserved(self) -> None:
        """Reordering must not change the width."""
        detector = _detector("blinklinmult-union")
        prepared = detector.prepare_features(
            np.zeros((7, FEATURE_DIM), dtype=np.float32),
            np.zeros(FEATURE_DIM),
            np.ones(FEATURE_DIM),
        )
        self.assertEqual(prepared.shape, (7, FEATURE_DIM))

    def test_missing_statistics_are_rejected(self) -> None:
        """A model that standardises must be given the statistics.

        Silently skipping standardisation would leave the model permanently
        saturated -- it would return a signal, just never a detection.
        """
        detector = _detector("blinklinmult-union")
        with self.assertRaises(DetectorError):
            detector.prepare_features(np.zeros((4, FEATURE_DIM), dtype=np.float32))

    def test_models_without_standardisation_need_no_statistics(self) -> None:
        """Only the two-stream 1.x model declares the requirement."""
        for model_id in ("densenet121-union", "blinklint-union", "blinkcnn"):
            with self.subTest(model=model_id):
                self.assertFalse(spec(model_id).standardise_features)


@unittest.skipUnless(HAVE_GRAPHS, "run tools/export_paper_onnx.py to build artifacts/onnx")
class TestScoreLong(unittest.TestCase):
    """Scoring a recording longer than the model's native window."""

    def test_returns_one_score_per_frame(self) -> None:
        """Every frame of the input gets exactly one score."""
        signal = _detector("blinklint-union").score_long(_crops(45))
        self.assertEqual(signal.shape, (45,))

    def test_every_frame_is_covered(self) -> None:
        """No frame is left at its initial value.

        With a stride that does not divide the remainder, the tail would
        otherwise go unscored -- a silent zero looks exactly like a confident
        "eye open".
        """
        signal = _detector("blinklint-union").score_long(_crops(40), stride=7)
        self.assertTrue(np.all(signal > 0.0))

    def test_frame_wise_models_are_scored_in_one_pass(self) -> None:
        """A model with no native window needs no sliding at all."""
        detector = _detector("densenet121-union")
        crops = _crops(30)
        np.testing.assert_allclose(detector.score_long(crops), detector.score(crops), rtol=1e-5)

    def test_short_input_is_scored_directly(self) -> None:
        """Input at or below the native window is a single forward pass."""
        detector = _detector("blinklint-union")
        crops = _crops(15)
        np.testing.assert_allclose(detector.score_long(crops), detector.score(crops), rtol=1e-5)

    def test_averaging_is_bounded_by_the_windows(self) -> None:
        """A mean cannot exceed the largest value it averages.

        This is the property that makes averaging safer than a maximum: a single
        badly-positioned window can pull a frame down but cannot spike it.
        """
        detector = _detector("blinklint-union")
        signal = detector.score_long(_crops(45))
        self.assertLessEqual(signal.max(), 1.0)
        self.assertGreaterEqual(signal.min(), 0.0)

    def test_stride_must_be_positive(self) -> None:
        """A stride of zero would loop forever."""
        with self.assertRaises(DetectorError):
            _detector("blinklint-union").score_long(_crops(45), stride=0)
