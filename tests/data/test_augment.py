"""Tests for eye-crop augmentation.

Two properties carry most of the weight. **Consistency across a window**: a
sample is one eye's whole sequence, and geometry and lighting are continuous in
time, so per-frame variation would manufacture exactly the chatter that
`box_jitter` measures and `temporal_smoothness` penalises. And
**reproducibility**: OmniLoader seeds from `(seed, epoch, index)` so a run can
be repeated, which only holds if nothing here draws from global state.
"""

from __future__ import annotations

import unittest

import torch

from blinklinmult.data.augment import (
    MAX_ROTATION_DEGREES,
    AugmentConfig,
    AugmentError,
    EyeAugmentation,
    _blur,
    augment_window,
)
from blinklinmult.data.schema import EYE_IMAGE


def window(frames: int = 6, size: int = 32, seed: int = 0) -> torch.Tensor:
    """A window of random crops in ``[0, 1]``."""
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(frames, 3, size, size, generator=generator)


def generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class TestAugmentConfig(unittest.TestCase):
    def test_zero_strength_is_disabled(self):
        self.assertFalse(AugmentConfig(strength=0.0).enabled)

    def test_every_transform_off_is_disabled(self):
        config = AugmentConfig(rotate=False, photometric=False, translate=False, blur=False)
        self.assertFalse(config.enabled)

    def test_a_negative_strength_is_rejected(self):
        with self.assertRaises(AugmentError):
            AugmentConfig(strength=-0.5)

    def test_a_strength_above_one_is_rejected(self):
        # The ranges are calibrated against the corpora; exceeding them would
        # teach invariance to variation that never occurs.
        with self.assertRaises(AugmentError):
            AugmentConfig(strength=1.5)


class TestAugmentWindow(unittest.TestCase):
    def test_the_shape_is_preserved(self):
        crops = window()
        self.assertEqual(augment_window(crops, AugmentConfig(), generator()).shape, crops.shape)

    def test_it_stays_in_range(self):
        result = augment_window(window(), AugmentConfig(), generator())
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_it_actually_changes_the_crops(self):
        crops = window()
        self.assertFalse(torch.allclose(augment_window(crops, AugmentConfig(), generator()), crops))

    def test_disabled_is_the_identity(self):
        crops = window()
        result = augment_window(crops, AugmentConfig(strength=0.0), generator())
        self.assertTrue(torch.equal(result, crops))

    def test_identical_frames_stay_identical(self):
        # The core property: one draw per window, not per frame. A head does not
        # jump orientation between adjacent frames.
        constant = window(frames=1).expand(6, 3, 32, 32).contiguous()
        result = augment_window(
            constant, AugmentConfig(photometric=False, blur=False), generator(3)
        )
        for index in range(1, 6):
            self.assertTrue(torch.allclose(result[0], result[index]), f"frame {index} differs")

    def test_photometric_is_also_window_consistent(self):
        # Lighting is continuous too: a per-frame draw would simulate a
        # flickering source and turn frame differences into noise.
        constant = window(frames=1).expand(5, 3, 32, 32).contiguous()
        result = augment_window(
            constant, AugmentConfig(rotate=False, translate=False, blur=False), generator(4)
        )
        for index in range(1, 5):
            self.assertTrue(torch.allclose(result[0], result[index]))

    def test_the_same_seed_reproduces_the_same_output(self):
        crops = window()
        first = augment_window(crops, AugmentConfig(), generator(7))
        second = augment_window(crops, AugmentConfig(), generator(7))
        self.assertTrue(torch.equal(first, second))

    def test_different_seeds_differ(self):
        crops = window()
        first = augment_window(crops, AugmentConfig(), generator(1))
        second = augment_window(crops, AugmentConfig(), generator(2))
        self.assertFalse(torch.equal(first, second))

    def test_a_lower_strength_changes_less(self):
        crops = window()
        gentle = augment_window(crops, AugmentConfig(strength=0.1), generator(5))
        strong = augment_window(crops, AugmentConfig(strength=1.0), generator(5))
        self.assertLess(float((gentle - crops).abs().mean()), float((strong - crops).abs().mean()))

    def test_a_non_window_input_is_rejected(self):
        with self.assertRaises(AugmentError):
            augment_window(torch.rand(3, 32, 32), AugmentConfig(), generator())

    def test_rotation_does_not_introduce_black_corners(self):
        # Border padding, not zeros: a black wedge is a feature the model could
        # key on, and it does not happen to real crops.
        flat = torch.full((4, 3, 32, 32), 0.5)
        result = augment_window(
            flat, AugmentConfig(photometric=False, blur=False, translate=False), generator(9)
        )
        self.assertGreater(float(result.min()), 0.4)

    def test_the_rotation_range_stays_inside_the_measured_roll(self):
        # Corpus roll spans about p05 -12 to p95 +13 degrees, so 12 covered the
        # *whole* observed range -- and sampling a distribution's extremes as
        # routine training noise is stronger than reproducing it. Softened to 6
        # after augmentation was measured to cost recall on every pairing
        # (TalkingFace: 3 misses to 9 under BCE, 4 to 5 under focal).
        #
        # The bound is two-sided on purpose: too small and the transform stops
        # doing anything, too large and it buries the brief lid movement that
        # *is* the signal.
        self.assertGreaterEqual(MAX_ROTATION_DEGREES, 3.0)
        self.assertLessEqual(MAX_ROTATION_DEGREES, 13.0)


class TestEyeAugmentation(unittest.TestCase):
    """The OmniLoader-facing transform."""

    def sample(self) -> dict:
        return {EYE_IMAGE: window()}

    def test_training_augments(self):
        original = self.sample()
        result = EyeAugmentation()(dict(original), training=True, generator=generator())
        self.assertFalse(torch.allclose(result[EYE_IMAGE], original[EYE_IMAGE]))

    def test_evaluation_is_untouched(self):
        # Validation and test must see the crops exactly as built, or the
        # benchmark measures augmentation rather than the model.
        original = self.sample()
        result = EyeAugmentation()(dict(original), training=False, generator=generator())
        self.assertTrue(torch.equal(result[EYE_IMAGE], original[EYE_IMAGE]))

    def test_the_input_sample_is_not_mutated(self):
        original = self.sample()
        keep = original[EYE_IMAGE].clone()
        EyeAugmentation()(original, training=True, generator=generator())
        self.assertTrue(torch.equal(original[EYE_IMAGE], keep))

    def test_no_generator_leaves_the_sample_alone(self):
        # Drawing from global state would make a run unreproducible; refusing is
        # safer than silently doing that.
        original = self.sample()
        result = EyeAugmentation()(dict(original), training=True, generator=None)
        self.assertTrue(torch.equal(result[EYE_IMAGE], original[EYE_IMAGE]))

    def test_a_sample_without_crops_passes_through(self):
        sample = {"something_else": torch.rand(2, 2)}
        self.assertEqual(EyeAugmentation()(sample, training=True, generator=generator()), sample)

    def test_a_single_frame_window_works(self):
        # The frame-wise path: a sample is one frame, still (T, C, H, W).
        sample = {EYE_IMAGE: window(frames=1)}
        result = EyeAugmentation()(dict(sample), training=True, generator=generator())
        self.assertEqual(result[EYE_IMAGE].shape, sample[EYE_IMAGE].shape)


class TestGeometryGate(unittest.TestCase):
    """Geometry must be dropped when a run reads handcrafted descriptors.

    152 of the 160 dimensions are pixel-space landmark coordinates describing
    the crop as it was built. Rotating the image leaves them describing the
    *unrotated* eye, so the model would get an image saying one thing and a
    feature vector saying another — silently, since nothing downstream checks
    that the two agree.
    """

    def test_rotation_counts_as_geometry(self):
        self.assertTrue(AugmentConfig(rotate=True, translate=False).geometric)

    def test_translation_counts_as_geometry(self):
        self.assertTrue(AugmentConfig(rotate=False, translate=True).geometric)

    def test_photometric_alone_is_not_geometry(self):
        # Brightness and blur leave every landmark exactly where it was.
        config = AugmentConfig(rotate=False, translate=False, photometric=True, blur=True)
        self.assertFalse(config.geometric)

    def test_zero_strength_is_not_geometry(self):
        self.assertFalse(AugmentConfig(strength=0.0).geometric)

    def test_without_geometry_keeps_the_photometric_half(self):
        stripped = AugmentConfig().without_geometry()
        self.assertFalse(stripped.rotate)
        self.assertFalse(stripped.translate)
        self.assertTrue(stripped.photometric)
        self.assertTrue(stripped.blur)

    def test_without_geometry_still_augments(self):
        crops = window()
        stripped = AugmentConfig().without_geometry()
        self.assertFalse(torch.allclose(augment_window(crops, stripped, generator(2)), crops))

    def test_without_geometry_leaves_landmarks_valid(self):
        # The property that matters: with geometry off, a constant-valued crop
        # is only rescaled, never resampled — so no pixel moves and every
        # landmark still points at what it pointed at.
        flat = torch.full((4, 3, 16, 16), 0.5)
        result = augment_window(
            flat, AugmentConfig(rotate=False, translate=False, blur=False), generator(6)
        )
        # A uniform image stays uniform under brightness/contrast alone.
        self.assertTrue(torch.allclose(result, torch.full_like(result, float(result[0, 0, 0, 0]))))

    def test_the_original_config_is_not_mutated(self):
        config = AugmentConfig()
        config.without_geometry()
        self.assertTrue(config.rotate)
        self.assertTrue(config.translate)


class TestBlur(unittest.TestCase):
    """The Gaussian blur behind ``AugmentConfig.blur``.

    Separable, so it runs as two 1-D convolutions rather than one 2-D kernel --
    cheaper, and the reason the kernel is built twice with different views.
    """

    def test_a_non_positive_sigma_is_a_no_op(self) -> None:
        """Zero blur must return the input untouched, not a copy through conv.

        The augmentation samples sigma from a range that can reach zero, and a
        convolution with a degenerate kernel would quietly alter the image.
        """
        images = torch.rand(4, 3, 8, 8)
        for sigma in (0.0, -1.0):
            with self.subTest(sigma=sigma):
                self.assertIs(_blur(images, sigma), images)

    def test_blurring_preserves_the_shape(self) -> None:
        """Padding must undo the kernel's reach exactly."""
        images = torch.rand(2, 3, 16, 16)
        self.assertEqual(_blur(images, 0.5).shape, images.shape)

    def test_a_flat_image_survives_unchanged(self) -> None:
        """A normalised kernel sums to one, so constant input is a fixed point.

        This is what catches an unnormalised kernel: it would brighten or darken
        every frame, which reads as a photometric bug rather than a blur one.
        """
        flat = torch.full((1, 3, 12, 12), 0.4)
        blurred = _blur(flat, 0.5)
        # The border is affected by zero-padding; the interior is not.
        torch.testing.assert_close(blurred[:, :, 3:-3, 3:-3], flat[:, :, 3:-3, 3:-3])

    def test_blurring_reduces_variation(self) -> None:
        """A blur is a low-pass filter, so sharp structure must soften."""
        images = torch.zeros(1, 3, 12, 12)
        images[:, :, 6, :] = 1.0
        self.assertLess(_blur(images, 0.8).std().item(), images.std().item())

    def test_a_larger_sigma_blurs_more(self) -> None:
        """Monotone in sigma, which is what makes the strength scaling work."""
        images = torch.zeros(1, 3, 16, 16)
        images[:, :, 8, 8] = 1.0
        gentle = _blur(images, 0.3).max().item()
        strong = _blur(images, 1.5).max().item()
        self.assertGreater(gentle, strong)

    def test_channels_stay_independent(self) -> None:
        """Grouped convolution: red must not bleed into green.

        Dropping ``groups=channels`` would mix them, which is invisible on a
        greyscale test image and wrong on a real one.
        """
        images = torch.zeros(1, 3, 10, 10)
        images[:, 0] = 1.0
        blurred = _blur(images, 0.5)
        self.assertGreater(blurred[0, 0].max().item(), 0.5)
        self.assertEqual(blurred[0, 1].max().item(), 0.0)
