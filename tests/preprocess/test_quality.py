"""Tests for the per-frame quality signals.

These exist to make bad data *findable*, so what matters is that the signals
separate the failure they were each added for: an eye leaving the frame, an eye
too small to hold detail, and a face turned to profile.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.preprocess.quality import (
    FRONTAL_ASPECT_MAX,
    GOOD_EYE_SPAN,
    SUSPECT_CONFIDENCE,
    blur_score,
    box_jitter,
    contour_fit,
    exposure_score,
    eye_aspect,
    is_suspect,
    on_screen_fraction,
    window_confidence,
)


def contour(width: float = 30.0, height: float = 10.0, origin=(100.0, 100.0)) -> np.ndarray:
    """An eight-point eyelid contour of a known width and height."""
    x0, y0 = origin
    return np.asarray(
        [
            [x0, y0 + height / 2],
            [x0 + width * 0.25, y0],
            [x0 + width * 0.75, y0],
            [x0 + width, y0 + height / 2],
            [x0 + width * 0.75, y0 + height],
            [x0 + width * 0.25, y0 + height],
            [x0 + width * 0.5, y0],
            [x0 + width * 0.5, y0 + height],
        ],
        dtype=np.float64,
    )


class TestEyeAspect(unittest.TestCase):
    def test_a_frontal_eye_scores_around_a_third(self):
        # Matches the corpus: frontal MPEblink crops have a median ratio 0.364.
        self.assertAlmostEqual(eye_aspect(contour(width=30.0, height=10.0)), 1 / 3, places=2)

    def test_a_profile_eye_scores_high_not_low(self):
        # Measured, and the opposite of the intuition: turning to profile
        # compresses the contour *horizontally* while its height persists, so
        # the ratio rises. Profile clips median 0.667 against frontal 0.364.
        narrowed = eye_aspect(contour(width=10.0, height=10.0))
        self.assertGreater(narrowed, FRONTAL_ASPECT_MAX)

    def test_a_degenerate_contour_is_zero(self):
        self.assertEqual(eye_aspect(contour(width=0.0, height=0.0)), 0.0)

    def test_an_empty_contour_is_zero(self):
        self.assertEqual(eye_aspect(np.zeros((0, 2))), 0.0)

    def test_it_is_scale_invariant(self):
        # A distant face and a close one at the same pose must score alike, or
        # the signal would just be measuring size twice.
        near = eye_aspect(contour(width=60.0, height=20.0))
        far = eye_aspect(contour(width=15.0, height=5.0))
        self.assertAlmostEqual(near, far, places=4)


class TestOnScreenFraction(unittest.TestCase):
    def test_a_contained_eye_is_fully_on_screen(self):
        self.assertEqual(on_screen_fraction(contour(origin=(100.0, 100.0)), 400, 400), 1.0)

    def test_an_eye_above_the_frame_is_zero(self):
        # MPEblink tracks faces past the frame edge; this is the real case.
        self.assertEqual(on_screen_fraction(contour(origin=(100.0, -300.0)), 400, 400), 0.0)

    def test_a_straddling_eye_is_partial(self):
        value = on_screen_fraction(contour(width=30.0, origin=(-15.0, 100.0)), 400, 400)
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)

    def test_an_empty_contour_is_zero(self):
        self.assertEqual(on_screen_fraction(np.zeros((0, 2)), 400, 400), 0.0)


FRONTAL_ASPECT = 0.36
"""The corpus's measured median for a frontal eye."""


class TestWindowConfidence(unittest.TestCase):
    def good(self, steps: int = 10):
        return (
            np.ones(steps),
            np.full(steps, GOOD_EYE_SPAN),
            np.full(steps, FRONTAL_ASPECT),
        )

    def test_a_clean_window_scores_one(self):
        self.assertAlmostEqual(window_confidence(*self.good()), 1.0, places=5)

    def test_an_offscreen_window_scores_low(self):
        steps = 10
        score = window_confidence(
            np.zeros(steps), np.full(steps, GOOD_EYE_SPAN), np.full(steps, FRONTAL_ASPECT)
        )
        self.assertLess(score, SUSPECT_CONFIDENCE)

    def test_a_profile_window_is_flagged(self):
        # The `test/183` case: fully on screen with an ordinary 21.5 px span,
        # so only the aspect can catch it -- and it is high, not low.
        steps = 10
        score = window_confidence(np.ones(steps), np.full(steps, 21.5), np.full(steps, 1.2))
        self.assertTrue(is_suspect(score))

    def test_a_frontal_window_is_not_flagged(self):
        # The other half of the claim: the rule must not condemn good data.
        self.assertFalse(is_suspect(window_confidence(*self.good())))

    def test_a_tiny_eye_is_flagged_even_when_frontal(self):
        steps = 10
        score = window_confidence(
            np.ones(steps), np.full(steps, 3.0), np.full(steps, FRONTAL_ASPECT)
        )
        self.assertTrue(is_suspect(score))

    def test_invalid_frames_are_excluded_not_scored_zero(self):
        # A window is judged on the frames it has; the mask already records how
        # many it lacks, and counting them twice would double-punish.
        steps = 10
        on_screen, span, aspect = self.good(steps)
        on_screen[5:] = 0.0
        valid = np.ones(steps, dtype=bool)
        valid[5:] = False
        self.assertAlmostEqual(
            window_confidence(on_screen, span, aspect, valid=valid), 1.0, places=5
        )

    def test_no_valid_frame_scores_zero(self):
        steps = 4
        self.assertEqual(
            window_confidence(*self.good(steps), valid=np.zeros(steps, dtype=bool)), 0.0
        )

    def test_the_score_stays_within_bounds(self):
        # An eye larger and rounder than the reference must not exceed 1.0, or
        # ranking by the score would put oddities at the top.
        steps = 6
        score = window_confidence(
            np.ones(steps), np.full(steps, 10 * GOOD_EYE_SPAN), np.full(steps, FRONTAL_ASPECT)
        )
        self.assertLessEqual(score, 1.0)

    def test_a_worse_window_never_scores_higher(self):
        steps = 8
        better = window_confidence(*self.good(steps))
        worse = window_confidence(
            np.full(steps, 0.5), np.full(steps, GOOD_EYE_SPAN / 2), np.full(steps, 1.4)
        )
        self.assertLess(worse, better)


class TestIsSuspect(unittest.TestCase):
    def test_a_clean_score_is_not_suspect(self):
        self.assertFalse(is_suspect(0.95))

    def test_a_low_score_is_suspect(self):
        self.assertTrue(is_suspect(0.1))


if __name__ == "__main__":
    unittest.main()


class TestBlurScore(unittest.TestCase):
    """Blur is the signal the geometry cannot see.

    Motion blur destroys the eyelid edge — the one thing blink detection reads —
    while leaving every geometric check intact: the landmarks still fit, the span
    is still ordinary, the eye is still on screen.
    """

    def sharp(self, size: int = 32) -> np.ndarray:
        # A hard edge, like a lid against sclera.
        crop = np.zeros((3, size, size), dtype=np.float32)
        crop[:, : size // 2, :] = 1.0
        return crop

    def blurred(self, size: int = 32) -> np.ndarray:
        # The same edge, ramped over the whole patch.
        ramp = np.linspace(0.0, 1.0, size, dtype=np.float32)
        return np.repeat(ramp[None, :, None], 3, axis=0).repeat(size, axis=2)

    def test_a_sharp_edge_scores_high(self):
        self.assertGreater(blur_score(self.sharp()), 0.5)

    def test_a_blurred_patch_scores_low(self):
        self.assertLess(blur_score(self.blurred()), 0.1)

    def test_sharp_beats_blurred(self):
        self.assertGreater(blur_score(self.sharp()), blur_score(self.blurred()))

    def test_a_flat_patch_scores_zero(self):
        self.assertEqual(blur_score(np.full((3, 32, 32), 0.5, dtype=np.float32)), 0.0)

    def test_a_degenerate_crop_is_not_sharp(self):
        # A missing patch must never score as usable.
        self.assertEqual(blur_score(np.zeros((3, 1, 1), dtype=np.float32)), 0.0)

    def test_it_accepts_a_single_channel(self):
        self.assertGreater(blur_score(self.sharp()[0]), 0.5)

    def test_the_score_stays_bounded(self):
        rng = np.random.default_rng(0)
        loud = rng.normal(0.0, 50.0, (3, 32, 32)).astype(np.float32)
        self.assertLessEqual(blur_score(loud), 1.0)

    def test_it_is_scale_invariant(self):
        # A hard edge measures 0.067 on a [0,1] crop and ~4300 on [0,255]; the
        # ratio against intensity variance makes one constant serve both.
        crop = self.sharp()
        self.assertAlmostEqual(blur_score(crop), blur_score(crop * 255.0), places=5)


class TestExposureScore(unittest.TestCase):
    """A near-black or blown-out crop has no lid edge, whatever its landmarks say."""

    def test_a_normal_crop_scores_high(self):
        rng = np.random.default_rng(0)
        self.assertGreater(exposure_score(rng.random((3, 32, 32)).astype(np.float32)), 0.5)

    def test_a_black_crop_scores_zero(self):
        self.assertEqual(exposure_score(np.zeros((3, 32, 32), dtype=np.float32)), 0.0)

    def test_a_saturated_crop_scores_zero(self):
        self.assertEqual(exposure_score(np.ones((3, 32, 32), dtype=np.float32)), 0.0)

    def test_it_is_scale_invariant(self):
        # The builder may store crops in [0, 1] or [0, 255]; one constant serves
        # both, so the same patch must score the same either way.
        rng = np.random.default_rng(0)
        crop = rng.random((3, 32, 32)).astype(np.float32)
        self.assertAlmostEqual(exposure_score(crop), exposure_score(crop * 255.0), places=5)

    def test_an_empty_crop_scores_zero(self):
        self.assertEqual(exposure_score(np.zeros((0,), dtype=np.float32)), 0.0)


class TestContourFit(unittest.TestCase):
    """The signal that catches a contour tracking onto skin.

    Thirty MPEblink clips pass every geometric check while the eye is edge-on and
    invisible: ordinary span, higher-than-usual contrast, landmarks that "fit".
    What fails is that the contour sits on flat skin rather than a lid edge.
    """

    def edged(self, size: int = 32) -> np.ndarray:
        crop = np.zeros((3, size, size), dtype=np.float32)
        crop[:, : size // 2, :] = 1.0
        return crop

    def test_a_contour_on_the_edge_scores_high(self):
        size = 32
        # Along the horizontal edge at row size//2.
        contour = np.stack([np.linspace(4, size - 4, 8), np.full(8, size // 2)], axis=-1)
        self.assertGreater(contour_fit(self.edged(size), contour), 0.5)

    def test_a_contour_on_flat_skin_scores_low(self):
        size = 32
        # Well away from the edge, in the flat region.
        contour = np.stack([np.linspace(4, size - 4, 8), np.full(8, 4)], axis=-1)
        self.assertLess(contour_fit(self.edged(size), contour), 0.5)

    def test_the_edge_beats_the_flat_region(self):
        size = 32
        on_edge = np.stack([np.linspace(4, size - 4, 8), np.full(8, size // 2)], axis=-1)
        on_skin = np.stack([np.linspace(4, size - 4, 8), np.full(8, 4)], axis=-1)
        crop = self.edged(size)
        self.assertGreater(contour_fit(crop, on_edge), contour_fit(crop, on_skin))

    def test_a_featureless_crop_scores_zero(self):
        contour = np.stack([np.linspace(4, 28, 8), np.full(8, 16)], axis=-1)
        self.assertEqual(contour_fit(np.full((3, 32, 32), 0.5, dtype=np.float32), contour), 0.0)

    def test_a_contour_outside_the_crop_is_a_fit_failure(self):
        contour = np.full((8, 2), 500.0)
        self.assertEqual(contour_fit(self.edged(), contour), 0.0)

    def test_an_empty_contour_scores_zero(self):
        self.assertEqual(contour_fit(self.edged(), np.zeros((0, 2))), 0.0)

    def test_the_score_stays_in_range(self):
        contour = np.stack([np.linspace(4, 28, 8), np.full(8, 16)], axis=-1)
        score = contour_fit(self.edged(), contour)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestBoxJitter(unittest.TestCase):
    """A skittering box means the detector lost lock, even if each crop looks fine."""

    def test_a_still_box_does_not_jitter(self):
        centres = np.tile([100.0, 100.0], (5, 1))
        spans = np.full(5, 20.0)
        self.assertTrue(np.allclose(box_jitter(centres, spans), 0.0))

    def test_a_jumping_box_scores_high(self):
        """Two eye-widths of movement saturates the signal at its maximum."""
        centres = np.asarray([[100.0, 100.0], [140.0, 100.0], [100.0, 100.0]])
        spans = np.full(3, 20.0)
        self.assertEqual(box_jitter(centres, spans)[1:].max(), 1.0)

    def test_it_is_relative_to_eye_size(self):
        # The same pixel displacement is negligible on a close face and severe
        # on a distant one. Both stay below saturation so the comparison tests
        # the scaling rather than the clip.
        centres = np.asarray([[100.0, 100.0], [110.0, 100.0]])
        near = box_jitter(centres, np.full(2, 100.0))
        far = box_jitter(centres, np.full(2, 25.0))
        self.assertLess(near[1], far[1])
        self.assertAlmostEqual(near[1], 0.1)
        self.assertAlmostEqual(far[1], 0.4)

    def test_the_first_frame_has_no_predecessor(self):
        centres = np.asarray([[100.0, 100.0], [200.0, 100.0]])
        self.assertEqual(box_jitter(centres, np.full(2, 20.0))[0], 0.0)

    def test_a_single_frame_yields_one_zero(self):
        self.assertEqual(box_jitter(np.asarray([[1.0, 1.0]]), np.asarray([20.0])).tolist(), [0.0])

    def test_an_empty_window_is_handled(self):
        self.assertEqual(box_jitter(np.zeros((0, 2)), np.zeros(0)).size, 0)


class TestBoxJitterStaysBounded(unittest.TestCase):
    """`eye_jitter` must honour the [0, 1] contract QUALITY_SIGNALS documents.

    It did not: a degenerate span divided by the 1e-6 floor produced
    494 959 584 on RN30, a value that would dominate any loss weighting or
    normalisation it reached.
    """

    def test_a_degenerate_span_scores_maximally_unreliable(self):
        centres = np.array([[0.0, 0.0], [5.0, 0.0]])
        spans = np.array([0.0, 0.0])
        jitter = box_jitter(centres, spans)
        self.assertEqual(jitter[0], 0.0)
        self.assertEqual(jitter[1], 1.0)

    def test_large_movement_saturates_rather_than_exploding(self):
        centres = np.array([[0.0, 0.0], [1000.0, 0.0]])
        spans = np.array([10.0, 10.0])
        self.assertEqual(box_jitter(centres, spans)[1], 1.0)

    def test_a_normal_step_is_unchanged(self):
        """Half an eye-width of movement must still read as 0.5."""
        centres = np.array([[0.0, 0.0], [5.0, 0.0]])
        spans = np.array([10.0, 10.0])
        self.assertAlmostEqual(box_jitter(centres, spans)[1], 0.5)

    def test_every_value_lies_in_the_unit_interval(self):
        rng = np.random.default_rng(0)
        centres = rng.uniform(-500, 500, (200, 2))
        spans = np.concatenate([np.zeros(50), rng.uniform(0, 40, 150)])
        jitter = box_jitter(centres, spans)
        self.assertTrue(np.all(jitter >= 0.0))
        self.assertTrue(np.all(jitter <= 1.0))
