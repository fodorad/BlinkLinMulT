"""Tests for the masked binary losses.

The masking is the point: a corpus that does not annotate a target contributes
an all-``False`` mask, and a loss that ignored it would train that head against
a placeholder on every sample of that corpus.
"""

from __future__ import annotations

import unittest

import torch

from blinklinmult.train.losses import (
    LossError,
    MaskedLoss,
    build_loss,
    duration_prior,
    masked_bce,
    masked_dice,
    masked_dice_bce,
    masked_focal,
    temporal_smoothness,
)

ALL_LOSSES = (masked_bce, masked_focal, masked_dice_bce)


class MaskedLossCase(unittest.TestCase):
    """Shared fixtures: confident-correct logits and their targets."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.logits = torch.tensor([[5.0, -5.0, 5.0, -5.0]])
        self.target = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
        self.full_mask = torch.ones_like(self.target, dtype=torch.bool)


class TestMaskingBehaviour(MaskedLossCase):
    def test_all_false_mask_gives_zero_loss(self):
        empty = torch.zeros_like(self.target, dtype=torch.bool)
        for loss_fn in (masked_bce, masked_focal):
            with self.subTest(loss=loss_fn.__name__):
                value = loss_fn(torch.randn_like(self.logits), self.target, empty)
                self.assertAlmostEqual(float(value), 0.0, places=5)

    def test_all_false_mask_keeps_the_gradient_path(self):
        # Boolean indexing would produce an empty tensor whose backward reaches
        # no parameter, which deadlocks DDP. Weighting keeps the graph intact.
        for loss_fn in ALL_LOSSES:
            with self.subTest(loss=loss_fn.__name__):
                logits = torch.randn(1, 4, requires_grad=True)
                empty = torch.zeros(1, 4, dtype=torch.bool)
                loss_fn(logits, self.target, empty).backward()
                self.assertIsNotNone(logits.grad)
                self.assertEqual(logits.grad.shape, logits.shape)

    def test_masked_positions_do_not_affect_the_loss(self):
        mask = torch.tensor([[True, True, False, False]])
        wrong = self.logits.clone()
        wrong[0, 2:] = -100.0  # catastrophically wrong, but masked out

        baseline = masked_bce(self.logits, self.target, mask)
        perturbed = masked_bce(wrong, self.target, mask)
        self.assertAlmostEqual(float(baseline), float(perturbed), places=6)

    def test_placeholder_targets_under_a_false_mask_are_harmless(self):
        # A real batch carries -1.0 placeholders where a corpus does not annotate.
        target = torch.tensor([[1.0, 0.0, -1.0, -1.0]])
        mask = torch.tensor([[True, True, False, False]])
        value = masked_bce(self.logits, target, mask)
        self.assertTrue(torch.isfinite(value))
        self.assertGreaterEqual(float(value), 0.0)

    def test_loss_does_not_depend_on_the_number_of_masked_positions(self):
        # Two batches with the same valid content but different amounts of
        # padding must score the same, or a batch's loss would depend on its mix.
        logits_a = torch.tensor([[2.0, -2.0]])
        target_a = torch.tensor([[1.0, 0.0]])
        mask_a = torch.ones(1, 2, dtype=torch.bool)

        logits_b = torch.tensor([[2.0, -2.0, 0.0, 0.0, 0.0]])
        target_b = torch.tensor([[1.0, 0.0, -1.0, -1.0, -1.0]])
        mask_b = torch.tensor([[True, True, False, False, False]])

        self.assertAlmostEqual(
            float(masked_bce(logits_a, target_a, mask_a)),
            float(masked_bce(logits_b, target_b, mask_b)),
            places=6,
        )


class TestMaskedBce(MaskedLossCase):
    def test_confident_correct_predictions_score_near_zero(self):
        value = masked_bce(self.logits, self.target, self.full_mask)
        self.assertLess(float(value), 0.01)

    def test_confident_wrong_predictions_score_high(self):
        value = masked_bce(-self.logits, self.target, self.full_mask)
        self.assertGreater(float(value), 4.0)

    def test_matches_the_reference_implementation_when_unmasked(self):
        logits = torch.randn(4, 6)
        target = (torch.rand(4, 6) > 0.5).float()
        mask = torch.ones_like(target, dtype=torch.bool)

        expected = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
        torch.testing.assert_close(masked_bce(logits, target, mask), expected, rtol=1e-4, atol=1e-6)

    def test_is_differentiable(self):
        logits = torch.randn(2, 4, requires_grad=True)
        masked_bce(logits, self.target.expand(2, 4), torch.ones(2, 4, dtype=torch.bool)).backward()
        self.assertTrue(logits.grad.abs().sum() > 0)


class TestMaskedFocal(MaskedLossCase):
    def test_gamma_zero_is_alpha_weighted_bce(self):
        logits = torch.randn(4, 6)
        target = (torch.rand(4, 6) > 0.5).float()
        mask = torch.ones_like(target, dtype=torch.bool)

        focal = masked_focal(logits, target, mask, gamma=0.0, alpha=0.5)
        # alpha=0.5 weights both classes equally, halving plain BCE.
        torch.testing.assert_close(
            focal, 0.5 * masked_bce(logits, target, mask), rtol=1e-4, atol=1e-6
        )

    def test_down_weights_easy_examples_relative_to_bce(self):
        # Easy negatives: the whole reason blink presence uses focal loss.
        logits = torch.full((1, 100), -6.0)
        target = torch.zeros(1, 100)
        mask = torch.ones(1, 100, dtype=torch.bool)

        self.assertLess(
            float(masked_focal(logits, target, mask)),
            float(masked_bce(logits, target, mask)),
        )

    def test_hard_examples_dominate_the_focal_loss(self):
        mask = torch.ones(1, 2, dtype=torch.bool)
        easy = masked_focal(torch.tensor([[6.0, 6.0]]), torch.tensor([[1.0, 1.0]]), mask)
        hard = masked_focal(torch.tensor([[-6.0, 6.0]]), torch.tensor([[1.0, 1.0]]), mask)
        self.assertGreater(float(hard), float(easy) * 10)

    def test_larger_gamma_focuses_harder(self):
        logits = torch.full((1, 50), -4.0)
        target = torch.zeros(1, 50)
        mask = torch.ones(1, 50, dtype=torch.bool)

        self.assertLess(
            float(masked_focal(logits, target, mask, gamma=4.0)),
            float(masked_focal(logits, target, mask, gamma=1.0)),
        )

    def test_negative_gamma_raises(self):
        with self.assertRaises(LossError):
            masked_focal(self.logits, self.target, self.full_mask, gamma=-1.0)

    def test_alpha_outside_the_unit_range_raises(self):
        with self.assertRaises(LossError):
            masked_focal(self.logits, self.target, self.full_mask, alpha=1.5)


class TestMaskedDice(MaskedLossCase):
    def test_perfect_overlap_scores_near_zero(self):
        logits = torch.tensor([[20.0, -20.0, 20.0, -20.0]])
        value = masked_dice(logits, self.target, self.full_mask)
        self.assertLess(float(value), 0.01)

    def test_inverted_prediction_scores_near_one(self):
        logits = torch.tensor([[-20.0, 20.0, -20.0, 20.0]])
        value = masked_dice(logits, self.target, self.full_mask)
        self.assertGreater(float(value), 0.9)

    def test_is_bounded(self):
        for _ in range(20):
            value = masked_dice(
                torch.randn(2, 8),
                (torch.rand(2, 8) > 0.5).float(),
                torch.ones(2, 8, dtype=torch.bool),
            )
            self.assertGreaterEqual(float(value), 0.0)
            self.assertLessEqual(float(value), 1.0 + 1e-5)

    def test_all_negative_target_is_handled(self):
        # No positives at all: Dice is degenerate but must stay finite.
        value = masked_dice(
            torch.full((1, 5), -10.0), torch.zeros(1, 5), torch.ones(1, 5, dtype=torch.bool)
        )
        self.assertTrue(torch.isfinite(value))


class TestMaskedDiceBce(MaskedLossCase):
    def test_is_the_sum_of_its_parts(self):
        logits = torch.randn(2, 6)
        target = (torch.rand(2, 6) > 0.5).float()
        mask = torch.ones_like(target, dtype=torch.bool)

        expected = masked_bce(logits, target, mask) + 0.5 * masked_dice(logits, target, mask)
        torch.testing.assert_close(
            masked_dice_bce(logits, target, mask), expected, rtol=1e-5, atol=1e-7
        )

    def test_weight_scales_the_dice_term(self):
        logits = torch.randn(2, 6)
        target = (torch.rand(2, 6) > 0.5).float()
        mask = torch.ones_like(target, dtype=torch.bool)

        bce_only = masked_dice_bce(logits, target, mask, dice_weight=0.0)
        torch.testing.assert_close(bce_only, masked_bce(logits, target, mask))

    def test_negative_weight_raises(self):
        with self.assertRaises(LossError):
            masked_dice_bce(self.logits, self.target, self.full_mask, dice_weight=-1.0)


class TestShapeValidation(MaskedLossCase):
    def test_mismatched_logits_and_target_raise(self):
        for loss_fn in ALL_LOSSES:
            with self.subTest(loss=loss_fn.__name__), self.assertRaises(LossError):
                loss_fn(torch.randn(1, 5), self.target, self.full_mask)

    def test_mismatched_mask_raises(self):
        for loss_fn in ALL_LOSSES:
            with self.subTest(loss=loss_fn.__name__), self.assertRaises(LossError):
                loss_fn(self.logits, self.target, torch.ones(1, 5, dtype=torch.bool))

    def test_multi_dimensional_targets_are_supported(self):
        # Eye state is (B, T, 2); blink presence is (B, T). One code path.
        logits = torch.randn(2, 4, 2)
        target = (torch.rand(2, 4, 2) > 0.5).float()
        mask = torch.ones_like(target, dtype=torch.bool)
        for loss_fn in ALL_LOSSES:
            with self.subTest(loss=loss_fn.__name__):
                self.assertEqual(loss_fn(logits, target, mask).ndim, 0)


class TestBuildLoss(MaskedLossCase):
    def test_builds_each_supported_loss(self):
        for name in ("bce", "focal", "dice_bce"):
            with self.subTest(loss=name):
                self.assertIsInstance(build_loss(name), MaskedLoss)

    def test_module_matches_its_function(self):
        module = build_loss("bce")
        torch.testing.assert_close(
            module(self.logits, self.target, self.full_mask),
            masked_bce(self.logits, self.target, self.full_mask),
        )

    def test_kwargs_are_forwarded(self):
        module = build_loss("focal", gamma=0.0, alpha=0.5)
        torch.testing.assert_close(
            module(self.logits, self.target, self.full_mask),
            masked_focal(self.logits, self.target, self.full_mask, gamma=0.0, alpha=0.5),
        )

    def test_unknown_loss_raises(self):
        with self.assertRaises(LossError) as ctx:
            build_loss("hinge")
        self.assertIn("hinge", str(ctx.exception))

    def test_unaccepted_kwarg_raises(self):
        # A typo'd loss argument must fail loudly, not be silently dropped.
        with self.assertRaises(LossError) as ctx:
            build_loss("bce", gamma=2.0)
        self.assertIn("gamma", str(ctx.exception))

    def test_wrong_loss_kwarg_raises(self):
        with self.assertRaises(LossError):
            build_loss("focal", dice_weight=0.5)


if __name__ == "__main__":
    unittest.main()


class TestTemporalSmoothness(unittest.TestCase):
    """A blink is continuous motion, so its signal should not chatter.

    Independent per-frame predictions have no reason to be smooth, and a jagged
    signal is what makes interval extraction brittle -- every spurious crossing
    of the operating point becomes a spurious event boundary.
    """

    def logits(self, values) -> torch.Tensor:
        return torch.logit(torch.tensor([values], dtype=torch.float32).clamp(1e-4, 1 - 1e-4))

    def test_a_flat_signal_is_free(self):
        signal = self.logits([0.5, 0.5, 0.5, 0.5])
        mask = torch.ones_like(signal, dtype=torch.bool)
        self.assertAlmostEqual(float(temporal_smoothness(signal, mask)), 0.0, places=5)

    def test_a_chattering_signal_is_penalised(self):
        smooth = self.logits([0.1, 0.2, 0.3, 0.4])
        jagged = self.logits([0.1, 0.9, 0.1, 0.9])
        mask = torch.ones_like(smooth, dtype=torch.bool)
        self.assertGreater(
            float(temporal_smoothness(jagged, mask)), float(temporal_smoothness(smooth, mask))
        )

    def test_a_margin_forgives_a_fast_closure(self):
        # A genuine blink closes quickly; the margin is what keeps the penalty
        # from fighting the signal it is meant to clean up.
        signal = self.logits([0.1, 0.35, 0.1, 0.35])
        mask = torch.ones_like(signal, dtype=torch.bool)
        self.assertAlmostEqual(float(temporal_smoothness(signal, mask, margin=0.5)), 0.0, places=5)

    def test_a_masked_gap_is_not_a_jump(self):
        # Jumping across a frame with no supervision is not evidence of chatter.
        signal = self.logits([0.1, 0.9, 0.1])
        mask = torch.tensor([[True, False, True]])
        self.assertAlmostEqual(float(temporal_smoothness(signal, mask)), 0.0, places=5)

    def test_a_single_frame_window_is_free(self):
        # The still-image case: no adjacent pair exists.
        signal = self.logits([0.7])
        mask = torch.ones_like(signal, dtype=torch.bool)
        self.assertEqual(float(temporal_smoothness(signal, mask)), 0.0)

    def test_it_rejects_disagreeing_shapes(self):
        with self.assertRaises(LossError):
            temporal_smoothness(torch.zeros(1, 4), torch.ones(1, 3, dtype=torch.bool))

    def test_it_rejects_a_non_sequence_input(self):
        with self.assertRaises(LossError):
            temporal_smoothness(torch.zeros(2, 3, 4), torch.ones(2, 3, 4, dtype=torch.bool))


class TestDurationPrior(unittest.TestCase):
    """Predicted closures should last about as long as real blinks do.

    Roughly 100-400 ms, or 3-12 frames at 30 fps. A model firing on one isolated
    frame, or holding a closure for seconds, is producing something that is not
    a blink whatever the frame-wise loss says.
    """

    def logits(self, values) -> torch.Tensor:
        return torch.logit(torch.tensor([values], dtype=torch.float32).clamp(1e-4, 1 - 1e-4))

    def test_a_plausible_blink_is_free(self):
        signal = self.logits([0.02] * 3 + [0.98] * 5 + [0.02] * 3)
        mask = torch.ones_like(signal, dtype=torch.bool)
        self.assertAlmostEqual(float(duration_prior(signal, mask)), 0.0, places=4)

    def test_a_window_with_no_closure_is_free(self):
        # Most windows contain no blink; demanding one would invent events.
        signal = self.logits([0.02] * 10)
        mask = torch.ones_like(signal, dtype=torch.bool)
        self.assertAlmostEqual(float(duration_prior(signal, mask)), 0.0, places=4)

    def test_an_implausibly_long_closure_is_penalised(self):
        signal = self.logits([0.98] * 30)
        mask = torch.ones_like(signal, dtype=torch.bool)
        self.assertGreater(float(duration_prior(signal, mask)), 0.0)

    def test_an_isolated_frame_is_penalised(self):
        signal = self.logits([0.02] * 5 + [0.98] + [0.02] * 5)
        mask = torch.ones_like(signal, dtype=torch.bool)
        self.assertGreater(float(duration_prior(signal, mask)), 0.0)

    def test_masked_frames_do_not_count_as_closed(self):
        signal = self.logits([0.98] * 30)
        mask = torch.zeros_like(signal, dtype=torch.bool)
        self.assertAlmostEqual(float(duration_prior(signal, mask)), 0.0, places=4)

    def test_it_rejects_inverted_bounds(self):
        signal = self.logits([0.5] * 4)
        mask = torch.ones_like(signal, dtype=torch.bool)
        with self.assertRaises(LossError):
            duration_prior(signal, mask, min_frames=10.0, max_frames=2.0)

    def test_it_is_differentiable(self):
        # It has to be usable as a loss term, which a hard interval extraction
        # would not be.
        signal = self.logits([0.98] * 30).requires_grad_(True)
        mask = torch.ones_like(signal, dtype=torch.bool)
        duration_prior(signal, mask).backward()
        self.assertIsNotNone(signal.grad)
