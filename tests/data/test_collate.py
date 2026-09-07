"""Tests for the batch-to-model boundary."""

from __future__ import annotations

import unittest

import torch

from blinklinmult.data.collate import (
    BatchError,
    apply_occlusion,
    check_eye_images,
    fold_time_into_batch,
    occluded_eyes,
    target_and_mask,
    unfold_time_from_batch,
    unpack_batch,
)
from blinklinmult.data.schema import (
    BLINK_PRESENCE,
    EYE_FEATURE,
    EYE_IMAGE,
    EYE_STATE,
    HEAD_POSE,
    LEFT,
    RIGHT,
    SAMPLE_KEY,
    UNKNOWN_EYE,
)


def eye_images(batch: int = 2, time: int = 4, size: int = 8) -> torch.Tensor:
    """One eye's crops as loaded from the HDF5: native image form."""
    return torch.rand(batch, time, 3, size, size)


class TestCheckEyeImages(unittest.TestCase):
    def test_a_matching_batch_passes_through_unchanged(self):
        images = eye_images(2, 4, 8)
        self.assertIs(check_eye_images(images, image_size=8), images)

    def test_the_loaded_form_is_already_an_image(self):
        # omniloader >= 1.1 keeps the structured shape end to end, so no
        # reshape happens at the model boundary.
        self.assertEqual(tuple(eye_images(2, 4, 8).shape), (2, 4, 3, 8, 8))

    def test_single_frame_window(self):
        self.assertEqual(tuple(check_eye_images(eye_images(2, 1, 8), 8).shape), (2, 1, 3, 8, 8))

    def test_wrong_image_size_raises_a_readable_error(self):
        with self.assertRaises(BatchError) as ctx:
            check_eye_images(eye_images(2, 4, 8), image_size=32)
        message = str(ctx.exception)
        self.assertIn("32", message)
        self.assertIn("rebuild", message)

    def test_a_flattened_batch_is_rejected(self):
        # What the pre-1.1 layout produced; it must not pass silently.
        with self.assertRaises(BatchError):
            check_eye_images(torch.rand(2, 4, 3 * 8 * 8), image_size=8)


class TestFoldUnfold(unittest.TestCase):
    def test_fold_collapses_batch_and_time(self):
        images = torch.rand(2, 4, 3, 8, 8)
        folded = fold_time_into_batch(images)
        self.assertEqual(tuple(folded.shape), (2 * 4, 3, 8, 8))

    def test_fold_preserves_crop_content(self):
        images = torch.rand(2, 3, 3, 8, 8)
        folded = fold_time_into_batch(images)
        # Time is the fastest axis: sample 0's frames come first, in order.
        torch.testing.assert_close(folded[0], images[0, 0])
        torch.testing.assert_close(folded[1], images[0, 1])
        torch.testing.assert_close(folded[3], images[1, 0])

    def test_unfold_restores_one_embedding_per_timestep(self):
        embedded = torch.rand(2 * 4, 16)
        unfolded = unfold_time_from_batch(embedded, batch=2, time=4)
        self.assertEqual(tuple(unfolded.shape), (2, 4, 16))

    def test_fold_unfold_round_trips_the_layout(self):
        images = torch.rand(2, 3, 3, 4, 4)
        folded = fold_time_into_batch(images)
        # An identity "embedding" of the flattened crop.
        embedded = folded.reshape(folded.shape[0], -1)
        unfolded = unfold_time_from_batch(embedded, batch=2, time=3)
        torch.testing.assert_close(unfolded, images.reshape(2, 3, -1))


class TestUnpackBatch(unittest.TestCase):
    def batch(self) -> dict:
        return {
            EYE_IMAGE: torch.rand(2, 4, 3, 4, 4),
            f"{EYE_IMAGE}_mask": torch.ones(2, 4, dtype=torch.bool),
            EYE_FEATURE: torch.rand(2, 4, 16),
            f"{EYE_FEATURE}_mask": torch.ones(2, 4, dtype=torch.bool),
        }

    def test_returns_inputs_in_the_requested_order(self):
        inputs, masks = unpack_batch(self.batch(), [EYE_FEATURE, EYE_IMAGE])
        self.assertEqual(inputs[0].shape[-1], 16)
        self.assertEqual(tuple(inputs[1].shape), (2, 4, 3, 4, 4))
        self.assertEqual(len(masks), 2)

    def test_single_feature(self):
        inputs, masks = unpack_batch(self.batch(), [EYE_IMAGE])
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(masks), 1)

    def test_missing_feature_raises(self):
        with self.assertRaises(BatchError) as ctx:
            unpack_batch(self.batch(), ["nope"])
        self.assertIn("nope", str(ctx.exception))

    def test_missing_mask_raises(self):
        batch = self.batch()
        del batch[f"{EYE_IMAGE}_mask"]
        with self.assertRaises(BatchError) as ctx:
            unpack_batch(batch, [EYE_IMAGE])
        self.assertIn("mask", str(ctx.exception))


class TestTargetAndMask(unittest.TestCase):
    def test_reads_a_one_dimensional_target(self):
        batch = {
            BLINK_PRESENCE: torch.rand(2, 4),
            f"{BLINK_PRESENCE}_mask": torch.ones(2, 4, dtype=torch.bool),
        }
        target, mask = target_and_mask(batch, BLINK_PRESENCE)
        self.assertEqual(target.shape, mask.shape)

    def test_eye_state_is_a_scalar_sequence_like_blink_presence(self):
        # One sample is one eye, so both targets are (B, T): one code path.
        batch = {
            EYE_STATE: torch.rand(2, 4),
            f"{EYE_STATE}_mask": torch.ones(2, 4, dtype=torch.bool),
        }
        target, mask = target_and_mask(batch, EYE_STATE)
        self.assertEqual(tuple(target.shape), (2, 4))
        self.assertEqual(target.shape, mask.shape)

    def test_the_mask_is_returned_unchanged(self):
        flags = torch.tensor([[True, False, True, True], [False, False, True, True]])
        batch = {EYE_STATE: torch.rand(2, 4), f"{EYE_STATE}_mask": flags}
        _, mask = target_and_mask(batch, EYE_STATE)
        torch.testing.assert_close(mask, flags)

    def test_an_all_false_mask_survives(self):
        batch = {
            BLINK_PRESENCE: torch.full((2, 4), -1.0),
            f"{BLINK_PRESENCE}_mask": torch.zeros(2, 4, dtype=torch.bool),
        }
        _, mask = target_and_mask(batch, BLINK_PRESENCE)
        self.assertFalse(mask.any())

    def test_missing_target_raises(self):
        with self.assertRaises(BatchError):
            target_and_mask({}, BLINK_PRESENCE)

    def test_missing_target_mask_raises(self):
        with self.assertRaises(BatchError):
            target_and_mask({BLINK_PRESENCE: torch.rand(2, 4)}, BLINK_PRESENCE)


if __name__ == "__main__":
    unittest.main()


class TestOccludedEyes(unittest.TestCase):
    """Self-occlusion: past a yaw threshold the far eye is behind the nose.

    Every geometric check still passes there — the landmarks track a face that
    is genuinely present — so this is information no other signal provides. The
    rule is per eye, not per frame: the *near* eye of a turned head is perfectly
    visible.
    """

    def pose(self, yaw: float, steps: int = 3) -> torch.Tensor:
        return torch.tensor([[[yaw, 0.0, 0.0]] * steps])

    def test_a_frontal_head_occludes_nothing(self):
        for side in (LEFT, RIGHT):
            self.assertFalse(bool(occluded_eyes(self.pose(0.0), [side]).any()), side)

    def test_a_turned_head_occludes_exactly_one_eye(self):
        # The whole point of a per-eye rule: one eye is hidden, its partner is
        # not, and the frame stays usable for the partner.
        left = occluded_eyes(self.pose(70.0), [LEFT])
        right = occluded_eyes(self.pose(70.0), [RIGHT])
        self.assertNotEqual(bool(left.all()), bool(right.all()))

    def test_the_other_direction_occludes_the_other_eye(self):
        positive = occluded_eyes(self.pose(70.0), [LEFT])
        negative = occluded_eyes(self.pose(-70.0), [LEFT])
        self.assertNotEqual(bool(positive.all()), bool(negative.all()))

    def test_yaw_below_the_threshold_is_kept(self):
        self.assertFalse(bool(occluded_eyes(self.pose(40.0), [LEFT], threshold=45.0).any()))

    def test_the_threshold_is_configurable(self):
        # It is a dataloader hyperparameter precisely so it can be swept without
        # re-preprocessing.
        pose = self.pose(35.0)
        self.assertFalse(bool(occluded_eyes(pose, [LEFT], threshold=45.0).any()))
        self.assertTrue(bool(occluded_eyes(pose, [LEFT], threshold=30.0).any()))

    def test_an_unknown_side_is_never_occluded(self):
        # MRL records no side; without knowing which eye it is, the rule cannot
        # say whether this is the far one, so it must not guess.
        self.assertFalse(bool(occluded_eyes(self.pose(80.0), [UNKNOWN_EYE]).any()))

    def test_it_is_per_timestep(self):
        # A head that turns mid-window occludes only the turned frames.
        pose = torch.tensor([[[0.0, 0.0, 0.0], [70.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
        occluded = occluded_eyes(pose, [LEFT])
        if not bool(occluded.any()):
            occluded = occluded_eyes(-pose, [LEFT])
        self.assertEqual(occluded.shape, (1, 3))
        self.assertEqual(int(occluded.sum()), 1)

    def test_a_bad_pose_shape_is_rejected(self):
        with self.assertRaises(BatchError):
            occluded_eyes(torch.zeros(2, 3), [LEFT, RIGHT])

    def test_mismatched_sides_are_rejected(self):
        with self.assertRaises(BatchError):
            occluded_eyes(torch.zeros(2, 3, 3), [LEFT])


class TestApplyOcclusion(unittest.TestCase):
    """The rule masks validity; it never edits the crops."""

    def batch(self, yaw: float, side: str = LEFT) -> dict:
        return {
            EYE_IMAGE: torch.rand(1, 2, 3, 8, 8),
            f"{EYE_IMAGE}_mask": torch.ones(1, 2, dtype=torch.bool),
            HEAD_POSE: torch.tensor([[[yaw, 0.0, 0.0]] * 2]),
            SAMPLE_KEY: [f"rec|000000|{side}"],
        }

    def test_a_frontal_batch_is_unchanged(self):
        batch = self.batch(0.0)
        self.assertTrue(bool(apply_occlusion(batch)[f"{EYE_IMAGE}_mask"].all()))

    def test_a_turned_batch_masks_one_side(self):
        masked_left = apply_occlusion(self.batch(80.0, LEFT))[f"{EYE_IMAGE}_mask"]
        masked_right = apply_occlusion(self.batch(80.0, RIGHT))[f"{EYE_IMAGE}_mask"]
        self.assertNotEqual(bool(masked_left.all()), bool(masked_right.all()))

    def test_the_original_batch_is_not_mutated(self):
        # A caller inspecting the same batch afterwards must see what the loader
        # produced, not what the rule decided.
        batch = self.batch(80.0)
        original = batch[f"{EYE_IMAGE}_mask"].clone()
        apply_occlusion(batch)
        self.assertTrue(bool((batch[f"{EYE_IMAGE}_mask"] == original).all()))

    def test_the_crops_are_untouched(self):
        batch = self.batch(80.0)
        updated = apply_occlusion(batch)
        self.assertTrue(bool((updated[EYE_IMAGE] == batch[EYE_IMAGE]).all()))

    def test_a_batch_without_pose_passes_through(self):
        # The still corpora have no face to estimate pose from.
        batch = {
            EYE_IMAGE: torch.rand(1, 1, 3, 8, 8),
            f"{EYE_IMAGE}_mask": torch.ones(1, 1, dtype=torch.bool),
        }
        self.assertIs(apply_occlusion(batch), batch)

    def test_an_already_masked_position_stays_masked(self):
        # The rule only ever removes validity; it must not resurrect a frame the
        # builder already marked unreadable.
        batch = self.batch(0.0)
        batch[f"{EYE_IMAGE}_mask"] = torch.zeros(1, 2, dtype=torch.bool)
        self.assertFalse(bool(apply_occlusion(batch)[f"{EYE_IMAGE}_mask"].any()))
