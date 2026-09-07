"""Tests for the shared preprocessing helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from blinklinmult.data.schema import SUBSETS
from blinklinmult.preprocess.common import (
    DEFAULT_SPLIT_RATIOS,
    PreprocessError,
    assign_splits,
    crop_square,
    eye_centre,
    eye_span,
    list_files,
    normalise_image,
    proportional_splits,
    split_of,
    stack_eye_window,
    subsample_evenly,
)


def gradient_image(height: int = 100, width: int = 120) -> np.ndarray:
    """An image whose pixel values encode their own coordinates."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width)[None, :] % 256
    image[..., 1] = np.arange(height)[:, None] % 256
    image[..., 2] = 42
    return image


class TestCropSquare(unittest.TestCase):
    def test_interior_crop_has_requested_size(self):
        patch = crop_square(gradient_image(), 50, 50, 20)
        self.assertEqual(patch.shape, (20, 20, 3))

    def test_interior_crop_is_centred(self):
        image = gradient_image()
        patch = crop_square(image, 50, 60, 10)
        # The crop starts at centre - size // 2.
        np.testing.assert_array_equal(patch, image[55:65, 45:55])

    def test_crop_at_the_left_edge_is_padded_not_shrunk(self):
        # 1.x bug: clipping to the image bounds returned a smaller patch, which a
        # later resize then stretched by a varying amount.
        patch = crop_square(gradient_image(), 2, 50, 20)
        self.assertEqual(patch.shape, (20, 20, 3))
        # The columns left of the image are zero-filled.
        self.assertTrue((patch[:, :8] == 0).all())

    def test_crop_at_the_bottom_right_is_padded(self):
        patch = crop_square(gradient_image(100, 120), 118, 98, 20)
        self.assertEqual(patch.shape, (20, 20, 3))
        self.assertTrue((patch[-8:, :] == 0).all())

    def test_crop_entirely_outside_returns_blank(self):
        patch = crop_square(gradient_image(), -500, -500, 16)
        self.assertEqual(patch.shape, (16, 16, 3))
        self.assertTrue((patch == 0).all())

    def test_crop_larger_than_image_is_padded(self):
        patch = crop_square(gradient_image(20, 20), 10, 10, 60)
        self.assertEqual(patch.shape, (60, 60, 3))

    def test_zero_size_raises(self):
        with self.assertRaises(PreprocessError):
            crop_square(gradient_image(), 10, 10, 0)

    def test_non_three_channel_image_raises(self):
        with self.assertRaises(PreprocessError):
            crop_square(np.zeros((10, 10), dtype=np.uint8), 5, 5, 4)

    def test_dtype_is_preserved(self):
        patch = crop_square(gradient_image(), 50, 50, 8)
        self.assertEqual(patch.dtype, np.uint8)


class TestEyeGeometry(unittest.TestCase):
    def test_centre_is_the_midpoint(self):
        self.assertEqual(eye_centre([10, 20, 30, 40]), (20, 30))

    def test_centre_accepts_an_array(self):
        self.assertEqual(eye_centre(np.array([0, 0, 10, 10])), (5, 5))

    def test_span_is_the_corner_distance(self):
        self.assertAlmostEqual(eye_span([0, 0, 3, 4]), 5.0)

    def test_span_of_coincident_corners_is_zero(self):
        self.assertAlmostEqual(eye_span([7, 7, 7, 7]), 0.0)


class TestNormaliseImage(unittest.TestCase):
    def test_output_is_channel_first(self):
        result = normalise_image(gradient_image(30, 40))
        self.assertEqual(result.shape, (3, 30, 40))

    def test_values_are_scaled_to_unit_range(self):
        result = normalise_image(np.full((4, 4, 3), 255, dtype=np.uint8))
        np.testing.assert_allclose(result, 1.0)

    def test_zero_stays_zero(self):
        result = normalise_image(np.zeros((4, 4, 3), dtype=np.uint8))
        np.testing.assert_allclose(result, 0.0)

    def test_dtype_is_float32(self):
        self.assertEqual(normalise_image(gradient_image()).dtype, np.float32)


class TestStackEyeWindow(unittest.TestCase):
    def frames(self, count: int, size: int = 8) -> list[np.ndarray]:
        return [np.full((3, size, size), i, dtype=np.float32) for i in range(count)]

    def test_shape_is_time_channels_height_width(self):
        # One eye per sample: no eye axis.
        self.assertEqual(stack_eye_window(self.frames(5)).shape, (5, 3, 8, 8))

    def test_frames_keep_their_order(self):
        stacked = stack_eye_window(self.frames(3))
        self.assertEqual(stacked[0].sum(), 0.0)
        self.assertEqual(stacked[1].sum(), 3 * 8 * 8)

    def test_single_frame_window(self):
        self.assertEqual(stack_eye_window(self.frames(1)).shape[0], 1)

    def test_ragged_frames_raise(self):
        with self.assertRaises(PreprocessError) as ctx:
            stack_eye_window([*self.frames(1), np.zeros((3, 4, 4), dtype=np.float32)])
        self.assertIn("same shape", str(ctx.exception))

    def test_wrong_channel_count_raises(self):
        with self.assertRaises(PreprocessError):
            stack_eye_window([np.zeros((1, 8, 8), dtype=np.float32)])

    def test_empty_window_raises(self):
        with self.assertRaises(PreprocessError):
            stack_eye_window([])

    def test_dtype_is_float32(self):
        self.assertEqual(stack_eye_window(self.frames(2)).dtype, np.float32)


class TestSplitOf(unittest.TestCase):
    def test_returns_a_known_split(self):
        self.assertIn(split_of("group-1"), SUBSETS)

    def test_is_deterministic(self):
        self.assertEqual(split_of("abc", salt="x"), split_of("abc", salt="x"))

    def test_salt_changes_the_assignment_distribution(self):
        groups = [f"g{i}" for i in range(200)]
        first = [split_of(g, salt="a") for g in groups]
        second = [split_of(g, salt="b") for g in groups]
        self.assertNotEqual(first, second)

    def test_all_weight_on_one_split(self):
        ratios = {"train": 0.0, "valid": 0.0, "test": 1.0}
        for index in range(30):
            self.assertEqual(split_of(f"g{index}", ratios), "test")

    def test_proportions_are_approximately_honoured(self):
        groups = [f"group-{i}" for i in range(4000)]
        assignments = [split_of(g, salt="ratio-test") for g in groups]
        for name, expected in DEFAULT_SPLIT_RATIOS.items():
            observed = assignments.count(name) / len(groups)
            self.assertAlmostEqual(observed, expected, delta=0.03)

    def test_ratios_not_summing_to_one_raise(self):
        with self.assertRaises(PreprocessError) as ctx:
            split_of("g", {"train": 0.5, "valid": 0.2, "test": 0.2})
        self.assertIn("sum to 1.0", str(ctx.exception))

    def test_negative_ratio_raises(self):
        with self.assertRaises(PreprocessError):
            split_of("g", {"train": 1.5, "valid": -0.5, "test": 0.0})

    def test_unknown_split_name_raises(self):
        with self.assertRaises(PreprocessError):
            split_of("g", {"train": 0.5, "holdout": 0.5})


class TestAssignSplits(unittest.TestCase):
    def test_every_group_receives_a_split(self):
        groups = [f"g{i}" for i in range(20)]
        assignment = assign_splits(groups, salt="t")
        self.assertEqual(set(assignment), set(groups))

    def test_duplicates_collapse(self):
        assignment = assign_splits(["a", "a", "b"], salt="t")
        self.assertEqual(len(assignment), 2)

    def test_a_group_lands_in_exactly_one_split(self):
        assignment = assign_splits([f"g{i}" for i in range(50)], salt="t")
        for value in assignment.values():
            self.assertIn(value, SUBSETS)

    def test_empty_input_raises(self):
        with self.assertRaises(PreprocessError):
            assign_splits([])

    def test_assignment_is_stable_when_groups_are_added(self):
        # A seeded shuffle would reassign everything; hashing does not.
        first = assign_splits([f"g{i}" for i in range(10)], salt="s")
        second = assign_splits([f"g{i}" for i in range(20)], salt="s")
        for group, split in first.items():
            self.assertEqual(second[group], split)


class TestListFiles(unittest.TestCase):
    """macOS writes a ``._name`` sidecar beside every real file.

    They carry the same extension and match the same glob, so an unfiltered
    listing of an unzipped MRL-Eye returns ~85k sidecars alongside ~85k images
    and the first one reaching a filename parser fails on a name that was never
    meant to be read.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_sidecars_are_skipped(self):
        (self.tmp / "s0001_00001_0_0_0_0_0_01.png").touch()
        (self.tmp / "._s0001_00001_0_0_0_0_0_01.png").touch()
        found = list_files(self.tmp, "*.png")
        self.assertEqual([p.name for p in found], ["s0001_00001_0_0_0_0_0_01.png"])

    def test_results_are_sorted(self):
        for name in ("c.png", "a.png", "b.png"):
            (self.tmp / name).touch()
        self.assertEqual(
            [p.name for p in list_files(self.tmp, "*.png")], ["a.png", "b.png", "c.png"]
        )

    def test_a_nested_glob_is_honoured(self):
        (self.tmp / "rec1").mkdir()
        (self.tmp / "rec1" / "a.tag").touch()
        (self.tmp / "rec1" / "._a.tag").touch()
        self.assertEqual(len(list_files(self.tmp, "*/*.tag")), 1)

    def test_no_matches_is_an_empty_list(self):
        self.assertEqual(list_files(self.tmp, "*.png"), [])

    def test_a_directory_of_only_sidecars_yields_nothing(self):
        (self.tmp / "._orphan.png").touch()
        self.assertEqual(list_files(self.tmp, "*.png"), [])


class TestSubsampleEvenly(unittest.TestCase):
    """MRL-Eye stores every closed-eye image before every open one.

    Taking the head of that list yields a single-class dataset, so a smoke run
    trains on data containing one label and looks perfectly healthy doing it.
    """

    def paths(self, count: int) -> list[Path]:
        return [Path(f"{index:04d}.png") for index in range(count)]

    def test_spreads_across_the_whole_list(self):
        # The corpus is label-sorted, so a sample must reach the far end.
        picked = subsample_evenly(self.paths(100), 5)
        self.assertEqual(
            [p.name for p in picked], ["0000.png", "0020.png", "0040.png", "0060.png", "0080.png"]
        )

    def test_both_classes_survive_a_label_sorted_list(self):
        labels = [0] * 60 + [1] * 40
        picked = subsample_evenly(self.paths(100), 10)
        chosen = [labels[int(p.stem)] for p in picked]
        self.assertIn(0, chosen)
        self.assertIn(1, chosen)

    def test_returns_the_requested_count(self):
        self.assertEqual(len(subsample_evenly(self.paths(100), 7)), 7)

    def test_a_limit_beyond_the_length_keeps_everything(self):
        items = self.paths(3)
        self.assertEqual(subsample_evenly(items, 10), items)

    def test_order_is_preserved(self):
        picked = subsample_evenly(self.paths(50), 8)
        self.assertEqual(picked, sorted(picked))

    def test_a_non_positive_limit_keeps_nothing(self):
        self.assertEqual(subsample_evenly(self.paths(10), 0), [])

    def test_never_repeats_an_item(self):
        picked = subsample_evenly(self.paths(10), 10)
        self.assertEqual(len(set(picked)), 10)


class TestProportionalSplits(unittest.TestCase):
    """Shuffling hits the requested ratios; hashing only approximates them.

    ``assign_splits`` assigns each group independently, so the proportions are
    a random draw. Eight recordings at 70/15/15 came out 5/3/0 --
    no test split at all. This divides instead.
    """

    def groups(self, count: int) -> list[str]:
        return [f"g{index:04d}" for index in range(count)]

    def test_the_counts_are_exact(self):
        from collections import Counter

        counts = Counter(proportional_splits(self.groups(100), salt="t").values())
        self.assertEqual(dict(counts), {"train": 70, "valid": 15, "test": 15})

    def test_every_group_is_assigned_exactly_once(self):
        groups = self.groups(50)
        result = proportional_splits(groups, salt="t")
        self.assertEqual(sorted(result), sorted(groups))

    def test_a_small_corpus_still_fills_every_split(self):
        # The failure this replaces: 8 groups, hashed, gave an empty test split.
        from collections import Counter

        counts = Counter(proportional_splits(self.groups(8), salt="t").values())
        self.assertEqual(sum(counts.values()), 8)
        for split in ("train", "valid", "test"):
            self.assertGreater(counts[split], 0, split)

    def test_the_division_is_reproducible(self):
        first = proportional_splits(self.groups(40), salt="t", seed=7)
        second = proportional_splits(self.groups(40), salt="t", seed=7)
        self.assertEqual(first, second)

    def test_a_different_seed_divides_differently(self):
        first = proportional_splits(self.groups(40), salt="t", seed=7)
        second = proportional_splits(self.groups(40), salt="t", seed=8)
        self.assertNotEqual(first, second)

    def test_input_order_does_not_matter(self):
        # The groups are sorted before shuffling; without that a set's
        # iteration order would make "seeded" reproducible in name only.
        forward = proportional_splits(self.groups(30), salt="t")
        backward = proportional_splits(list(reversed(self.groups(30))), salt="t")
        self.assertEqual(forward, backward)

    def test_duplicate_groups_collapse(self):
        result = proportional_splits(["a", "a", "b"], salt="t")
        self.assertEqual(len(result), 2)

    def test_custom_ratios_are_honoured(self):
        from collections import Counter

        counts = Counter(
            proportional_splits(
                self.groups(10), {"train": 0.5, "valid": 0.2, "test": 0.3}, salt="t"
            ).values()
        )
        self.assertEqual(dict(counts), {"train": 5, "valid": 2, "test": 3})

    def test_no_groups_raises(self):
        with self.assertRaises(PreprocessError):
            proportional_splits([], salt="t")

    def test_invalid_ratios_raise(self):
        with self.assertRaises(PreprocessError):
            proportional_splits(self.groups(10), {"train": 0.5}, salt="t")
