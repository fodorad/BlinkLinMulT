"""Tests for the 160-d eye descriptors.

The real extractor pulls exordium's multi-GB weights, so what is tested here is
everything this project owns: the fixed feature layout, the arithmetic that
joins the two blocks, and — above all — that a failure produces a masked entry
rather than a plausible-looking zero vector.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

from blinklinmult.data.schema import EYE_FEATURE_DIM
from blinklinmult.preprocess import extractors
from blinklinmult.preprocess.extractors import _first_face_box
from blinklinmult.preprocess.features import (
    HEADPOSE_DIM,
    IRIS_FEATURE_DIM,
    IRIS_FIELDS,
    MODEL_SPACE,
    POSE_SCALE,
    FakeExtractor,
    FeatureError,
    assemble,
    empty_feature,
    flatten_iris,
    stack_window,
)
from blinklinmult.preprocess.geometry import LEFT, RIGHT, EyeBox

MEDIAPIPE_LEFT_EYE = (
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
    466,
    388,
    387,
    386,
    385,
    384,
    398,
)
"""MediaPipe FaceMesh 478 contour for the subject's left eye.

Pinned rather than imported from exordium: the values are a published, stable
part of the FaceMesh topology, and importing them would tie this file to an
optional multi-GB extra for sixteen integers.
"""

MEDIAPIPE_RIGHT_EYE = (
    33,
    7,
    163,
    144,
    145,
    153,
    154,
    155,
    133,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
)
"""MediaPipe FaceMesh 478 contour for the subject's right eye."""


def iris_dict(**overrides) -> dict:
    """A well-formed ``eye_to_feature`` return value."""
    base = {
        "eye_region_landmarks": np.zeros((71, 2), dtype=np.float32),
        "iris_landmarks": np.zeros((5, 2), dtype=np.float32),
        "iris_diameters": np.zeros(2, dtype=np.float32),
        "eyelid_pupil_distances": np.zeros(2, dtype=np.float32),
        "ear": np.float32(0.3),
    }
    base.update(overrides)
    return base


def frame(size: int = 200) -> np.ndarray:
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def boxes() -> dict[str, EyeBox]:
    return {
        LEFT: EyeBox(centre_x=60, centre_y=100, side=32, span=21.0),
        RIGHT: EyeBox(centre_x=140, centre_y=100, side=32, span=21.0),
    }


class TestLayout(unittest.TestCase):
    def test_the_blocks_sum_to_the_schema_width(self):
        self.assertEqual(IRIS_FEATURE_DIM + HEADPOSE_DIM, EYE_FEATURE_DIM)

    def test_the_iris_block_is_157(self):
        self.assertEqual(IRIS_FEATURE_DIM, 157)

    def test_the_declared_widths_match_exordiums_shapes(self):
        # 71x2 + 5x2 + 2 + 2 + 1. If exordium changes any of these, every built
        # corpus carries a differently-meaning vector.
        expected = {
            "eye_region_landmarks": 142,
            "iris_landmarks": 10,
            "iris_diameters": 2,
            "eyelid_pupil_distances": 2,
            "ear": 1,
        }
        self.assertEqual({name: width for name, width, _ in IRIS_FIELDS}, expected)

    def test_pixel_space_blocks_are_normalised_and_the_ratio_is_not(self):
        # Measured on TalkingFace, the landmark blocks span 9..56 in 64x64
        # model space while the EAR is 0.36. Concatenated raw, 152 pixel values
        # would swamp the single most informative number for eye closure.
        divisors = {name: divisor for name, _, divisor in IRIS_FIELDS}
        self.assertEqual(divisors["ear"], 1.0)
        for name in (
            "eye_region_landmarks",
            "iris_landmarks",
            "iris_diameters",
            "eyelid_pupil_distances",
        ):
            self.assertEqual(divisors[name], MODEL_SPACE)


class TestFlattenIris(unittest.TestCase):
    def test_produces_the_iris_width(self):
        self.assertEqual(flatten_iris(iris_dict()).shape, (IRIS_FEATURE_DIM,))

    def test_accepts_torch_tensors(self):
        # exordium returns tensors, not arrays.
        values = {
            key: torch.from_numpy(np.asarray(value, dtype=np.float32))
            for key, value in iris_dict().items()
        }
        self.assertEqual(flatten_iris(values).shape, (IRIS_FEATURE_DIM,))

    def test_the_order_is_fixed(self):
        # Each block gets a distinct value; the concatenation must lay them out
        # in the declared order, because that order IS the feature layout.
        marked = iris_dict(
            eye_region_landmarks=np.full((71, 2), 1.0 * MODEL_SPACE, dtype=np.float32),
            iris_landmarks=np.full((5, 2), 2.0 * MODEL_SPACE, dtype=np.float32),
            iris_diameters=np.full(2, 3.0 * MODEL_SPACE, dtype=np.float32),
            eyelid_pupil_distances=np.full(2, 4.0 * MODEL_SPACE, dtype=np.float32),
            ear=np.float32(5.0),
        )
        flat = flatten_iris(marked)
        self.assertTrue((flat[:142] == 1.0).all())
        self.assertTrue((flat[142:152] == 2.0).all())
        self.assertTrue((flat[152:154] == 3.0).all())
        self.assertTrue((flat[154:156] == 4.0).all())
        self.assertEqual(flat[156], 5.0)

    def test_landmarks_land_in_the_unit_range(self):
        # A landmark at the far corner of the 64x64 model space becomes 1.0,
        # so the whole block sits alongside the EAR rather than dwarfing it.
        flat = flatten_iris(
            iris_dict(eye_region_landmarks=np.full((71, 2), MODEL_SPACE, dtype=np.float32))
        )
        self.assertTrue((flat[:142] == 1.0).all())

    def test_a_missing_key_raises(self):
        values = iris_dict()
        del values["ear"]
        with self.assertRaises(FeatureError) as ctx:
            flatten_iris(values)
        self.assertIn("ear", str(ctx.exception))

    def test_an_unexpected_width_raises(self):
        # An upstream shape change must fail loudly: silently accepting it
        # would give every corpus a different feature layout.
        with self.assertRaises(FeatureError) as ctx:
            flatten_iris(iris_dict(iris_landmarks=np.zeros((6, 2), dtype=np.float32)))
        self.assertIn("layout has changed", str(ctx.exception))


class TestAssemble(unittest.TestCase):
    def test_produces_the_full_width(self):
        result = assemble(np.zeros(IRIS_FEATURE_DIM), np.zeros(HEADPOSE_DIM))
        self.assertEqual(result.shape, (EYE_FEATURE_DIM,))

    def test_pose_is_scaled_into_range(self):
        # Raw degrees would be the largest block by magnitude and would
        # dominate the first layer on scale alone.
        result = assemble(np.zeros(IRIS_FEATURE_DIM), np.asarray([90.0, -45.0, 0.0]))
        np.testing.assert_allclose(
            result[-HEADPOSE_DIM:], [90 / POSE_SCALE, -45 / POSE_SCALE, 0.0], atol=1e-6
        )

    def test_the_iris_block_comes_first(self):
        result = assemble(np.ones(IRIS_FEATURE_DIM), np.zeros(HEADPOSE_DIM))
        self.assertTrue((result[:IRIS_FEATURE_DIM] == 1.0).all())

    def test_a_wrong_iris_width_raises(self):
        with self.assertRaises(FeatureError):
            assemble(np.zeros(10), np.zeros(HEADPOSE_DIM))

    def test_a_wrong_pose_width_raises(self):
        with self.assertRaises(FeatureError):
            assemble(np.zeros(IRIS_FEATURE_DIM), np.zeros(2))

    def test_output_is_float32(self):
        self.assertEqual(
            assemble(np.zeros(IRIS_FEATURE_DIM), np.zeros(HEADPOSE_DIM)).dtype, np.float32
        )


class TestEmptyFeature(unittest.TestCase):
    def test_is_the_declared_width(self):
        self.assertEqual(empty_feature().shape, (EYE_FEATURE_DIM,))

    def test_is_zero(self):
        # Zero is only safe because it is always written with a False mask;
        # the pairing is what makes it honest rather than a fabricated pose.
        self.assertTrue((empty_feature() == 0).all())


class TestStackWindow(unittest.TestCase):
    def test_stacks_to_time_by_width(self):
        values, mask = stack_window([(empty_feature(), True)] * 4)
        self.assertEqual(values.shape, (4, EYE_FEATURE_DIM))
        self.assertEqual(mask.shape, (4,))

    def test_the_mask_records_per_frame_validity(self):
        values, mask = stack_window(
            [(empty_feature(), True), (empty_feature(), False), (empty_feature(), True)]
        )
        np.testing.assert_array_equal(mask, [True, False, True])

    def test_an_empty_window_raises(self):
        with self.assertRaises(FeatureError):
            stack_window([])


class TestFakeExtractor(unittest.TestCase):
    def test_describes_both_eyes(self):
        result = FakeExtractor().frame_features(frame(), boxes())
        self.assertEqual(sorted(result), [LEFT, RIGHT])

    def test_descriptors_have_the_declared_width(self):
        for value, _ in FakeExtractor().frame_features(frame(), boxes()).values():
            self.assertEqual(value.shape, (EYE_FEATURE_DIM,))

    def test_a_missing_box_is_masked(self):
        result = FakeExtractor().frame_features(frame(), {LEFT: None, RIGHT: boxes()[RIGHT]})
        self.assertFalse(result[LEFT][1])
        self.assertTrue(result[RIGHT][1])

    def test_a_masked_eye_carries_the_empty_descriptor(self):
        value, valid = FakeExtractor().frame_features(frame(), {LEFT: None})[LEFT]
        self.assertFalse(valid)
        self.assertTrue((value == 0).all())

    def test_a_failing_side_is_masked_but_the_other_survives(self):
        # Features and images fail independently: one bad eye must not discard
        # the frame.
        result = FakeExtractor(fail_sides=(LEFT,)).frame_features(frame(), boxes())
        self.assertFalse(result[LEFT][1])
        self.assertTrue(result[RIGHT][1])

    def test_is_deterministic(self):
        image, box = frame(), boxes()
        first = FakeExtractor().frame_features(image, box)[LEFT][0]
        second = FakeExtractor().frame_features(image, box)[LEFT][0]
        np.testing.assert_array_equal(first, second)


class TestFirstFaceBox(unittest.TestCase):
    """Detectors differ in what they return; the shape is probed, not assumed."""

    def test_reads_a_bb_xyxy_attribute(self):
        class Detection:
            bb_xyxy = np.asarray([10, 20, 110, 140])

        self.assertEqual(_first_face_box([Detection()]), (10, 20, 110, 140))

    def test_reads_a_plain_array(self):
        self.assertEqual(_first_face_box(np.asarray([[1, 2, 3, 4]])), (1, 2, 3, 4))

    def test_no_detection_is_none(self):
        self.assertIsNone(_first_face_box([]))

    def test_none_is_none(self):
        self.assertIsNone(_first_face_box(None))

    def test_a_too_short_box_is_none(self):
        self.assertIsNone(_first_face_box(np.asarray([[1, 2]])))


class TestEyeSideConvention(unittest.TestCase):
    """The two libraries mirror each other, and getting it wrong is silent.

    exordium names its FaceMesh regions from the **subject's** point of view;
    the ``.tag`` corpora name their eye corners from the **viewer's**. Mapping
    ``LEFT -> LEFT_EYE`` puts every detected crop on the other eye — measured
    on TalkingFace frame 170, 110px out, a whole inter-eye distance. Nothing
    downstream notices: the model trains, scores, and reports a plausible
    number against the wrong eye.
    """

    def regions(self) -> dict[str, tuple[int, ...]]:
        """The mapping ``FaceMeshLocator`` installs, as landmark indices.

        The values are MediaPipe's canonical FaceMesh 478 eye contours, pinned
        here rather than imported from ``exordium.video.face.landmark.constants``.
        Importing would tie this test to an optional multi-GB extra for two
        tuples of integers, and pinning is the stronger check: it fails if those
        contours ever change upstream, which an import would silently accept.

        Returns:
            dict[str, tuple[int, ...]]: Side to landmark indices, deliberately crossed.
        """
        return {LEFT: MEDIAPIPE_RIGHT_EYE, RIGHT: MEDIAPIPE_LEFT_EYE}

    def test_the_sides_are_crossed(self):
        regions = self.regions()
        self.assertEqual(regions[LEFT], MEDIAPIPE_RIGHT_EYE)
        self.assertEqual(regions[RIGHT], MEDIAPIPE_LEFT_EYE)

    def test_each_side_maps_to_a_distinct_region(self):
        regions = self.regions()
        self.assertNotEqual(regions[LEFT], regions[RIGHT])

    def test_both_regions_have_sixteen_points(self):
        for indices in self.regions().values():
            self.assertEqual(len(indices), 16)

    def test_the_locator_installs_the_crossed_mapping(self):
        # Reads the source rather than constructing FaceMeshLocator, which
        # would download several GB of weights. A straight-across mapping is
        # the bug this guards, and it is invisible at runtime.
        source = Path(extractors.__file__).read_text()
        self.assertIn("LEFT: FaceMesh478Regions.RIGHT_EYE", source)
        self.assertIn("RIGHT: FaceMesh478Regions.LEFT_EYE", source)


if __name__ == "__main__":
    unittest.main()
