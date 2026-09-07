"""Tests for locating eye regions.

The property under test is that the annotated route and the detected route
produce boxes on the same definition — same centre, same scale rule — so a
model trained on one can be evaluated on the other, and the cost of automatic
localisation is measurable rather than confounded by a different crop.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.preprocess.geometry import (
    EYE_CROP_SCALE,
    EYE_SPAN_RATIO,
    LEFT,
    MIN_EYE_SPAN,
    RIGHT,
    EyeBox,
    GeometryError,
    box_from_corners,
    box_from_eye_centres,
    box_from_landmarks,
    boxes_from_annotation,
    pose_from_keypoints,
)


def corners(x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    return np.asarray([x1, y1, x2, y2], dtype=np.int32)


class Record:
    """The two fields :func:`boxes_from_annotation` reads."""

    def __init__(self, left: np.ndarray, right: np.ndarray):
        self.left_eye_corners = left
        self.right_eye_corners = right


class TestBoxFromCorners(unittest.TestCase):
    def test_side_scales_with_the_eye(self):
        # A 100px-wide eye at 1.5x gives a 150px box.
        box = box_from_corners(corners(100, 200, 200, 200))
        self.assertIsNotNone(box)
        self.assertEqual(box.side, int(round(100 * EYE_CROP_SCALE)))

    def test_box_is_centred_on_the_eye(self):
        box = box_from_corners(corners(100, 200, 200, 200))
        self.assertEqual((box.centre_x, box.centre_y), (150, 200))

    def test_a_tilted_eye_uses_the_true_distance(self):
        # 3-4-5: the span is 5, not the horizontal 3.
        box = box_from_corners(corners(0, 0, 30, 40))
        self.assertAlmostEqual(box.span, 50.0, places=4)

    def test_a_degenerate_annotation_yields_no_box(self):
        # Two coincident points: cropping would magnify a single pixel.
        self.assertIsNone(box_from_corners(corners(10, 10, 10, 10)))

    def test_below_the_minimum_span_yields_no_box(self):
        self.assertIsNone(box_from_corners(corners(0, 0, int(MIN_EYE_SPAN) - 1, 0)))

    def test_the_scale_is_configurable(self):
        wide = box_from_corners(corners(0, 0, 100, 0), scale=2.0)
        self.assertEqual(wide.side, 200)


class TestEyeBox(unittest.TestCase):
    def test_xyxy_spans_exactly_the_side(self):
        box = EyeBox(centre_x=50, centre_y=50, side=20, span=10.0)
        x1, y1, x2, y2 = box.xyxy
        self.assertEqual((x2 - x1, y2 - y1), (20, 20))

    def test_crop_returns_a_square_patch(self):
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        patch = EyeBox(centre_x=100, centre_y=100, side=32, span=21.0).crop(frame)
        self.assertEqual(patch.shape, (32, 32, 3))

    def test_a_box_off_the_frame_edge_is_padded_not_clipped(self):
        # Constant scale matters: a clipped crop would be resized by a
        # different factor and the eye would come out a different size.
        frame = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
        patch = EyeBox(centre_x=2, centre_y=2, side=32, span=21.0).crop(frame)
        self.assertEqual(patch.shape, (32, 32, 3))


class TestVisibleFraction(unittest.TestCase):
    """A box may sit off-frame: MPEblink annotates faces past the frame edge."""

    def test_a_contained_box_is_fully_visible(self):
        box = EyeBox(centre_x=100, centre_y=100, side=20, span=10.0)
        self.assertEqual(box.visible_fraction(200, 200), 1.0)

    def test_a_box_entirely_above_the_frame_is_zero(self):
        # The real case: MPEblink puts eye landmarks at y = -226 for a head
        # tracked past the top edge. Cropping gives a black patch, which must
        # not then carry the tracklet's blink label.
        box = EyeBox(centre_x=800, centre_y=-226, side=60, span=30.0)
        self.assertEqual(box.visible_fraction(998, 1920), 0.0)

    def test_a_half_overlapping_box_is_about_half(self):
        # Centred on the left edge: half its width is outside.
        box = EyeBox(centre_x=0, centre_y=100, side=40, span=20.0)
        self.assertAlmostEqual(box.visible_fraction(200, 200), 0.5, places=2)

    def test_a_corner_box_multiplies_both_axes(self):
        # Half out horizontally and half out vertically leaves a quarter.
        box = EyeBox(centre_x=0, centre_y=0, side=40, span=20.0)
        self.assertAlmostEqual(box.visible_fraction(200, 200), 0.25, places=2)


class TestShiftedInto(unittest.TestCase):
    """A visible eye near the frame edge keeps real pixels, not black padding."""

    def test_a_contained_box_is_unchanged(self):
        box = EyeBox(centre_x=100, centre_y=100, side=20, span=10.0)
        moved = box.shifted_into(200, 200)
        self.assertEqual((moved.centre_x, moved.centre_y), (100, 100))

    def test_a_box_over_the_left_edge_slides_right(self):
        box = EyeBox(centre_x=5, centre_y=100, side=40, span=20.0)
        self.assertEqual(box.shifted_into(200, 200).xyxy[0], 0)

    def test_a_box_over_the_bottom_edge_slides_up(self):
        box = EyeBox(centre_x=100, centre_y=195, side=40, span=20.0)
        self.assertEqual(box.shifted_into(200, 200).xyxy[3], 200)

    def test_the_side_never_changes(self):
        # Clipping instead of shifting would make the box non-square, and the
        # later resize would then stretch it differently from a mid-frame crop.
        box = EyeBox(centre_x=-10, centre_y=-10, side=40, span=20.0)
        self.assertEqual(box.shifted_into(200, 200).side, 40)

    def test_a_shifted_box_is_fully_visible(self):
        box = EyeBox(centre_x=2, centre_y=198, side=40, span=20.0)
        self.assertEqual(box.shifted_into(200, 200).visible_fraction(200, 200), 1.0)

    def test_a_box_larger_than_the_frame_is_clamped(self):
        # Cannot be made to fit; the honest outcome is a padded crop at the
        # origin rather than a silently resized one.
        box = EyeBox(centre_x=50, centre_y=50, side=300, span=150.0)
        self.assertEqual(box.shifted_into(200, 200).xyxy[:2], (0, 0))


class TestBoxFromLandmarks(unittest.TestCase):
    def test_uses_the_landmark_extent(self):
        points = np.asarray([[100, 200], [150, 195], [200, 200], [150, 210]], dtype=np.float32)
        box = box_from_landmarks(points)
        self.assertIsNotNone(box)
        self.assertEqual(box.centre_x, 150)

    def test_agrees_with_the_annotated_route_on_the_same_eye(self):
        # The two routes must define the crop identically, or a model trained
        # on annotations cannot be evaluated with detection.
        eye = corners(100, 200, 200, 200)
        from_corners = box_from_corners(eye)
        from_points = box_from_landmarks(
            np.asarray([[100, 200], [200, 200], [150, 198]], dtype=np.float32)
        )
        self.assertEqual(from_points.side, from_corners.side)
        self.assertEqual(from_points.centre_x, from_corners.centre_x)

    def test_degenerate_landmarks_yield_no_box(self):
        points = np.asarray([[10, 10], [10, 10]], dtype=np.float32)
        self.assertIsNone(box_from_landmarks(points))

    def test_a_malformed_array_raises(self):
        with self.assertRaises(GeometryError):
            box_from_landmarks(np.zeros((3, 3), dtype=np.float32))

    def test_a_single_point_raises(self):
        with self.assertRaises(GeometryError):
            box_from_landmarks(np.zeros((1, 2), dtype=np.float32))


class TestBoxesFromAnnotation(unittest.TestCase):
    def test_both_eyes_are_located(self):
        boxes = boxes_from_annotation(
            Record(corners(100, 200, 150, 200), corners(300, 200, 350, 200))
        )
        self.assertEqual(sorted(boxes), [LEFT, RIGHT])
        self.assertEqual(boxes[LEFT].centre_x, 125)
        self.assertEqual(boxes[RIGHT].centre_x, 325)

    def test_one_missing_eye_does_not_lose_the_other(self):
        boxes = boxes_from_annotation(Record(corners(0, 0, 0, 0), corners(300, 200, 350, 200)))
        self.assertIsNone(boxes[LEFT])
        self.assertIsNotNone(boxes[RIGHT])


class TestCropScale(unittest.TestCase):
    def test_the_default_carries_lid_and_brow_without_the_other_eye(self):
        # Measured on TalkingFace: median eye span 50px, face width 246px. The
        # crop must be wider than the eye and much narrower than the face.
        span, face_width = 50.0, 246.0
        side = span * EYE_CROP_SCALE
        self.assertGreater(side, span)
        self.assertLess(side, face_width / 2)


if __name__ == "__main__":
    unittest.main()


class TestBoxFromEyeCentres(unittest.TestCase):
    """Crop boxes built from two eye centres alone.

    The streaming path: a face detector's pose output gives one point per eye,
    so the eye's span has to be inferred from the distance between them. These
    pin the properties that inference rests on.
    """

    LEFT_CENTRE = np.array([100.0, 200.0])
    RIGHT_CENTRE = np.array([200.0, 200.0])
    """A level pair 100 px apart, so the arithmetic is checkable by hand."""

    def test_the_box_sits_on_the_requested_eye(self) -> None:
        """Centre comes straight from the keypoint, not from between them."""
        left = box_from_eye_centres(self.LEFT_CENTRE, self.RIGHT_CENTRE, which=LEFT)
        right = box_from_eye_centres(self.LEFT_CENTRE, self.RIGHT_CENTRE, which=RIGHT)
        self.assertEqual((left.centre_x, left.centre_y), (100, 200))
        self.assertEqual((right.centre_x, right.centre_y), (200, 200))

    def test_span_is_the_calibrated_fraction_of_inter_eye_distance(self) -> None:
        """100 px apart at ratio 0.412 gives a 41.2 px span.

        Hand-derivable, so a change to the constant fails here loudly rather
        than shifting every crop in the streaming pipeline silently.
        """
        box = box_from_eye_centres(self.LEFT_CENTRE, self.RIGHT_CENTRE, which=LEFT)
        self.assertAlmostEqual(box.span, 100.0 * EYE_SPAN_RATIO, places=6)

    def test_side_follows_the_shared_crop_scale(self) -> None:
        """The same scale rule as every other path, so crops are comparable."""
        box = box_from_eye_centres(self.LEFT_CENTRE, self.RIGHT_CENTRE, which=LEFT)
        self.assertEqual(box.side, round(100.0 * EYE_SPAN_RATIO * EYE_CROP_SCALE))

    def test_scaling_the_face_scales_the_box(self) -> None:
        """Twice the inter-eye distance gives twice the span.

        This is the property the whole approximation rests on: a face closer to
        the camera must get a proportionally larger crop, or the eye is framed
        differently at different distances.
        """
        near = box_from_eye_centres(np.array([0.0, 0.0]), np.array([200.0, 0.0]), which=LEFT)
        far = box_from_eye_centres(np.array([0.0, 0.0]), np.array([100.0, 0.0]), which=LEFT)
        self.assertAlmostEqual(near.span / far.span, 2.0, places=6)

    def test_a_tilted_face_still_works(self) -> None:
        """Span uses the true distance, not the horizontal component."""
        box = box_from_eye_centres(np.array([0.0, 0.0]), np.array([60.0, 80.0]), which=RIGHT)
        self.assertAlmostEqual(box.span, 100.0 * EYE_SPAN_RATIO, places=6)

    def test_coincident_centres_give_no_box(self) -> None:
        """A collapsed detection must not become a tiny crop of nothing."""
        point = np.array([50.0, 50.0])
        self.assertIsNone(box_from_eye_centres(point, point, which=LEFT))

    def test_the_two_boxes_do_not_overlap_on_a_normal_face(self) -> None:
        """Left and right crops must frame different eyes.

        With span at 0.412 of the inter-eye distance and the crop scale at 2.0,
        each box is ~0.82 of that distance wide, so they abut without crossing.
        """
        left = box_from_eye_centres(self.LEFT_CENTRE, self.RIGHT_CENTRE, which=LEFT)
        right = box_from_eye_centres(self.LEFT_CENTRE, self.RIGHT_CENTRE, which=RIGHT)
        self.assertLess(left.centre_x + left.side / 2, right.centre_x + right.side / 2)
        self.assertGreater(right.centre_x - right.side / 2, left.centre_x - left.side / 2)

    def test_aspect_is_absent_rather_than_invented(self) -> None:
        """Two centres cannot imply an eye contour, so aspect stays at zero."""
        box = box_from_eye_centres(self.LEFT_CENTRE, self.RIGHT_CENTRE, which=LEFT)
        self.assertEqual(box.aspect, 0.0)

    def test_an_unknown_side_is_refused(self) -> None:
        """Cropping the wrong eye is the failure this guards against."""
        with self.assertRaises(GeometryError):
            box_from_eye_centres(self.LEFT_CENTRE, self.RIGHT_CENTRE, which="middle")


class TestPoseFromKeypoints(unittest.TestCase):
    """Head rotation estimated from five facial keypoints.

    Roll and pitch track 6DRepNet at correlation +0.94 on real video; these
    tests pin the geometry and the sign conventions that agreement depends on.
    """

    @staticmethod
    def _face(nose_x: float = 0.0, nose_y: float = 50.0, roll_degrees: float = 0.0):
        """Build a synthetic face with controllable nose position and roll.

        Args:
            nose_x (float): Nose offset from the eye midpoint, positive right.
            nose_y (float): Nose height between the eye line and the mouth.
            roll_degrees (float): Rotation applied about the face centre.

        Returns:
            np.ndarray: ``(5, 2)`` keypoints.
        """
        points = np.array(
            [[-50.0, 0.0], [50.0, 0.0], [nose_x, nose_y], [-30.0, 100.0], [30.0, 100.0]]
        )
        angle = np.radians(roll_degrees)
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        return points @ rotation.T

    def test_a_frontal_face_reads_as_frontal(self) -> None:
        """Symmetric keypoints must give all three angles at zero."""
        angles = pose_from_keypoints(self._face())
        np.testing.assert_allclose(angles, [0.0, 0.0, 0.0], atol=1e-5)

    def test_roll_is_the_eye_line_angle(self) -> None:
        """A face rotated 15 degrees reports 15 degrees of roll."""
        self.assertAlmostEqual(
            float(pose_from_keypoints(self._face(roll_degrees=15.0))[2]), 15.0, places=3
        )

    def test_roll_sign_follows_image_coordinates(self) -> None:
        """Opposite rotations give opposite signs, not the same magnitude."""
        positive = float(pose_from_keypoints(self._face(roll_degrees=20.0))[2])
        negative = float(pose_from_keypoints(self._face(roll_degrees=-20.0))[2])
        self.assertAlmostEqual(positive, -negative, places=3)

    def test_a_nose_toward_image_left_is_positive_yaw(self) -> None:
        """The convention the occlusion rule depends on.

        ``pipeline.YAW_LIMIT`` treats positive yaw as occluding the viewer's
        *right* eye, because positive yaw turns the nose toward image-left.
        Getting this backwards would suppress the visible eye and keep the
        hidden one -- a demo that looks plausible and is exactly wrong.
        """
        self.assertGreater(float(pose_from_keypoints(self._face(nose_x=-30.0))[0]), 0.0)
        self.assertLess(float(pose_from_keypoints(self._face(nose_x=30.0))[0]), 0.0)

    def test_yaw_grows_with_the_turn(self) -> None:
        """Monotone, which is what an occlusion threshold needs."""
        angles = [float(pose_from_keypoints(self._face(nose_x=-x))[0]) for x in (0, 10, 20, 40)]
        self.assertEqual(angles, sorted(angles))

    def test_pitch_moves_with_the_nose_height(self) -> None:
        """A nose nearer the eye line is a different pitch from one nearer the mouth."""
        high = float(pose_from_keypoints(self._face(nose_y=30.0))[1])
        low = float(pose_from_keypoints(self._face(nose_y=70.0))[1])
        self.assertNotAlmostEqual(high, low, places=2)
        self.assertGreater(high, low)

    def test_degenerate_keypoints_give_zeros(self) -> None:
        """Coincident eyes carry no geometry; zeros beat an exception here.

        A single unreadable frame must not end a stream, and the caller records
        its own validity alongside.
        """
        collapsed = np.zeros((5, 2))
        np.testing.assert_allclose(pose_from_keypoints(collapsed), [0.0, 0.0, 0.0])

    def test_the_wrong_keypoint_count_is_refused(self) -> None:
        """A different layout would produce plausible wrong angles."""
        with self.assertRaises(GeometryError):
            pose_from_keypoints(np.zeros((478, 2)))

    def test_the_result_is_yaw_pitch_roll(self) -> None:
        """Order matches ``ExordiumExtractor.head_pose``, so the two are swappable."""
        angles = pose_from_keypoints(self._face(nose_x=-30.0, roll_degrees=10.0))
        self.assertEqual(angles.shape, (3,))
        self.assertGreater(float(angles[0]), 0.0)
        self.assertAlmostEqual(float(angles[2]), 10.0, places=1)
