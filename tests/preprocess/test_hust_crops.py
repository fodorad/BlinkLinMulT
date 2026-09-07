"""Tests for locating HUST-LEBW's shipped eye crops in their source frames.

The corpus ships both the full frames and per-eye crops cut from them. The crops
are pixel copies, so their position is recoverable exactly — measured at a
correlation of 1.0000 on every crop of a sample clip. That is what identifies
which of several faces is the annotated subject, and what lets a too-tight crop
be re-cut with context.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from blinklinmult.preprocess.hust_crops import (
    RECROP_SCALE,
    CropError,
    containing_box,
    crop_files,
    locate,
    recrop,
)


def frame(height: int = 200, width: int = 400) -> np.ndarray:
    """A frame with enough structure that a match is unambiguous."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (height, width, 3), dtype=np.uint8)


class TestCropFiles(unittest.TestCase):
    """A missing directory is how the corpus records an occluded eye."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def write(self, directory: str, indices: list[int], eye: str = "zuo") -> Path:
        path = self.root / directory
        path.mkdir(parents=True, exist_ok=True)
        for index in indices:
            cv2.imwrite(str(path / f"{index}eye_{eye}.bmp"), frame(8, 8))
        return path

    def test_it_returns_crops_in_frame_order(self):
        # Filenames sort lexically as 1, 10, 2 — the index must drive the order.
        path = self.write("zuo", [1, 2, 10])
        self.assertEqual([index for index, _ in crop_files(path)], [1, 2, 10])

    def test_a_missing_directory_is_empty_not_an_error(self):
        # `train/unblink/100` ships no `you` directory: that eye was occluded,
        # so it simply has no sample.
        self.assertEqual(crop_files(self.root / "you"), [])

    def test_non_crop_files_are_ignored(self):
        path = self.write("zuo", [1])
        (path / "feature.txt").write_text("not a crop")
        (path / "feature_hog.txt").write_text("nor this")
        self.assertEqual(len(crop_files(path)), 1)

    def test_both_eye_directories_are_recognised(self):
        self.assertEqual(len(crop_files(self.write("you", [1, 2], eye="you"))), 2)


class TestLocate(unittest.TestCase):
    """The crops are pixel copies, so the match is exact or it is wrong."""

    def test_it_finds_the_exact_position(self):
        source = frame()
        crop = source[40:80, 100:150]
        x, y, score = locate(source, crop)
        self.assertEqual((x, y), (100, 40))
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_a_crop_from_elsewhere_scores_low(self):
        # A genuine crop scores ~1.0; anything materially below means it did not
        # come from this frame and locating it would be a guess.
        rng = np.random.default_rng(7)
        unrelated = rng.integers(0, 255, (30, 30, 3), dtype=np.uint8)
        self.assertLess(locate(frame(), unrelated)[2], 0.9)

    def test_a_crop_larger_than_its_frame_is_rejected(self):
        with self.assertRaises(CropError):
            locate(frame(20, 20), frame(40, 40))

    def test_a_whole_frame_matches_itself(self):
        source = frame(30, 30)
        x, y, score = locate(source, source)
        self.assertEqual((x, y), (0, 0))
        self.assertAlmostEqual(score, 1.0, places=4)


class TestRecrop(unittest.TestCase):
    """The shipped crops frame the eye alone — too tight for lid landmarks."""

    def test_it_doubles_the_box_by_default(self):
        wide = recrop(frame(200, 400), x=100, y=50, width=40, height=40)
        self.assertEqual(wide.shape[:2], (80, 80))

    def test_it_keeps_the_centre(self):
        source = frame(200, 400)
        wide = recrop(source, x=100, y=50, width=40, height=40)
        # Centre was (120, 70); doubled box spans 80 px, so it starts at (80, 30).
        self.assertTrue(np.array_equal(wide, source[30:110, 80:160]))

    def test_the_scale_is_configurable(self):
        wide = recrop(frame(200, 400), x=100, y=50, width=40, height=40, scale=3.0)
        self.assertEqual(wide.shape[:2], (120, 120))

    def test_an_edge_crop_is_clamped_not_padded(self):
        # Real pixels beat zeros: an off-centre window of the face describes the
        # eye better than a centred window half full of black.
        wide = recrop(frame(200, 400), x=0, y=0, width=40, height=40)
        self.assertEqual(wide.shape[0], 60)
        self.assertEqual(wide.shape[1], 60)

    def test_a_non_positive_scale_is_rejected(self):
        with self.assertRaises(CropError):
            recrop(frame(), x=10, y=10, width=10, height=10, scale=0.0)

    def test_the_default_scale_is_two(self):
        self.assertEqual(RECROP_SCALE, 2.0)


class TestContainingBox(unittest.TestCase):
    """A film frame holds several people; the eye says which is the subject."""

    def test_it_picks_the_face_containing_the_eye(self):
        boxes = [(0, 0, 100, 100), (200, 200, 300, 300)]
        self.assertEqual(containing_box(boxes, x=210, y=210, width=20, height=20), boxes[1])

    def test_nested_boxes_resolve_to_the_tightest(self):
        # A group shot can nest boxes; the smallest containing one is the face.
        group, face = (0, 0, 400, 400), (100, 100, 200, 200)
        self.assertEqual(containing_box([group, face], x=140, y=140, width=20, height=20), face)

    def test_an_eye_outside_every_face_yields_nothing(self):
        self.assertIsNone(containing_box([(0, 0, 50, 50)], x=300, y=300, width=10, height=10))

    def test_no_detections_yields_nothing(self):
        self.assertIsNone(containing_box([], x=10, y=10, width=10, height=10))

    def test_the_eye_centre_decides_not_its_corner(self):
        # A crop straddling a box edge belongs to the face its centre is in.
        boxes = [(100, 100, 300, 300)]
        self.assertEqual(containing_box(boxes, x=90, y=140, width=40, height=20), boxes[0])
