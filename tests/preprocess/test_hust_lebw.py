"""Tests for the HUST-LEBW corpus reader.

Three properties carry this corpus, and each of them was a real defect the
files exposed rather than a hypothetical:

* the **landmark file** is the frame list, not the directory listing — five test
  clips hold more BMPs than annotated frames, one 39 for 13;
* the clip length is **whichever of 13/ and 10/ annotates more** — five clips
  have an empty ``land_13.txt`` and a complete ``land_10.txt``;
* the annotated eye midpoint **picks the subject's face** — 9% of frames hold
  more than one, and a party scene in ``train/blink/7`` holds three.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from blinklinmult.preprocess.hust_lebw import (
    CLIP_LENGTHS,
    LABELS,
    TIME_DIM,
    VALID_FRACTION,
    Clip,
    parse_landmarks,
    split_clips,
)


class TestParseLandmarks(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, text: str) -> Path:
        path = self.tmp / "land_13.txt"
        path.write_text(text)
        return path

    def test_reads_frame_id_and_four_coordinates(self):
        landmarks = parse_landmarks(self.write("2\t1075\t285\t1302\t324\n"))
        self.assertEqual(sorted(landmarks), [2])
        np.testing.assert_allclose(landmarks[2], [1075, 285, 1302, 324])

    def test_scientific_notation_is_read(self):
        # The corpus writes some coordinates as 2.655000e+02.
        landmarks = parse_landmarks(self.write("58452\t833\t2.655000e+02\t971\t2.535000e+02\n"))
        np.testing.assert_allclose(landmarks[58452], [833, 265.5, 971, 253.5])

    def test_blank_lines_are_ignored(self):
        self.assertEqual(len(parse_landmarks(self.write("1\t1\t2\t3\t4\n\n\n"))), 1)

    def test_a_truncated_row_is_skipped_not_fatal(self):
        # 5 rows of 17154 are malformed. Aborting the corpus over 0.03% of its
        # annotation would be the wrong trade; those frames fall through to
        # detection like any other.
        landmarks = parse_landmarks(self.write("1\t1\t2\t3\t4\n2\t67\t406\t149\n3\t5\t6\t7\t8\n"))
        self.assertEqual(sorted(landmarks), [1, 3])

    def test_the_not_annotated_marker_is_skipped(self):
        # The corpus writes a bare "-1 -1" for a frame it did not annotate.
        self.assertEqual(parse_landmarks(self.write("-1\t-1\n")), {})

    def test_a_non_numeric_row_is_skipped(self):
        landmarks = parse_landmarks(self.write("1\t1\t2\t3\t4\nx\ty\tz\tw\tv\n"))
        self.assertEqual(sorted(landmarks), [1])

    def test_a_missing_file_is_empty(self):
        self.assertEqual(parse_landmarks(self.tmp / "absent.txt"), {})


class TestClipFrameIds(unittest.TestCase):
    def clip(self, landmarks: dict[int, list[float]], frames: list[int]) -> Clip:
        return Clip(
            split="train",
            label="blink",
            clip_id="1",
            directory=Path("x"),
            landmarks={k: np.asarray(v, dtype=np.float32) for k, v in landmarks.items()},
            frames={f: Path(f"{f}.bmp") for f in frames},
        )

    def test_only_frames_that_are_both_annotated_and_present(self):
        # Five test clips hold more BMPs than annotated frames -- one has 39
        # files for 13 landmark rows -- so globbing would build a window out of
        # frames the annotation never covered.
        clip = self.clip({1: [0, 0, 1, 1], 2: [0, 0, 1, 1]}, [1, 2, 3, 4, 5])
        self.assertEqual(clip.frame_ids, [1, 2])

    def test_an_annotated_frame_with_no_file_is_dropped(self):
        clip = self.clip({1: [0, 0, 1, 1], 9: [0, 0, 1, 1]}, [1])
        self.assertEqual(clip.frame_ids, [1])

    def test_frames_are_ordered(self):
        clip = self.clip({3: [0, 0, 1, 1], 1: [0, 0, 1, 1]}, [1, 3])
        self.assertEqual(clip.frame_ids, [1, 3])

    def test_a_long_clip_is_cut_to_the_window(self):
        many = {i: [0, 0, 1, 1] for i in range(30)}
        self.assertEqual(len(self.clip(many, list(range(30))).frame_ids), TIME_DIM)

    def test_a_short_clip_keeps_what_it_has(self):
        # 209 clips are shorter than 13 frames. They pad and mask rather than
        # being dropped.
        clip = self.clip({1: [0, 0, 1, 1], 2: [0, 0, 1, 1]}, [1, 2])
        self.assertEqual(len(clip.frame_ids), 2)

    def test_the_sample_id_separates_class_and_split(self):
        # blink/1 and unblink/1 are unrelated clips; the corpus numbers each
        # class independently from 1.
        clip = self.clip({1: [0, 0, 1, 1]}, [1])
        self.assertEqual(clip.sample_id, "train_blink_1")


class TestSplitClips(unittest.TestCase):
    def clips(self, count: int) -> list[Path]:
        return [Path(str(index)) for index in range(1, count + 1)]

    def test_test_is_honoured_as_given(self):
        assigned = split_clips(self.clips(10), "test")
        self.assertTrue(all(split == "test" for _, split in assigned))

    def test_validation_is_the_tail_of_train(self):
        # The clips are film-ordered, so a contiguous tail keeps a film's clips
        # together where a random draw would split an actor across the boundary.
        assigned = split_clips(self.clips(100), "train")
        splits = [split for _, split in assigned]
        self.assertEqual(splits[:85], ["train"] * 85)
        self.assertEqual(splits[85:], ["valid"] * 15)

    def test_the_real_class_sizes_divide_as_documented(self):
        for count, valid in ((254, 38), (194, 29)):
            with self.subTest(count=count):
                assigned = split_clips(self.clips(count), "train")
                self.assertEqual(sum(1 for _, s in assigned if s == "valid"), valid)

    def test_every_clip_is_assigned_exactly_once(self):
        assigned = split_clips(self.clips(50), "train")
        self.assertEqual(len(assigned), 50)
        self.assertEqual(len({clip for clip, _ in assigned}), 50)

    def test_both_splits_are_populated(self):
        splits = {split for _, split in split_clips(self.clips(20), "train")}
        self.assertEqual(splits, {"train", "valid"})


class TestConstants(unittest.TestCase):
    def test_the_window_is_thirteen_frames(self):
        self.assertEqual(TIME_DIM, 13)

    def test_thirteen_is_tried_before_ten(self):
        # 668 clips use 13/; the 5 with an unusable land_13.txt fall back to 10/.
        self.assertEqual(CLIP_LENGTHS, ("13", "10"))

    def test_the_labels_map_to_blink_presence(self):
        self.assertEqual(LABELS, {"blink": 1.0, "unblink": 0.0})

    def test_the_validation_share_is_fifteen_percent(self):
        self.assertAlmostEqual(VALID_FRACTION, 0.15)


if __name__ == "__main__":
    unittest.main()
