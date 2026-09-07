"""Tests for MRL Eye filename decoding.

The label inversion is the point of this file: MRL encodes ``0 = closed,
1 = open`` while this project labels closure everywhere. Getting it backwards
would silently train the model inverted on the largest corpus in the benchmark.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from blinklinmult.preprocess.common import PreprocessError
from blinklinmult.preprocess.mrl_names import OPEN_EYE_CODE, parse_filename


def name(
    subject: str = "s0001",
    number: int = 1,
    gender: int = 0,
    glasses: int = 0,
    eye_state: int = 0,
    reflection: int = 0,
    lighting: int = 0,
    sensor: int = 1,
) -> Path:
    """Build an MRL filename with the given fields."""
    return Path(
        f"{subject}_{number:05d}_{gender}_{glasses}_{eye_state}_"
        f"{reflection}_{lighting}_{sensor:02d}.png"
    )


class TestParseFilename(unittest.TestCase):
    def test_subject_and_number(self):
        sample = parse_filename(name(subject="s0037", number=1234))
        self.assertEqual(sample.subject, "s0037")
        self.assertEqual(sample.image_number, 1234)

    def test_mrl_zero_means_closed_so_our_label_is_one(self):
        # MRL: 0 = closed. This project labels closure, so closed -> 1.0.
        self.assertEqual(parse_filename(name(eye_state=0)).closed, 1.0)

    def test_mrl_one_means_open_so_our_label_is_zero(self):
        self.assertEqual(parse_filename(name(eye_state=OPEN_EYE_CODE)).closed, 0.0)

    def test_the_two_states_are_opposites(self):
        closed = parse_filename(name(eye_state=0)).closed
        opened = parse_filename(name(eye_state=1)).closed
        self.assertNotEqual(closed, opened)
        self.assertEqual(closed + opened, 1.0)

    def test_attribute_fields(self):
        sample = parse_filename(name(gender=1, glasses=1, reflection=2, lighting=1, sensor=3))
        self.assertEqual(sample.gender, 1)
        self.assertEqual(sample.glasses, 1)
        self.assertEqual(sample.reflection, 2)
        self.assertEqual(sample.lighting, 1)
        self.assertEqual(sample.sensor, 3)

    def test_sample_id_combines_subject_and_number(self):
        self.assertEqual(parse_filename(name("s0002", 42)).sample_id, "s0002_00042")

    def test_sample_ids_are_unique_within_a_subject(self):
        first = parse_filename(name("s0001", 1)).sample_id
        second = parse_filename(name("s0001", 2)).sample_id
        self.assertNotEqual(first, second)

    def test_too_few_fields_raises(self):
        with self.assertRaises(PreprocessError) as ctx:
            parse_filename(Path("s0001_00001_0.png"))
        self.assertIn("8", str(ctx.exception))

    def test_too_many_fields_raises(self):
        with self.assertRaises(PreprocessError):
            parse_filename(Path("s0001_00001_0_0_0_0_0_01_extra.png"))

    def test_non_integer_field_raises(self):
        with self.assertRaises(PreprocessError) as ctx:
            parse_filename(Path("s0001_abcde_0_0_0_0_0_01.png"))
        self.assertIn("non-integer", str(ctx.exception))

    def test_extension_is_ignored(self):
        for suffix in (".png", ".jpg", ".bmp"):
            sample = parse_filename(Path(str(name()).replace(".png", suffix)))
            self.assertEqual(sample.subject, "s0001")


if __name__ == "__main__":
    unittest.main()


class TestSubjectSplits(unittest.TestCase):
    """MRL's split is declared, not hashed.

    Hashing assigns each subject independently, which is a random draw rather
    than a proportional division. MRL's subjects differ in size by an order of
    magnitude, so that gave 55/3/42 percent by image count against a 70/15/15
    target — 2816 validation images against a 35k test split.
    """

    def test_every_subject_has_exactly_one_split(self):
        from blinklinmult.preprocess.mrl import SPLITS

        assigned = [s for names in SPLITS.values() for s in names.split()]
        self.assertEqual(len(assigned), len(set(assigned)))

    def test_all_thirty_seven_subjects_are_covered(self):
        from blinklinmult.preprocess.mrl import SUBJECT_SPLIT

        self.assertEqual(len(SUBJECT_SPLIT), 37)
        self.assertEqual(sorted(SUBJECT_SPLIT), [f"s{i:04d}" for i in range(1, 38)])

    def test_no_split_is_empty(self):
        from blinklinmult.preprocess.mrl import SPLITS

        for split, names in SPLITS.items():
            self.assertTrue(names.split(), split)

    def test_lookup_returns_the_declared_split(self):
        from blinklinmult.preprocess.mrl import subject_splits

        result = subject_splits(["s0001", "s0002", "s0003"])
        self.assertEqual(result["s0001"], "test")
        self.assertEqual(result["s0002"], "train")
        self.assertEqual(result["s0003"], "valid")

    def test_an_unknown_subject_raises(self):
        from blinklinmult.preprocess.common import PreprocessError
        from blinklinmult.preprocess.mrl import subject_splits

        with self.assertRaises(PreprocessError) as ctx:
            subject_splits(["s9999"])
        self.assertIn("s9999", str(ctx.exception))

    def test_the_division_is_roughly_seventy_fifteen_fifteen_by_subject(self):
        from collections import Counter

        from blinklinmult.preprocess.mrl import SUBJECT_SPLIT

        counts = Counter(SUBJECT_SPLIT.values())
        self.assertEqual(counts["train"], 24)
        self.assertEqual(counts["valid"], 7)
        self.assertEqual(counts["test"], 6)
