"""Tests for RN's recording identity and splits.

RN restarts its numbering inside every split directory, so ``train/rn15/1``,
``val/rn15/1``, and ``test/rn15/1`` are three different recordings all sitting
in a directory called ``1``. Naming them by that directory alone gave all three
the same id and their sample keys collided inside the HDF5 — which is what these
tests exist to stop coming back.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from blinklinmult.preprocess.common import PreprocessError
from blinklinmult.preprocess.rn import RATES, layout, recording_name, split_of_path

RAW = Path("data/raw/RN")


def tag(split: str, rate: str, number: str) -> Path:
    return RAW / split / rate / number / "cam.tag"


class TestSplitOfPath(unittest.TestCase):
    def test_each_split_directory_maps(self):
        self.assertEqual(split_of_path(tag("train", "rn15", "1"), RAW), "train")
        self.assertEqual(split_of_path(tag("val", "rn15", "1"), RAW), "valid")
        self.assertEqual(split_of_path(tag("test", "rn15", "1"), RAW), "test")

    def test_an_unknown_split_directory_raises(self):
        with self.assertRaises(PreprocessError):
            split_of_path(tag("holdout", "rn15", "1"), RAW)

    def test_a_path_outside_the_corpus_raises(self):
        with self.assertRaises(PreprocessError):
            split_of_path(Path("/elsewhere/x.tag"), RAW)


class TestRecordingName(unittest.TestCase):
    def test_the_same_number_in_two_splits_gets_two_names(self):
        # The collision that crashed a real build.
        train = recording_name(tag("train", "rn15", "1"), RAW)
        test = recording_name(tag("test", "rn15", "1"), RAW)
        self.assertNotEqual(train, test)

    def test_the_name_carries_the_split(self):
        self.assertEqual(recording_name(tag("train", "rn15", "7"), RAW), "train_7")
        self.assertEqual(recording_name(tag("val", "rn30", "7"), RAW), "valid_7")

    def test_names_are_unique_across_a_whole_rate(self):
        # Runs against the real corpus when present; skipped otherwise so the
        # suite stays runnable without the raw data.
        for rate in sorted(RATES):
            corpus = layout(rate)
            if not corpus.raw_dir.is_dir():
                self.skipTest(f"{corpus.raw_dir} not present")

            from blinklinmult.preprocess.common import list_files

            paths = list_files(corpus.raw_dir, corpus.tag_glob)
            names = [recording_name(p, corpus.raw_dir) for p in paths]
            self.assertEqual(len(names), len(set(names)), f"rn{rate} has duplicate ids")


class TestLayout(unittest.TestCase):
    def test_the_two_rates_are_separate_corpora(self):
        self.assertEqual(layout(15).name, "rn15")
        self.assertEqual(layout(30).name, "rn30")

    def test_each_rate_derives_its_own_window(self):
        # Half a second is 8 frames at 15 fps and 15 at 30.
        self.assertEqual(layout(15).window, 23)
        self.assertEqual(layout(30).window, 45)

    def test_the_stride_is_half_the_window(self):
        self.assertEqual(layout(15).stride, 11)
        self.assertEqual(layout(30).stride, 22)

    def test_each_rate_writes_its_own_h5(self):
        self.assertEqual(layout(15).h5_path.name, "rn15.h5")
        self.assertEqual(layout(30).h5_path.name, "rn30.h5")


if __name__ == "__main__":
    unittest.main()
