"""Tests for the corpus inspection helpers.

These back the per-corpus notebook, whose whole job is to show a human what a
built corpus contains. A helper that silently returns nothing would make a
corpus look empty rather than raise, so the checks here are mostly about the
absent and empty cases.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from blinklinmult.data.inspect import (
    count_events,
    gather_field,
    occlusion_share,
    populated_splits,
    read_field,
    sample_keys,
)
from blinklinmult.data.schema import DatasetSpec
from blinklinmult.data.writer import H5Writer

IMAGE_SIZE = 8
TIME_DIM = 4


class CorpusCase(unittest.TestCase):
    """A tiny real corpus on disk, so the helpers meet real h5 semantics."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "corpus.h5"
        self.write(splits=("train", "test"))

    def spec(self, **overrides) -> DatasetSpec:
        defaults = {
            "name": "rn30",
            "fps": 8.0,
            "window_seconds": 0.5,
            "image_size": IMAGE_SIZE,
            "has_blink_presence": True,
            "has_eye_state": True,
            "has_blink_ids": True,
            "has_head_pose": True,
            "quality_signals": ("eye_blur",),
        }
        return DatasetSpec(**{**defaults, **overrides})

    def write(self, splits: tuple[str, ...], per_split: int = 4) -> None:
        spec = self.spec()
        with H5Writer(spec, self.path) as writer:
            for position, subset in enumerate(splits):
                for index in range(per_split):
                    ids = np.full(TIME_DIM, -1, dtype=np.int32)
                    # One event per sample, numbered so two samples of the same
                    # recording share an id and two recordings do not.
                    ids[1:3] = index % 2
                    writer.add(
                        subset=subset,
                        # Split-scoped: sample keys must be unique across the
                        # whole file, not just within a split.
                        video_id=f"{subset}_rec{index % 2}",
                        # Unique per (recording, sample): the two recordings
                        # interleave, so a shared frame group would collide.
                        frame_group=f"{(position * per_split + index) * TIME_DIM:06d}",
                        eye_side="left",
                        eye_images=np.random.rand(TIME_DIM, 3, IMAGE_SIZE, IMAGE_SIZE).astype(
                            np.float32
                        ),
                        blink_presence=np.zeros(TIME_DIM, dtype=np.float32),
                        eye_state=np.zeros(TIME_DIM, dtype=np.float32),
                        blink_ids=ids,
                        head_pose=np.tile([index * 20.0, 0.0, 0.0], (TIME_DIM, 1)),
                        quality_signals={"eye_blur": np.full(TIME_DIM, 0.5, dtype=np.float32)},
                    )


class TestPopulatedSplits(CorpusCase):
    def test_it_lists_only_populated_splits(self):
        with h5py.File(self.path) as handle:
            self.assertEqual(populated_splits(handle), ["train", "test"])

    def test_the_order_is_conventional_not_alphabetical(self):
        # "test" sorts before "train"; a report must not present it that way.
        with h5py.File(self.path) as handle:
            self.assertEqual(populated_splits(handle)[0], "train")


class TestSampleKeys(CorpusCase):
    def test_it_returns_the_requested_count(self):
        with h5py.File(self.path) as handle:
            self.assertEqual(len(sample_keys(handle, "train", 3)), 3)

    def test_it_spreads_rather_than_taking_a_head_slice(self):
        # Keys are sorted, so a head slice is one recording and its statistics
        # are not the corpus's.
        with h5py.File(self.path) as handle:
            keys = sample_keys(handle, "train", 2)
            self.assertNotEqual(keys, list(handle["train"])[:2])

    def test_an_offset_moves_to_a_different_set(self):
        with h5py.File(self.path) as handle:
            self.assertNotEqual(
                sample_keys(handle, "train", 2, offset=0),
                sample_keys(handle, "train", 2, offset=1),
            )

    def test_the_offset_wraps_rather_than_running_out(self):
        # The browser must keep working however many times it is re-run.
        with h5py.File(self.path) as handle:
            self.assertEqual(len(sample_keys(handle, "train", 2, offset=99)), 2)

    def test_a_zero_count_returns_nothing(self):
        with h5py.File(self.path) as handle:
            self.assertEqual(sample_keys(handle, "train", 0), [])


class TestReadField(CorpusCase):
    def test_it_reads_a_present_field(self):
        with h5py.File(self.path) as handle:
            key = sample_keys(handle, "train", 1)[0]
            self.assertEqual(read_field(handle, "train", key, "eye_state").shape, (TIME_DIM,))

    def test_an_absent_field_is_none_not_an_error(self):
        # Corpora declare different subsets; a missing field is normal.
        with h5py.File(self.path) as handle:
            key = sample_keys(handle, "train", 1)[0]
            self.assertIsNone(read_field(handle, "train", key, "eye_contour_fit"))

    def test_images_are_upcast_from_float16(self):
        with h5py.File(self.path) as handle:
            key = sample_keys(handle, "train", 1)[0]
            self.assertEqual(read_field(handle, "train", key, "eye_image").dtype, np.float32)


class TestGatherField(CorpusCase):
    def test_it_flattens_across_samples(self):
        with h5py.File(self.path) as handle:
            values = gather_field(handle, "train", "eye_state")
            self.assertEqual(values.size, 4 * TIME_DIM)

    def test_an_absent_field_yields_an_empty_array(self):
        with h5py.File(self.path) as handle:
            self.assertEqual(gather_field(handle, "train", "eye_contour_fit").size, 0)

    def test_the_limit_caps_the_read(self):
        with h5py.File(self.path) as handle:
            self.assertEqual(gather_field(handle, "train", "eye_state", limit=2).size, 2 * TIME_DIM)


class TestOcclusionShare(unittest.TestCase):
    def test_it_counts_both_directions_of_yaw(self):
        # Yaw sign says which eye is occluded, not whether one is.
        self.assertAlmostEqual(occlusion_share(np.asarray([-60.0, 60.0, 0.0]), 45.0), 2 / 3)

    def test_nothing_beyond_the_threshold_is_zero(self):
        self.assertEqual(occlusion_share(np.asarray([10.0, -20.0]), 45.0), 0.0)

    def test_an_empty_input_is_zero(self):
        self.assertEqual(occlusion_share(np.asarray([]), 45.0), 0.0)


class TestCountEvents(CorpusCase):
    def test_it_counts_distinct_events(self):
        # Two recordings, each with its own event 0 or 1 -> two distinct.
        with h5py.File(self.path) as handle:
            self.assertEqual(count_events(handle, "train"), 2)

    def test_ids_are_scoped_to_their_recording(self):
        # Event 0 of rec0 and event 0 of rec1 are different blinks; counting the
        # raw id would merge them.
        with h5py.File(self.path) as handle:
            keys = sample_keys(handle, "train", 4)
            videos = {read_field(handle, "train", key, "video_id").decode() for key in keys}
            self.assertGreater(len(videos), 1)
            self.assertGreaterEqual(count_events(handle, "train"), len(videos))
