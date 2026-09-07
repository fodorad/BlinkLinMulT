"""Tests for per-clip shards.

The point of sharding is that an interrupted build resumes instead of
restarting, so these check the property that makes resumption trustworthy: a
shard exists only if its clip finished, and a completed clip is never redone.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from blinklinmult.data.schema import DatasetSpec
from blinklinmult.data.shards import (
    completed_clips,
    merge_shards,
    pending,
    shard_attrs,
    shard_dir,
    shard_path,
)
from blinklinmult.data.writer import H5Writer

IMAGE_SIZE = 8
TIME_DIM = 2


class ShardCase(unittest.TestCase):
    """A corpus path in a temporary tree."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.h5_path = Path(self._tmp.name) / "corpus.h5"
        shard_dir(self.h5_path).mkdir(parents=True)

    def spec(self) -> DatasetSpec:
        return DatasetSpec(
            name="mpeblink",
            fps=4.0,
            window_seconds=0.5,
            image_size=IMAGE_SIZE,
            has_blink_presence=True,
        )

    def write_shard(self, clip_id: str, samples: int = 2, subset: str = "train") -> Path:
        path = shard_path(self.h5_path, clip_id)
        with H5Writer(self.spec(), path) as writer:
            for index in range(samples):
                writer.add(
                    subset=subset,
                    video_id=clip_id,
                    frame_group=f"{index:06d}",
                    eye_side="left",
                    eye_images=np.random.rand(TIME_DIM, 3, IMAGE_SIZE, IMAGE_SIZE).astype(
                        np.float32
                    ),
                    blink_presence=np.zeros(TIME_DIM, dtype=np.float32),
                )
        return path


class TestCompletedClips(ShardCase):
    def test_no_shards_means_nothing_completed(self):
        self.assertEqual(completed_clips(self.h5_path), set())

    def test_a_written_shard_is_completed(self):
        self.write_shard("test_1")
        self.assertEqual(completed_clips(self.h5_path), {"test_1"})

    def test_a_missing_directory_is_not_an_error(self):
        # The first run of a build, before anything exists.
        fresh = Path(self._tmp.name) / "other" / "corpus.h5"
        self.assertEqual(completed_clips(fresh), set())

    def test_an_abandoned_tmp_does_not_count_as_complete(self):
        # H5Writer renames into place only on a clean close, so a killed build
        # leaves a .tmp. Counting it would skip a clip that never finished.
        (shard_dir(self.h5_path) / "test_9.h5.tmp").write_bytes(b"partial")
        self.assertEqual(completed_clips(self.h5_path), set())


class TestPending(ShardCase):
    def entries(self) -> list[tuple[str, str]]:
        return [("test", "1"), ("test", "2"), ("test", "3")]

    def clip_id(self, entry: tuple[str, str]) -> str:
        return f"{entry[0]}_{entry[1]}"

    def test_everything_is_pending_on_a_fresh_build(self):
        remaining = pending(self.entries(), self.h5_path, self.clip_id)
        self.assertEqual(len(remaining), 3)

    def test_a_completed_clip_is_skipped(self):
        self.write_shard("test_2")
        remaining = pending(self.entries(), self.h5_path, self.clip_id)
        self.assertEqual([self.clip_id(e) for e in remaining], ["test_1", "test_3"])

    def test_the_original_order_is_kept(self):
        remaining = pending(self.entries(), self.h5_path, self.clip_id)
        self.assertEqual([self.clip_id(e) for e in remaining], ["test_1", "test_2", "test_3"])

    def test_nothing_pending_when_all_are_done(self):
        for entry in self.entries():
            self.write_shard(self.clip_id(entry))
        self.assertEqual(pending(self.entries(), self.h5_path, self.clip_id), [])


class TestMergeShards(ShardCase):
    def test_it_combines_every_sample(self):
        self.write_shard("test_1", samples=2)
        self.write_shard("test_2", samples=3)
        written = merge_shards(self.h5_path, shard_attrs(self.h5_path))
        self.assertEqual(written, 5)
        with h5py.File(self.h5_path) as handle:
            self.assertEqual(len(handle["train"]), 5)

    def test_samples_from_different_shards_coexist(self):
        self.write_shard("test_1", samples=1)
        self.write_shard("test_2", samples=1)
        merge_shards(self.h5_path, shard_attrs(self.h5_path))
        with h5py.File(self.h5_path) as handle:
            keys = list(handle["train"])
        self.assertTrue(any(k.startswith("test_1") for k in keys))
        self.assertTrue(any(k.startswith("test_2") for k in keys))

    def test_splits_are_preserved(self):
        self.write_shard("test_1", samples=2, subset="train")
        self.write_shard("test_2", samples=1, subset="test")
        merge_shards(self.h5_path, shard_attrs(self.h5_path))
        with h5py.File(self.h5_path) as handle:
            self.assertEqual(len(handle["train"]), 2)
            self.assertEqual(len(handle["test"]), 1)

    def test_the_root_attributes_survive(self):
        # The merged file must be indistinguishable from a single-pass build,
        # or the loader would read a corpus with no schema version.
        self.write_shard("test_1")
        attrs = shard_attrs(self.h5_path)
        merge_shards(self.h5_path, attrs)
        with h5py.File(self.h5_path) as handle:
            self.assertEqual(handle.attrs["schema_version"], attrs["schema_version"])
            self.assertEqual(handle.attrs["dataset"], attrs["dataset"])

    def test_it_cleans_up_the_shards(self):
        self.write_shard("test_1")
        merge_shards(self.h5_path, shard_attrs(self.h5_path))
        self.assertFalse(shard_dir(self.h5_path).exists())

    def test_it_can_keep_the_shards(self):
        self.write_shard("test_1")
        merge_shards(self.h5_path, shard_attrs(self.h5_path), remove=False)
        self.assertTrue(shard_dir(self.h5_path).exists())

    def test_merging_nothing_is_an_error(self):
        # Silently writing an empty corpus would look like a successful build.
        with self.assertRaises(FileNotFoundError):
            merge_shards(self.h5_path, {})

    def test_an_empty_shard_contributes_nothing(self):
        # A video with no tracklets still gets a shard, so it is not retried on
        # every resume; it must merge to zero samples rather than fail.
        with H5Writer(self.spec(), shard_path(self.h5_path, "test_empty")):
            pass
        self.write_shard("test_1", samples=2)
        self.assertEqual(merge_shards(self.h5_path, shard_attrs(self.h5_path)), 2)


class TestShardAttrs(ShardCase):
    def test_it_reads_from_a_shard(self):
        self.write_shard("test_1")
        self.assertEqual(shard_attrs(self.h5_path)["dataset"], "mpeblink")

    def test_no_shards_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            shard_attrs(self.h5_path)
