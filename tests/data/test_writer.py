"""Tests for writing samples straight into a corpus's HDF5 file.

The contract under test is the one OmniLoader reads: group layout, dtypes, and
the pairing of every value with its own validity mask. The other property that
matters is atomicity — a run that fails must leave no file at all, rather than
a partial one a later run would mistake for finished.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from blinklinmult.data.schema import (
    BLINK_ID,
    BLINK_IDS,
    BLINK_PRESENCE,
    EYE_FEATURE,
    EYE_IMAGE,
    EYE_STATE,
    HEAD_POSE,
    NO_BLINK,
    QUALITY_SIGNALS,
    SAMPLE_KEY,
    SCHEMA_VERSION,
    SOURCE_KEY,
    DatasetSpec,
)
from blinklinmult.data.writer import H5Writer, WriterError

IMAGE_SIZE = 8
TIME_DIM = 4


class WriterCase(unittest.TestCase):
    """A writer over a temporary tree."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.h5_path = self.root / "corpus.h5"

    def spec(self, **overrides) -> DatasetSpec:
        defaults = {
            "name": "talkingface",
            "fps": 8.0,  # -> TIME_DIM frames at 0.5s
            "window_seconds": 0.5,
            "image_size": IMAGE_SIZE,
            "has_blink_presence": True,
            "has_eye_state": True,
        }
        return DatasetSpec(**{**defaults, **overrides})

    def images(self, time_dim: int = TIME_DIM) -> np.ndarray:
        return np.random.rand(time_dim, 3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    def labels(self, time_dim: int = TIME_DIM, positive: bool = False) -> np.ndarray:
        values = np.zeros(time_dim, dtype=np.float32)
        if positive:
            values[1] = 1.0
        return values

    def write_one(self, writer: H5Writer, **overrides) -> str:
        defaults = {
            "subset": "test",
            "video_id": "talking",
            "frame_group": "000000",
            "eye_side": "left",
            "eye_images": self.images(),
            "blink_presence": self.labels(),
            "eye_state": self.labels(),
        }
        return writer.add(**{**defaults, **overrides})


class TestLayout(WriterCase):
    def test_writes_the_file_on_a_clean_exit(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            self.write_one(writer)
        self.assertTrue(self.h5_path.is_file())

    def test_samples_land_under_their_subset(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(writer)
        with h5py.File(self.h5_path, "r") as handle:
            self.assertIn(key, handle["test"])

    def test_the_sample_key_encodes_video_group_and_side(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(writer, video_id="talking", frame_group="000042", eye_side="right")
        self.assertEqual(key, "talking|000042|right")

    def test_every_value_carries_a_mask(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(writer)
        with h5py.File(self.h5_path, "r") as handle:
            group = handle["test"][key]
            for name in (EYE_IMAGE, BLINK_PRESENCE, EYE_STATE):
                self.assertIn(f"{name}_mask", group, name)

    def test_provenance_fields_are_stored(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(writer)
        with h5py.File(self.h5_path, "r") as handle:
            group = handle["test"][key]
            self.assertEqual(group[SAMPLE_KEY][()].decode(), key)
            self.assertEqual(group[SOURCE_KEY][()].decode(), "talkingface")
            self.assertIn(BLINK_ID, group)

    def test_images_are_stored_as_float16(self):
        # The crops dominate the file; fp16 halves it at a precision finer than
        # 8-bit pixel data carries anyway.
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(writer)
        with h5py.File(self.h5_path, "r") as handle:
            self.assertEqual(handle["test"][key][EYE_IMAGE].dtype, np.float16)

    def test_images_keep_their_native_shape(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(writer)
        with h5py.File(self.h5_path, "r") as handle:
            shape = handle["test"][key][EYE_IMAGE].shape
        self.assertEqual(shape, (TIME_DIM, 3, IMAGE_SIZE, IMAGE_SIZE))


class TestRootAttributes(WriterCase):
    def test_records_what_the_file_is(self):
        with H5Writer(self.spec(), self.h5_path, config_yaml="name: x", git_sha="abc") as writer:
            self.write_one(writer)
        with h5py.File(self.h5_path, "r") as handle:
            self.assertEqual(int(handle.attrs["schema_version"]), SCHEMA_VERSION)
            self.assertEqual(str(handle.attrs["dataset"]), "talkingface")
            self.assertEqual(str(handle.attrs["git_sha"]), "abc")
            self.assertEqual(str(handle.attrs["builder_config_yaml"]), "name: x")
            self.assertEqual(int(handle.attrs["time_dim"]), TIME_DIM)

    def test_a_still_corpus_records_sentinels_not_nulls(self):
        # h5py cannot store None, so absent values become -1.
        spec = DatasetSpec(
            name="cew",
            fps=None,
            window_seconds=None,
            image_size=IMAGE_SIZE,
            has_eye_state=True,
        )
        with H5Writer(spec, self.h5_path) as writer:
            writer.add(
                subset="train",
                video_id="v",
                frame_group="000000",
                eye_side="left",
                eye_images=self.images(1),
                eye_state=self.labels(1),
            )
        with h5py.File(self.h5_path, "r") as handle:
            self.assertEqual(float(handle.attrs["fps"]), -1.0)
            self.assertEqual(int(handle.attrs["feature_dim"]), -1)


class TestTimeFitting(WriterCase):
    def test_a_short_window_is_padded_and_masked(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(
                writer,
                eye_images=self.images(2),
                blink_presence=self.labels(2),
                eye_state=self.labels(2),
            )
        with h5py.File(self.h5_path, "r") as handle:
            group = handle["test"][key]
            self.assertEqual(group[EYE_IMAGE].shape[0], TIME_DIM)
            np.testing.assert_array_equal(
                group[f"{EYE_IMAGE}_mask"][()], [True, True, False, False]
            )

    def test_padding_is_invalid_even_if_the_source_mask_said_otherwise(self):
        # The caller's mask describes real frames; it cannot vouch for padding.
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(
                writer,
                eye_images=self.images(2),
                eye_image_mask=np.ones(2, dtype=bool),
                blink_presence=self.labels(2),
                eye_state=self.labels(2),
            )
        with h5py.File(self.h5_path, "r") as handle:
            self.assertFalse(handle["test"][key][f"{EYE_IMAGE}_mask"][()][2:].any())

    def test_a_long_window_is_truncated(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(
                writer,
                eye_images=self.images(TIME_DIM + 3),
                blink_presence=self.labels(TIME_DIM + 3),
                eye_state=self.labels(TIME_DIM + 3),
            )
        with h5py.File(self.h5_path, "r") as handle:
            self.assertEqual(handle["test"][key][EYE_IMAGE].shape[0], TIME_DIM)

    def test_a_supplied_mask_survives(self):
        mask = np.asarray([True, False, True, True])
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(writer, eye_state_mask=mask)
        with h5py.File(self.h5_path, "r") as handle:
            np.testing.assert_array_equal(handle["test"][key][f"{EYE_STATE}_mask"][()], mask)


class TestFeatures(WriterCase):
    def spec(self, **overrides):
        return super().spec(**{"feature_dim": 6, **overrides})

    def test_features_are_written_with_their_mask(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            key = self.write_one(
                writer,
                eye_features=np.random.rand(TIME_DIM, 6).astype(np.float32),
                eye_feature_mask=np.asarray([True, True, False, True]),
            )
        with h5py.File(self.h5_path, "r") as handle:
            group = handle["test"][key]
            self.assertEqual(group[EYE_FEATURE].shape, (TIME_DIM, 6))
            np.testing.assert_array_equal(
                group[f"{EYE_FEATURE}_mask"][()], [True, True, False, True]
            )

    def test_a_declared_stream_that_is_missing_raises(self):
        with self.assertRaises(WriterError) as ctx:
            with H5Writer(self.spec(), self.h5_path) as writer:
                self.write_one(writer)
        self.assertIn("feature_dim", str(ctx.exception))

    def test_a_wrong_feature_width_raises(self):
        with self.assertRaises(WriterError) as ctx:
            with H5Writer(self.spec(), self.h5_path) as writer:
                self.write_one(writer, eye_features=np.zeros((TIME_DIM, 3), dtype=np.float32))
        self.assertIn("expected (T, 6)", str(ctx.exception))


class TestValidation(WriterCase):
    def test_a_duplicate_key_raises(self):
        # An HDF5 group collision would silently drop a sample.
        with self.assertRaises(WriterError) as ctx:
            with H5Writer(self.spec(), self.h5_path) as writer:
                self.write_one(writer)
                self.write_one(writer)
        self.assertIn("duplicate", str(ctx.exception))

    def test_an_unknown_subset_raises(self):
        with self.assertRaises(WriterError):
            with H5Writer(self.spec(), self.h5_path) as writer:
                self.write_one(writer, subset="holdout")

    def test_a_wrong_image_shape_raises(self):
        with self.assertRaises(WriterError) as ctx:
            with H5Writer(self.spec(), self.h5_path) as writer:
                self.write_one(writer, eye_images=np.zeros((TIME_DIM, 3, 5, 5), dtype=np.float32))
        self.assertIn("expected", str(ctx.exception))

    def test_a_declared_target_that_is_missing_raises(self):
        with self.assertRaises(WriterError) as ctx:
            with H5Writer(self.spec(), self.h5_path) as writer:
                writer.add(
                    subset="test",
                    video_id="v",
                    frame_group="000000",
                    eye_side="left",
                    eye_images=self.images(),
                    blink_presence=self.labels(),
                )
        self.assertIn(EYE_STATE, str(ctx.exception))

    def test_writing_before_opening_raises(self):
        writer = H5Writer(self.spec(), self.h5_path)
        with self.assertRaises(WriterError):
            self.write_one(writer)


class TestAtomicity(WriterCase):
    def test_a_failed_run_leaves_no_file(self):
        # The property that makes "built or absent" true: a partial file would
        # be indistinguishable from a finished one on the next run.
        with self.assertRaises(RuntimeError):
            with H5Writer(self.spec(), self.h5_path) as writer:
                self.write_one(writer)
                raise RuntimeError("interrupted")

        self.assertFalse(self.h5_path.exists())
        self.assertFalse(self.h5_path.with_suffix(".h5.tmp").exists())

    def test_no_temporary_file_survives_a_success(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            self.write_one(writer)
        self.assertFalse(self.h5_path.with_suffix(".h5.tmp").exists())


class TestStats(WriterCase):
    def test_counts_samples_per_split(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            self.write_one(writer, eye_side="left")
            self.write_one(writer, eye_side="right")
        self.assertEqual(writer.stats.per_subset["test"], 2)

    def test_records_the_blink_fraction(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            self.write_one(writer, eye_side="left", blink_presence=self.labels(positive=True))
            self.write_one(writer, eye_side="right", blink_presence=self.labels())
        self.assertAlmostEqual(writer.stats.positive_fraction["test"], 0.5)

    def test_reports_the_file_size(self):
        with H5Writer(self.spec(), self.h5_path) as writer:
            self.write_one(writer)
        self.assertGreater(writer.stats.bytes_written, 0)


if __name__ == "__main__":
    unittest.main()


class TestFrameConditioningFields(WriterCase):
    """Head pose, confidence, and blink ids must survive a round trip.

    They are declared as OmniLoader *features* rather than quality extras
    because a quality field is written but never loaded — which is why head
    pose was previously recoverable only as `eye_feature[:, -3:] * 90` and no
    analysis could read it.
    """

    def conditioned_spec(self) -> DatasetSpec:
        return self.spec(has_head_pose=True, quality_signals=QUALITY_SIGNALS, has_blink_ids=True)

    def pose(self, time_dim: int = TIME_DIM) -> np.ndarray:
        # Degrees, not the /90 normalised form.
        return np.stack(
            [
                np.linspace(-60.0, 60.0, time_dim),
                np.zeros(time_dim),
                np.full(time_dim, 5.0),
            ],
            axis=-1,
        ).astype(np.float32)

    def write_conditioned(self, writer: H5Writer, **overrides) -> str:
        defaults = {
            "head_pose": self.pose(),
            "quality_signals": {
                name: np.linspace(0.0, 1.0, TIME_DIM).astype(np.float32) for name in QUALITY_SIGNALS
            },
            "blink_ids": np.asarray([-1, 3, 3, -1], dtype=np.int32),
        }
        return self.write_one(writer, **{**defaults, **overrides})

    def test_head_pose_round_trips_in_degrees(self):
        spec = self.conditioned_spec()
        with H5Writer(spec, self.h5_path) as writer:
            key = self.write_conditioned(writer)
        with h5py.File(self.h5_path) as handle:
            stored = np.asarray(handle["test"][key][HEAD_POSE])
        self.assertEqual(stored.shape, (TIME_DIM, 3))
        # Degrees survive: the extremes are not squashed into [-1, 1].
        self.assertAlmostEqual(float(stored[0, 0]), -60.0, places=3)
        self.assertAlmostEqual(float(stored[-1, 0]), 60.0, places=3)

    def test_every_quality_signal_round_trips(self):
        # Stored separately, never pre-combined: the weighting is a dataloader
        # decision made once the distributions have been seen.
        spec = self.conditioned_spec()
        with H5Writer(spec, self.h5_path) as writer:
            key = self.write_conditioned(writer)
        with h5py.File(self.h5_path) as handle:
            for name in QUALITY_SIGNALS:
                stored = np.asarray(handle["test"][key][name])
                self.assertEqual(stored.shape, (TIME_DIM,), name)
                self.assertAlmostEqual(float(stored[-1]), 1.0, places=3, msg=name)

    def test_a_missing_signal_is_rejected(self):
        spec = self.conditioned_spec()
        partial = {name: np.zeros(TIME_DIM, dtype=np.float32) for name in QUALITY_SIGNALS[:-1]}
        with H5Writer(spec, self.h5_path) as writer, self.assertRaises(WriterError):
            self.write_conditioned(writer, quality_signals=partial)

    def test_blink_ids_keep_their_identity(self):
        spec = self.conditioned_spec()
        with H5Writer(spec, self.h5_path) as writer:
            key = self.write_conditioned(writer)
        with h5py.File(self.h5_path) as handle:
            stored = np.asarray(handle["test"][key][BLINK_IDS])
        self.assertEqual(stored.tolist(), [-1, 3, 3, -1])

    def test_short_windows_pad_with_no_blink_not_zero(self):
        # Zero is a valid event id, so zero-padding would invent event 0 on
        # every window shorter than the time dimension.
        spec = self.conditioned_spec()
        with H5Writer(spec, self.h5_path) as writer:
            key = self.write_conditioned(
                writer,
                eye_images=self.images(2),
                blink_presence=self.labels(2),
                eye_state=self.labels(2),
                head_pose=self.pose(2),
                quality_signals={name: np.zeros(2, dtype=np.float32) for name in QUALITY_SIGNALS},
                blink_ids=np.asarray([7, 7], dtype=np.int32),
            )
        with h5py.File(self.h5_path) as handle:
            stored = np.asarray(handle["test"][key][BLINK_IDS])
        self.assertEqual(stored.tolist(), [7, 7, NO_BLINK, NO_BLINK])

    def test_a_declared_field_must_be_supplied(self):
        spec = self.conditioned_spec()
        with H5Writer(spec, self.h5_path) as writer, self.assertRaises(WriterError):
            self.write_one(writer)  # no head_pose

    def test_an_undeclared_corpus_writes_nothing(self):
        # CEW and MRL have no face box, so no pose — they must stay writable.
        spec = self.spec()
        with H5Writer(spec, self.h5_path) as writer:
            key = self.write_one(writer)
        with h5py.File(self.h5_path) as handle:
            self.assertNotIn(HEAD_POSE, handle["test"][key])

    def test_the_root_attrs_announce_the_fields(self):
        # The loader decides what to declare from these.
        spec = self.conditioned_spec()
        with H5Writer(spec, self.h5_path) as writer:
            self.write_conditioned(writer)
        with h5py.File(self.h5_path) as handle:
            self.assertTrue(bool(handle.attrs["has_head_pose"]))
            self.assertTrue(bool(handle.attrs["has_blink_ids"]))
            self.assertEqual(list(handle.attrs["quality_signals"]), list(QUALITY_SIGNALS))
