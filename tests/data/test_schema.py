"""Tests for the per-corpus dataset contract."""

from __future__ import annotations

import unittest

from blinklinmult.data.schema import (
    BLINK_PRESENCE,
    DATASETS,
    EYE_FEATURE,
    EYE_IMAGE,
    EYE_SIDES,
    EYE_STATE,
    IMAGE_DATASETS,
    LEFT,
    RIGHT,
    SUBSETS,
    UNKNOWN_EYE,
    VIDEO_DATASETS,
    BuildStats,
    DatasetSpec,
    SchemaError,
    build_sample_id,
    parse_sample_id,
)


def video_spec(**overrides) -> DatasetSpec:
    """A spec for a video corpus annotating both tasks."""
    defaults = {
        "name": "rn30",
        "fps": 30.0,
        "window_seconds": 0.5,
        "image_size": 64,
        "has_blink_presence": True,
        "has_eye_state": True,
    }
    return DatasetSpec(**{**defaults, **overrides})


def still_spec(**overrides) -> DatasetSpec:
    """A spec for a still-image corpus annotating eye state only."""
    defaults = {
        "name": "cew",
        "fps": None,
        "window_seconds": None,
        "image_size": 64,
        "has_blink_presence": False,
        "has_eye_state": True,
    }
    return DatasetSpec(**{**defaults, **overrides})


class TestConstants(unittest.TestCase):
    def test_subsets_are_the_three_splits(self):
        self.assertEqual(SUBSETS, ("train", "valid", "test"))

    def test_supported_datasets(self):
        self.assertEqual(
            set(DATASETS),
            {"talkingface", "rn15", "rn30", "cew", "mrl", "hust_lebw", "mpeblink"},
        )

    def test_rn_is_split_by_rate(self):
        # The two rates derive different frame counts from one window duration,
        # so they cannot share a declaration.
        self.assertIn("rn15", DATASETS)
        self.assertIn("rn30", DATASETS)
        self.assertNotIn("rn", DATASETS)

    def test_video_and_image_corpora_partition_the_datasets(self):
        self.assertEqual(set(VIDEO_DATASETS) | set(IMAGE_DATASETS), set(DATASETS))
        self.assertFalse(set(VIDEO_DATASETS) & set(IMAGE_DATASETS))

    def test_retired_corpora_are_absent(self):
        # RT-BENE/RT-GENE and ZJU were in the 1.x paper but not the v2 benchmark.
        for name in ("rtgene", "rtbene", "zju"):
            self.assertNotIn(name, DATASETS)

    def test_keys_are_distinct(self):
        keys = {EYE_IMAGE, EYE_FEATURE, BLINK_PRESENCE, EYE_STATE}
        self.assertEqual(len(keys), 4)


class TestDatasetSpec(unittest.TestCase):
    def test_video_spec_is_valid(self):
        spec = video_spec()
        self.assertTrue(spec.has_blink_presence)
        self.assertTrue(spec.has_eye_state)

    def test_still_spec_is_valid(self):
        self.assertEqual(still_spec().time_dim, 1)

    def test_unknown_name_raises(self):
        with self.assertRaises(SchemaError) as ctx:
            video_spec(name="zju")
        self.assertIn("Unknown dataset", str(ctx.exception))

    def test_non_positive_window_seconds_raises(self):
        with self.assertRaises(SchemaError):
            video_spec(window_seconds=0.0)

    def test_zero_image_size_raises(self):
        with self.assertRaises(SchemaError):
            video_spec(image_size=0)

    def test_negative_feature_dim_raises(self):
        with self.assertRaises(SchemaError):
            video_spec(feature_dim=-5)

    def test_non_positive_fps_raises(self):
        with self.assertRaises(SchemaError):
            video_spec(fps=0.0)

    def test_supervising_no_task_raises(self):
        with self.assertRaises(SchemaError) as ctx:
            video_spec(has_blink_presence=False, has_eye_state=False)
        self.assertIn("neither", str(ctx.exception))

    def test_blink_presence_without_an_fps_raises(self):
        # Blink presence is scored over a window, and a still corpus has none.
        # Not because a still cannot show a closed eye -- it can, and that is
        # exactly its eye_state label -- but because there is no window to
        # aggregate, so the target would be a second name for the first.
        with self.assertRaises(SchemaError) as ctx:
            still_spec(has_blink_presence=True)
        self.assertIn("window", str(ctx.exception))

    def test_a_video_corpus_without_a_window_raises(self):
        with self.assertRaises(SchemaError):
            video_spec(window_seconds=None)

    def test_has_eye_feature_reflects_feature_dim(self):
        self.assertFalse(video_spec().has_eye_feature)
        self.assertTrue(video_spec(feature_dim=160).has_eye_feature)

    def test_time_dim_is_derived_from_the_rate(self):
        # One window duration, two frame counts: that is the whole point.
        self.assertEqual(video_spec(fps=30.0, window_seconds=0.5).time_dim, 15)
        self.assertEqual(video_spec(fps=15.0, window_seconds=0.5).time_dim, 8)
        # Rounds half up: banker's rounding would give 12 here, silently
        # shortening the window relative to 30 fps.
        self.assertEqual(video_spec(fps=25.0, window_seconds=0.5).time_dim, 13)

    def test_a_still_corpus_is_one_frame(self):
        self.assertEqual(still_spec().time_dim, 1)

    def test_is_video_reflects_the_rate(self):
        self.assertTrue(video_spec().is_video)
        self.assertFalse(still_spec().is_video)

    def test_with_window_rederives_the_frame_count(self):
        spec = video_spec(fps=30.0, window_seconds=0.5)
        self.assertEqual(spec.with_window(1.0).time_dim, 30)
        # The original is untouched.
        self.assertEqual(spec.time_dim, 15)

    def test_with_window_leaves_a_still_corpus_alone(self):
        spec = still_spec()
        self.assertIs(spec.with_window(1.0), spec)

    def test_feature_keys_without_eye_features(self):
        self.assertEqual(video_spec().feature_keys, [EYE_IMAGE])

    def test_feature_keys_with_eye_features(self):
        self.assertEqual(video_spec(feature_dim=160).feature_keys, [EYE_IMAGE, EYE_FEATURE])

    def test_target_keys_reflect_what_is_annotated(self):
        self.assertEqual(video_spec().target_keys, [BLINK_PRESENCE, EYE_STATE])
        self.assertEqual(still_spec().target_keys, [EYE_STATE])

    def test_specs_are_frozen(self):
        with self.assertRaises(Exception):
            video_spec().fps = 60.0


class TestFromDict(unittest.TestCase):
    def test_builds_from_a_mapping(self):
        spec = DatasetSpec.from_dict({"name": "mrl", "has_eye_state": True})
        self.assertEqual(spec.name, "mrl")

    def test_unknown_key_raises(self):
        with self.assertRaises(SchemaError) as ctx:
            DatasetSpec.from_dict({"name": "mrl", "time_dim": 1, "has_eye_state": True, "typo": 1})
        self.assertIn("typo", str(ctx.exception))

    def test_missing_name_raises(self):
        with self.assertRaises(SchemaError) as ctx:
            DatasetSpec.from_dict({"has_eye_state": True})
        self.assertIn("name", str(ctx.exception))

    def test_validation_still_applies(self):
        with self.assertRaises(SchemaError):
            DatasetSpec.from_dict({"name": "cew", "time_dim": 1})


class TestBuildStats(unittest.TestCase):
    def test_n_samples_sums_the_splits(self):
        stats = BuildStats(dataset="cew", per_subset={"train": 10, "valid": 3, "test": 4})
        self.assertEqual(stats.n_samples, 17)

    def test_defaults_are_empty(self):
        stats = BuildStats()
        self.assertEqual(stats.n_samples, 0)
        self.assertEqual(stats.per_subset, {})


if __name__ == "__main__":
    unittest.main()


class TestSampleId(unittest.TestCase):
    def test_round_trips(self):
        sample_id = build_sample_id("rn30_1", "000881", LEFT)
        self.assertEqual(parse_sample_id(sample_id), ("rn30_1", "000881", LEFT))

    def test_the_two_eyes_of_a_window_differ_only_in_the_side(self):
        left = build_sample_id("v", "000010", LEFT)
        right = build_sample_id("v", "000010", RIGHT)
        self.assertNotEqual(left, right)
        self.assertEqual(parse_sample_id(left)[:2], parse_sample_id(right)[:2])

    def test_unknown_side_is_representable(self):
        # MRL ships one eye without saying which.
        sample_id = build_sample_id("s0001", "000042", UNKNOWN_EYE)
        self.assertEqual(parse_sample_id(sample_id)[2], UNKNOWN_EYE)

    def test_every_declared_side_is_accepted(self):
        for side in EYE_SIDES:
            with self.subTest(side=side):
                build_sample_id("v", "000000", side)

    def test_unrecognised_side_raises(self):
        with self.assertRaises(SchemaError):
            build_sample_id("v", "000000", "middle")

    def test_a_separator_in_a_component_raises(self):
        # Would make the id ambiguous to parse.
        with self.assertRaises(SchemaError):
            build_sample_id("a|b", "000000", LEFT)

    def test_malformed_id_raises(self):
        with self.assertRaises(SchemaError):
            parse_sample_id("no-separators-here")


class TestPerSignalQualityDeclaration(unittest.TestCase):
    """Quality signals are declared per signal, not all-or-nothing.

    A still-image corpus has one frame and so no box jitter; MRL-Eye does not
    record which eye a sample is, so it can have no symmetry either. Forcing the
    full set would make those corpora undeclarable.
    """

    def test_a_subset_is_accepted(self):
        spec = DatasetSpec(
            name="cew",
            window_seconds=None,
            has_eye_state=True,
            quality_signals=("eye_blur", "eye_exposure"),
        )
        self.assertEqual(spec.quality_signals, ("eye_blur", "eye_exposure"))

    def test_no_signals_is_the_default(self):
        spec = DatasetSpec(name="cew", window_seconds=None, has_eye_state=True)
        self.assertFalse(spec.has_quality_signals)

    def test_any_signal_sets_the_flag(self):
        spec = DatasetSpec(
            name="cew", window_seconds=None, has_eye_state=True, quality_signals=("eye_blur",)
        )
        self.assertTrue(spec.has_quality_signals)

    def test_an_unknown_signal_is_rejected(self):
        with self.assertRaises(SchemaError):
            DatasetSpec(
                name="cew",
                window_seconds=None,
                has_eye_state=True,
                quality_signals=("eye_sharpness",),
            )
