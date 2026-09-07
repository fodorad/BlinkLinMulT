"""Tests for the OmniLoader schema bridge."""

from __future__ import annotations

import unittest

import torch

from blinklinmult.data.omni import (
    build_schemas,
    dataset_schema,
    eye_image_shape,
    unified_specs,
)
from blinklinmult.data.schema import (
    BLINK_PRESENCE,
    EYE_EMBEDDING,
    EYE_FEATURE,
    EYE_IMAGE,
    EYE_STATE,
    TARGET_PLACEHOLDER,
    DatasetSpec,
    SchemaError,
)


def video(**overrides) -> DatasetSpec:
    defaults = {
        "name": "rn30",
        "fps": 30.0,
        "window_seconds": 0.5,
        "image_size": 64,
        "has_blink_presence": True,
        "has_eye_state": True,
    }
    return DatasetSpec(**{**defaults, **overrides})


def slow_video(**overrides) -> DatasetSpec:
    """A 15 fps corpus: half the frames of `video` for the same duration."""
    defaults = {
        "name": "rn15",
        "fps": 15.0,
        "window_seconds": 0.5,
        "image_size": 64,
        "has_blink_presence": True,
        "has_eye_state": True,
    }
    return DatasetSpec(**{**defaults, **overrides})


def still(**overrides) -> DatasetSpec:
    defaults = {
        "name": "cew",
        "fps": None,
        "window_seconds": None,
        "image_size": 64,
        "has_eye_state": True,
    }
    return DatasetSpec(**{**defaults, **overrides})


def clips(**overrides) -> DatasetSpec:
    defaults = {
        "name": "hust_lebw",
        "fps": 30.0,
        "window_seconds": 0.5,
        "image_size": 64,
        "has_blink_presence": True,
        "has_eye_state": False,
    }
    return DatasetSpec(**{**defaults, **overrides})


class TestEyeImageShape(unittest.TestCase):
    def test_is_channels_height_width(self):
        # One eye per sample: no eye axis in the tensor.
        self.assertEqual(eye_image_shape(64), (3, 64, 64))

    def test_tracks_the_crop_size(self):
        self.assertEqual(eye_image_shape(32), (3, 32, 32))


class TestDatasetSchema(unittest.TestCase):
    def test_eye_image_is_always_a_feature(self):
        schema = dataset_schema(video(), time_dim=15, image_size=64, feature_dim=None)
        self.assertEqual([s.name for s in schema.features], [EYE_IMAGE])

    def test_eye_image_is_declared_as_a_structured_sequence(self):
        # Declared as an image, not as a wide feature axis: omniloader >= 1.1
        # carries the structured shape through padding, masking and collate.
        schema = dataset_schema(video(), time_dim=15, image_size=64, feature_dim=None)
        spec = schema.features[0]
        self.assertTrue(spec.is_sequence)
        self.assertEqual(spec.time_dim, 15)
        self.assertEqual(spec.shape, eye_image_shape(64))
        self.assertIsNone(spec.feature_dim)

    def test_the_declared_value_shape_is_an_image_sequence(self):
        schema = dataset_schema(video(), time_dim=15, image_size=64, feature_dim=None)
        self.assertEqual(schema.features[0].value_shape, (15, 3, 64, 64))
        # One flag per timestep; an image is valid or not as a whole.
        self.assertEqual(schema.features[0].mask_shape, (15,))

    def test_eye_features_are_declared_when_supplied(self):
        schema = dataset_schema(video(feature_dim=160), time_dim=15, image_size=64, feature_dim=160)
        self.assertEqual([s.name for s in schema.features], [EYE_IMAGE, EYE_FEATURE])

    def test_targets_reflect_what_the_corpus_annotates(self):
        both = dataset_schema(video(), 15, 64, None)
        self.assertEqual([s.name for s in both.targets], [BLINK_PRESENCE, EYE_STATE])

        state_only = dataset_schema(still(), 15, 64, None)
        self.assertEqual([s.name for s in state_only.targets], [EYE_STATE])

        blink_only = dataset_schema(clips(), 15, 64, None)
        self.assertEqual([s.name for s in blink_only.targets], [BLINK_PRESENCE])

    def test_eye_state_is_a_scalar_sequence(self):
        # One value per timestep for this sample's single eye.
        schema = dataset_schema(still(), 15, 64, None)
        spec = next(s for s in schema.targets if s.name == EYE_STATE)
        self.assertIsNone(spec.feature_dim)
        self.assertEqual(spec.time_dim, 15)

    def test_blink_presence_is_a_scalar_sequence(self):
        schema = dataset_schema(video(), 15, 64, None)
        spec = next(s for s in schema.targets if s.name == BLINK_PRESENCE)
        self.assertIsNone(spec.feature_dim)
        self.assertEqual(spec.time_dim, 15)

    def test_targets_use_the_out_of_range_placeholder(self):
        schema = dataset_schema(video(), 15, 64, None)
        for spec in schema.targets:
            self.assertEqual(spec.placeholder, TARGET_PLACEHOLDER)
            self.assertTrue(spec.placeholder < 0 or spec.placeholder > 1)

    def test_dtypes_are_float32(self):
        schema = dataset_schema(video(), 15, 64, None)
        for spec in [*schema.features, *schema.targets]:
            self.assertEqual(spec.dtype, torch.float32)

    def test_a_still_corpus_is_declared_at_the_shared_window(self):
        # time_dim 1 natively, but the run's shared T is what the schema declares;
        # OmniLoader pads and masks the difference.
        schema = dataset_schema(still(), time_dim=15, image_size=64, feature_dim=None)
        self.assertEqual(schema.features[0].time_dim, 15)

    def test_feature_width_disagreement_raises(self):
        with self.assertRaises(SchemaError) as ctx:
            dataset_schema(video(feature_dim=160), 15, 64, feature_dim=64)
        self.assertIn("comparable", str(ctx.exception))

    def test_declared_features_with_no_shared_width_raises(self):
        with self.assertRaises(SchemaError):
            dataset_schema(video(feature_dim=160), 15, 64, feature_dim=None)


class TestUnifiedSpecs(unittest.TestCase):
    def test_time_dim_defaults_to_the_longest(self):
        time_dim, _, _ = unified_specs([video(), still()])
        self.assertEqual(time_dim, 15)

    def test_rates_resolve_to_the_longest_frame_count(self):
        # 30 fps needs 15 frames for 0.5s, 15 fps needs 8; the shorter is padded
        # rather than the longer truncated.
        time_dim, _, _ = unified_specs([video(), slow_video()])
        self.assertEqual(time_dim, 15)

    def test_explicit_time_dim_wins(self):
        time_dim, _, _ = unified_specs([video(), still()], time_dim=8)
        self.assertEqual(time_dim, 8)

    def test_a_shortened_window_warns_about_truncation(self):
        with self.assertLogs("blinklinmult.data.omni", level="WARNING") as logs:
            unified_specs([video()], time_dim=4)
        self.assertIn("cropped on read", "".join(logs.output))

    def test_still_mode_does_not_warn_about_truncation(self):
        # In still mode the frame is chosen before OmniLoader sees the sample,
        # so nothing is cropped -- warning would be false, and a warning the
        # reader learns to ignore is worse than none.
        with self.assertNoLogs("blinklinmult.data.omni", level="WARNING"):
            unified_specs([video()], time_dim=1, stills=True)

    def test_agreeing_image_sizes_are_inferred(self):
        _, image_size, _ = unified_specs([video(), still()])
        self.assertEqual(image_size, 64)

    def test_disagreeing_image_sizes_raise(self):
        with self.assertRaises(SchemaError) as ctx:
            unified_specs([video(image_size=64), still(image_size=32)])
        self.assertIn("re-extract", str(ctx.exception))

    def test_explicit_image_size_resolves_a_disagreement(self):
        _, image_size, _ = unified_specs(
            [video(image_size=64), still(image_size=32)], image_size=64
        )
        self.assertEqual(image_size, 64)

    def test_feature_dim_is_none_when_no_corpus_supplies_features(self):
        _, _, feature_dim = unified_specs([video(), still()])
        self.assertIsNone(feature_dim)

    def test_feature_dim_is_taken_from_the_supplying_corpus(self):
        _, _, feature_dim = unified_specs([video(feature_dim=160), still()])
        self.assertEqual(feature_dim, 160)

    def test_disagreeing_feature_widths_raise(self):
        with self.assertRaises(SchemaError) as ctx:
            unified_specs([video(feature_dim=160), clips(feature_dim=64)])
        self.assertIn("differing widths", str(ctx.exception))

    def test_empty_list_raises(self):
        with self.assertRaises(SchemaError):
            unified_specs([])

    def test_zero_time_dim_raises(self):
        with self.assertRaises(SchemaError):
            unified_specs([video()], time_dim=0)


class TestBuildSchemas(unittest.TestCase):
    def test_returns_one_schema_per_spec_in_order(self):
        specs = [video(), still(), clips()]
        schemas, time_dim, image_size, feature_dim = build_schemas(specs)
        self.assertEqual(len(schemas), 3)
        self.assertEqual(time_dim, 15)
        self.assertEqual(image_size, 64)
        self.assertIsNone(feature_dim)

    def test_every_schema_shares_the_resolved_shape(self):
        schemas, time_dim, image_size, _ = build_schemas([video(), still(), clips()])
        for schema in schemas:
            for spec in [*schema.features, *schema.targets]:
                self.assertEqual(spec.time_dim, time_dim)
        self.assertEqual(schemas[0].features[0].shape, eye_image_shape(image_size))

    def test_the_union_of_targets_spans_both_tasks(self):
        schemas, _, _, _ = build_schemas([still(), clips()])
        names = {s.name for schema in schemas for s in schema.targets}
        # Neither corpus alone annotates both, but together they do -- which is
        # exactly what OmniLoader's masking makes trainable in one run.
        self.assertEqual(names, {EYE_STATE, BLINK_PRESENCE})

    def test_shared_names_agree_on_shape_across_corpora(self):
        # OmniLoader rejects specs that share a name and disagree; this is the
        # property build_schemas exists to guarantee.
        schemas, _, _, _ = build_schemas([video(), still(), clips()])
        by_name: dict[str, tuple] = {}
        for schema in schemas:
            for spec in [*schema.features, *schema.targets]:
                shape = (spec.value_shape, spec.dtype)
                if spec.name in by_name:
                    self.assertEqual(by_name[spec.name], shape)
                by_name[spec.name] = shape


if __name__ == "__main__":
    unittest.main()


class TestEmbeddingIsDeclared(unittest.TestCase):
    """An undeclared field is dropped by the collate, silently.

    Measured 2026-08-30: the datamodule logged "reading cached embeddings" while
    every batch arrived without one, so the encoder ran anyway and a 14.5x
    speedup was lost. The log said it worked; the tensor said otherwise.
    """

    def _spec(self):
        return DatasetSpec(
            name="rn30",
            fps=30.0,
            window_seconds=1.5,
            image_size=64,
            has_blink_presence=True,
            has_eye_state=True,
        )

    def test_it_is_absent_when_no_cache_is_configured(self):
        schema = dataset_schema(self._spec(), 45, 64, None, None)
        self.assertNotIn(EYE_EMBEDDING, [f.name for f in schema.features])

    def test_it_is_declared_when_a_width_is_given(self):
        schema = dataset_schema(self._spec(), 45, 64, None, 256)
        self.assertIn(EYE_EMBEDDING, [f.name for f in schema.features])

    def test_the_declared_width_matches(self):
        schema = dataset_schema(self._spec(), 45, 64, None, 256)
        spec = next(f for f in schema.features if f.name == EYE_EMBEDDING)
        self.assertEqual(spec.feature_dim, 256)

    def test_build_schemas_passes_the_width_to_every_corpus(self):
        schemas, *_ = build_schemas([self._spec()], embedding_dim=256)
        for schema in schemas:
            self.assertIn(EYE_EMBEDDING, [f.name for f in schema.features])
