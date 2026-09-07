"""End-to-end tests for the OmniLoader-backed datamodule.

These build real HDF5 files and read them back through
:class:`omniloader.OmniLoader`. That is the point: the builder's layout and
OmniLoader's reader are two halves of one contract, and a test that mocked
either side would not check that they agree.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from blinklinmult.data.datamodule import (
    MAX_WORKERS,
    STRATEGIES,
    BlinkDataModule,
    CachedEmbeddingDataset,
    DataModuleError,
    EmbeddingCacheError,
    _subsample,
    build_strategy,
    pinning_helps,
    worker_count,
)
from blinklinmult.data.schema import (
    BLINK_PRESENCE,
    EYE_EMBEDDING,
    EYE_FEATURE_DIM,
    EYE_IMAGE,
    EYE_STATE,
    LEFT,
    RIGHT,
    SAMPLE_KEY,
    DatasetSpec,
)
from blinklinmult.data.writer import H5Writer
from blinklinmult.train.config import DataConfig

IMAGE_SIZE = 32
TIME_DIM = 4


class CorpusFixture(unittest.TestCase):
    """Builds real per-corpus HDF5 files in a temporary tree."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def video_spec(self) -> DatasetSpec:
        return DatasetSpec(
            name="rn30",
            fps=8.0,  # -> TIME_DIM frames at 0.5s
            window_seconds=0.5,
            image_size=IMAGE_SIZE,
            has_blink_presence=True,
            has_eye_state=True,
        )

    def slow_spec(self) -> DatasetSpec:
        """A half-rate corpus: same duration, half the frames."""
        return DatasetSpec(
            name="rn15",
            fps=4.0,
            window_seconds=0.5,
            image_size=IMAGE_SIZE,
            has_blink_presence=True,
            has_eye_state=True,
        )

    def still_spec(self) -> DatasetSpec:
        return DatasetSpec(
            name="cew",
            fps=None,
            window_seconds=None,
            image_size=IMAGE_SIZE,
            has_blink_presence=False,
            has_eye_state=True,
        )

    def make_corpus(
        self, spec: DatasetSpec, per_split: int = 3, splits: tuple[str, ...] = ()
    ) -> None:
        """Write and build one corpus."""
        splits = splits or ("train", "valid", "test")
        processed = self.root / "data" / "processed" / spec.name
        processed.mkdir(parents=True, exist_ok=True)
        with H5Writer(spec, processed / f"{spec.name}.h5") as writer:
            for subset in splits:
                for index in range(per_split):
                    labels = np.zeros(spec.time_dim, dtype=np.float32)
                    if index % 2 == 0:
                        labels[0] = 1.0
                    for eye_side in (LEFT, RIGHT):
                        writer.add(
                            subset=subset,
                            video_id=f"{spec.name}_{subset}_rec{index}",
                            frame_group=f"{index:06d}",
                            eye_side=eye_side,
                            eye_images=np.random.rand(
                                spec.time_dim, 3, spec.image_size, spec.image_size
                            ).astype(np.float32),
                            blink_presence=labels if spec.has_blink_presence else None,
                            eye_state=(
                                np.zeros(spec.time_dim, dtype=np.float32)
                                if spec.has_eye_state
                                else None
                            ),
                        )

    def datamodule(self, specs: list[DatasetSpec], **overrides) -> BlinkDataModule:
        defaults = {
            "datasets": [s.name for s in specs],
            "window_seconds": 0.5,
            "batch_size": 2,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        }
        config = DataConfig(**{**defaults, **overrides})
        return BlinkDataModule(config, specs, root=self.root)


class TestWorkerCount(unittest.TestCase):
    """Workers are capped, not forced.

    An earlier version clamped every request to 0, measured on a 123 MB corpus.
    Re-measured over the 54 GB frame-wise mix, workers are worth ~4x -- so the
    cap now protects against oversubscription rather than forbidding workers.
    The symptom of getting this wrong is silence: a run that merely looks slow.
    """

    def test_a_request_below_the_cap_is_honoured(self):
        # The regression that matters: an earlier clamp returned MAX_WORKERS
        # unconditionally, so *every* setting silently loaded in-process and a
        # benchmark sweeping the value measured the same thing five times.
        self.assertEqual(worker_count(2), 2)

    def test_a_request_above_the_cap_is_clamped(self):
        self.assertEqual(worker_count(64), MAX_WORKERS)

    def test_zero_stays_in_process(self):
        self.assertEqual(worker_count(0), 0)

    def test_the_cap_leaves_cores_for_training(self):
        # Measured throughput peaks at 10 workers on this 10-core machine and
        # falls again by 12; the cap sits below the peak so the training process
        # is not starved by its own loaders.
        self.assertGreater(MAX_WORKERS, 0)
        self.assertLessEqual(MAX_WORKERS, 8)


class TestBuildStrategy(unittest.TestCase):
    def test_every_named_strategy_builds(self):
        for name in STRATEGIES:
            with self.subTest(strategy=name):
                self.assertIsNotNone(build_strategy(name, [100, 50]))

    def test_kwargs_are_forwarded(self):
        self.assertIsNotNone(build_strategy("temperature", [100, 50], {"temperature": 3.0}))

    def test_unknown_strategy_raises(self):
        with self.assertRaises(DataModuleError):
            build_strategy("magic", [10])

    def test_an_unfitting_kwarg_names_the_strategy(self):
        """A wrong `strategy_kwargs` key must be actionable from the message.

        Without this the failure surfaces as a bare ``TypeError`` from inside
        OmniLoader, naming neither the offending key nor the strategy that
        rejected it -- and the caller only has a YAML file to work from.
        """
        with self.assertRaises(DataModuleError) as caught:
            build_strategy("proportional", [10, 20], {"not_a_real_knob": 1})
        message = str(caught.exception)
        self.assertIn("strategy_kwargs", message)
        self.assertIn("proportional", message)


class TestSingleCorpus(CorpusFixture):
    def test_setup_opens_every_split(self):
        spec = self.video_spec()
        self.make_corpus(spec)
        module = self.datamodule([spec])
        module.setup(None)
        self.assertEqual(set(module.loaders), {"train", "valid", "test"})

    def test_a_batch_carries_the_declared_keys_and_masks(self):
        spec = self.video_spec()
        self.make_corpus(spec)
        module = self.datamodule([spec])
        module.setup("fit")

        batch = next(iter(module.train_dataloader()))
        for key in (EYE_IMAGE, BLINK_PRESENCE, EYE_STATE):
            with self.subTest(key=key):
                self.assertIn(key, batch)
                self.assertIn(f"{key}_mask", batch)

    def test_batch_shapes_match_the_resolved_schema(self):
        spec = self.video_spec()
        self.make_corpus(spec)
        module = self.datamodule([spec])
        module.setup("fit")

        batch = next(iter(module.train_dataloader()))
        self.assertEqual(tuple(batch[EYE_IMAGE].shape), (2, TIME_DIM, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertEqual(batch[BLINK_PRESENCE].shape, (2, TIME_DIM))
        # One eye per sample, so eye_state is a scalar sequence.
        self.assertEqual(batch[EYE_STATE].shape, (2, TIME_DIM))

    def test_resolved_shape_is_exposed_for_the_model(self):
        spec = self.video_spec()
        self.make_corpus(spec)
        module = self.datamodule([spec])
        self.assertEqual(module.time_dim, TIME_DIM)
        self.assertEqual(module.image_size, IMAGE_SIZE)
        self.assertIsNone(module.feature_dim)

    def test_an_unbuilt_corpus_raises_with_the_make_target(self):
        module = self.datamodule([self.video_spec()])
        with self.assertRaises(DataModuleError) as ctx:
            module.setup("fit")
        self.assertIn("make preprocess-rn30", str(ctx.exception))

    def test_no_datasets_raises(self):
        with self.assertRaises(DataModuleError):
            BlinkDataModule(DataConfig(datasets=["cew"]), [], root=self.root)


class TestJointCorpora(CorpusFixture):
    def setUp(self) -> None:
        super().setUp()
        self.video = self.video_spec()
        self.still = self.still_spec()
        self.make_corpus(self.video)
        self.make_corpus(self.still)

    def test_both_corpora_contribute(self):
        module = self.datamodule([self.video, self.still])
        module.setup("fit")
        self.assertEqual(len(module.loaders["train"]), 12)

    def test_the_union_schema_spans_both_tasks(self):
        module = self.datamodule([self.video, self.still], batch_size=6)
        module.setup("fit")
        batch = next(iter(module.train_dataloader()))
        # CEW annotates no blinks, but the key is present for every sample --
        # that is what lets one head see a mixed batch.
        self.assertIn(BLINK_PRESENCE, batch)
        self.assertIn(EYE_STATE, batch)

    def test_a_still_corpus_is_padded_to_the_shared_window(self):
        module = self.datamodule([self.video, self.still], batch_size=6)
        module.setup("fit")
        batch = next(iter(module.train_dataloader()))
        # Every sample is TIME_DIM long regardless of its corpus's native length.
        self.assertEqual(batch[EYE_IMAGE].shape[1], TIME_DIM)

    def test_some_blink_masks_are_false_in_a_mixed_batch(self):
        module = self.datamodule([self.video, self.still], batch_size=6, strategy="round_robin")
        module.setup("fit")

        seen_false = False
        for batch in module.train_dataloader():
            if not batch[f"{BLINK_PRESENCE}_mask"].all():
                seen_false = True
                break
        # CEW samples must arrive with blink presence masked out; without this
        # the blink head would train on placeholders.
        self.assertTrue(seen_false)

    def test_eye_state_is_supervised_by_both_corpora(self):
        module = self.datamodule([self.video, self.still], batch_size=6)
        module.setup("fit")
        batch = next(iter(module.train_dataloader()))
        self.assertTrue(batch[f"{EYE_STATE}_mask"].any())

    def test_masked_targets_carry_the_placeholder(self):
        module = self.datamodule([self.video, self.still], batch_size=6, strategy="round_robin")
        module.setup("fit")
        for batch in module.train_dataloader():
            mask = batch[f"{BLINK_PRESENCE}_mask"]
            if not mask.all():
                placeholder_values = batch[BLINK_PRESENCE][~mask]
                # Out of [0, 1], so a placeholder reaching a metric is visibly
                # wrong rather than plausible.
                self.assertTrue((placeholder_values < 0).all())
                break


class TestSplits(CorpusFixture):
    def test_a_corpus_missing_a_split_is_skipped_not_fatal(self):
        # TalkingFace is held out entirely for testing; its train split is empty.
        video = self.video_spec()
        self.make_corpus(video, splits=("test",))
        module = self.datamodule([video])
        module.setup(None)

        self.assertIn("test", module.loaders)
        self.assertNotIn("train", module.loaders)

    def test_requesting_an_absent_split_raises(self):
        video = self.video_spec()
        self.make_corpus(video, splits=("test",))
        module = self.datamodule([video])
        module.setup(None)
        with self.assertRaises(DataModuleError):
            module.train_dataloader()

    def test_a_corpus_with_no_train_split_still_joins_the_test_split(self):
        video = self.video_spec()
        still = self.still_spec()
        self.make_corpus(video)
        self.make_corpus(still, splits=("test",))

        module = self.datamodule([video, still])
        module.setup(None)
        self.assertEqual(len(module.loaders["train"]), 6)
        self.assertEqual(len(module.loaders["test"]), 12)


class TestDataloaders(CorpusFixture):
    def setUp(self) -> None:
        super().setUp()
        self.spec = self.video_spec()
        self.make_corpus(self.spec, per_split=4)

    def test_validation_and_test_are_deterministic(self):
        module = self.datamodule([self.spec])
        module.setup(None)

        first = [b[SAMPLE_KEY] for b in module.val_dataloader()]
        second = [b[SAMPLE_KEY] for b in module.val_dataloader()]
        self.assertEqual(first, second)

    def test_every_eval_sample_is_seen_exactly_once(self):
        module = self.datamodule([self.spec])
        module.setup("test")
        seen = [sid for batch in module.test_dataloader() for sid in batch[SAMPLE_KEY]]
        self.assertEqual(len(seen), 8)
        self.assertEqual(len(set(seen)), 8)

    def test_set_epoch_does_not_raise(self):
        module = self.datamodule([self.spec])
        module.setup("fit")
        module.set_epoch(3)

    def test_tensors_are_float32_for_the_model(self):
        # Stored as float16 to halve the file; a model in fp32 needs fp32 in.
        module = self.datamodule([self.spec])
        module.setup("fit")
        batch = next(iter(module.train_dataloader()))
        self.assertEqual(batch[EYE_IMAGE].dtype, torch.float32)

    def test_masks_are_boolean(self):
        module = self.datamodule([self.spec])
        module.setup("fit")
        batch = next(iter(module.train_dataloader()))
        self.assertEqual(batch[f"{EYE_IMAGE}_mask"].dtype, torch.bool)


class TestPartialCorpora(CorpusFixture):
    """A corpus that does not carry every split.

    TalkingFace and MPEblink are evaluation-only -- they have a test split and
    nothing else -- so a missing split is a normal corpus shape here, not a
    broken build. It must be skipped with a log line rather than raising.
    """

    def test_a_missing_split_is_skipped_not_fatal(self):
        """An eval-only corpus must still open."""
        spec = self.video_spec()
        self.make_corpus(spec, splits=("test",))
        module = self.datamodule([spec])
        module.setup(None)
        self.assertIn("test", module.loaders)
        self.assertNotIn("train", module.loaders)

    def test_subsampling_a_corpus_keeps_it_usable(self):
        """`fit_fractions` is keyed by *corpus*, and must not empty its split.

        A fraction that rounded to zero samples would produce an empty loader
        and a run that trains on nothing while reporting success.
        """
        spec = self.video_spec()
        self.make_corpus(spec, per_split=6)
        module = self.datamodule([spec], fit_fractions={spec.name: 0.5})
        module.setup(None)
        self.assertIn("train", module.loaders)
        self.assertGreater(len(module.loaders["train"].datasets), 0)


if __name__ == "__main__":
    unittest.main()


class TestMixedRates(CorpusFixture):
    """Corpora at different rates cover the same duration in different frames."""

    def test_the_shorter_corpus_is_padded_and_masked(self):
        fast, slow = self.video_spec(), self.slow_spec()
        self.assertEqual(fast.time_dim, TIME_DIM)
        self.assertEqual(slow.time_dim, TIME_DIM // 2)

        self.make_corpus(fast)
        self.make_corpus(slow)

        module = self.datamodule([fast, slow], batch_size=4, strategy="round_robin")
        module.setup("fit")
        # The run resolves to the longest frame count; nothing is truncated.
        self.assertEqual(module.time_dim, TIME_DIM)

        saw_padding = False
        for batch in module.train_dataloader():
            if not batch[f"{EYE_IMAGE}_mask"].all():
                saw_padding = True
                break
        self.assertTrue(saw_padding)


class TestStillMode(CorpusFixture):
    """A video corpus served one frame at a time, for the frame-wise model.

    The wiring these tests protect is that ``stills: true`` reaches OmniLoader
    as a genuine ``T = 1`` stream drawn from the *windowed* files -- no second
    build, and no reliance on OmniLoader cropping windows to their first frame,
    which would silently discard every blink that starts later in a window.
    """

    def blinking_corpus(self, spec: DatasetSpec, closed_at: int) -> None:
        """Write a corpus whose eye closes at one known frame of each window."""
        processed = self.root / "data" / "processed" / spec.name
        processed.mkdir(parents=True, exist_ok=True)
        with H5Writer(spec, processed / f"{spec.name}.h5") as writer:
            for subset in ("train", "valid", "test"):
                for index in range(6):
                    labels = np.zeros(spec.time_dim, dtype=np.float32)
                    labels[closed_at] = 1.0
                    for eye_side in (LEFT, RIGHT):
                        writer.add(
                            subset=subset,
                            video_id=f"{spec.name}_{subset}_rec{index}",
                            frame_group=f"{index:06d}",
                            eye_side=eye_side,
                            eye_images=np.random.rand(
                                spec.time_dim, 3, spec.image_size, spec.image_size
                            ).astype(np.float32),
                            blink_presence=labels,
                            eye_state=labels,
                        )

    def still_module(self, spec: DatasetSpec, **overrides) -> BlinkDataModule:
        return self.datamodule([spec], stills=True, still_stride=2, **overrides)

    def test_a_video_corpus_is_served_as_single_frames(self):
        spec = self.video_spec()
        self.blinking_corpus(spec, closed_at=1)
        module = self.still_module(spec)
        module.setup("fit")

        self.assertEqual(module.time_dim, 1)
        batch = next(iter(module.train_dataloader()))
        self.assertEqual(batch[EYE_IMAGE].shape[1], 1)
        self.assertEqual(batch[EYE_STATE].shape[1], 1)

    def test_a_blink_late_in_the_window_still_reaches_the_batch(self):
        # The property that distinguishes real frame selection from OmniLoader
        # cropping every window to frame 0: with the closure at the LAST frame,
        # a cropping implementation would serve nothing but open eyes.
        spec = self.video_spec()
        last = spec.time_dim - 1
        self.blinking_corpus(spec, closed_at=last)
        module = self.still_module(spec)
        module.setup("fit")

        seen = torch.cat([b[EYE_STATE][b[f"{EYE_STATE}_mask"]] for b in module.train_dataloader()])
        self.assertTrue(bool((seen > 0.5).any()))

    def test_the_training_split_is_class_balanced(self):
        spec = self.video_spec()
        self.blinking_corpus(spec, closed_at=1)
        module = self.still_module(spec)
        module.setup("fit")

        seen = torch.cat([b[EYE_STATE][b[f"{EYE_STATE}_mask"]] for b in module.train_dataloader()])
        closed = int((seen > 0.5).sum())
        self.assertEqual(closed, seen.numel() - closed)

    def test_a_still_corpus_passes_through_unwrapped(self):
        # Already one frame per sample: wrapping it would be a no-op at best.
        spec = self.still_spec()
        self.make_corpus(spec)
        module = self.still_module(spec)
        module.setup("fit")
        self.assertEqual(module.time_dim, 1)


class TestAugmentationGate(CorpusFixture):
    """Geometric augmentation is dropped when the run reads descriptors.

    The handcrafted vector is 152 dimensions of pixel-space landmark
    coordinates describing the crop as built. Rotating the image would leave
    them describing the unrotated eye — an inconsistency nothing downstream
    checks for, so it must not be created.
    """

    def module_with(self, spec: DatasetSpec, **overrides) -> BlinkDataModule:
        self.make_corpus(spec)
        module = self.datamodule([spec], augment={"strength": 1.0}, **overrides)
        module.setup("fit")
        return module

    def test_an_image_only_corpus_keeps_geometry(self):
        module = self.module_with(self.still_spec())
        transform = module._augmentation()
        self.assertIsNotNone(transform)
        self.assertTrue(transform.config.rotate)

    def test_a_model_reading_descriptors_drops_geometry(self):
        module = self.module_with(self.still_spec(), model_reads_features=True)
        transform = module._augmentation()
        self.assertIsNotNone(transform)
        self.assertFalse(transform.config.rotate)
        self.assertFalse(transform.config.translate)

    def test_a_corpus_supplying_descriptors_is_not_enough(self):
        # The gate keys on what the *model* reads, not what the corpora happen
        # to supply. RN15 and RN30 carry a 160-d vector, but BlinkCNN takes eye
        # crops alone -- gating on the corpora would drop rotation from a run
        # that could never be harmed by it.
        module = self.module_with(self.still_spec())
        module.feature_dim = EYE_FEATURE_DIM
        self.assertTrue(module._augmentation().config.rotate)

    def test_it_keeps_the_photometric_half(self):
        # Dropping geometry must not disable augmentation entirely: brightness
        # and contrast are safe alongside a descriptor and are what the corpora
        # most differ in.
        module = self.module_with(self.still_spec(), model_reads_features=True)
        self.assertTrue(module._augmentation().config.photometric)

    def test_no_augmentation_configured_means_no_transform(self):
        spec = self.still_spec()
        self.make_corpus(spec)
        module = self.datamodule([spec])
        module.setup("fit")
        self.assertIsNone(module._augmentation())


class TestPinningHelps(unittest.TestCase):
    """`pin_memory` is a CUDA optimisation MPS does not implement.

    PyTorch warns once per loader and then ignores the flag, so leaving it on is
    inert rather than harmful -- but a benchmark log full of ignorable warnings
    is one where a real warning goes unread.
    """

    def test_it_tracks_cuda_availability(self):
        self.assertEqual(pinning_helps(), torch.cuda.is_available())

    def test_it_returns_a_bool(self):
        """The value is passed straight to DataLoader, which type-checks it."""
        self.assertIsInstance(pinning_helps(), bool)


class TestSubsample(unittest.TestCase):
    """Per-corpus thinning of the fitting splits, never the test split."""

    def test_it_keeps_the_requested_share(self):
        self.assertEqual(len(_subsample(list(range(100)), 0.1, "corpus", "train")), 10)

    def test_it_spreads_across_the_split(self):
        """A stride, not a prefix: a prefix would keep whole recordings."""
        kept = _subsample(list(range(100)), 0.1, "corpus", "valid")
        self.assertEqual([kept[i] for i in range(len(kept))], list(range(0, 100, 10)))

    def test_a_full_fraction_returns_the_dataset_unchanged(self):
        data = list(range(50))
        self.assertIs(_subsample(data, 1.0, "corpus", "train"), data)

    def test_it_always_keeps_at_least_one_sample(self):
        self.assertGreaterEqual(len(_subsample(list(range(3)), 0.01, "corpus", "train")), 1)


class TestStridedSubsetCoverage(unittest.TestCase):
    """A thinned training split is eventually seen in full."""

    def test_training_shifts_each_epoch(self):
        kept = _subsample(list(range(100)), 0.1, "corpus", "train")
        first = [kept[i] for i in range(len(kept))]
        kept.set_epoch(1)
        self.assertNotEqual([kept[i] for i in range(len(kept))], first)

    def test_enough_epochs_cover_the_whole_corpus(self):
        """The point of shifting: 10% x 10 epochs sees all of it, not a tenth."""
        kept = _subsample(list(range(100)), 0.1, "corpus", "train")
        seen: set[int] = set()
        for epoch in range(10):
            kept.set_epoch(epoch)
            seen.update(kept[i] for i in range(len(kept)))
        self.assertEqual(seen, set(range(100)))

    def test_the_epoch_size_never_changes(self):
        """Shifting must not change the cost of an epoch."""
        kept = _subsample(list(range(97)), 0.1, "corpus", "train")
        sizes = set()
        for epoch in range(5):
            kept.set_epoch(epoch)
            sizes.add(len(kept))
        self.assertEqual(len(sizes), 1)

    def test_validation_stays_fixed(self):
        """A moving validation subset would change what early stopping compares."""
        kept = _subsample(list(range(100)), 0.1, "corpus", "valid")
        first = [kept[i] for i in range(len(kept))]
        kept.set_epoch(7)
        self.assertEqual([kept[i] for i in range(len(kept))], first)

    def test_indices_stay_in_range_when_the_offset_wraps(self):
        kept = _subsample(list(range(100)), 0.1, "corpus", "train")
        kept.set_epoch(999)
        values = [kept[i] for i in range(len(kept))]
        self.assertTrue(all(0 <= v < 100 for v in values))


class TestCachedEmbeddingDataset(unittest.TestCase):
    """Serving precomputed embeddings in place of a live encoder pass.

    The wrapper takes any indexable dataset, so it tests without Lightning. What
    matters is its refusal: a cache that does not cover every sample must stop
    the run rather than encode some samples live and read others from disk,
    which would train on two different feature distributions at once.
    """

    def _shard(self, keys: list[str]) -> dict:
        """A cache shard covering the given sample keys.

        Args:
            keys (list[str]): Sample ids the shard holds.

        Returns:
            dict: A shard in the layout the reader produces.
        """
        return {
            "keys": np.asarray(keys),
            "embedding": np.arange(len(keys) * 4, dtype="float32").reshape(len(keys), 4),
        }

    def _dataset(self, keys: list[str]) -> list[dict]:
        """A minimal indexable dataset of samples carrying ids."""
        return [{SAMPLE_KEY: key} for key in keys]

    def test_a_cached_embedding_is_attached_to_the_sample(self) -> None:
        """The point of the cache: the sample gains its embedding."""
        keys = ["a|0|left", "b|0|left"]
        wrapped = CachedEmbeddingDataset(self._dataset(keys), self._shard(keys), "rn30")
        sample = wrapped[1]
        self.assertIn(EYE_EMBEDDING, sample)
        self.assertEqual(tuple(sample[EYE_EMBEDDING].shape), (4,))

    def test_rows_are_matched_by_key_not_by_position(self) -> None:
        """A shard written in a different order must still line up.

        Matching by index would silently pair each sample with another
        sample's embedding -- a corpus that trains and scores plausibly wrong.
        """
        dataset = self._dataset(["b|0|left", "a|0|left"])
        shard = self._shard(["a|0|left", "b|0|left"])
        wrapped = CachedEmbeddingDataset(dataset, shard, "rn30")
        np.testing.assert_allclose(wrapped[0][EYE_EMBEDDING].numpy(), shard["embedding"][1])

    def test_a_sample_missing_from_the_cache_is_refused(self) -> None:
        """A stale cache must fail loudly, not fall back to live encoding."""
        wrapped = CachedEmbeddingDataset(
            self._dataset(["a|0|left", "unknown|0|left"]),
            self._shard(["a|0|left"]),
            "rn30",
        )
        with self.assertRaises(EmbeddingCacheError) as caught:
            _ = wrapped[1]
        self.assertIn("rebuild", str(caught.exception))

    def test_the_error_names_the_corpus(self) -> None:
        """A multi-corpus run needs to know which cache is stale."""
        wrapped = CachedEmbeddingDataset(
            self._dataset(["missing|0|left"]), self._shard(["a|0|left"]), "hust_lebw"
        )
        with self.assertRaises(EmbeddingCacheError) as caught:
            _ = wrapped[0]
        self.assertIn("hust_lebw", str(caught.exception))
