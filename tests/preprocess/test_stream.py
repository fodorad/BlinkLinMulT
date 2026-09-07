"""Tests for the single-pass recording sweep.

Two properties carry the design. **A frame is decoded once**, however many
overlapping windows reference it — that is what makes streaming cheaper than
the per-window extraction it replaces. And **the annotation stays attached to
the image it describes**, which is where the positional-versus-timestamp
choice matters: on TalkingFace the two clocks disagree (25 fps in the ``.tag``,
30 in the ``.avi`` container), and picking wrongly silently drops frames.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from blinklinmult.data.schema import EYE_FEATURE_DIM
from blinklinmult.preprocess.features import FakeExtractor, empty_feature
from blinklinmult.preprocess.geometry import LEFT, RIGHT
from blinklinmult.preprocess.stream import (
    FrameCache,
    StreamError,
    _is_contiguous,
    build_cache,
    decode_annotated_frames,
    frame_count,
)

IMAGE_SIZE = 8
FRAME_SIZE = 120


def write_video(path: Path, frames: int, fps: float = 25.0) -> Path:
    """Write a tiny video whose Nth frame is filled with the value N.

    Args:
        path (Path): Destination file.
        frames (int): How many frames to write.
        fps (float): Declared frame rate.

    Returns:
        Path: The written file.
    """
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (FRAME_SIZE, FRAME_SIZE)
    )
    for index in range(frames):
        # A flat grey whose level identifies the frame, so a mis-ordered or
        # dropped frame is visible in the pixels rather than only in the count.
        writer.write(np.full((FRAME_SIZE, FRAME_SIZE, 3), index + 1, dtype=np.uint8))
    writer.release()
    return path


class Record:
    """The fields the stream reads off an annotation."""

    def __init__(self, frame_id: int):
        self.frame_id = frame_id
        self.left_eye_corners = np.asarray([30, 50, 60, 50], dtype=np.int32)
        self.right_eye_corners = np.asarray([70, 50, 100, 50], dtype=np.int32)


class Tag:
    """A minimal stand-in for a parsed ``.tag`` file."""

    def __init__(self, frames: int):
        self.records = [Record(index) for index in range(frames)]


class VideoCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestFrameCount(VideoCase):
    def test_counts_what_actually_decodes(self):
        path = write_video(self.tmp / "v.avi", frames=12)
        self.assertEqual(frame_count(path), 12)

    def test_a_missing_video_raises(self):
        with self.assertRaises(StreamError):
            frame_count(self.tmp / "absent.avi")


class TestDecodeAnnotatedFrames(VideoCase):
    def timestamps(self, count: int, step: float) -> dict[int, float]:
        return {index: (index + 1) * step for index in range(count)}

    def test_matching_counts_map_positionally(self):
        path = write_video(self.tmp / "v.avi", frames=10)
        # A deliberately mismatched clock: 0.04s steps against a 25 fps video
        # is fine positionally, and is exactly TalkingFace's situation.
        pairs = list(decode_annotated_frames(path, self.timestamps(10, 0.04)))
        self.assertEqual(len(pairs), 10)
        self.assertEqual([frame_id for frame_id, _ in pairs], list(range(10)))

    def test_positional_mapping_loses_no_frame(self):
        # The bug this guards: nearest-timestamp matching against a container
        # whose declared fps differs from the annotation's collapsed 834 of
        # TalkingFace's 5000 frames onto shared ids.
        path = write_video(self.tmp / "v.avi", frames=50)
        pairs = list(decode_annotated_frames(path, self.timestamps(50, 0.04)))
        self.assertEqual(len({frame_id for frame_id, _ in pairs}), 50)

    def test_frames_arrive_in_order_and_intact(self):
        path = write_video(self.tmp / "v.avi", frames=6)
        pairs = list(decode_annotated_frames(path, self.timestamps(6, 0.04)))
        for expected, (frame_id, frame) in enumerate(pairs):
            self.assertEqual(frame_id, expected)
            self.assertEqual(int(frame[0, 0, 0]), expected + 1)

    def test_fewer_annotations_than_frames_falls_back_to_timestamps(self):
        # The clocks must agree for timestamp matching to mean anything, so the
        # annotation is timed at the video's own rate.
        path = write_video(self.tmp / "v.avi", frames=10, fps=25.0)
        pairs = list(decode_annotated_frames(path, self.timestamps(4, 0.04)))
        self.assertLessEqual(len(pairs), 4)


class TestClockAgreement(unittest.TestCase):
    """Timestamp matching assumes one clock; RN proved that assumption wrong.

    RN's containers declare 30 fps for footage annotated at ~16.7, and every
    recording fell into timestamp matching because its ``.txt`` carries six more
    entries than the video has frames. The labels then slid up to a second along
    the recording -- blink frames came out showing open eyes -- and nothing
    failed. These tests are the guard that would have caught it.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def timestamps(self, count: int, step: float) -> dict[int, float]:
        return {i: i * step for i in range(count)}

    def gapped(self, step: float) -> dict[int, float]:
        """A non-contiguous annotation, which cannot be mapped positionally."""
        return {i: i * step for i in (0, 2, 5, 9)}

    def test_a_disagreeing_clock_raises_rather_than_drifting(self):
        # 25 fps container, 12.5 fps annotation. The ids are gapped so
        # positional is unavailable and the timestamp path is genuinely
        # reached -- which is where the drift would happen.
        path = write_video(self.tmp / "v.avi", frames=10, fps=25.0)
        with self.assertRaises(StreamError) as error:
            list(decode_annotated_frames(path, self.gapped(0.08)))
        self.assertIn("fps", str(error.exception))

    def test_positional_skips_the_check_entirely(self):
        # Nothing is timed when frames map one to one, so a disagreeing
        # container rate is irrelevant.
        path = write_video(self.tmp / "v.avi", frames=10, fps=25.0)
        pairs = list(decode_annotated_frames(path, self.gapped(0.08), positional=True))
        self.assertEqual(len(pairs), 4)

    def test_agreeing_clocks_pass(self):
        # Gapped ids timed at the video's own 25 fps: frame i sits at i/25 s, so
        # the implied rate matches the container and matching on time is sound.
        path = write_video(self.tmp / "v.avi", frames=10, fps=25.0)
        timestamps = {i: i / 25.0 for i in (0, 2, 5, 9)}
        pairs = list(decode_annotated_frames(path, timestamps))
        self.assertLessEqual(len(pairs), 4)

    def test_a_contiguous_annotation_prefers_positional(self):
        # The RN case: the container's clock is wrong, but the annotation names
        # frames 0..N-1, so positional is available and exact. It must be chosen
        # rather than the run aborting on the bad clock.
        path = write_video(self.tmp / "v.avi", frames=10, fps=25.0)
        pairs = list(decode_annotated_frames(path, self.timestamps(4, 0.08)))
        self.assertEqual([frame_id for frame_id, _ in pairs], [0, 1, 2, 3])


class TestAnnotationAlignment(unittest.TestCase):
    """The annotation decides which frames are wanted, not the timestamp file.

    RN ships a ``.txt`` with exactly six timestamps more than its video has
    frames. Comparing against *that* count failed the positional test on all
    107 recordings and sent every one down the drifting timestamp path, while
    the ``.tag`` count matched the video exactly in 96 of them.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def timestamps(self, count: int, step: float) -> dict[int, float]:
        return {i: i * step for i in range(count)}

    def test_a_trailing_timestamp_tail_still_maps_positionally(self):
        path = write_video(self.tmp / "v.avi", frames=10, fps=25.0)
        # 16 timestamps, 10 annotated frames, 10 decoded: positional.
        timestamps = {i: i * 0.04 for i in range(16)}
        pairs = list(decode_annotated_frames(path, timestamps, annotated_ids=list(range(10))))
        self.assertEqual([frame_id for frame_id, _ in pairs], list(range(10)))

    def test_positional_attaches_frame_i_to_annotation_i(self):
        # write_video paints frame i with the value i+1, so the pixel proves
        # which decoded frame carries which id.
        path = write_video(self.tmp / "v.avi", frames=6, fps=25.0)
        timestamps = {i: i * 0.04 for i in range(12)}
        pairs = list(decode_annotated_frames(path, timestamps, annotated_ids=list(range(6))))
        for frame_id, frame in pairs:
            self.assertEqual(int(frame[0, 0, 0]), frame_id + 1)

    def test_the_strategy_can_be_forced(self):
        path = write_video(self.tmp / "v.avi", frames=10)
        pairs = list(decode_annotated_frames(path, self.timestamps(10, 0.04), positional=True))
        self.assertEqual(len(pairs), 10)

    def test_no_timestamps_raises(self):
        path = write_video(self.tmp / "v.avi", frames=4)
        with self.assertRaises(StreamError):
            list(decode_annotated_frames(path, {}))

    def test_a_missing_video_raises(self):
        with self.assertRaises(StreamError):
            list(decode_annotated_frames(self.tmp / "absent.avi", {0: 0.0}))


class TestFrameCache(unittest.TestCase):
    def cache(self, **overrides) -> FrameCache:
        defaults = {"image_size": IMAGE_SIZE, "feature_dim": EYE_FEATURE_DIM}
        return FrameCache(**{**defaults, **overrides})

    def crop(self) -> np.ndarray:
        return np.random.rand(3, IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)

    def test_window_images_stack_in_frame_order(self):
        cache = self.cache()
        for frame_id in range(3):
            cache.add(frame_id, LEFT, np.full((3, IMAGE_SIZE, IMAGE_SIZE), frame_id, np.float32))
        images, mask = cache.window_images([0, 1, 2], LEFT)
        self.assertEqual(images.shape, (3, 3, IMAGE_SIZE, IMAGE_SIZE))
        self.assertTrue(mask.all())
        self.assertEqual([float(v[0, 0, 0]) for v in images], [0.0, 1.0, 2.0])

    def test_a_missing_frame_is_blank_and_masked(self):
        # A frame whose eye could not be located must not silently become a
        # black eye the model trains on.
        cache = self.cache()
        cache.add(0, LEFT, self.crop())
        images, mask = cache.window_images([0, 1], LEFT)
        np.testing.assert_array_equal(mask, [True, False])
        self.assertTrue((images[1] == 0).all())

    def test_the_two_eyes_are_cached_independently(self):
        cache = self.cache()
        cache.add(0, LEFT, self.crop())
        _, left = cache.window_images([0], LEFT)
        _, right = cache.window_images([0], RIGHT)
        self.assertTrue(left.all())
        self.assertFalse(right.any())

    def test_window_features_stack_with_their_mask(self):
        cache = self.cache()
        cache.add(0, LEFT, self.crop(), (np.ones(EYE_FEATURE_DIM, np.float32), True))
        cache.add(1, LEFT, self.crop(), (empty_feature(), False))
        values, mask = cache.window_features([0, 1], LEFT)
        self.assertEqual(values.shape, (2, EYE_FEATURE_DIM))
        np.testing.assert_array_equal(mask, [True, False])

    def test_a_frame_with_no_descriptor_is_masked(self):
        cache = self.cache()
        cache.add(0, LEFT, self.crop())
        values, mask = cache.window_features([0], LEFT)
        self.assertFalse(mask.any())
        self.assertTrue((values == 0).all())

    def test_asking_a_feature_free_cache_for_features_raises(self):
        with self.assertRaises(StreamError):
            self.cache(feature_dim=None).window_features([0], LEFT)


class TestBuildCache(VideoCase):
    def test_caches_both_eyes_of_every_annotated_frame(self):
        path = write_video(self.tmp / "v.avi", frames=5)
        timestamps = {index: (index + 1) * 0.04 for index in range(5)}
        cache = build_cache(path, Tag(5), timestamps, IMAGE_SIZE, FakeExtractor())
        self.assertEqual(len(cache.crops), 10)
        self.assertEqual(len(cache.features), 10)

    def test_crops_are_resized_to_the_configured_size(self):
        path = write_video(self.tmp / "v.avi", frames=3)
        timestamps = {index: (index + 1) * 0.04 for index in range(3)}
        cache = build_cache(path, Tag(3), timestamps, IMAGE_SIZE, FakeExtractor())
        images, _ = cache.window_images([0], LEFT)
        self.assertEqual(images.shape[1:], (3, IMAGE_SIZE, IMAGE_SIZE))

    def test_each_frame_is_described_once(self):
        # The property the whole redesign rests on: overlapping windows must
        # not re-run the detector over the same pixels.
        path = write_video(self.tmp / "v.avi", frames=8)
        timestamps = {index: (index + 1) * 0.04 for index in range(8)}
        extractor = FakeExtractor()
        build_cache(path, Tag(8), timestamps, IMAGE_SIZE, extractor)
        self.assertEqual(extractor.calls, 8)

    def test_without_an_extractor_only_crops_are_cached(self):
        path = write_video(self.tmp / "v.avi", frames=4)
        timestamps = {index: (index + 1) * 0.04 for index in range(4)}
        cache = build_cache(path, Tag(4), timestamps, IMAGE_SIZE, None)
        self.assertEqual(len(cache.crops), 8)
        self.assertEqual(cache.features, {})
        self.assertIsNone(cache.feature_dim)

    def test_crops_are_normalised(self):
        path = write_video(self.tmp / "v.avi", frames=2)
        timestamps = {index: (index + 1) * 0.04 for index in range(2)}
        cache = build_cache(path, Tag(2), timestamps, IMAGE_SIZE, None)
        images, _ = cache.window_images([0], LEFT)
        self.assertLessEqual(float(images.max()), 1.0)
        self.assertGreaterEqual(float(images.min()), 0.0)


if __name__ == "__main__":
    unittest.main()


class TestWindowSignals(unittest.TestCase):
    """Assembling per-frame pose and quality across a window.

    A frame the decoder could not read must contribute a neutral value rather
    than break the window: the corpus records its own validity mask alongside,
    so a zero here is "unknown", not "measured as zero".
    """

    def _cache(self) -> FrameCache:
        """A cache holding one readable frame out of two."""
        cache = FrameCache(image_size=8)
        cache.crops[(0, LEFT)] = np.full((8, 8, 3), 128, dtype=np.uint8)
        cache.boxes[(0, LEFT)] = (10.0, 20.0, 30.0)
        cache.poses[0] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        return cache

    def test_pose_falls_back_to_zeros_for_a_missing_frame(self) -> None:
        """A dropped frame is frontal-by-default, which is the least-wrong guess."""
        poses = self._cache().window_pose([0, 1])
        np.testing.assert_allclose(poses[0], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(poses[1], [0.0, 0.0, 0.0])

    def test_pose_has_one_row_per_frame(self) -> None:
        """The window's shape must match its frame list exactly."""
        self.assertEqual(self._cache().window_pose([0, 1, 2]).shape, (3, 3))

    def test_quality_reports_every_signal(self) -> None:
        """Each is stored as its own corpus field, so all must be present."""
        signals = self._cache().window_quality([0, 1], LEFT)
        for name in ("eye_blur", "eye_exposure"):
            self.assertIn(name, signals)
            self.assertEqual(signals[name].shape, (2,))

    def test_quality_is_zero_where_the_crop_is_missing(self) -> None:
        """No crop means no measurement, and zero is the documented filler."""
        signals = self._cache().window_quality([0, 1], LEFT)
        self.assertEqual(float(signals["eye_blur"][1]), 0.0)
        self.assertEqual(float(signals["eye_exposure"][1]), 0.0)

    def test_a_readable_crop_produces_a_real_measurement(self) -> None:
        """Otherwise the fallback would be indistinguishable from a real read."""
        signals = self._cache().window_quality([0], LEFT)
        self.assertEqual(signals["eye_blur"].shape, (1,))


class TestContiguity(unittest.TestCase):
    """Whether a set of frame ids covers a recording without gaps."""

    def test_an_empty_set_is_not_contiguous(self) -> None:
        """Nothing cannot cover a recording, and must not divide by zero."""
        self.assertFalse(_is_contiguous(np.array([], dtype=np.int64)))

    def test_a_run_from_zero_is_contiguous(self) -> None:
        """The case a full decode produces."""
        self.assertTrue(_is_contiguous(np.arange(5)))

    def test_a_gap_breaks_contiguity(self) -> None:
        """A skipped frame means the annotation and the video disagree."""
        self.assertFalse(_is_contiguous(np.array([0, 1, 3, 4])))

    def test_not_starting_at_zero_breaks_contiguity(self) -> None:
        """An offset start means frame ids and decode order are not aligned."""
        self.assertFalse(_is_contiguous(np.array([1, 2, 3])))
