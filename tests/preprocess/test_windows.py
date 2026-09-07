"""Tests for fixed-length window cutting.

There is one sampling policy — :func:`sliding_windows` — and every split uses
it. The blink-centred, class-balanced training sampler 1.x used was removed
along with its tests; see :mod:`blinklinmult.preprocess.windows` for why.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.data.schema import LEFT, RIGHT
from blinklinmult.preprocess.annotation import NO_BLINK, TagFile, TagRecord
from blinklinmult.preprocess.windows import (
    WindowError,
    sliding_windows,
    window_frames,
)


def record(frame_id: int, blink_id: int = NO_BLINK, closed: bool = False) -> TagRecord:
    """Build one record without going through the file parser."""
    return TagRecord(
        frame_id=frame_id,
        blink_id=blink_id,
        non_frontal_face=False,
        left_fully_closed=closed,
        left_not_visible=False,
        right_fully_closed=closed,
        right_not_visible=False,
        face_xywh=np.zeros(4, dtype=np.int32),
        left_eye_corners=np.zeros(4, dtype=np.int32),
        right_eye_corners=np.zeros(4, dtype=np.int32),
    )


def tag_from_ids(blink_ids: list[int], video_id: str = "vid") -> TagFile:
    """Build a TagFile from a list of per-frame blink ids."""
    return TagFile(
        [
            record(index, blink_id=bid, closed=bid != NO_BLINK)
            for index, bid in enumerate(blink_ids)
        ],
        video_id,
    )


def open_tag(length: int) -> TagFile:
    """A recording with no blinks at all."""
    return tag_from_ids([NO_BLINK] * length)


def blink_at_ten(video_id: str = "vid") -> TagFile:
    """A 20-frame recording with one blink at frame 10."""
    ids = [NO_BLINK] * 20
    ids[10] = 1
    return tag_from_ids(ids, video_id)


class TestFrameGroup(unittest.TestCase):
    def test_group_is_the_first_frame_id(self):
        window = sliding_windows(blink_at_ten("rn30_1"), window=6)[1]
        self.assertEqual(window.frame_group, f"{int(window.frame_ids[0]):06d}")

    def test_group_is_stable_across_rebuilds(self):
        first = sliding_windows(blink_at_ten("v"), window=6)[1]
        second = sliding_windows(blink_at_ten("v"), window=6)[1]
        self.assertEqual(first.frame_group, second.frame_group)


class TestPerEyeAccessors(unittest.TestCase):
    def window(self):
        return sliding_windows(blink_at_ten(), window=6)[1]

    def test_eye_state_is_a_scalar_sequence_per_side(self):
        window = self.window()
        for side in (LEFT, RIGHT):
            with self.subTest(side=side):
                self.assertEqual(window.eye_state_for(side).shape, (6,))

    def test_the_two_sides_read_different_columns(self):
        window = self.window()
        # The fixture closes both eyes on a blink frame, so the columns agree in
        # value; what matters is that each side reads its own column.
        np.testing.assert_array_equal(window.eye_state_for(LEFT), window.eye_state[:, 0])
        np.testing.assert_array_equal(window.eye_state_for(RIGHT), window.eye_state[:, 1])

    def test_validity_is_per_side(self):
        window = self.window()
        np.testing.assert_array_equal(window.validity_for(LEFT), window.validity[:, 0])

    def test_an_unknown_side_raises(self):
        with self.assertRaises(WindowError):
            self.window().eye_state_for("unknown")


class TestWindowFrames(unittest.TestCase):
    def test_derives_frames_from_the_rate(self):
        self.assertEqual(window_frames(30.0, 0.5), 15)
        self.assertEqual(window_frames(15.0, 0.5), 8)

    def test_rounds_half_up(self):
        # Banker's rounding would give 12 and silently shorten the window.
        self.assertEqual(window_frames(25.0, 0.5), 13)

    def test_never_returns_zero(self):
        self.assertEqual(window_frames(30.0, 0.001), 1)

    def test_non_positive_arguments_raise(self):
        with self.assertRaises(WindowError):
            window_frames(0.0, 0.5)
        with self.assertRaises(WindowError):
            window_frames(30.0, 0.0)


class TestSlidingWindows(unittest.TestCase):
    def test_non_overlapping_by_default(self):
        windows = sliding_windows(open_tag(30), window=10)
        self.assertEqual([w.start_index for w in windows], [0, 10, 20])

    def test_explicit_stride(self):
        windows = sliding_windows(open_tag(20), window=10, stride=5)
        self.assertEqual([w.start_index for w in windows], [0, 5, 10])

    def test_stride_of_one_covers_every_position(self):
        windows = sliding_windows(open_tag(15), window=10, stride=1)
        self.assertEqual(len(windows), 6)

    def test_trailing_partial_window_is_dropped(self):
        # 33 frames at window 10 gives 3 full windows; frames 30-32 cannot fill
        # a fourth and are dropped rather than zero-padded.
        windows = sliding_windows(open_tag(33), window=10)
        self.assertEqual(len(windows), 3)
        self.assertEqual(int(windows[-1].frame_ids[-1]), 29)

    def test_window_equal_to_recording_yields_one(self):
        self.assertEqual(len(sliding_windows(open_tag(10), window=10)), 1)

    def test_zero_stride_raises(self):
        with self.assertRaises(WindowError):
            sliding_windows(open_tag(30), window=10, stride=0)

    def test_window_longer_than_recording_raises(self):
        with self.assertRaises(WindowError):
            sliding_windows(open_tag(5), window=10)

    def test_labels_track_the_source_annotation(self):
        ids = [NO_BLINK] * 20
        ids[3] = ids[4] = 1
        windows = sliding_windows(tag_from_ids(ids), window=10)
        self.assertTrue(windows[0].has_blink)
        self.assertFalse(windows[1].has_blink)


class TestTheSweepIsTheOnlyPolicy(unittest.TestCase):
    """The properties that removing the 1.x training sampler bought.

    Both are about the *distribution* the model is fit on rather than about any
    single window, so neither is visible from the functions above.
    """

    def blinking(self, length: int = 300, period: int = 60) -> TagFile:
        """A recording that blinks briefly and regularly, as a real one does."""
        ids = [NO_BLINK] * length
        for blink, start in enumerate(range(20, length - 10, period), start=1):
            for offset in range(4):
                ids[start + offset] = blink
        return tag_from_ids(ids)

    def test_the_natural_class_prior_survives(self):
        # The point of the change. 1.x drew one window per blink plus an equal
        # number of blink-free ones, fixing training at 50% positive while the
        # swept evaluation splits sat nearer 10%. A sweep reports whatever the
        # recording actually contains.
        windows = sliding_windows(self.blinking(), window=15, stride=7)
        positive = sum(w.has_blink for w in windows) / len(windows)
        self.assertLess(positive, 0.5)

    def test_blinks_appear_at_every_offset_in_the_window(self):
        # 1.x centred every training window on its blink, so the model never saw
        # a closure cut by a window edge -- which is most of what a deployed
        # sweep produces. Here the same blink lands at many offsets.
        windows = sliding_windows(self.blinking(), window=15, stride=1)
        offsets = {int(np.flatnonzero(w.labels > 0.5)[0]) for w in windows if w.has_blink}
        self.assertGreater(len(offsets), 1)

    def test_the_sweep_is_deterministic(self):
        # No RNG anywhere in the path: two builds of one corpus agree exactly,
        # where the old negative draw depended on a seed.
        tag = self.blinking()
        first = sliding_windows(tag, window=15, stride=7)
        second = sliding_windows(tag, window=15, stride=7)
        self.assertEqual([w.start_index for w in first], [w.start_index for w in second])

    def test_every_annotated_frame_is_reachable(self):
        # A 50% overlap covers the recording, so no blink is skipped merely
        # because it fell between two windows.
        tag = self.blinking()
        covered = {int(f) for w in sliding_windows(tag, window=15, stride=7) for f in w.frame_ids}
        blinking_frames = {
            int(record.frame_id) for record in tag.records if record.blink_id != NO_BLINK
        }
        self.assertTrue(blinking_frames <= covered)


if __name__ == "__main__":
    unittest.main()


class TestWindowGuards(unittest.TestCase):
    """The refusals, which are what keep a corpus build honest."""

    def _tag(self, length: int = 10) -> TagFile:
        """A tag file of plain open-eye frames.

        Args:
            length (int): Frames to generate.

        Returns:
            TagFile: The annotation.
        """
        return tag_from_ids([NO_BLINK] * length)

    def test_a_window_longer_than_the_recording_is_refused(self) -> None:
        """Padding to length would invent frames the camera never saw."""
        with self.assertRaises(WindowError):
            list(sliding_windows(self._tag(5), window=10))

    def test_a_non_positive_window_is_refused(self) -> None:
        """Zero frames is not a window, and a negative one is a caller bug."""
        for length in (0, -1):
            with self.subTest(window=length), self.assertRaises(WindowError):
                list(sliding_windows(self._tag(), window=length))

    def test_an_unknown_eye_side_is_refused(self) -> None:
        """Validity is annotated per side, so a typo must not silently pick one.

        Returning the left eye's validity for an unrecognised label would mask
        the wrong eye and be invisible in the resulting corpus.
        """
        window = next(iter(sliding_windows(self._tag(), window=4)))
        with self.assertRaises(WindowError):
            window.validity_for("middle")

    def test_a_window_spanning_two_blinks_is_refused(self) -> None:
        """Every annotated event must be countable in exactly one sample.

        A window holding two blinks would be labelled with one of them, so the
        other vanishes from the count -- a silent recall loss.
        """
        tag = tag_from_ids([1, 1, 2])
        window = next(iter(sliding_windows(tag, window=3)))
        with self.assertRaises(WindowError):
            _ = window.blink_id

    def test_first_blink_id_tolerates_two_blinks(self) -> None:
        """The evaluation sweep expects overlap, so it takes the earlier one."""
        tag = tag_from_ids([1, 1, 2])
        window = next(iter(sliding_windows(tag, window=3)))
        self.assertEqual(window.first_blink_id, 1)
