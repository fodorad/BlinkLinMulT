"""Tests for the sampling protocol's guarantees.

**One policy for every split**: an annotation-blind, 50%-overlapping sweep. What
that buys, and what it gives up, is the subject of this module.

1.x sampled training separately — one window centred on each blink plus an equal
number of blink-free ones, with a guard band keeping negatives clear of blink
motion. That bought a strict "one blink per window" property and a balanced
training set, and it cost two things that matter more:

* the model was fit at a **50% blink prior** and evaluated near 10%, because a
  swept recording is overwhelmingly blink-free;
* every training blink sat at the **window's centre**, so the model never saw a
  closure cut by a window edge — most of what a deployed sweep produces.

The sweep gives up the one-blink-per-window guarantee: two blinks half a second
apart genuinely land in one window (9 of 713 on TalkingFace), which is natural
double-blinking rather than an annotation fault. ``first_blink_id`` records the
earlier event for provenance instead of refusing the sample.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.data.schema import LEFT, RIGHT
from blinklinmult.preprocess.annotation import NO_BLINK, TagFile, TagRecord
from blinklinmult.preprocess.windows import sliding_windows


def record(frame_id: int, blink_id: int = NO_BLINK, visible: bool = True) -> TagRecord:
    """One annotated frame."""
    return TagRecord(
        frame_id=frame_id,
        blink_id=blink_id,
        non_frontal_face=False,
        left_fully_closed=blink_id != NO_BLINK,
        left_not_visible=not visible,
        right_fully_closed=blink_id != NO_BLINK,
        right_not_visible=not visible,
        face_xywh=np.zeros(4, dtype=np.int32),
        left_eye_corners=np.zeros(4, dtype=np.int32),
        right_eye_corners=np.zeros(4, dtype=np.int32),
    )


def tag_from_ids(blink_ids: list[int], video_id: str = "vid") -> TagFile:
    """A recording whose per-frame blink ids are given."""
    return TagFile([record(i, bid) for i, bid in enumerate(blink_ids)], video_id)


def recording_with_blinks(length: int, blinks: dict[int, tuple[int, int]]) -> TagFile:
    """A recording with blinks at the given ``id -> (start, stop)`` spans."""
    ids = [NO_BLINK] * length
    for blink_id, (start, stop) in blinks.items():
        for i in range(start, stop):
            ids[i] = blink_id
    return tag_from_ids(ids)


def tag_window(tag: TagFile, start: int, length: int):
    """Cut one window by index, for tests about a window rather than a policy."""
    return sliding_windows(tag, window=length, stride=max(1, length))[start]


class TestEveryBlinkIsReachable(unittest.TestCase):
    """A sweep covers the recording, so no blink is lost between windows.

    This replaces 1.x's "exactly one window per blink". The count is no longer
    one, but every event still reaches the model, which is what the claim
    "the test set contains N blinks" actually needs.
    """

    def test_every_blinking_frame_lands_in_some_window(self):
        tag = recording_with_blinks(120, {1: (10, 14), 2: (50, 55), 3: (100, 104)})
        covered = {
            int(frame)
            for window in sliding_windows(tag, window=15, stride=7)
            for frame in window.frame_ids
        }
        blinking = {int(r.frame_id) for r in tag.records if r.blink_id != NO_BLINK}
        self.assertTrue(blinking <= covered)

    def test_every_blink_appears_in_at_least_one_window(self):
        tag = recording_with_blinks(120, {1: (10, 14), 2: (50, 55), 3: (100, 104)})
        seen = {
            window.first_blink_id
            for window in sliding_windows(tag, window=15, stride=7)
            if window.has_blink
        }
        self.assertEqual(seen, {1, 2, 3})

    def test_a_blink_near_the_end_is_still_covered(self):
        # The trailing partial window is dropped, so a blink in the last frames
        # is the case most at risk of being silently lost.
        tag = recording_with_blinks(60, {1: (44, 48)})
        windows = [w for w in sliding_windows(tag, window=15, stride=7) if w.has_blink]
        self.assertTrue(windows)


class TestSweepBlinkId(unittest.TestCase):
    """``first_blink_id`` keeps a sample attributable without refusing it."""

    def test_the_first_blink_is_reported_when_a_window_spans_two(self):
        tag = recording_with_blinks(40, {1: (10, 12), 2: (14, 16)})
        window = tag_window(tag, start=0, length=20)
        self.assertEqual(window.first_blink_id, 1)

    def test_a_blink_free_window_reports_no_blink(self):
        tag = recording_with_blinks(120, {1: (50, 54)})
        window = tag_window(tag, start=0, length=15)
        self.assertEqual(window.first_blink_id, NO_BLINK)

    def test_it_never_raises(self):
        # The whole point: a sweep must not abort three quarters of the way
        # through a multi-hour build because two blinks were close together.
        tag = recording_with_blinks(60, {1: (10, 12), 2: (14, 16), 3: (18, 20)})
        for window in sliding_windows(tag, window=15, stride=3):
            self.assertIsInstance(window.first_blink_id, int)


class TestTargetValidity(unittest.TestCase):
    def test_an_invisible_eye_is_marked_invalid(self):
        # The annotator could not see it, so the frame carries no usable
        # eye-state label and must not be trained against.
        records = [record(i) for i in range(20)]
        records[5] = record(5, visible=False)
        tag = TagFile(records, "vid")

        window = tag_window(tag, start=0, length=15)
        self.assertFalse(window.validity_for(LEFT)[5])
        self.assertTrue(window.validity_for(LEFT)[4])

    def test_validity_is_tracked_per_eye(self):
        records = [record(i) for i in range(20)]
        tag = TagFile(records, "vid")
        window = tag_window(tag, start=0, length=15)
        self.assertEqual(window.validity_for(LEFT).shape, (15,))
        self.assertEqual(window.validity_for(RIGHT).shape, (15,))


if __name__ == "__main__":
    unittest.main()
