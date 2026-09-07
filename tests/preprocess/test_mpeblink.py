"""Tests for the MPEblink 2.0 reader.

Three properties carry this corpus, and each is something the annotation does
differently from every other corpus here:

* blinks arrive as ``[start, end, category]`` **segments**, not a per-frame
  vector, and the category is deliberately ignored;
* a tracklet's visible frames are **not contiguous** — people leave shot and
  return — so a window must never bridge an absence;
* eyes are **derived from WFLW landmarks**, since the corpus annotates faces.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from blinklinmult.data.schema import LEFT, NO_BLINK, RIGHT
from blinklinmult.preprocess.mpeblink import (
    EYE_LANDMARKS,
    IBUG_EYE_LANDMARKS,
    MIN_EYE_SPAN,
    PUPILS,
    SPLIT_DIRS,
    MPEblinkError,
    Tracklet,
    eye_box,
    number_events,
    rasterise,
    read_tracklets,
    runs,
    sweep,
    window_blink_id,
)


def landmarks(eye_span: float = 30.0, centre: tuple[float, float] = (100.0, 100.0)):
    """A WFLW point set whose eyes have a known span, pupils at their middle."""
    points = np.zeros((98, 2), dtype=np.float64)
    for side, offset in ((LEFT, -40.0), (RIGHT, 40.0)):
        start, stop = EYE_LANDMARKS[side]
        xs = np.linspace(centre[0] + offset, centre[0] + offset + eye_span, stop - start)
        points[start:stop, 0] = xs
        points[start:stop, 1] = centre[1]
        points[PUPILS[side]] = points[start:stop].mean(axis=0)
    return points


class TestRasterise(unittest.TestCase):
    def test_a_segment_becomes_a_run_of_ones(self):
        labels = rasterise([[2, 4, 0]], length=8)
        np.testing.assert_array_equal(labels, [0, 0, 1, 1, 1, 0, 0, 0])

    def test_both_endpoints_are_inclusive(self):
        # The annotation names the first and last blinking frame, so [2, 4]
        # covers three frames, not two.
        self.assertEqual(int(rasterise([[2, 4, 0]], length=8).sum()), 3)

    def test_the_category_is_ignored(self):
        # The authors' own converter reads elements [0] and [1] only, so all
        # 17 711 events count equally whatever their third field says.
        for category in (0, 1, 2, 10):
            with self.subTest(category=category):
                labels = rasterise([[1, 2, category]], length=5)
                self.assertEqual(int(labels.sum()), 2)

    def test_overlapping_segments_merge(self):
        # 86 of the corpus's segments overlap a neighbour; rasterising is
        # idempotent so they cost nothing.
        labels = rasterise([[1, 3, 0], [2, 4, 0]], length=6)
        np.testing.assert_array_equal(labels, [0, 1, 1, 1, 1, 0])

    def test_a_segment_past_the_end_is_clipped(self):
        labels = rasterise([[6, 99, 0]], length=8)
        self.assertEqual(int(labels.sum()), 2)

    def test_a_null_segment_is_skipped(self):
        # 11 entries in the corpus carry a null category and some are malformed;
        # one bad row must not abort a 921-video build.
        self.assertEqual(int(rasterise([[None, None, 0], [1, 2, 0]], length=5).sum()), 2)

    def test_no_segments_is_all_zero(self):
        self.assertEqual(int(rasterise([], length=5).sum()), 0)


class TestEyeBox(unittest.TestCase):
    def test_the_box_is_centred_on_the_eye(self):
        box = eye_box(landmarks(centre=(200.0, 150.0)), LEFT)
        self.assertIsNotNone(box)
        self.assertAlmostEqual(box.centre_y, 150, delta=2)

    def test_the_side_scales_with_the_eye(self):
        small = eye_box(landmarks(eye_span=20.0), LEFT)
        large = eye_box(landmarks(eye_span=60.0), LEFT)
        self.assertLess(small.side, large.side)

    def test_the_two_eyes_are_apart(self):
        points = landmarks()
        left, right = eye_box(points, LEFT), eye_box(points, RIGHT)
        self.assertNotEqual(left.centre_x, right.centre_x)

    def test_the_centre_is_the_pupil(self):
        # The eyelid contour moves as the lid closes; the pupil does not, so a
        # crop centred on it stays still through the blink it is meant to show.
        points = landmarks()
        # Inside the contour, but off its mean -- which is what a real pupil is.
        points[PUPILS[LEFT]] = points[PUPILS[LEFT]] + np.asarray([5.0, 0.0])
        box = eye_box(points, LEFT)
        self.assertEqual(box.centre_x, int(round(points[PUPILS[LEFT]][0])))
        self.assertEqual(box.centre_y, int(round(points[PUPILS[LEFT]][1])))

    def test_a_pupil_outside_its_contour_falls_back(self):
        # 85 of 104 168 pupils land outside their own eyelid contour: a bad
        # point, not a bad eye, so the contour still says where the eye is.
        points = landmarks()
        points[PUPILS[LEFT]] = np.asarray([9000.0, 9000.0])
        box = eye_box(points, LEFT)
        start, stop = EYE_LANDMARKS[LEFT]
        expected = points[start:stop].mean(axis=0)
        self.assertEqual(box.centre_x, int(round(expected[0])))

    def test_the_side_is_measured_corner_to_corner(self):
        # The same measure every other corpus uses. Taking the contour's
        # bounding-box diagonal instead would fold in the eyelid's vertical
        # travel, so the crop would shrink as the eye closes.
        points = landmarks(eye_span=30.0)
        # Push a mid-contour point far above the corners: a diagonal-based
        # side would grow, a corner-based one must not.
        start, stop = EYE_LANDMARKS[LEFT]
        flat = eye_box(points, LEFT).side
        points[start + 2, 1] -= 25.0
        self.assertEqual(eye_box(points, LEFT).side, flat)

    def test_the_size_still_comes_from_the_contour(self):
        # Only the centre moved to the pupil; the side must still scale with
        # the face, or crops stop being comparable across corpora.
        small = eye_box(landmarks(eye_span=20.0), LEFT)
        large = eye_box(landmarks(eye_span=60.0), LEFT)
        self.assertLess(small.side, large.side)

    def test_a_tiny_eye_is_refused(self):
        # Films put faces at every scale; a distant face leaves an eye a few
        # pixels wide, where a crop would be interpolation noise.
        self.assertIsNone(eye_box(landmarks(eye_span=MIN_EYE_SPAN / 2), LEFT))


def ibug(eye_span: float = 30.0, centre: tuple[float, float] = (100.0, 100.0), columns: int = 3):
    """A 68-point iBUG set, as the corpus's long test clips ship it."""
    points = np.zeros((68, columns), dtype=np.float64)
    for side, offset in ((LEFT, -40.0), (RIGHT, 40.0)):
        start, stop = IBUG_EYE_LANDMARKS[side]
        points[start:stop, 0] = np.linspace(
            centre[0] + offset, centre[0] + offset + eye_span, stop - start
        )
        points[start:stop, 1] = centre[1]
    return points


class TestIbugFallback(unittest.TestCase):
    """30 of 212 test clips carry only the 68-point scheme."""

    def test_a_68_point_set_still_yields_a_box(self):
        box = eye_box(ibug(), LEFT)
        self.assertIsNotNone(box)

    def test_the_third_column_is_ignored(self):
        # The field is (68, 3); the extra column is not a coordinate, and
        # treating it as one would move every centre.
        wide, flat = eye_box(ibug(columns=3), LEFT), eye_box(ibug(columns=2), LEFT)
        self.assertEqual((wide.centre_x, wide.centre_y), (flat.centre_x, flat.centre_y))

    def test_the_two_eyes_are_apart(self):
        points = ibug()
        self.assertNotEqual(eye_box(points, LEFT).centre_x, eye_box(points, RIGHT).centre_x)

    def test_the_side_scales_with_the_eye(self):
        small, large = eye_box(ibug(eye_span=20.0), LEFT), eye_box(ibug(eye_span=60.0), LEFT)
        self.assertLess(small.side, large.side)

    def test_it_agrees_with_wflw_on_the_same_eye(self):
        # Both schemes describe the same eye, so the crop must not depend on
        # which one a clip happens to ship.
        wflw, other = eye_box(landmarks(eye_span=30.0), LEFT), eye_box(ibug(eye_span=30.0), LEFT)
        self.assertEqual(wflw.side, other.side)
        self.assertAlmostEqual(wflw.centre_x, other.centre_x, delta=1)

    def test_an_unknown_point_count_is_refused(self):
        self.assertIsNone(eye_box(np.zeros((5, 2)), LEFT))


class TestFrameEdge(unittest.TestCase):
    """An overhanging box is not an absent eye.

    The crop carries twice the eye's width in context, so it leaves the frame
    long before the eye does: over 244 000 crops, 1.06% of boxes overflow but
    only 0.47% hold an eye genuinely off-screen. Gating on the box would
    discard 1 423 evaluable eyes.
    """

    def test_a_visible_eye_near_the_edge_is_kept(self):
        # The eye sits inside the frame; only its context box overhangs.
        points = landmarks(eye_span=30.0, centre=(35.0, 100.0))
        self.assertIsNotNone(eye_box(points, RIGHT, frame_shape=(200, 200)))

    def test_that_box_is_slid_fully_into_the_frame(self):
        points = landmarks(eye_span=30.0, centre=(35.0, 100.0))
        box = eye_box(points, RIGHT, frame_shape=(200, 200))
        self.assertEqual(box.visible_fraction(200, 200), 1.0)

    def test_sliding_preserves_the_scale(self):
        # The whole point of shifting rather than clipping.
        points = landmarks(eye_span=30.0, centre=(35.0, 100.0))
        free = eye_box(points, RIGHT)
        edged = eye_box(points, RIGHT, frame_shape=(200, 200))
        self.assertEqual(free.side, edged.side)

    def test_an_eye_off_the_frame_is_refused(self):
        # MPEblink annotates heads tracked past the frame edge, so this is real:
        # a black patch carrying a blink label is worse than no patch.
        points = landmarks(eye_span=30.0, centre=(100.0, -400.0))
        self.assertIsNone(eye_box(points, LEFT, frame_shape=(200, 200)))

    def test_without_a_frame_shape_nothing_is_gated(self):
        points = landmarks(eye_span=30.0, centre=(100.0, -400.0))
        self.assertIsNotNone(eye_box(points, LEFT))

    def test_a_mid_frame_eye_is_untouched(self):
        points = landmarks(eye_span=30.0, centre=(300.0, 300.0))
        free = eye_box(points, LEFT)
        framed = eye_box(points, LEFT, frame_shape=(600, 600))
        self.assertEqual((free.centre_x, free.centre_y), (framed.centre_x, framed.centre_y))


class TestRuns(unittest.TestCase):
    def test_one_unbroken_run(self):
        self.assertEqual(list(runs(np.asarray([1, 1, 1], dtype=bool))), [(0, 3)])

    def test_a_gap_splits_the_run(self):
        mask = np.asarray([1, 1, 0, 0, 1, 1, 1], dtype=bool)
        self.assertEqual(list(runs(mask)), [(0, 2), (4, 7)])

    def test_a_run_reaching_the_end_is_closed(self):
        self.assertEqual(list(runs(np.asarray([0, 1, 1], dtype=bool))), [(1, 3)])

    def test_an_empty_mask_yields_nothing(self):
        self.assertEqual(list(runs(np.zeros(5, dtype=bool))), [])


class TestSweep(unittest.TestCase):
    def test_windows_step_by_the_stride(self):
        starts = sweep(np.ones(20, dtype=bool), window=8, stride=4)
        self.assertEqual(starts, [0, 4, 8, 12])

    def test_no_window_bridges_an_absence(self):
        # The property that matters: 281 of 1 247 sampled tracklets have gaps,
        # and a window spanning one would splice together frames seconds apart.
        mask = np.asarray([1] * 10 + [0] * 5 + [1] * 10, dtype=bool)
        for start in sweep(mask, window=6, stride=3):
            self.assertTrue(mask[start : start + 6].all())

    def test_a_run_shorter_than_the_window_is_skipped(self):
        mask = np.asarray([1] * 4 + [0] + [1] * 12, dtype=bool)
        self.assertTrue(all(start >= 5 for start in sweep(mask, window=6, stride=3)))

    def test_a_fully_invisible_tracklet_yields_nothing(self):
        self.assertEqual(sweep(np.zeros(30, dtype=bool), window=8, stride=4), [])


class TestReadTracklets(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, payload: dict) -> Path:
        path = self.tmp / "annotation_WFLW.json"
        path.write_text(json.dumps(payload))
        return path

    def payload(self, people: int = 2, length: int = 6) -> dict:
        data = {"length": length, "width": 1920, "height": 1080}
        for index in range(people):
            data[f"person{index}"] = {
                "bbox": [[0, 0, 10, 10]] * length,
                "landmark_WFLW": [landmarks().tolist()] * length,
                "blink": [[1, 2, 0]],
            }
        return data

    def test_every_person_becomes_a_tracklet(self):
        tracklets = read_tracklets(self.write(self.payload(people=3)), "test_1")
        self.assertEqual(len(tracklets), 3)

    def test_the_sample_prefix_separates_people(self):
        tracklets = read_tracklets(self.write(self.payload(people=2)), "test_1")
        prefixes = {t.sample_prefix for t in tracklets}
        self.assertEqual(len(prefixes), 2)
        self.assertTrue(all(p.startswith("test_1-") for p in prefixes))

    def test_the_blink_vector_matches_the_video_length(self):
        tracklet = read_tracklets(self.write(self.payload(length=9)), "v")[0]
        self.assertEqual(len(tracklet.blinks), 9)

    def test_non_person_keys_are_ignored(self):
        payload = self.payload(people=1)
        payload["width"] = 1920
        self.assertEqual(len(read_tracklets(self.write(payload), "v")), 1)

    def test_a_clip_without_wflw_falls_back_to_the_68_point_field(self):
        # test/183..212 -- 30 clips, thousands of annotated blinks -- ship only
        # `landmark`. Reading WFLW alone would drop 14% of the test split.
        payload = self.payload(people=1, length=4)
        payload["person0"]["landmark"] = payload["person0"].pop("landmark_WFLW")
        tracklet = read_tracklets(self.write(payload), "v")[0]
        self.assertTrue(tracklet.visible().all())

    def test_wflw_wins_when_both_are_present(self):
        payload = self.payload(people=1, length=4)
        payload["person0"]["landmark"] = [np.zeros((68, 3)).tolist()] * 4
        tracklet = read_tracklets(self.write(payload), "v")[0]
        self.assertEqual(len(tracklet.landmarks[0]), 98)

    def test_a_zero_length_video_raises(self):
        with self.assertRaises(MPEblinkError):
            read_tracklets(self.write({"length": 0}), "v")

    def test_unreadable_json_raises(self):
        path = self.tmp / "annotation_WFLW.json"
        path.write_text("{not json")
        with self.assertRaises(MPEblinkError):
            read_tracklets(path, "v")


class TestVisibility(unittest.TestCase):
    def tracklet(self, boxes, marks) -> Tracklet:
        return Tracklet(
            video_id="v",
            person="person0",
            boxes=boxes,
            landmarks=marks,
            blinks=np.zeros(len(boxes), dtype=np.float32),
            length=len(boxes),
        )

    def test_a_frame_needs_both_a_box_and_landmarks(self):
        points = landmarks().tolist()
        visible = self.tracklet(
            [[0, 0, 1, 1], None, [0, 0, 1, 1]], [points, points, None]
        ).visible()
        np.testing.assert_array_equal(visible, [True, False, False])


class TestSplits(unittest.TestCase):
    def test_the_corpus_split_is_honoured(self):
        # 540/169/212 as shipped; re-deriving it would make the numbers
        # incomparable with the authors'.
        self.assertEqual(SPLIT_DIRS, {"train": "train", "val": "valid", "test": "test"})


if __name__ == "__main__":
    unittest.main()


class TestNumberEvents(unittest.TestCase):
    """Blinks must be individuable, not just detectable.

    `rasterise` answers "is this frame inside a blink"; `number_events` answers
    "which blink". Without it MPEblink carries no usable `blink_id`, so a double
    blink cannot be told from one long closure and no event count can be
    supervised -- on the corpus that is 80% of the benchmark.
    """

    def test_each_segment_gets_its_own_id(self):
        ids = number_events([[1, 2], [5, 6]], 8)
        self.assertEqual(ids[1], 0)
        self.assertEqual(ids[2], 0)
        self.assertEqual(ids[5], 1)
        self.assertEqual(ids[6], 1)

    def test_frames_outside_every_event_are_marked(self):
        ids = number_events([[1, 2]], 5)
        self.assertEqual(ids[0], NO_BLINK)
        self.assertEqual(ids[3], NO_BLINK)
        self.assertEqual(ids[4], NO_BLINK)

    def test_adjacent_events_stay_distinct(self):
        # The whole point: two back-to-back blinks must not merge into one.
        ids = number_events([[1, 2], [3, 4]], 6)
        self.assertNotEqual(ids[2], ids[3])

    def test_it_agrees_with_the_rasterised_label(self):
        # A frame has an id exactly where it is annotated as blinking; the id
        # refines the label rather than disagreeing with it.
        segments = [[1, 3], [6, 7]]
        labels = rasterise(segments, 10)
        ids = number_events(segments, 10)
        self.assertTrue(np.array_equal(labels > 0.5, ids != NO_BLINK))

    def test_it_agrees_when_segments_overlap(self):
        # 86 pairs in the corpus overlap. Both functions resolve them the same
        # way, so the id stays a refinement of the label.
        segments = [[1, 4], [3, 6]]
        labels = rasterise(segments, 8)
        ids = number_events(segments, 8)
        self.assertTrue(np.array_equal(labels > 0.5, ids != NO_BLINK))

    def test_segments_out_of_range_are_clipped(self):
        ids = number_events([[-2, 1], [6, 99]], 8)
        self.assertEqual(ids[0], 0)
        self.assertEqual(ids[7], 1)

    def test_a_malformed_segment_is_skipped(self):
        ids = number_events([[1], [None, 3], [5, 6]], 8)
        self.assertTrue(np.all(ids[:5] == NO_BLINK))
        self.assertEqual(ids[5], 0)

    def test_no_segments_yields_no_ids(self):
        self.assertTrue(np.all(number_events([], 5) == NO_BLINK))


class TestWindowBlinkId(unittest.TestCase):
    """A window spanning two events must pick one, stably."""

    def test_the_first_event_wins(self):
        # Stability as the window slides: majority or last would flip the id
        # mid-blink and split one event across two identities.
        self.assertEqual(window_blink_id(np.asarray([0, 0, 1, 1, 1], dtype=np.int32)), 0)

    def test_a_blink_free_window_has_no_id(self):
        self.assertEqual(window_blink_id(np.full(4, NO_BLINK, dtype=np.int32)), NO_BLINK)

    def test_leading_background_is_skipped(self):
        self.assertEqual(window_blink_id(np.asarray([NO_BLINK, NO_BLINK, 3, 3], dtype=np.int32)), 3)

    def test_an_empty_window_has_no_id(self):
        self.assertEqual(window_blink_id(np.zeros(0, dtype=np.int32)), NO_BLINK)
