"""Tests for event-level scoring.

The protocol is the literature's, so the tests pin the properties papers rely
on: a frame with no prediction is not a confident negative, two detections of
one blink are not two hits, and the four matching criteria genuinely disagree.
A test suite that cannot tell ``any`` from ``iou75`` is not testing them.
"""

from __future__ import annotations

import unittest

import numpy as np

from blinklinmult.train.events import (
    BLINK_AP_TIOU,
    CRITERIA,
    EventError,
    annotated_intervals,
    average_overlapping,
    average_precision,
    blink_ap,
    blink_ap_summary,
    event_metrics,
    froc,
    interpolated_average_precision,
    match,
    matches,
    pool_curves,
    segment_iou,
    temporal_iou,
    to_exclusive,
    to_intervals,
)


def signal_of(values: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """A fully covered signal."""
    array = np.asarray(values, dtype=np.float64)
    return array, np.ones_like(array, dtype=bool)


class TestAverageOverlapping(unittest.TestCase):
    def test_a_frame_under_several_windows_gets_their_mean(self):
        signal, mask = average_overlapping(
            probabilities=[0.2, 0.8, 0.5], frame_ids=[0, 0, 0], n_frames=1
        )
        self.assertAlmostEqual(float(signal[0]), 0.5)
        self.assertTrue(mask[0])

    def test_an_uncovered_frame_is_masked_not_zero(self):
        # A zero is a confident "no blink"; an uncovered frame is a different
        # claim, and merging the two would invent negatives.
        signal, mask = average_overlapping([0.9], [0], n_frames=3)
        np.testing.assert_array_equal(mask, [True, False, False])
        self.assertEqual(float(signal[0]), 0.9)

    def test_the_mask_matches_coverage_exactly(self):
        _, mask = average_overlapping([0.1, 0.2], [0, 2], n_frames=4)
        np.testing.assert_array_equal(mask, [True, False, True, False])

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(EventError):
            average_overlapping([0.1, 0.2], [0], n_frames=2)

    def test_a_frame_id_past_the_recording_raises(self):
        with self.assertRaises(EventError):
            average_overlapping([0.1], [5], n_frames=2)

    def test_no_predictions_leaves_everything_masked(self):
        signal, mask = average_overlapping([], [], n_frames=3)
        self.assertFalse(mask.any())
        self.assertEqual(signal.shape, (3,))


class TestToIntervals(unittest.TestCase):
    def test_consecutive_frames_merge_into_one_interval(self):
        signal, mask = signal_of([0.1, 0.9, 0.9, 0.9, 0.1])
        self.assertEqual(to_intervals(signal, mask, 0.5), [(1, 3)])

    def test_a_gap_splits_a_run(self):
        signal, mask = signal_of([0.9, 0.9, 0.1, 0.9, 0.9])
        self.assertEqual(to_intervals(signal, mask, 0.5), [(0, 1), (3, 4)])

    def test_an_uncovered_frame_breaks_a_run(self):
        # With no prediction there, joining across the gap would assert a
        # continuity the model never claimed.
        signal = np.asarray([0.9, 0.9, 0.9, 0.9])
        mask = np.asarray([True, True, False, True])
        self.assertEqual(to_intervals(signal, mask, 0.5), [(0, 1), (3, 3)])

    def test_a_run_reaching_the_end_is_closed(self):
        signal, mask = signal_of([0.1, 0.9, 0.9])
        self.assertEqual(to_intervals(signal, mask, 0.5), [(1, 2)])

    def test_a_run_starting_at_zero_is_found(self):
        signal, mask = signal_of([0.9, 0.1])
        self.assertEqual(to_intervals(signal, mask, 0.5), [(0, 0)])

    def test_nothing_above_threshold_is_empty(self):
        signal, mask = signal_of([0.1, 0.2])
        self.assertEqual(to_intervals(signal, mask, 0.5), [])

    def test_a_mismatched_mask_raises(self):
        with self.assertRaises(EventError):
            to_intervals(np.zeros(3), np.ones(2, dtype=bool), 0.5)


class TestTemporalIou(unittest.TestCase):
    def test_identical_intervals_score_one(self):
        self.assertAlmostEqual(temporal_iou((10, 20), (10, 20)), 1.0)

    def test_disjoint_intervals_score_zero(self):
        self.assertEqual(temporal_iou((0, 5), (10, 15)), 0.0)

    def test_touching_but_not_overlapping_scores_zero(self):
        self.assertEqual(temporal_iou((0, 5), (6, 10)), 0.0)

    def test_a_hand_checked_partial_overlap(self):
        # (0..9) and (5..14): intersection 5..9 = 5 frames, union 0..14 = 15.
        self.assertAlmostEqual(temporal_iou((0, 9), (5, 14)), 5 / 15)

    def test_containment_is_the_length_ratio(self):
        # (0..9) contains (4..5): intersection 2, union 10.
        self.assertAlmostEqual(temporal_iou((0, 9), (4, 5)), 0.2)

    def test_it_is_symmetric(self):
        self.assertAlmostEqual(temporal_iou((0, 9), (5, 14)), temporal_iou((5, 14), (0, 9)))


class TestCriteriaDisagree(unittest.TestCase):
    """The four criteria must be distinguishable, or reporting all four is noise."""

    def test_a_partial_overlap_separates_them(self):
        # Prediction (0..9) against annotation (6..15): intersection 6..9 = 4,
        # union 0..15 = 16, IoU = 0.25. Above 0.2, below 0.5.
        predicted, annotated = (0, 9), (6, 15)
        self.assertAlmostEqual(temporal_iou(predicted, annotated), 0.25)

        self.assertTrue(matches(predicted, annotated, "any"))
        self.assertTrue(matches(predicted, annotated, "iou20"))
        self.assertFalse(matches(predicted, annotated, "iou50"))
        self.assertFalse(matches(predicted, annotated, "iou75"))

    def test_a_one_frame_touch_hits_only_any(self):
        predicted, annotated = (0, 10), (10, 20)
        self.assertTrue(matches(predicted, annotated, "any"))
        for criterion in ("iou20", "iou50", "iou75"):
            self.assertFalse(matches(predicted, annotated, criterion), criterion)

    def test_an_exact_match_hits_every_criterion(self):
        for criterion in CRITERIA:
            self.assertTrue(matches((5, 10), (5, 10), criterion), criterion)

    def test_an_unknown_criterion_raises(self):
        with self.assertRaises(EventError):
            matches((0, 1), (0, 1), "iou90")


class TestMatch(unittest.TestCase):
    def test_two_predictions_on_one_blink_is_one_hit_and_one_false_alarm(self):
        # The property that stops a jittery model inflating its own recall.
        result = match([(10, 12), (14, 16)], [(9, 17)], "any")
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 0)

    def test_one_prediction_spanning_two_blinks_hits_only_one(self):
        # Matching is one-to-one, as in MPEblink's Hungarian assignment. A
        # single detection covering two annotated blinks means the model failed
        # to separate them -- precisely the "consecutive rapid eyeblink" failure
        # the literature measures -- so it scores one hit and one miss rather
        # than being credited with both.
        result = match([(0, 30)], [(5, 10), (20, 25)], "any")
        self.assertEqual(result.true_positives, 1)
        self.assertEqual(result.false_negatives, 1)
        self.assertEqual(result.false_positives, 0)

    def test_separating_two_close_blinks_scores_both(self):
        # The counterpart: a model that resolves the pair gets full credit.
        result = match([(5, 10), (20, 25)], [(5, 10), (20, 25)], "any")
        self.assertEqual(result.true_positives, 2)
        self.assertEqual(result.false_negatives, 0)

    def test_a_prediction_touching_nothing_is_a_false_alarm(self):
        result = match([(100, 110)], [(0, 5)], "any")
        self.assertEqual(result.false_positives, 1)
        self.assertEqual(result.false_negatives, 1)
        self.assertEqual(result.true_positives, 0)

    def test_an_undetected_blink_is_a_miss(self):
        result = match([], [(0, 5), (10, 15)], "any")
        self.assertEqual(result.false_negatives, 2)
        self.assertEqual(result.recall, 0.0)

    def test_the_match_map_records_which_prediction_hit(self):
        result = match([(0, 5), (10, 15)], [(10, 15)], "any")
        self.assertEqual(result.matched, {0: [1]})

    def test_perfect_detection_scores_one_everywhere(self):
        annotated = [(0, 5), (10, 15)]
        result = match(annotated, annotated, "iou75")
        self.assertEqual(result.recall, 1.0)
        self.assertEqual(result.precision, 1.0)
        self.assertEqual(result.f1, 1.0)
        self.assertEqual(result.false_positives, 0)

    def test_rates_are_zero_rather_than_undefined_when_empty(self):
        result = match([], [], "any")
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.precision, 0.0)
        self.assertEqual(result.f1, 0.0)

    def test_false_alarms_are_normalised_by_duration(self):
        result = match([(0, 5), (10, 15)], [], "any")
        self.assertAlmostEqual(result.false_alarms_per_minute(2.0), 1.0)

    def test_a_zero_length_recording_reports_no_rate(self):
        result = match([(0, 5)], [], "any")
        self.assertEqual(result.false_alarms_per_minute(0.0), 0.0)


class TestPoolCurves(unittest.TestCase):
    """A corpus curve is the sum of its recordings' counts, not their mean.

    The same rule the single operating point already uses: averaging rates
    would give a thirty-second clip the weight of a five-minute recording.
    """

    def curve(self, tp, fp, fn, thresholds=(0.25, 0.5, 0.75)) -> dict[str, np.ndarray]:
        hits, alarms, misses = (np.asarray(v, dtype=float) for v in (tp, fp, fn))
        return {
            "thresholds": np.asarray(thresholds, dtype=float),
            "tp": hits,
            "fp": alarms,
            "fn": misses,
            "recall": hits / np.maximum(hits + misses, 1),
            "precision": hits / np.maximum(hits + alarms, 1),
            "f1": np.zeros_like(hits),
            "false_alarms_per_minute": alarms,
        }

    def test_counts_are_summed_at_each_threshold(self):
        pooled = pool_curves(
            [
                self.curve([90, 90, 90], [0, 0, 0], [10, 10, 10]),
                self.curve([0, 0, 0], [0, 0, 0], [1, 1, 1]),
            ],
            minutes=10.0,
        )
        for value in pooled["recall"]:
            self.assertAlmostEqual(float(value), 90 / 101)

    def test_the_false_alarm_rate_uses_the_total_duration(self):
        pooled = pool_curves([self.curve([0, 0, 0], [6, 6, 6], [0, 0, 0])], minutes=2.0)
        for value in pooled["false_alarms_per_minute"]:
            self.assertAlmostEqual(float(value), 3.0)

    def test_the_thresholds_are_preserved(self):
        pooled = pool_curves([self.curve([1, 1], [0, 0], [0, 0], (0.3, 0.7))], minutes=1.0)
        np.testing.assert_allclose(pooled["thresholds"], [0.3, 0.7])

    def test_no_curves_pools_to_nothing(self):
        self.assertEqual(pool_curves([], minutes=1.0), {})

    def test_mismatched_thresholds_raise(self):
        # Summing index-by-index is only meaningful when the indices mean the
        # same operating point in every recording.
        with self.assertRaises(EventError):
            pool_curves(
                [
                    self.curve([1, 1], [0, 0], [0, 0], (0.3, 0.7)),
                    self.curve([1, 1], [0, 0], [0, 0], (0.4, 0.8)),
                ],
                minutes=1.0,
            )

    def test_a_zero_duration_does_not_divide_by_zero(self):
        pooled = pool_curves([self.curve([1, 1], [2, 2], [0, 0], (0.3, 0.7))], minutes=0.0)
        self.assertTrue(np.all(pooled["false_alarms_per_minute"] == 0.0))

    def test_the_pooled_curve_feeds_average_precision(self):
        pooled = pool_curves([self.curve([1, 1, 0], [0, 0, 0], [0, 0, 1])], minutes=1.0)
        self.assertGreaterEqual(average_precision(pooled), 0.0)


class TestFroc(unittest.TestCase):
    def curve(self) -> dict[str, np.ndarray]:
        signal, mask = signal_of([0.1, 0.4, 0.9, 0.9, 0.4, 0.1, 0.6, 0.2])
        return froc(signal, mask, [(2, 3)], minutes=1.0, criterion="any")

    def test_recall_never_increases_with_the_threshold(self):
        recall = self.curve()["recall"]
        self.assertTrue(np.all(np.diff(recall) <= 1e-9))

    def test_false_alarms_are_not_monotonic(self):
        # Worth pinning, because the intuition is wrong. Event counts are not
        # monotonic in the threshold: at a very low threshold the whole
        # recording merges into ONE interval covering the blink -- 1 TP, 0 FP.
        # Raising it splits that interval in two, and the half that misses the
        # blink becomes a false alarm. Merging, not just thresholding, decides
        # the count, which is exactly why the FROC curve is swept rather than a
        # single operating point being assumed to bound the others.
        signal, mask = signal_of([0.1, 0.4, 0.9, 0.9, 0.4, 0.1, 0.6, 0.2])
        curve = froc(signal, mask, [(2, 3)], minutes=1.0, criterion="any")
        rate = curve["false_alarms_per_minute"]
        self.assertGreater(float(rate.max()), float(rate[0]))

    def test_a_low_threshold_detects_everything(self):
        signal, mask = signal_of([0.1, 0.9, 0.9, 0.1])
        curve = froc(signal, mask, [(1, 2)], 1.0, "any", thresholds=np.asarray([0.05]))
        self.assertEqual(float(curve["recall"][0]), 1.0)

    def test_a_high_threshold_predicts_nothing(self):
        signal, mask = signal_of([0.1, 0.9, 0.9, 0.1])
        curve = froc(signal, mask, [(1, 2)], 1.0, "any", thresholds=np.asarray([0.99]))
        self.assertEqual(float(curve["recall"][0]), 0.0)
        self.assertEqual(float(curve["false_alarms_per_minute"][0]), 0.0)

    def test_every_series_is_aligned(self):
        curve = self.curve()
        length = curve["thresholds"].size
        for key in ("recall", "precision", "f1", "false_alarms_per_minute"):
            self.assertEqual(curve[key].size, length, key)


class TestAveragePrecision(unittest.TestCase):
    # A confident frame must sit strictly above every swept threshold, and a
    # background frame strictly below. `DEFAULT_THRESHOLDS` spans [0.01, 0.99]
    # and `to_intervals` compares with a strict `>`, so 0.99 is *not* above the
    # last swept point -- these fixtures use 1.0 and 0.0 to stay clear of both
    # ends rather than encoding the sweep's bounds into the expectation.
    CONFIDENT = 1.0
    BACKGROUND = 0.0

    def perfect(self):
        return signal_of([self.BACKGROUND, self.CONFIDENT, self.CONFIDENT, self.BACKGROUND])

    def test_a_perfect_detector_scores_high(self):
        signal, mask = self.perfect()
        curve = froc(signal, mask, [(1, 2)], 1.0, "any")
        self.assertGreater(average_precision(curve), 0.9)

    def test_a_perfect_flat_curve_scores_one(self):
        # A detector holding recall at 1.0 across every threshold is the best
        # possible result. Without the recall-0 anchor the trapezoid collapses
        # and reports 0.0 -- the best case scored as the worst.
        signal, mask = self.perfect()
        curve = froc(signal, mask, [(1, 2)], 1.0, "any")
        self.assertTrue(np.all(curve["recall"] == 1.0))
        self.assertAlmostEqual(average_precision(curve), 1.0)

    def test_a_detector_that_finds_nothing_scores_zero(self):
        curve = {"recall": np.asarray([0.0, 0.0]), "precision": np.asarray([0.0, 0.0])}
        self.assertEqual(average_precision(curve), 0.0)

    def test_an_empty_curve_is_zero(self):
        curve = {"recall": np.asarray([]), "precision": np.asarray([])}
        self.assertEqual(average_precision(curve), 0.0)


class TestAnnotatedIntervals(unittest.TestCase):
    def test_a_run_becomes_one_interval(self):
        self.assertEqual(annotated_intervals([-1, 3, 3, 3, -1]), [(1, 3)])

    def test_two_blinks_are_two_intervals(self):
        self.assertEqual(annotated_intervals([1, 1, -1, 2, 2]), [(0, 1), (3, 4)])

    def test_a_reused_id_after_a_gap_is_two_events(self):
        # "Sometimes people blink twice very fast, which are considered as two
        # blinks" -- the Eyeblink8 annotation notes.
        self.assertEqual(annotated_intervals([1, 1, -1, 1, 1]), [(0, 1), (3, 4)])

    def test_adjacent_ids_split_without_a_gap(self):
        self.assertEqual(annotated_intervals([1, 1, 2, 2]), [(0, 1), (2, 3)])

    def test_no_blinks_is_empty(self):
        self.assertEqual(annotated_intervals([-1, -1]), [])

    def test_a_blink_at_the_end_is_closed(self):
        self.assertEqual(annotated_intervals([-1, 7, 7]), [(1, 2)])


class TestEventMetrics(unittest.TestCase):
    def metrics(self) -> dict[str, float]:
        signal, mask = signal_of([0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])
        return event_metrics(signal, mask, [(1, 3)], minutes=1.0)

    def test_every_criterion_is_reported(self):
        metrics = self.metrics()
        for criterion in CRITERIA:
            self.assertIn(f"event/{criterion}/f1", metrics, criterion)

    def test_counts_and_rates_are_both_present(self):
        metrics = self.metrics()
        for name in ("tp", "fp", "fn", "recall", "precision", "f1", "fa_per_min"):
            self.assertIn(f"event/any/{name}", metrics, name)

    def test_descriptive_counts_give_the_rates_context(self):
        metrics = self.metrics()
        self.assertEqual(metrics["event/n_annotated"], 1.0)
        self.assertEqual(metrics["event/frames_covered"], 8.0)
        self.assertEqual(metrics["event/mean_blink_frames"], 3.0)

    def test_a_perfect_prediction_scores_one_under_every_criterion(self):
        # If a later disagreement appears, it is a real model difference and
        # not an artefact of how the criteria are wired.
        metrics = self.metrics()
        for criterion in CRITERIA:
            self.assertEqual(metrics[f"event/{criterion}/recall"], 1.0, criterion)
            self.assertEqual(metrics[f"event/{criterion}/fp"], 0.0, criterion)

    def test_a_recording_with_no_blinks_still_reports(self):
        signal, mask = signal_of([0.1, 0.1, 0.1])
        metrics = event_metrics(signal, mask, [], minutes=1.0)
        self.assertEqual(metrics["event/n_annotated"], 0.0)
        self.assertNotIn("event/mean_blink_frames", metrics)


if __name__ == "__main__":
    unittest.main()


class TestSegmentIou(unittest.TestCase):
    """The authors' convention: exclusive endpoints, no ``+1``."""

    def test_identical_segments_score_one(self):
        value = segment_iou((10, 20), np.asarray([[10.0, 20.0]]))
        self.assertAlmostEqual(float(value[0]), 1.0)

    def test_disjoint_segments_score_zero(self):
        value = segment_iou((10, 20), np.asarray([[30.0, 40.0]]))
        self.assertAlmostEqual(float(value[0]), 0.0)

    def test_half_overlap(self):
        # [10,20) vs [15,25): intersection 5, union 15.
        value = segment_iou((10, 20), np.asarray([[15.0, 25.0]]))
        self.assertAlmostEqual(float(value[0]), 5 / 15)

    def test_it_differs_from_the_inclusive_convention(self):
        # The reason this function exists: on a short blink the two conventions
        # disagree enough to move a detection across a tIoU threshold.
        exclusive = float(segment_iou((10, 14), np.asarray([[10.0, 14.0]]))[0])
        self.assertAlmostEqual(exclusive, 1.0)
        self.assertAlmostEqual(temporal_iou((10, 14), (12, 16)), 3 / 7)
        shifted = float(segment_iou((10, 14), np.asarray([[12.0, 16.0]]))[0])
        self.assertNotAlmostEqual(shifted, 3 / 7)

    def test_no_candidates_yields_an_empty_array(self):
        self.assertEqual(segment_iou((0, 5), np.zeros((0, 2))).size, 0)


class TestBlinkAp(unittest.TestCase):
    """MPEblink's headline metric, matched to the authors' implementation."""

    def test_a_perfect_detector_scores_one(self):
        annotated = {"a": [(10, 20), (40, 50)]}
        predicted = {"a": [(10, 20, 0.9), (40, 50, 0.8)]}
        scores = blink_ap(annotated, predicted)
        self.assertAlmostEqual(float(scores.mean()), 1.0, places=6)

    def test_nothing_predicted_scores_zero(self):
        self.assertEqual(float(blink_ap({"a": [(1, 5)]}, {}).mean()), 0.0)

    def test_nothing_annotated_scores_zero(self):
        self.assertEqual(float(blink_ap({}, {"a": [(1, 5, 0.9)]}).mean()), 0.0)

    def test_the_sweep_has_ten_thresholds(self):
        # 0.5:0.05:0.95, so index 0 is @0.5, index 5 is @0.75.
        self.assertEqual(len(blink_ap({"a": [(1, 5)]}, {"a": [(1, 5, 0.5)]})), 10)

    def test_the_threshold_grid_matches_the_authors(self):
        # `np.linspace(0.5, 0.95, 10)` verbatim from their `action_ap`. A
        # different grid would silently produce an incomparable headline.
        self.assertEqual(len(BLINK_AP_TIOU), 10)
        self.assertAlmostEqual(float(BLINK_AP_TIOU[0]), 0.5)
        self.assertAlmostEqual(float(BLINK_AP_TIOU[5]), 0.75)
        self.assertAlmostEqual(float(BLINK_AP_TIOU[-1]), 0.95)

    def test_a_loose_match_passes_low_thresholds_only(self):
        # tIoU 0.6: a true positive at 0.5 and 0.55, a false positive above.
        annotated = {"a": [(0, 10)]}
        predicted = {"a": [(0, 6, 0.9)]}
        scores = blink_ap(annotated, predicted)
        self.assertGreater(scores[0], 0.0)
        self.assertEqual(float(scores[-1]), 0.0)

    def test_a_duplicate_is_counted_as_a_false_positive(self):
        # The lock: only one prediction may claim a blink. The second is a
        # false positive -- but ranked *below* the hit it lands after recall
        # has saturated, and VOC interpolation ignores the tail. Verified
        # against the authors' own code, which likewise returns 1.0 here.
        annotated = {"a": [(10, 20)]}
        doubled = blink_ap(annotated, {"a": [(10, 20, 0.9), (10, 20, 0.8)]})
        self.assertAlmostEqual(float(doubled.mean()), 1.0, places=6)

    def test_a_duplicate_ranked_above_the_hit_costs_precision(self):
        # Where the lock bites: a spurious detection scored *higher* than the
        # true one pushes precision down at the recall point that matters.
        annotated = {"a": [(10, 20)]}
        clean = blink_ap(annotated, {"a": [(10, 20, 0.5)]})
        noisy = blink_ap(annotated, {"a": [(10, 20, 0.5), (60, 70, 0.9)]})
        self.assertLess(float(noisy.mean()), float(clean.mean()))

    def test_instances_are_scored_separately(self):
        # The multi-person property: a prediction on person A must never be
        # credited against person B's blink, even in the same video.
        annotated = {"v-p0": [(10, 20)], "v-p1": [(10, 20)]}
        crossed = blink_ap(annotated, {"v-p0": [(10, 20, 0.9), (10, 20, 0.8)]})
        self.assertLess(float(crossed.mean()), 1.0)

    def test_a_prediction_on_an_unannotated_instance_is_a_false_positive(self):
        annotated = {"a": [(10, 20)]}
        clean = blink_ap(annotated, {"a": [(10, 20, 0.9)]})
        noisy = blink_ap(annotated, {"a": [(10, 20, 0.9)], "b": [(1, 5, 0.95)]})
        self.assertLess(float(noisy.mean()), float(clean.mean()))

    def test_ranking_is_global_not_per_instance(self):
        # AP integrates one corpus-wide curve, so a confident detection in one
        # clip outranks a doubtful one in another.
        annotated = {"a": [(10, 20)], "b": [(10, 20)]}
        good_first = blink_ap(annotated, {"a": [(10, 20, 0.99)], "b": [(60, 70, 0.10)]})
        bad_first = blink_ap(annotated, {"a": [(10, 20, 0.10)], "b": [(60, 70, 0.99)]})
        self.assertGreater(float(good_first.mean()), float(bad_first.mean()))

    def test_scores_stay_within_bounds(self):
        annotated = {"a": [(10, 20), (40, 50)]}
        predicted = {"a": [(10, 20, 0.9), (41, 49, 0.7), (80, 90, 0.3)]}
        scores = blink_ap(annotated, predicted)
        self.assertTrue(bool(np.all(scores >= 0.0)))
        self.assertTrue(bool(np.all(scores <= 1.0)))


class TestBlinkApSummary(unittest.TestCase):
    def test_it_names_the_published_figures(self):
        summary = blink_ap_summary(np.linspace(0.5, 0.95, 10))
        self.assertAlmostEqual(summary["blink_ap50"], 0.5)
        self.assertAlmostEqual(summary["blink_ap75"], 0.75)
        self.assertAlmostEqual(summary["blink_ap95"], 0.95)

    def test_the_headline_is_the_mean(self):
        scores = np.linspace(0.5, 0.95, 10)
        self.assertAlmostEqual(blink_ap_summary(scores)["blink_ap"], float(scores.mean()))


class TestInterpolatedAveragePrecision(unittest.TestCase):
    def test_a_perfect_curve_integrates_to_one(self):
        precision = np.asarray([1.0, 1.0, 1.0])
        recall = np.asarray([1 / 3, 2 / 3, 1.0])
        self.assertAlmostEqual(interpolated_average_precision(precision, recall), 1.0)

    def test_precision_is_made_monotone(self):
        # A dip in precision is filled from the right, as VOC does.
        precision = np.asarray([1.0, 0.5, 1.0])
        recall = np.asarray([1 / 3, 2 / 3, 1.0])
        self.assertAlmostEqual(interpolated_average_precision(precision, recall), 1.0)


class TestToExclusive(unittest.TestCase):
    """The boundary between this project's convention and the authors'."""

    def test_an_inclusive_span_keeps_its_length(self):
        # Frames 10..14 are five frames, so the half-open form must be [10, 15).
        self.assertEqual(to_exclusive((10, 14)), (10.0, 15.0))

    def test_a_single_frame_has_length_one(self):
        # The case that silently cost 2.4 AP: without the +1 this is length
        # zero, its IoU against itself is 0/0, and a perfect hit scores as a
        # miss. Thresholding a signal readily produces one-frame intervals.
        self.assertEqual(to_exclusive((7, 7)), (7.0, 8.0))

    def test_a_one_frame_blink_is_detected_perfectly(self):
        self.assertAlmostEqual(
            float(blink_ap({"a": [(10, 10)]}, {"a": [(10, 10, 1.0)]}).mean()), 1.0
        )

    def test_a_one_frame_iou_against_itself_is_one(self):
        overlap = segment_iou(to_exclusive((10, 10)), np.asarray([to_exclusive((10, 10))]))
        self.assertAlmostEqual(float(overlap[0]), 1.0)


class TestHysteresis(unittest.TestCase):
    """Two thresholds separate "is this a blink" from "where does it start".

    A single cut has to be low enough to catch the shallow start of a closure
    and high enough not to fire on noise, and no single value does both. The
    high threshold decides whether a run is an event; the low one decides its
    extent.
    """

    def test_it_recovers_the_ramp_around_a_peak(self):
        # A blink: closing (0.4), closed (0.9), opening (0.4). A single cut at
        # 0.5 keeps only the deepest frame; hysteresis keeps the whole closure.
        signal, mask = signal_of([0.05, 0.4, 0.9, 0.4, 0.05])
        single = to_intervals(signal, mask, 0.5)
        both = to_intervals(signal, mask, 0.5, low_threshold=0.2)
        self.assertEqual(single, [(2, 2)])
        self.assertEqual(both, [(1, 3)])

    def test_a_run_that_never_peaks_is_not_an_event(self):
        # An incomplete closure: it crosses the low threshold but never the
        # high one, so it must not be reported as a blink.
        signal, mask = signal_of([0.05, 0.3, 0.35, 0.3, 0.05])
        self.assertEqual(to_intervals(signal, mask, 0.5, low_threshold=0.2), [])

    def test_two_peaks_in_one_low_run_stay_one_event(self):
        # A double blink is *not* separated by this rule alone -- the extractor
        # reports the span the signal stayed elevated. Pinned so the limitation
        # is explicit rather than assumed.
        signal, mask = signal_of([0.05, 0.9, 0.3, 0.9, 0.05])
        self.assertEqual(to_intervals(signal, mask, 0.5, low_threshold=0.2), [(1, 3)])

    def test_separated_events_stay_separate(self):
        signal, mask = signal_of([0.9, 0.05, 0.05, 0.9])
        self.assertEqual(to_intervals(signal, mask, 0.5, low_threshold=0.2), [(0, 0), (3, 3)])

    def test_no_low_threshold_is_the_old_behaviour(self):
        # Every existing caller must be bit-identical.
        signal, mask = signal_of([0.05, 0.4, 0.9, 0.4, 0.05])
        self.assertEqual(
            to_intervals(signal, mask, 0.5), to_intervals(signal, mask, 0.5, low_threshold=None)
        )

    def test_a_low_threshold_at_or_above_the_high_one_is_ignored(self):
        # It could only shrink a run, which is not what hysteresis means.
        signal, mask = signal_of([0.05, 0.4, 0.9, 0.4, 0.05])
        plain = to_intervals(signal, mask, 0.5)
        self.assertEqual(to_intervals(signal, mask, 0.5, low_threshold=0.5), plain)
        self.assertEqual(to_intervals(signal, mask, 0.5, low_threshold=0.7), plain)

    def test_an_uncovered_frame_still_breaks_a_run(self):
        # Coverage wins over hysteresis: with no prediction there, joining
        # across the gap would assert a continuity the model never claimed.
        signal = np.asarray([0.4, 0.9, 0.4, 0.4, 0.9, 0.4])
        mask = np.asarray([True, True, True, False, True, True])
        self.assertEqual(to_intervals(signal, mask, 0.5, low_threshold=0.2), [(0, 2), (4, 5)])

    def test_nothing_above_the_high_threshold_yields_nothing(self):
        signal, mask = signal_of([0.1, 0.2, 0.1])
        self.assertEqual(to_intervals(signal, mask, 0.5, low_threshold=0.05), [])
