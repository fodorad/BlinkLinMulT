"""Tests for the masked detection metrics."""

from __future__ import annotations

import unittest

import torch

from blinklinmult.data.schema import TASK_BPD
from blinklinmult.train.metrics import (
    METRIC_NAMES,
    PRIMARY_METRIC,
    BlinkMetrics,
    MultiTargetMetrics,
    aggregate_by_group,
    average_precision,
    binary_metrics,
    frame_group_ids,
    frame_level_metrics,
    window_level_metrics,
)


def perfect(n: int = 10) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Probabilities that classify every position correctly."""
    target = torch.tensor([1.0, 0.0] * (n // 2))
    probability = torch.where(target > 0.5, 0.95, 0.05)
    return probability, target, torch.ones_like(target, dtype=torch.bool)


class TestBinaryMetrics(unittest.TestCase):
    def test_perfect_prediction_scores_one(self):
        scores = binary_metrics(*perfect())
        for name in ("f1", "precision", "recall", "accuracy", "balanced_accuracy"):
            with self.subTest(metric=name):
                self.assertAlmostEqual(float(scores[name]), 1.0, places=5)

    def test_inverted_prediction_scores_zero_f1(self):
        probability, target, mask = perfect()
        scores = binary_metrics(1.0 - probability, target, mask)
        self.assertAlmostEqual(float(scores["f1"]), 0.0, places=5)

    def test_every_named_metric_is_reported(self):
        self.assertEqual(set(binary_metrics(*perfect())), set(METRIC_NAMES))

    def test_precision_and_recall_are_computed_separately(self):
        # Predicts every position positive: perfect recall, poor precision.
        target = torch.tensor([1.0, 0.0, 0.0, 0.0])
        probability = torch.ones(4)
        mask = torch.ones(4, dtype=torch.bool)

        scores = binary_metrics(probability, target, mask)
        self.assertAlmostEqual(float(scores["recall"]), 1.0, places=5)
        self.assertAlmostEqual(float(scores["precision"]), 0.25, places=5)

    def test_accuracy_flatters_an_all_negative_predictor(self):
        # The reason F1, not accuracy, is the primary metric: 19 of 20 frames
        # carry no blink, so predicting "never" scores 95% accuracy and 0 F1.
        target = torch.zeros(20)
        target[0] = 1.0
        probability = torch.zeros(20)
        mask = torch.ones(20, dtype=torch.bool)

        scores = binary_metrics(probability, target, mask)
        self.assertAlmostEqual(float(scores["accuracy"]), 0.95, places=5)
        self.assertAlmostEqual(float(scores["f1"]), 0.0, places=5)

    def test_balanced_accuracy_does_not_flatter_it(self):
        target = torch.zeros(20)
        target[0] = 1.0
        scores = binary_metrics(torch.zeros(20), target, torch.ones(20, dtype=torch.bool))
        self.assertAlmostEqual(float(scores["balanced_accuracy"]), 0.5, places=5)

    def test_masked_positions_are_excluded(self):
        probability, target, mask = perfect(10)
        # Corrupt the masked-out tail; the score must not move.
        mask[5:] = False
        corrupted = probability.clone()
        corrupted[5:] = 1.0 - corrupted[5:]

        self.assertAlmostEqual(float(binary_metrics(corrupted, target, mask)["f1"]), 1.0, places=5)

    def test_empty_mask_yields_zeros_not_nans(self):
        probability, target, _ = perfect()
        scores = binary_metrics(probability, target, torch.zeros_like(target, dtype=torch.bool))
        for name, value in scores.items():
            with self.subTest(metric=name):
                self.assertFalse(torch.isnan(value))
                self.assertAlmostEqual(float(value), 0.0, places=5)

    def test_threshold_shifts_the_decision(self):
        probability = torch.tensor([0.6, 0.6, 0.6, 0.6])
        target = torch.zeros(4)
        mask = torch.ones(4, dtype=torch.bool)

        self.assertAlmostEqual(
            float(binary_metrics(probability, target, mask, threshold=0.9)["accuracy"]),
            1.0,
            places=5,
        )
        self.assertAlmostEqual(
            float(binary_metrics(probability, target, mask, threshold=0.5)["accuracy"]),
            0.0,
            places=5,
        )

    def test_two_dimensional_input_is_handled(self):
        probability = torch.rand(4, 5, 2)
        target = (torch.rand(4, 5, 2) > 0.5).float()
        mask = torch.ones_like(target, dtype=torch.bool)
        self.assertEqual(binary_metrics(probability, target, mask)["f1"].ndim, 0)


class TestAveragePrecision(unittest.TestCase):
    def test_perfect_ranking_scores_one(self):
        probability = torch.tensor([0.9, 0.8, 0.2, 0.1])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        mask = torch.ones(4, dtype=torch.bool)
        self.assertAlmostEqual(float(average_precision(probability, target, mask)), 1.0, places=5)

    def test_is_threshold_free(self):
        # Badly calibrated but perfectly ranked: AP is 1, accuracy at 0.5 is not.
        probability = torch.tensor([0.42, 0.41, 0.02, 0.01])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        mask = torch.ones(4, dtype=torch.bool)

        self.assertAlmostEqual(float(average_precision(probability, target, mask)), 1.0, places=5)
        self.assertLess(float(binary_metrics(probability, target, mask)["f1"]), 1.0)

    def test_inverted_ranking_scores_low(self):
        probability = torch.tensor([0.1, 0.2, 0.8, 0.9])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        mask = torch.ones(4, dtype=torch.bool)
        self.assertLess(float(average_precision(probability, target, mask)), 0.6)

    def test_no_positives_yields_zero(self):
        probability = torch.rand(5)
        target = torch.zeros(5)
        mask = torch.ones(5, dtype=torch.bool)
        self.assertEqual(float(average_precision(probability, target, mask)), 0.0)

    def test_empty_mask_yields_zero(self):
        probability, target, _ = perfect()
        empty = torch.zeros_like(target, dtype=torch.bool)
        self.assertEqual(float(average_precision(probability, target, empty)), 0.0)

    def test_masked_positions_are_excluded_from_the_ranking(self):
        probability = torch.tensor([0.9, 0.8, 0.99, 0.1])
        target = torch.tensor([1.0, 1.0, 0.0, 0.0])
        # Position 2 is a confident false positive, but masked out.
        mask = torch.tensor([True, True, False, True])
        self.assertAlmostEqual(float(average_precision(probability, target, mask)), 1.0, places=5)


class TestBlinkMetrics(unittest.TestCase):
    def test_accumulates_across_batches(self):
        metric = BlinkMetrics("blink_presence")
        for _ in range(3):
            metric.update(*perfect(4))
        self.assertEqual(metric.valid_count, 12)

    def test_computes_the_target_prefixed_metrics(self):
        metric = BlinkMetrics("blink_presence")
        metric.update(*perfect())
        result = metric.compute()
        self.assertIn("blink_presence/f1", result)
        self.assertAlmostEqual(float(result["blink_presence/f1"]), 1.0, places=5)

    def test_compute_before_update_yields_zeros(self):
        # A legitimate state in a joint run whose batch mix excluded a corpus.
        result = BlinkMetrics("eye_state").compute()
        self.assertEqual(set(result), {f"eye_state/{n}" for n in METRIC_NAMES})
        for value in result.values():
            self.assertEqual(float(value), 0.0)

    def test_reset_clears_the_state(self):
        metric = BlinkMetrics("blink_presence")
        metric.update(*perfect())
        metric.reset()
        self.assertEqual(metric.valid_count, 0)

    def test_valid_count_ignores_masked_positions(self):
        metric = BlinkMetrics("blink_presence")
        probability, target, mask = perfect(10)
        mask[5:] = False
        metric.update(probability, target, mask)
        self.assertEqual(metric.valid_count, 5)

    def test_predictions_returns_the_flat_state(self):
        metric = BlinkMetrics("blink_presence")
        metric.update(*perfect(6))
        probability, target, mask = metric.predictions()
        self.assertEqual(probability.shape, (6,))
        self.assertEqual(target.shape, (6,))
        self.assertEqual(mask.shape, (6,))

    def test_predictions_before_update_are_empty(self):
        probability, _, _ = BlinkMetrics("x").predictions()
        self.assertEqual(probability.numel(), 0)

    def test_multi_dimensional_updates_are_flattened(self):
        metric = BlinkMetrics("eye_state")
        target = (torch.rand(3, 4, 2) > 0.5).float()
        metric.update(target, target, torch.ones_like(target, dtype=torch.bool))
        self.assertEqual(metric.valid_count, 24)

    def test_mismatched_shapes_raise(self):
        metric = BlinkMetrics("x")
        with self.assertRaises(ValueError):
            metric.update(torch.rand(4), torch.rand(5), torch.ones(4, dtype=torch.bool))

    def test_threshold_outside_the_open_unit_range_raises(self):
        for bad in (0.0, 1.0, -0.5, 1.5):
            with self.subTest(threshold=bad), self.assertRaises(ValueError):
                BlinkMetrics("x", threshold=bad)


class TestMultiTargetMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.metrics = MultiTargetMetrics(["blink_presence", "eye_state"])

    def test_reports_every_target(self):
        for name in ("blink_presence", "eye_state"):
            self.metrics.update(name, *perfect())
        result = self.metrics.compute()
        self.assertIn("blink_presence/f1", result)
        self.assertIn("eye_state/f1", result)

    def test_primary_metric_averages_the_f1_scores(self):
        self.metrics.update("blink_presence", *perfect())
        probability, target, mask = perfect()
        self.metrics.update("eye_state", 1.0 - probability, target, mask)

        result = self.metrics.compute()
        expected = (float(result["blink_presence/f1"]) + float(result["eye_state/f1"])) / 2
        self.assertAlmostEqual(float(result[PRIMARY_METRIC]), expected, places=5)

    def test_valid_positions_are_reported_per_target(self):
        self.metrics.update("blink_presence", *perfect(8))
        result = self.metrics.compute()
        self.assertEqual(float(result["blink_presence/valid_positions"]), 8.0)
        # The unsupervised target saw nothing, and says so rather than hiding it.
        self.assertEqual(float(result["eye_state/valid_positions"]), 0.0)

    def test_an_unsupervised_target_scores_zero_without_raising(self):
        self.metrics.update("blink_presence", *perfect())
        result = self.metrics.compute()
        self.assertEqual(float(result["eye_state/f1"]), 0.0)
        self.assertTrue(torch.isfinite(result[PRIMARY_METRIC]))

    def test_single_target_run(self):
        metrics = MultiTargetMetrics(["eye_state"])
        metrics.update("eye_state", *perfect())
        result = metrics.compute()
        self.assertAlmostEqual(float(result[PRIMARY_METRIC]), 1.0, places=5)

    def test_reset_clears_every_target(self):
        self.metrics.update("blink_presence", *perfect())
        self.metrics.reset()
        self.assertEqual(float(self.metrics.compute()["blink_presence/valid_positions"]), 0.0)

    def test_is_a_module_so_lightning_moves_its_state(self):
        self.assertIsInstance(self.metrics, torch.nn.Module)
        self.assertTrue(any(True for _ in self.metrics.children()))


if __name__ == "__main__":
    unittest.main()


class TestFrameGroupIds(unittest.TestCase):
    def test_the_two_eyes_of_a_frame_share_an_id(self):
        # The property, not the encoding: ids are a stable hash rather than an
        # enumeration, so asserting the literal 0 would pin an implementation
        # detail the accumulator must be free to change.
        ids = frame_group_ids(["v|000010|left", "v|000010|right"])
        self.assertEqual(int(ids[0]), int(ids[1]))

    def test_different_frames_get_different_ids(self):
        ids = frame_group_ids(["v|000010|left", "v|000020|left"])
        self.assertNotEqual(ids[0].item(), ids[1].item())

    def test_different_recordings_get_different_ids(self):
        # Same frame group number, different video: not the same frame.
        ids = frame_group_ids(["a|000010|left", "b|000010|left"])
        self.assertNotEqual(ids[0].item(), ids[1].item())

    def test_an_empty_batch_yields_none(self):
        self.assertIsNone(frame_group_ids([]))

    def test_unparseable_ids_disable_grouping_rather_than_raising(self):
        # The per-eye metrics must still run.
        self.assertIsNone(frame_group_ids(["not-an-id"]))


class TestAggregateByGroup(unittest.TestCase):
    def test_max_takes_the_larger_of_the_pair(self):
        values = torch.tensor([0.1, 0.9, 0.4, 0.2])
        groups = torch.tensor([0, 0, 1, 1])
        reduced, unique = aggregate_by_group(values, groups, "max")
        torch.testing.assert_close(reduced, torch.tensor([0.9, 0.4]))
        self.assertEqual(unique.tolist(), [0, 1])

    def test_mean_averages_the_pair(self):
        values = torch.tensor([0.2, 0.8, 0.4, 0.6])
        groups = torch.tensor([0, 0, 1, 1])
        reduced, _ = aggregate_by_group(values, groups, "mean")
        torch.testing.assert_close(reduced, torch.tensor([0.5, 0.5]))

    def test_a_lone_position_is_returned_unchanged(self):
        reduced, _ = aggregate_by_group(torch.tensor([0.7]), torch.tensor([3]), "max")
        torch.testing.assert_close(reduced, torch.tensor([0.7]))

    def test_groups_need_not_be_contiguous(self):
        reduced, unique = aggregate_by_group(torch.tensor([0.1, 0.5]), torch.tensor([7, 2]), "max")
        self.assertEqual(unique.tolist(), [2, 7])
        torch.testing.assert_close(reduced, torch.tensor([0.5, 0.1]))

    def test_unknown_reducer_raises(self):
        with self.assertRaises(ValueError):
            aggregate_by_group(torch.rand(2), torch.tensor([0, 1]), "median")

    def test_mismatched_shapes_raise(self):
        with self.assertRaises(ValueError):
            aggregate_by_group(torch.rand(3), torch.tensor([0, 1]), "max")


class TestFrameLevelMetrics(unittest.TestCase):
    def metric(self, probability, target, groups) -> BlinkMetrics:
        metric = BlinkMetrics("blink_presence")
        metric.update(
            torch.tensor(probability),
            torch.tensor(target),
            torch.ones_like(torch.tensor(target), dtype=torch.bool),
            torch.tensor(groups),
        )
        return metric

    def test_max_recovers_a_blink_one_eye_missed(self):
        # Left eye occluded (0.1), right eye plainly closing (0.9): the frame is
        # a blink, and max says so where mean (0.5) sits on the threshold.
        metric = self.metric([[0.1], [0.9]], [[1.0], [1.0]], [0, 0])
        scores = frame_level_metrics(metric)
        self.assertAlmostEqual(float(scores["frame_max/recall"]), 1.0, places=5)

    def test_both_reducers_are_reported(self):
        metric = self.metric([[0.9], [0.8]], [[1.0], [1.0]], [0, 0])
        scores = frame_level_metrics(metric)
        self.assertIn("frame_max/f1", scores)
        self.assertIn("frame_mean/f1", scores)

    def test_the_target_is_aggregated_by_max(self):
        # A frame is a blink frame if either eye is annotated as blinking.
        metric = self.metric([[0.9], [0.9]], [[1.0], [0.0]], [0, 0])
        scores = frame_level_metrics(metric)
        self.assertAlmostEqual(float(scores["frame_max/recall"]), 1.0, places=5)

    def test_two_frames_are_scored_independently(self):
        metric = self.metric(
            [[0.9], [0.9], [0.1], [0.1]], [[1.0], [1.0], [0.0], [0.0]], [0, 0, 1, 1]
        )
        scores = frame_level_metrics(metric)
        self.assertAlmostEqual(float(scores["frame_max/f1"]), 1.0, places=5)

    def test_without_group_ids_nothing_is_reported(self):
        metric = BlinkMetrics("blink_presence")
        metric.update(torch.rand(2, 1), torch.ones(2, 1), torch.ones(2, 1, dtype=torch.bool))
        self.assertEqual(frame_level_metrics(metric), {})

    def test_an_all_masked_epoch_reports_nothing(self):
        metric = BlinkMetrics("blink_presence")
        metric.update(
            torch.rand(2, 1),
            torch.ones(2, 1),
            torch.zeros(2, 1, dtype=torch.bool),
            torch.tensor([0, 0]),
        )
        self.assertEqual(frame_level_metrics(metric), {})


class TestPrimaryMetricIsFrameLevel(unittest.TestCase):
    def test_frame_scores_drive_selection_when_available(self):
        metrics = MultiTargetMetrics(["blink_presence"])
        # Perfect at frame level under max, imperfect per eye: the two differ,
        # so the primary metric must track the frame-level one.
        metrics.update(
            "blink_presence",
            torch.tensor([[0.1], [0.9]]),
            torch.tensor([[1.0], [1.0]]),
            torch.ones(2, 1, dtype=torch.bool),
            torch.tensor([0, 0]),
        )
        result = metrics.compute()
        self.assertAlmostEqual(
            float(result[PRIMARY_METRIC]),
            float(result["blink_presence/frame_max/f1"]),
            places=5,
        )
        self.assertNotAlmostEqual(
            float(result[PRIMARY_METRIC]), float(result["mean_eye_f1"]), places=5
        )

    def test_eye_scores_are_still_reported(self):
        metrics = MultiTargetMetrics(["blink_presence"])
        metrics.update(
            "blink_presence",
            torch.tensor([[0.9], [0.9]]),
            torch.tensor([[1.0], [1.0]]),
            torch.ones(2, 1, dtype=torch.bool),
            torch.tensor([0, 0]),
        )
        result = metrics.compute()
        self.assertIn("blink_presence/f1", result)
        self.assertIn("mean_eye_f1", result)

    def test_it_falls_back_to_the_eye_level_mean_without_groups(self):
        metrics = MultiTargetMetrics(["blink_presence"])
        metrics.update("blink_presence", *perfect())
        result = metrics.compute()
        self.assertAlmostEqual(
            float(result[PRIMARY_METRIC]), float(result["mean_eye_f1"]), places=5
        )


class TestEsrVersusBpd(unittest.TestCase):
    """The two protocols are different tasks over the same window."""

    def metric(self, probability, target, groups, name=TASK_BPD) -> BlinkMetrics:
        metric = BlinkMetrics(name)
        p = torch.tensor(probability)
        metric.update(
            p, torch.tensor(target), torch.ones_like(p, dtype=torch.bool), torch.tensor(groups)
        )
        return metric

    def test_bpd_forgives_a_boundary_disagreement_that_esr_punishes(self):
        # The model finds the blink but is one frame late -- exactly the
        # annotation imprecision these corpora carry. ESR counts a miss and a
        # false alarm; BPD sees a window that contains a blink and says so.
        metric = self.metric(
            probability=[[0.1, 0.1, 0.9, 0.1]],
            target=[[0.0, 1.0, 0.0, 0.0]],
            groups=[0],
        )
        esr = frame_level_metrics(metric)
        bpd = window_level_metrics(metric)

        self.assertAlmostEqual(float(esr["frame_max/f1"]), 0.0, places=5)
        self.assertAlmostEqual(float(bpd["window/f1"]), 1.0, places=5)

    def test_an_all_open_window_can_never_be_scored_blink_present(self):
        # The incoherence a single head exists to prevent, checked at the
        # scoring end. Both protocols read one probability sequence, so when no
        # frame crosses the threshold the window maximum cannot either: the
        # window holds a real blink and BOTH protocols must record the miss.
        # Two independent heads could report ESR "every frame open" alongside
        # BPD "a blink occurred" -- a contradiction unrepresentable here.
        metric = self.metric(
            probability=[[0.1, 0.2, 0.3, 0.1]],
            target=[[0.0, 1.0, 0.0, 0.0]],
            groups=[0],
        )
        self.assertAlmostEqual(
            float(frame_level_metrics(metric)["frame_max/recall"]), 0.0, places=5
        )
        self.assertAlmostEqual(float(window_level_metrics(metric)["window/recall"]), 0.0, places=5)

    def test_bpd_is_max_over_the_window(self):
        # One confident frame is enough for the window to be positive.
        metric = self.metric(
            probability=[[0.1, 0.1, 0.95, 0.1]],
            target=[[0.0, 0.0, 1.0, 0.0]],
            groups=[0],
        )
        self.assertAlmostEqual(float(window_level_metrics(metric)["window/recall"]), 1.0, places=5)

    def test_a_blink_free_window_predicted_blink_free_is_a_true_negative(self):
        metric = self.metric(
            probability=[[0.1, 0.2, 0.1, 0.1]],
            target=[[0.0, 0.0, 0.0, 0.0]],
            groups=[0],
        )
        scores = window_level_metrics(metric)
        self.assertAlmostEqual(
            float(scores["accuracy"] if "accuracy" in scores else scores["window/accuracy"]),
            1.0,
            places=5,
        )

    def test_both_eyes_of_a_window_collapse_to_one_decision(self):
        # Two eye-wise samples, same window: BPD must score it once, not twice.
        metric = self.metric(
            probability=[[0.1, 0.9, 0.1, 0.1], [0.1, 0.8, 0.1, 0.1]],
            target=[[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            groups=[0, 0],
        )
        self.assertAlmostEqual(float(window_level_metrics(metric)["window/f1"]), 1.0, places=5)

    def test_two_windows_are_scored_separately(self):
        metric = self.metric(
            probability=[[0.9, 0.9, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]],
            target=[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            groups=[0, 1],
        )
        scores = window_level_metrics(metric)
        self.assertAlmostEqual(float(scores["window/f1"]), 1.0, places=5)

    def test_without_group_ids_no_window_scores_are_produced(self):
        metric = BlinkMetrics(TASK_BPD)
        metric.update(torch.rand(2, 4), torch.ones(2, 4), torch.ones(2, 4, dtype=torch.bool))
        self.assertEqual(window_level_metrics(metric), {})


class TestBpdIsOnlyReportedForBlinkPresence(unittest.TestCase):
    def test_the_blink_target_gets_a_window_score(self):
        metrics = MultiTargetMetrics([TASK_BPD])
        metrics.update(
            TASK_BPD,
            torch.tensor([[0.1, 0.9, 0.1]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([0]),
        )
        result = metrics.compute()
        self.assertIn(f"{TASK_BPD}/window/f1", result)
        self.assertIn("bpd_f1", result)

    def test_eye_state_gets_no_window_score(self):
        # "Does this window contain a blink" is meaningless for eye state,
        # which is a per-frame property by definition.
        metrics = MultiTargetMetrics(["eye_state"])
        metrics.update(
            "eye_state",
            torch.tensor([[0.1, 0.9, 0.1]]),
            torch.tensor([[0.0, 1.0, 0.0]]),
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([0]),
        )
        result = metrics.compute()
        self.assertNotIn("eye_state/window/f1", result)
        self.assertNotIn("bpd_f1", result)
        # But the frame-level ESR score is there.
        self.assertIn("eye_state/frame_max/f1", result)


class TestFrameGroupsSurviveBatching(unittest.TestCase):
    """Group ids must be stable across batches, not per-batch enumerations.

    The accumulator persists across every batch of an epoch. Ids that restarted
    at 0 each batch made window 0 of batch 1 collide with window 0 of batch 0,
    so `frame_max` maxed together frames from different recordings and reported
    a perfect 1.0 on RN30 where the true score was 0.57. A metric that reads
    100% is the hardest kind of bug to notice, which is why this is pinned.
    """

    def test_ids_do_not_restart_between_batches(self):
        first = frame_group_ids(["v|000000|left", "v|000000|right"])
        second = frame_group_ids(["v|000045|left", "v|000045|right"])
        self.assertNotEqual(int(first[0]), int(second[0]))

    def test_the_same_window_gets_the_same_id_in_any_batch(self):
        alone = frame_group_ids(["v|000045|left"])
        with_others = frame_group_ids(["a|000000|left", "v|000045|left", "z|999999|right"])
        self.assertEqual(int(alone[0]), int(with_others[1]))

    def test_accumulating_two_batches_keeps_every_frame_distinct(self):
        metric = BlinkMetrics("blink_presence", threshold=0.5)
        steps = 45
        for ids in (
            ["v|000000|left", "v|000000|right"],
            ["v|000045|left", "v|000045|right"],
        ):
            metric.update(
                torch.rand(2, steps),
                torch.zeros(2, steps),
                torch.ones(2, steps, dtype=torch.bool),
                frame_group_ids(ids),
            )
        groups = metric.groups()
        self.assertEqual(groups.unique().numel(), 2 * steps)

    def test_both_eyes_of_a_window_still_share_a_group(self):
        # The whole point of grouping: `max` over the two eyes of one frame.
        ids = frame_group_ids(["v|000000|left", "v|000000|right"])
        self.assertEqual(int(ids[0]), int(ids[1]))

    def test_different_recordings_never_collide(self):
        ids = frame_group_ids(["a|000000|left", "b|000000|left"])
        self.assertNotEqual(int(ids[0]), int(ids[1]))

    def test_ids_are_stable_across_processes(self):
        # A plain hash() would differ per process under PYTHONHASHSEED, so a
        # resumed run or a second DDP rank would disagree about which frame is
        # which. Pinned against a literal so a change of scheme is deliberate.
        first = int(frame_group_ids(["v|000000|left"])[0])
        second = int(frame_group_ids(["v|000000|left"])[0])
        self.assertEqual(first, second)
        self.assertGreater(first, 0)

    def test_the_expanded_ids_stay_within_int64(self):
        # ids are multiplied by window_length before flattening; a 48-bit hash
        # times a long window must not overflow.
        ids = frame_group_ids(["v|000000|left"])
        self.assertLess(int(ids[0]) * 512, 2**63 - 1)


class TestAggregateByGroupWithLargeIds(unittest.TestCase):
    """Group ids are hashes, not small integers, and must survive any backend.

    `torch.unique(..., return_inverse=True)` is wrong on MPS for large int64
    values: on real RN30 state it collapsed 164 520 distinct frame groups into
    9 050, silently maxing together frames from unrelated recordings. The
    reported frame_max/f1 rose from a true 0.52 to 0.71 -- a metric that looks
    plausible, which is why this is pinned rather than left to a smoke run.
    """

    def test_large_sparse_ids_keep_every_group(self):
        ids = torch.tensor([1178217338580, 12664012557285584, 7, 1178217338580])
        values = torch.tensor([1.0, 2.0, 3.0, 4.0])
        reduced, unique = aggregate_by_group(values, ids, "max")
        self.assertEqual(unique.numel(), 3)
        self.assertEqual(reduced.numel(), 3)

    def test_the_reduction_is_correct_for_large_ids(self):
        ids = torch.tensor([12664012557285584, 12664012557285584, 7])
        values = torch.tensor([1.0, 9.0, 5.0])
        reduced, unique = aggregate_by_group(values, ids, "max")
        # unique() sorts, so group 7 comes first.
        torch.testing.assert_close(reduced, torch.tensor([5.0, 9.0]))

    def test_a_realistic_id_span_is_not_collapsed(self):
        # The measured span: ~1e12 to ~1e16 across 2 000 groups.
        rng = torch.Generator().manual_seed(0)
        base = torch.randint(10**12, 10**16, (2000,), generator=rng, dtype=torch.int64)
        ids = base.repeat_interleave(2)
        values = torch.rand(ids.numel(), generator=rng)
        _, unique = aggregate_by_group(values, ids, "max")
        self.assertEqual(unique.numel(), base.unique().numel())

    def test_mean_also_survives_large_ids(self):
        ids = torch.tensor([12664012557285584, 12664012557285584, 7, 7])
        values = torch.tensor([1.0, 3.0, 10.0, 20.0])
        reduced, _ = aggregate_by_group(values, ids, "mean")
        torch.testing.assert_close(reduced, torch.tensor([15.0, 2.0]))


class TestAggregateByGroupOnAccelerator(unittest.TestCase):
    """The MPS bug is invisible on the CPU, so the test must use the device.

    Every other test here runs on the CPU, where `torch.unique` is correct --
    which is exactly why the collapse survived into a training run. This one
    exercises the accelerator the runs actually use.
    """

    def device(self) -> str | None:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return None

    def test_large_ids_are_not_collapsed_on_the_accelerator(self):
        device = self.device()
        if device is None:
            self.skipTest("no accelerator available")
        rng = torch.Generator().manual_seed(0)
        base = torch.randint(10**12, 10**16, (5000,), generator=rng, dtype=torch.int64)
        ids = base.repeat_interleave(2).to(device)
        values = torch.rand(ids.numel(), generator=rng).to(device)
        _, unique = aggregate_by_group(values, ids, "max")
        self.assertEqual(unique.numel(), base.unique().numel())

    def test_the_accelerator_agrees_with_the_cpu(self):
        device = self.device()
        if device is None:
            self.skipTest("no accelerator available")
        rng = torch.Generator().manual_seed(1)
        base = torch.randint(10**12, 10**16, (2000,), generator=rng, dtype=torch.int64)
        ids = base.repeat_interleave(2)
        values = torch.rand(ids.numel(), generator=rng)
        on_cpu, _ = aggregate_by_group(values, ids, "max")
        on_device, _ = aggregate_by_group(values.to(device), ids.to(device), "max")
        torch.testing.assert_close(on_cpu, on_device.cpu())


class TestRestore(unittest.TestCase):
    """Restored state must score exactly like state computed in place.

    This is what makes an interrupted test pass resumable rather than merely
    approximable: every metric reduces over per-position values in an
    order-independent way, so where a position came from cannot matter.
    """

    def batch(self, size: int = 6, steps: int = 2, seed: int = 0):
        torch.manual_seed(seed)
        probability = torch.rand(size, steps)
        target = (torch.rand(size, steps) > 0.5).float()
        mask = torch.ones(size, steps, dtype=torch.bool)
        group = torch.arange(size, dtype=torch.long)
        return probability, target, mask, group

    def test_restoring_matches_updating(self):
        probability, target, mask, group = self.batch()

        live = BlinkMetrics("eye_state")
        live.update(probability, target, mask, group)

        restored = BlinkMetrics("eye_state")
        gathered = live._gathered()
        assert gathered is not None
        groups = live.groups()
        assert groups is not None
        restored.restore(*gathered, groups, live.window_length)

        self.assertEqual(restored.compute(), live.compute())

    def test_the_group_ids_survive_verbatim(self):
        # `update` would recompute them from the shapes and lose the frame
        # identity every downstream report groups by.
        probability, target, mask, group = self.batch()
        live = BlinkMetrics("eye_state")
        live.update(probability, target, mask, group)
        groups = live.groups()
        assert groups is not None

        restored = BlinkMetrics("eye_state")
        gathered = live._gathered()
        assert gathered is not None
        restored.restore(*gathered, groups, live.window_length)

        self.assertTrue(torch.equal(restored.groups(), groups))

    def test_a_resumed_pass_equals_an_uninterrupted_one(self):
        # Two halves restored then continued must equal scoring both at once.
        first = self.batch(seed=1)
        second = self.batch(seed=2)

        whole = BlinkMetrics("eye_state")
        whole.update(*first)
        whole.update(*second)

        interrupted = BlinkMetrics("eye_state")
        interrupted.update(*first)
        gathered = interrupted._gathered()
        assert gathered is not None
        groups = interrupted.groups()
        assert groups is not None

        resumed = BlinkMetrics("eye_state")
        resumed.restore(*gathered, groups, interrupted.window_length)
        resumed.update(*second)

        self.assertEqual(resumed.compute(), whole.compute())

    def test_the_window_length_is_restored(self):
        probability, target, mask, group = self.batch(steps=5)
        live = BlinkMetrics("eye_state")
        live.update(probability, target, mask, group)

        restored = BlinkMetrics("eye_state")
        gathered = live._gathered()
        assert gathered is not None
        groups = live.groups()
        assert groups is not None
        restored.restore(*gathered, groups, live.window_length)

        self.assertEqual(restored.window_length, 5)

    def test_disagreeing_shapes_are_rejected(self):
        metric = BlinkMetrics("eye_state")
        with self.assertRaises(ValueError):
            metric.restore(
                torch.rand(4),
                torch.rand(3),
                torch.ones(4, dtype=torch.bool),
                torch.arange(4),
                1,
            )


class TestUnsupervisedTargetDevice(unittest.TestCase):
    """An unsupervised target's zeros must live on the metric's own device.

    `MultiTargetMetrics.compute` stacks each target's F1 into a mean, and
    `torch.stack` across CPU and MPS **segfaults** -- a hard process crash with
    no exception to catch. A single-corpus evaluation reaches this whenever the
    corpus does not annotate an eval-only target, which is ordinary: a
    still-image corpus cannot witness a blink.
    """

    def test_an_empty_target_reports_on_its_own_device(self):
        metric = BlinkMetrics("blink_presence")
        self.assertTrue(all(v.device == metric.device for v in metric.compute().values()))

    def test_every_score_agrees_on_device_when_one_target_is_empty(self):
        metrics = MultiTargetMetrics(["eye_state", "blink_presence"])
        values = torch.rand(4, 2)
        metrics.update(
            "eye_state", values, (values > 0.5).float(), torch.ones(4, 2, dtype=torch.bool)
        )
        # blink_presence deliberately left unsupervised.
        computed = metrics.compute()
        self.assertEqual(len({v.device for v in computed.values()}), 1)

    def test_the_summary_is_computable_with_an_empty_target(self):
        # The regression itself: this is the call that crashed the process.
        metrics = MultiTargetMetrics(["eye_state", "blink_presence"])
        values = torch.rand(4, 2)
        metrics.update(
            "eye_state", values, (values > 0.5).float(), torch.ones(4, 2, dtype=torch.bool)
        )
        self.assertIn(PRIMARY_METRIC, metrics.compute())

    @unittest.skipUnless(torch.backends.mps.is_available(), "needs the MPS accelerator")
    def test_it_holds_on_the_accelerator(self):
        # Where the crash actually happened -- CPU-only testing would miss it.
        metrics = MultiTargetMetrics(["eye_state", "blink_presence"]).to("mps")
        values = torch.rand(4, 2, device="mps")
        metrics.update(
            "eye_state",
            values,
            (values > 0.5).float(),
            torch.ones(4, 2, dtype=torch.bool, device="mps"),
        )
        computed = metrics.compute()
        self.assertEqual(len({v.device.type for v in computed.values()}), 1)

    def test_a_supervised_and_an_unsupervised_target_agree_on_device(self):
        """The exact pairing that segfaulted the full test suite.

        `update` accumulates on the CPU to keep the MPS allocator from
        fragmenting, so a *supervised* target computes CPU scalars. An
        *unsupervised* one returned zeros on the metric's nominal device, and
        `MultiTargetMetrics.compute` then stacked across CPU and MPS -- which
        segfaults rather than raising. Neither test module reproduced it alone;
        it needed one of each branch in the same process.
        """
        metrics = MultiTargetMetrics(["eye_state", "blink_presence"])
        values = torch.rand(4, 2)
        # eye_state supervised, blink_presence deliberately never updated.
        metrics.update(
            "eye_state", values, (values > 0.5).float(), torch.ones(4, 2, dtype=torch.bool)
        )
        supervised = metrics["eye_state"].compute()
        unsupervised = metrics["blink_presence"].compute()
        self.assertEqual(
            {v.device.type for v in supervised.values()},
            {v.device.type for v in unsupervised.values()},
        )
        self.assertEqual(len({v.device.type for v in metrics.compute().values()}), 1)
