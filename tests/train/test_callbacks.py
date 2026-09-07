"""Tests for the training callbacks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from blinklinmult.data.schema import BLINK_PRESENCE, EYE_STATE
from blinklinmult.train.callbacks import (
    SIGNALS_TARGET_KEY,
    AcceleratorCacheLimiter,
    EpochPropagator,
    EventReport,
    PerDatasetReport,
    PlotCallback,
    PredictionWriter,
    TestCheckpointer,
    TestStateReleaser,
    TimeTrackingCallback,
    _pr_curve,
    load_signals,
)
from blinklinmult.train.config import ModelConfig, TrainConfig
from blinklinmult.train.events import CRITERIA, DEFAULT_THRESHOLDS
from blinklinmult.train.module import BlinkLightningModule

IMAGE_SIZE = 32
TIME_DIM = 3


class FakeTrainer:
    """The subset of the Trainer surface the callbacks touch."""

    def __init__(self, epoch: int = 0, datamodule=None):
        self.current_epoch = epoch
        self.datamodule = datamodule


class RecordingDataModule:
    """Records the epochs propagated to it."""

    def __init__(self):
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(epoch)


def module(targets: list[str] | None = None) -> BlinkLightningModule:
    """A real module, so the callbacks read the real metric state."""
    task = "joint" if targets is None else "eye_state"
    return BlinkLightningModule(
        model_config=ModelConfig(
            family="lint",
            backbone_pretrained=False,
            backbone_output_dim=4,
            d_model=8,
            num_heads=2,
            cmt_num_layers=1,
            head_hidden_dim=4,
        ),
        train_config=TrainConfig(task=task, loss="bce", max_epochs=1),
        image_size=IMAGE_SIZE,
    )


def populate(model: BlinkLightningModule, datasets: list[str] | None = None) -> None:
    """Fill a module's test state as a real test epoch would."""
    batch_size = len(datasets) if datasets else 4
    for name in model.target_names:
        # One eye per sample, so every target is (B, T).
        width = (batch_size, TIME_DIM)
        model.test_metrics.update(
            name,
            torch.rand(*width),
            (torch.rand(*width) > 0.5).float(),
            torch.ones(*width, dtype=torch.bool),
        )
    model.test_sample_ids = [f"s{i}" for i in range(batch_size)]
    model.test_datasets = datasets or ["rn30"] * batch_size


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestTimeTrackingCallback(TempDirCase):
    def run_epochs(self, count: int = 2) -> TimeTrackingCallback:
        callback = TimeTrackingCallback(self.tmp)
        model = module()
        trainer = FakeTrainer()
        callback.on_fit_start(trainer, model)
        for epoch in range(count):
            trainer.current_epoch = epoch
            callback.on_train_epoch_start(trainer, model)
            callback.on_train_epoch_end(trainer, model)
        callback.on_fit_end(trainer, model)
        return callback

    def test_writes_a_timing_summary(self):
        self.run_epochs()
        self.assertTrue((self.tmp / "time.json").is_file())

    def test_records_one_entry_per_epoch(self):
        self.run_epochs(3)
        data = json.loads((self.tmp / "time.json").read_text())
        self.assertEqual(len(data["epochs"]), 3)
        self.assertEqual([e["epoch"] for e in data["epochs"]], [0, 1, 2])

    def test_reports_a_total_and_a_mean(self):
        self.run_epochs(2)
        data = json.loads((self.tmp / "time.json").read_text())
        self.assertGreaterEqual(data["total_seconds"], 0.0)
        self.assertGreaterEqual(data["mean_epoch_seconds"], 0.0)

    def test_a_run_with_no_epochs_does_not_divide_by_zero(self):
        callback = TimeTrackingCallback(self.tmp)
        model = module()
        trainer = FakeTrainer()
        callback.on_fit_start(trainer, model)
        callback.on_fit_end(trainer, model)
        data = json.loads((self.tmp / "time.json").read_text())
        self.assertEqual(data["mean_epoch_seconds"], 0.0)


class TestEpochPropagator(unittest.TestCase):
    def test_propagates_the_epoch_to_the_datamodule(self):
        # Without this the mixing draw and the augmentations repeat every epoch,
        # silently reducing an N-epoch run to one epoch seen N times.
        datamodule = RecordingDataModule()
        EpochPropagator().on_train_epoch_start(
            FakeTrainer(epoch=4, datamodule=datamodule), module()
        )
        self.assertEqual(datamodule.epochs, [4])

    def test_tolerates_an_absent_datamodule(self):
        EpochPropagator().on_train_epoch_start(FakeTrainer(datamodule=None), module())

    def test_tolerates_a_datamodule_without_set_epoch(self):
        EpochPropagator().on_train_epoch_start(FakeTrainer(datamodule=object()), module())


class TestPredictionWriter(TempDirCase):
    def test_writes_one_csv_per_target(self):
        model = module()
        populate(model)
        PredictionWriter(self.tmp).on_test_epoch_end(FakeTrainer(), model)

        for name in (BLINK_PRESENCE, EYE_STATE):
            with self.subTest(target=name):
                self.assertTrue((self.tmp / f"test_predictions_{name}.csv").is_file())

    def test_csv_carries_probability_target_and_validity(self):
        import pandas as pd

        model = module()
        populate(model)
        PredictionWriter(self.tmp).on_test_epoch_end(FakeTrainer(), model)

        frame = pd.read_csv(self.tmp / f"test_predictions_{BLINK_PRESENCE}.csv")
        self.assertEqual(list(frame.columns), ["probability", "target", "valid"])
        self.assertEqual(len(frame), 4 * TIME_DIM)

    def test_no_predictions_writes_nothing_and_does_not_raise(self):
        PredictionWriter(self.tmp).on_test_epoch_end(FakeTrainer(), module())
        self.assertEqual(list(self.tmp.glob("*.csv")), [])


class TestPerDatasetReport(TempDirCase):
    def test_writes_a_breakdown_per_corpus(self):
        model = module()
        populate(model, datasets=["rn30", "rn30", "cew", "cew"])
        PerDatasetReport(self.tmp).on_test_epoch_end(FakeTrainer(), model)

        report = json.loads((self.tmp / "test_per_dataset.json").read_text())
        self.assertEqual(set(report[BLINK_PRESENCE]), {"rn30", "cew"})

    def test_each_corpus_reports_every_metric(self):
        model = module()
        populate(model, datasets=["rn30"] * 4)
        PerDatasetReport(self.tmp).on_test_epoch_end(FakeTrainer(), model)

        report = json.loads((self.tmp / "test_per_dataset.json").read_text())
        scores = report[BLINK_PRESENCE]["rn30"]
        for key in ("f1", "precision", "recall", "average_precision", "valid_positions"):
            self.assertIn(key, scores)

    def test_valid_positions_sum_to_the_whole_epoch(self):
        model = module()
        populate(model, datasets=["rn30", "rn30", "cew", "cew"])
        PerDatasetReport(self.tmp).on_test_epoch_end(FakeTrainer(), model)

        report = json.loads((self.tmp / "test_per_dataset.json").read_text())
        total = sum(v["valid_positions"] for v in report[BLINK_PRESENCE].values())
        self.assertEqual(total, 4 * TIME_DIM)

    def test_no_dataset_ids_writes_nothing(self):
        model = module()
        populate(model)
        model.test_datasets = []
        PerDatasetReport(self.tmp).on_test_epoch_end(FakeTrainer(), model)
        self.assertFalse((self.tmp / "test_per_dataset.json").exists())

    def test_no_predictions_writes_nothing(self):
        model = module()
        model.test_datasets = ["cew"]
        PerDatasetReport(self.tmp).on_test_epoch_end(FakeTrainer(), model)
        self.assertFalse((self.tmp / "test_per_dataset.json").exists())


class TestPlotCallback(TempDirCase):
    def test_writes_a_precision_recall_figure(self):
        model = module()
        populate(model)
        PlotCallback(self.tmp).on_test_epoch_end(FakeTrainer(), model)
        self.assertTrue((self.tmp / "test_precision_recall.png").is_file())

    def test_no_predictions_writes_nothing(self):
        PlotCallback(self.tmp).on_test_epoch_end(FakeTrainer(), module())
        self.assertFalse((self.tmp / "test_precision_recall.png").exists())

    def test_an_all_masked_target_is_skipped(self):
        model = module()
        for name in model.target_names:
            width = (2, TIME_DIM)
            model.test_metrics.update(
                name,
                torch.rand(*width),
                torch.zeros(*width),
                torch.zeros(*width, dtype=torch.bool),
            )
        PlotCallback(self.tmp).on_test_epoch_end(FakeTrainer(), model)
        self.assertFalse((self.tmp / "test_precision_recall.png").exists())


class TestPrCurve(unittest.TestCase):
    def test_perfect_ranking_holds_precision_at_one(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0])
        precision, recall = _pr_curve(scores, labels)
        self.assertAlmostEqual(float(precision[1]), 1.0, places=5)
        self.assertAlmostEqual(float(recall[1]), 1.0, places=5)

    def test_recall_is_monotonically_non_decreasing(self):
        scores = torch.rand(50)
        labels = (torch.rand(50) > 0.5).float()
        _, recall = _pr_curve(scores, labels)
        self.assertTrue((recall[1:] >= recall[:-1] - 1e-6).all())

    def test_no_positives_yields_empty_curves(self):
        precision, recall = _pr_curve(torch.rand(5), torch.zeros(5))
        self.assertEqual(len(precision), 0)
        self.assertEqual(len(recall), 0)

    @unittest.skipUnless(
        torch.backends.mps.is_available() or torch.cuda.is_available(),
        "needs an accelerator",
    )
    def test_it_runs_on_an_accelerator(self):
        # The rank vector is built with torch.arange, which defaults to CPU:
        # every real training run ends on an accelerator, so a device mismatch
        # here crashed the whole test epoch after the model had finished.
        device = "mps" if torch.backends.mps.is_available() else "cuda"
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1], device=device)
        labels = torch.tensor([1.0, 1.0, 0.0, 0.0], device=device)
        precision, recall = _pr_curve(scores, labels)
        self.assertAlmostEqual(float(precision[1]), 1.0, places=5)
        self.assertAlmostEqual(float(recall[1]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()


class TestEventReport(TempDirCase):
    """Blink detection scored as events, the way the literature reports it."""

    def populate_recording(
        self,
        model: BlinkLightningModule,
        starts: list[int],
        video_id: str = "talking",
    ) -> None:
        """Fill test state with windows of one recording, at known frames."""
        blink = torch.zeros(len(starts), TIME_DIM)
        # A blink in the middle of the first window, so there is something to
        # detect and something to miss.
        blink[0, 1:3] = 1.0

        for name in model.target_names:
            model.test_metrics.update(
                name,
                blink.clone(),
                blink.clone(),
                torch.ones(len(starts), TIME_DIM, dtype=torch.bool),
            )
        model.test_sample_ids = [f"{video_id}|{start:06d}|left" for start in starts]
        model.test_datasets = ["talkingface"] * len(starts)

    def test_writes_the_report(self):
        model = module()
        self.populate_recording(model, [0, 2, 4])
        EventReport(self.tmp, fps=25.0).on_test_epoch_end(FakeTrainer(), model)
        self.assertTrue((self.tmp / "test_events.json").is_file())

    def test_every_criterion_is_reported(self):
        model = module()
        self.populate_recording(model, [0, 2, 4])
        EventReport(self.tmp, fps=25.0).on_test_epoch_end(FakeTrainer(), model)

        report = json.loads((self.tmp / "test_events.json").read_text())
        for criterion in CRITERIA:
            self.assertIn(f"event/{criterion}/f1", report["_corpus"], criterion)

    def test_a_perfect_prediction_is_detected(self):
        # The predictions here ARE the targets, so every annotated blink must
        # be found under every criterion.
        model = module()
        self.populate_recording(model, [0, 2, 4])
        EventReport(self.tmp, fps=25.0).on_test_epoch_end(FakeTrainer(), model)

        summary = json.loads((self.tmp / "test_events.json").read_text())["_corpus"]
        for criterion in CRITERIA:
            self.assertEqual(summary[f"event/{criterion}/recall"], 1.0, criterion)
            self.assertEqual(summary[f"event/{criterion}/fp"], 0.0, criterion)

    def test_recordings_are_scored_separately(self):
        # False alarms are per minute, and a duration only means something for
        # one continuous timeline.
        model = module()
        self.populate_recording(model, [0, 2])
        model.test_sample_ids = ["a|000000|left", "b|000000|left"]
        EventReport(self.tmp, fps=25.0).on_test_epoch_end(FakeTrainer(), model)

        report = json.loads((self.tmp / "test_events.json").read_text())
        self.assertIn("a", report)
        self.assertIn("b", report)

    def test_the_corpus_row_sums_counts_rather_than_averaging_rates(self):
        # Averaging rates would weigh a three-second clip like a five-minute
        # recording; the corpus row must be count-weighted.
        report = {
            "long": {
                "event/any/tp": 90.0,
                "event/any/fp": 0.0,
                "event/any/fn": 10.0,
                "event/minutes": 10.0,
            },
            "short": {
                "event/any/tp": 0.0,
                "event/any/fp": 0.0,
                "event/any/fn": 1.0,
                "event/minutes": 0.1,
            },
        }
        total, _ = EventReport._aggregate(report)
        # 90 of 101 hits, not the mean of 0.9 and 0.0.
        self.assertAlmostEqual(total["event/any/recall"], 90 / 101)

    def _curve(self, tp, fp, fn, thresholds=(0.25, 0.5, 0.75)):
        """A FROC curve with the given per-threshold counts."""
        import numpy as np

        hits = np.asarray(tp, dtype=float)
        alarms = np.asarray(fp, dtype=float)
        misses = np.asarray(fn, dtype=float)
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

    def test_the_froc_scalars_survive_aggregation(self):
        # These went missing from the corpus row while sitting present in every
        # per-recording row, which kept the headline FROC numbers out of MLflow.
        report = {
            "a": {
                "event/any/tp": 1.0,
                "event/any/fp": 0.0,
                "event/any/fn": 0.0,
                "event/minutes": 1.0,
            }
        }
        curves = {"any": [self._curve([1, 1, 0], [0, 0, 0], [0, 0, 1])]}
        total, pooled = EventReport._aggregate(report, curves)
        self.assertIn("event/any/average_precision", total)
        self.assertIn("event/any/best_f1", total)
        self.assertIn("any", pooled)

    def test_pooled_curves_sum_counts_at_each_threshold(self):
        # The same count-weighting the single operating point uses: a curve from
        # a short clip must not pull the corpus curve as hard as a long one.
        report = {
            "long": {
                "event/any/tp": 0.0,
                "event/any/fp": 0.0,
                "event/any/fn": 0.0,
                "event/minutes": 10.0,
            },
            "short": {
                "event/any/tp": 0.0,
                "event/any/fp": 0.0,
                "event/any/fn": 0.0,
                "event/minutes": 1.0,
            },
        }
        curves = {
            "any": [
                self._curve([90, 90, 90], [0, 0, 0], [10, 10, 10]),
                self._curve([0, 0, 0], [0, 0, 0], [1, 1, 1]),
            ]
        }
        _, pooled = EventReport._aggregate(report, curves)
        # 90 of 101 at every threshold, not the mean of 0.9 and 0.0.
        for value in pooled["any"]["recall"]:
            self.assertAlmostEqual(float(value), 90 / 101)

    def test_aggregating_without_curves_still_works(self):
        # The counts path must not depend on the curves being supplied.
        report = {
            "a": {
                "event/any/tp": 1.0,
                "event/any/fp": 1.0,
                "event/any/fn": 0.0,
                "event/minutes": 2.0,
            }
        }
        total, pooled = EventReport._aggregate(report)
        self.assertAlmostEqual(total["event/any/recall"], 1.0)
        self.assertEqual(pooled, {})

    def test_no_samples_writes_nothing_and_does_not_raise(self):
        EventReport(self.tmp).on_test_epoch_end(FakeTrainer(), module())
        self.assertFalse((self.tmp / "test_events.json").exists())

    def test_unparseable_sample_ids_are_skipped(self):
        model = module()
        populate(model)  # ids are "s0", "s1", ... which carry no frame group
        EventReport(self.tmp).on_test_epoch_end(FakeTrainer(), model)
        self.assertFalse((self.tmp / "test_events.json").exists())

    def test_a_run_without_blink_presence_is_skipped(self):
        model = module(targets=[EYE_STATE])
        self.populate_recording(model, [0, 2])
        EventReport(self.tmp).on_test_epoch_end(FakeTrainer(), model)
        self.assertFalse((self.tmp / "test_events.json").exists())


class TestEventReportBlinkAp(TestEventReport):
    """MPEblink's Blink-AP, the number the published table is compared against."""

    def test_blink_ap_is_reported(self):
        model = module()
        self.populate_recording(model, [0, 2, 4])
        EventReport(self.tmp, fps=25.0).on_test_epoch_end(FakeTrainer(), model)

        summary = json.loads((self.tmp / "test_events.json").read_text())["_corpus"]
        for key in ("blink_ap", "blink_ap50", "blink_ap75", "blink_ap95"):
            self.assertIn(f"event/{key}", summary, key)

    def test_a_perfect_prediction_scores_one(self):
        # The predictions are the targets, so AP must be exactly 1.0 -- this is
        # what caught the inclusive/exclusive interval mismatch, which scored a
        # perfect detector at 0.9755.
        model = module()
        self.populate_recording(model, [0, 2, 4])
        EventReport(self.tmp, fps=25.0).on_test_epoch_end(FakeTrainer(), model)

        summary = json.loads((self.tmp / "test_events.json").read_text())["_corpus"]
        self.assertAlmostEqual(summary["event/blink_ap"], 1.0, places=5)
        self.assertAlmostEqual(summary["event/blink_ap50"], 1.0, places=5)

    def test_instances_are_kept_apart(self):
        # Two people must not have their blinks pooled into one timeline: the
        # multi-person property MPEblink exists to measure.
        model = module()
        self.populate_recording(model, [0, 2])
        model.test_sample_ids = ["v-person0|000000|left", "v-person1|000000|left"]
        EventReport(self.tmp, fps=25.0).on_test_epoch_end(FakeTrainer(), model)

        report = json.loads((self.tmp / "test_events.json").read_text())
        self.assertIn("v-person0", report)
        self.assertIn("v-person1", report)
        self.assertIn("event/blink_ap", report["_corpus"])

    def test_nothing_annotated_omits_the_metric(self):
        # No annotation means no AP to report; a fabricated 0.0 would look like
        # a failed detector rather than an absent target.
        model = module()
        blank = torch.zeros(2, TIME_DIM)
        for name in model.target_names:
            model.test_metrics.update(
                name, blank.clone(), blank.clone(), torch.ones(2, TIME_DIM, dtype=torch.bool)
            )
        model.test_sample_ids = ["v|000000|left", "v|000002|left"]
        model.test_datasets = ["talkingface"] * 2
        EventReport(self.tmp, fps=25.0).on_test_epoch_end(FakeTrainer(), model)

        summary = json.loads((self.tmp / "test_events.json").read_text())["_corpus"]
        self.assertNotIn("event/blink_ap", summary)


class TestEventThresholdTuning(TestEventReport):
    """The operating point is fitted on validation, not assumed to be 0.5.

    Measured on RN30: focal-loss arms peaked at 0.35 and 0.30 and lost 11.5 and
    16.5 points of event F1 to the fixed default, while a BCE arm peaked at
    0.50 and lost nothing. Reporting all three at 0.5 compares calibration
    rather than detection.
    """

    def test_it_defaults_to_tuning(self):
        self.assertTrue(EventReport(self.tmp).tune_threshold)

    def test_the_fallback_is_used_before_any_fitting(self):
        report = EventReport(self.tmp, threshold=0.5)
        self.assertEqual(report.operating_point, 0.5)

    def test_a_fitted_threshold_overrides_the_fallback(self):
        report = EventReport(self.tmp, threshold=0.5)
        report.fitted_threshold = 0.3
        self.assertEqual(report.operating_point, 0.3)

    def test_tuning_can_be_switched_off(self):
        model = module()
        self.populate_recording(model, [0, 2, 4])
        model.valid_sample_ids = list(model.test_sample_ids)
        report = EventReport(self.tmp, fps=25.0, tune_threshold=False)
        report.on_validation_epoch_end(FakeTrainer(), model)
        self.assertIsNone(report.fitted_threshold)
        self.assertEqual(report.operating_point, 0.5)

    def test_no_validation_ids_leaves_the_fallback_in_place(self):
        # A run whose validation split carried no ids must still test, at the
        # default, rather than crash or report nothing.
        model = module()
        model.valid_sample_ids = []
        report = EventReport(self.tmp, fps=25.0)
        report.on_validation_epoch_end(FakeTrainer(), model)
        self.assertEqual(report.operating_point, 0.5)

    def test_the_report_uses_the_fitted_point(self):
        model = module()
        self.populate_recording(model, [0, 2, 4])
        report = EventReport(self.tmp, fps=25.0)
        report.fitted_threshold = 0.3
        report.on_test_epoch_end(FakeTrainer(), model)
        written = json.loads((self.tmp / "test_events.json").read_text())
        self.assertIn("_corpus", written)


class TestByRecordingLayout(unittest.TestCase):
    """`_by_recording` must return arrays, not boxed Python lists.

    The grouped structure spans every frame of the test split, and `.tolist()`
    boxes each float32 into a 24-byte Python float in an 8-byte list slot --
    measured at 98.6 bytes per frame against 16.7 as arrays. On a 2.4M-frame
    benchmark pass that difference is most of a gigabyte, so the layout is a
    correctness-adjacent property worth pinning.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.report = EventReport(self.tmp, fps=25.0)

    def grouped(self, per_sample: int = 1):
        count = 12
        sample_ids = [f"rec{i % 3}|{i:06d}|left" for i in range(count)]
        torch.manual_seed(0)
        probability = torch.rand(count * per_sample)
        target = (torch.rand(count * per_sample) > 0.5).float()
        return self.report._by_recording(sample_ids, probability, target, per_sample)

    def test_the_values_are_arrays(self):
        for values in self.grouped().values():
            for key in ("p", "t", "f"):
                self.assertIsInstance(values[key], np.ndarray, f"{key} should be an array")

    def test_frame_ids_stay_integers(self):
        # `_score` indexes a signal with these, so a float dtype would break it
        # somewhere far from here.
        for values in self.grouped().values():
            self.assertTrue(np.issubdtype(values["f"].dtype, np.integer))

    def test_each_recording_keeps_only_its_own_frames(self):
        grouped = self.grouped()
        self.assertEqual(set(grouped), {"rec0", "rec1", "rec2"})
        for name, values in grouped.items():
            index = int(name.removeprefix("rec"))
            expected = [i for i in range(12) if i % 3 == index]
            self.assertEqual(values["f"].tolist(), expected)

    def test_a_window_expands_to_consecutive_frames(self):
        # per_sample > 1 is the video-corpus case: a window's Nth position is
        # frame start+N, which is what lets overlapping windows be averaged.
        grouped = self.report._by_recording(["rec|000010|left"], torch.rand(4), torch.zeros(4), 4)
        self.assertEqual(grouped["rec"]["f"].tolist(), [10, 11, 12, 13])

    def test_the_arrays_stay_aligned_with_their_frames(self):
        count = 6
        sample_ids = [f"rec|{i:06d}|left" for i in range(count)]
        probability = torch.arange(count, dtype=torch.float32)
        grouped = self.report._by_recording(sample_ids, probability, torch.zeros(count), 1)
        self.assertEqual(grouped["rec"]["p"].tolist(), list(range(count)))

    def test_an_unparseable_id_is_skipped_not_fatal(self):
        grouped = self.report._by_recording(
            ["good|000000|left", "nonsense"], torch.rand(2), torch.zeros(2), 1
        )
        self.assertEqual(set(grouped), {"good"})

    def test_the_consumers_accept_what_it_returns(self):
        # The point of the layout change is that `_score` and `_minutes` take
        # arrays as readily as lists; this is what would catch a regression.
        values = next(iter(self.grouped().values()))
        self.assertGreater(self.report._minutes(values), 0.0)
        self.assertIn("event/any/f1", self.report._score(values))


class TestTestStateReleaser(unittest.TestCase):
    """The test accumulator must be freed once the reporters have read it.

    `on_test_epoch_end` deliberately leaves it populated so the writer, the
    per-corpus report, the event report, and the plotter can all read it. That
    leaves nothing to release it, so a full pass holds every test prediction
    until the process exits.
    """

    def test_it_clears_the_accumulated_state(self):
        model = module()
        populate(model)
        self.assertGreater(model.test_metrics[model.target_names[0]].valid_count, 0)

        TestStateReleaser().on_test_end(FakeTrainer(), model)

        self.assertEqual(model.test_metrics[model.target_names[0]].valid_count, 0)
        self.assertEqual(model.test_sample_ids, [])
        self.assertEqual(model.test_datasets, [])

    def test_the_module_can_still_compute_its_epoch_metrics(self):
        """The release must not precede `LightningModule.on_test_epoch_end`.

        Lightning runs every callback's `on_test_epoch_end` *before* the
        module's own, so releasing there emptied the accumulator before the
        epoch metrics were computed -- reporting `test/mean_f1 = 0.0` against a
        real per-corpus F1 of 0.968. Releasing in `on_test_end` is what fixes
        it, and this asserts the ordering rather than the hook name.
        """
        model = module()
        populate(model)

        # Every callback's epoch-end hook, in the order Lightning calls them.
        tmp = Path(tempfile.mkdtemp())
        for callback in (PredictionWriter(tmp), PerDatasetReport(tmp), TestStateReleaser()):
            hook = getattr(callback, "on_test_epoch_end", None)
            if hook is not None:
                hook(FakeTrainer(), model)

        # Now the module's own hook, which is what logs the epoch metrics.
        computed = model.test_metrics.compute()
        self.assertGreater(float(computed[f"{model.target_names[0]}/valid_positions"]), 0.0)

    def test_it_runs_after_a_reporter_has_read_the_state(self):
        # Ordering is the whole contract: released too early and the reports
        # come out empty.
        model = module()
        populate(model, ["talkingface", "talkingface"])
        tmp = Path(tempfile.mkdtemp())
        PerDatasetReport(tmp).on_test_epoch_end(FakeTrainer(), model)
        TestStateReleaser().on_test_end(FakeTrainer(), model)
        self.assertTrue((tmp / "test_per_dataset.json").exists())

    def test_releasing_twice_is_harmless(self):
        model = module()
        populate(model)
        releaser = TestStateReleaser()
        releaser.on_test_end(FakeTrainer(), model)
        releaser.on_test_end(FakeTrainer(), model)
        self.assertEqual(model.test_sample_ids, [])


class TestAcceleratorCacheLimiter(unittest.TestCase):
    """The allocator pool must be bounded during a long evaluation pass.

    MPS caches freed blocks per shape instead of returning them. Measured on a
    bare conv over 300 identical batches, `current_allocated_memory` stayed at
    0.0 MB -- nothing retained -- while `driver_allocated_memory` reached
    1118 MB, and one `empty_cache()` returned it to 11 MB. Over the 34 906-batch
    benchmark pass the pool hit the 42.4 GiB watermark and a routine 45 MiB
    convolution failed.
    """

    def setUp(self):
        self.dropped: list[int] = []

    def limiter(self, every: int) -> AcceleratorCacheLimiter:
        limiter = AcceleratorCacheLimiter(every_n_batches=every)
        limiter._drop = lambda: self.dropped.append(1)  # count, do not allocate
        return limiter

    def run_batches(self, limiter: AcceleratorCacheLimiter, count: int) -> None:
        for index in range(count):
            limiter.on_test_batch_end(FakeTrainer(), None, None, None, index)

    def test_it_drops_on_the_configured_cadence(self):
        self.run_batches(self.limiter(100), 1000)
        # Batches 100, 200, ... 900 -- batch 0 is excluded, see below.
        self.assertEqual(len(self.dropped), 9)

    def test_it_does_not_drop_on_the_first_batch(self):
        # Nothing has accumulated yet, and dropping there would only cost a
        # re-allocation.
        self.run_batches(self.limiter(100), 1)
        self.assertEqual(self.dropped, [])

    def test_a_cadence_of_one_drops_every_batch(self):
        self.run_batches(self.limiter(1), 5)
        self.assertEqual(len(self.dropped), 4)

    def test_validation_is_bounded_too(self):
        # The eval-only flow validates over the full unstrided split before
        # testing, which is long enough to grow the pool on its own.
        limiter = self.limiter(50)
        for index in range(200):
            limiter.on_validation_batch_end(FakeTrainer(), None, None, None, index)
        self.assertEqual(len(self.dropped), 3)

    def test_a_nonsense_cadence_is_rejected(self):
        with self.assertRaises(ValueError):
            AcceleratorCacheLimiter(every_n_batches=0)

    def test_dropping_the_cache_is_safe_to_call(self):
        # The real `_drop`, on whatever accelerator this host has -- it must be
        # a no-op rather than an error when none is available.
        AcceleratorCacheLimiter()._drop()


class TestThresholdCache(unittest.TestCase):
    """The fitted operating point must survive between runs.

    Fitting it costs a full validation pass -- 21 minutes on the frame-wise
    benchmark -- and produces exactly one number. A test pass that dies partway
    should be retryable without paying for that again.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def report(self, **kwargs) -> EventReport:
        return EventReport(self.tmp, fps=25.0, **kwargs)

    def test_a_fitted_point_is_written(self):
        report = self.report()
        report.fitted_threshold = 0.35
        report._save_threshold(0.72)
        self.assertTrue(report.threshold_path.is_file())

    def test_it_round_trips(self):
        first = self.report()
        first.fitted_threshold = 0.35
        first._save_threshold(0.72)

        second = self.report()
        self.assertEqual(second.load_cached_threshold(), 0.35)
        self.assertEqual(second.operating_point, 0.35)

    def test_no_cache_returns_none(self):
        self.assertIsNone(self.report().load_cached_threshold())

    def test_a_cache_for_another_criterion_is_refused(self):
        # A threshold fitted against `iou50` describes a different notion of
        # "detected" than one fitted against `any`; reusing it silently would
        # report a number no run could reproduce.
        first = self.report(tuning_criterion="iou50")
        first.fitted_threshold = 0.35
        first._save_threshold(0.72)

        second = self.report(tuning_criterion="any")
        self.assertIsNone(second.load_cached_threshold())

    def test_a_cache_for_another_target_is_refused(self):
        first = self.report(target=EYE_STATE)
        first.fitted_threshold = 0.35
        first._save_threshold(0.72)

        second = self.report(target=BLINK_PRESENCE)
        self.assertIsNone(second.load_cached_threshold())

    def test_a_corrupt_cache_falls_back_to_fitting(self):
        report = self.report()
        report.threshold_path.write_text("not json{")
        self.assertIsNone(report.load_cached_threshold())

    def test_a_refused_cache_leaves_the_fallback_threshold(self):
        report = self.report(tuning_criterion="any")
        other = self.report(tuning_criterion="iou50")
        other.fitted_threshold = 0.35
        other._save_threshold(0.72)

        report.load_cached_threshold()
        self.assertEqual(report.operating_point, 0.5)

    def test_fitting_writes_the_cache(self):
        # The end-to-end path: a real validation epoch must leave a cache behind
        # without anyone calling `_save_threshold` explicitly.
        model = module()
        starts = [0, 2, 4]
        blink = torch.zeros(len(starts), TIME_DIM)
        blink[0, 1:3] = 1.0
        for name in model.target_names:
            model.valid_metrics.update(
                name,
                blink.clone(),
                blink.clone(),
                torch.ones(len(starts), TIME_DIM, dtype=torch.bool),
            )
        model.valid_sample_ids = [f"talking|{start:06d}|left" for start in starts]

        report = self.report()
        report.on_validation_epoch_end(FakeTrainer(), model)
        if report.fitted_threshold is not None:
            self.assertTrue(report.threshold_path.is_file())


class TestTestCheckpointer(unittest.TestCase):
    """A killed test pass must resume without changing the reported numbers.

    The claim is exactness, not approximation: the test loader is deterministic
    (`shuffle=False`, no sampler, `drop_last=False`) and every metric is an
    order-independent reduction over per-position values, so restored
    predictions score identically to recomputed ones. These tests are what pin
    that claim down.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def scored(self, model: BlinkLightningModule) -> dict[str, float]:
        return {k: float(v) for k, v in model.test_metrics.compute().items()}

    def test_a_finished_pass_leaves_no_state_behind(self):
        # Directly answers "is the disk space released": clear() removes it.
        model = module()
        populate(model)
        checkpointer = TestCheckpointer(self.tmp, enabled=True)
        checkpointer.save(model, batches=4)
        self.assertTrue(checkpointer.state_path.is_file())
        checkpointer.clear()
        self.assertFalse(checkpointer.state_path.exists())

    def test_clearing_a_missing_state_is_harmless(self):
        TestCheckpointer(self.tmp, enabled=True).clear()

    def test_it_writes_nothing_when_disabled(self):
        model = module()
        populate(model)
        checkpointer = TestCheckpointer(self.tmp, enabled=False)
        checkpointer.on_test_batch_end(FakeTrainer(), model, None, None, 2000)
        self.assertFalse(checkpointer.state_path.exists())

    def test_restoring_reproduces_the_metrics_exactly(self):
        # The heart of it: score a pass, checkpoint it, restore into a fresh
        # module, and the computed metrics must be identical.
        original = module()
        populate(original)
        expected = self.scored(original)

        checkpointer = TestCheckpointer(self.tmp, enabled=True)
        checkpointer.save(original, batches=7)

        resumed = module()
        TestCheckpointer(self.tmp, enabled=True).on_test_start(FakeTrainer(), resumed)

        self.assertEqual(self.scored(resumed), expected)

    def test_it_reports_how_far_the_pass_got(self):
        model = module()
        populate(model)
        TestCheckpointer(self.tmp, enabled=True).save(model, batches=7)

        resumed = module()
        checkpointer = TestCheckpointer(self.tmp, enabled=True)
        checkpointer.on_test_start(FakeTrainer(), resumed)
        self.assertEqual(checkpointer.skip_batches, 7)
        self.assertEqual(resumed.skip_test_batches, 7)

    def test_the_sample_provenance_survives(self):
        model = module()
        populate(model, ["talkingface", "rn30"])
        TestCheckpointer(self.tmp, enabled=True).save(model, batches=2)

        resumed = module()
        TestCheckpointer(self.tmp, enabled=True).on_test_start(FakeTrainer(), resumed)
        self.assertEqual(resumed.test_datasets, model.test_datasets)
        self.assertEqual(resumed.test_sample_ids, model.test_sample_ids)

    def test_state_for_other_targets_is_refused(self):
        # Merging predictions from a differently-scoped run would corrupt the
        # benchmark in a way nothing downstream could detect.
        model = module()
        populate(model)
        TestCheckpointer(self.tmp, enabled=True).save(model, batches=3)
        state = torch.load(
            TestCheckpointer(self.tmp).state_path, map_location="cpu", weights_only=False
        )
        state["key"] = "something|else"
        torch.save(state, TestCheckpointer(self.tmp).state_path)

        resumed = module()
        checkpointer = TestCheckpointer(self.tmp, enabled=True)
        checkpointer.on_test_start(FakeTrainer(), resumed)
        self.assertEqual(checkpointer.skip_batches, 0)

    def test_a_corrupt_state_falls_back_to_a_full_pass(self):
        TestCheckpointer(self.tmp).state_path.write_bytes(b"not a torch file")
        resumed = module()
        checkpointer = TestCheckpointer(self.tmp, enabled=True)
        checkpointer.on_test_start(FakeTrainer(), resumed)
        self.assertEqual(checkpointer.skip_batches, 0)
        self.assertEqual(resumed.skip_test_batches, 0)

    def test_no_state_means_a_full_pass(self):
        resumed = module()
        checkpointer = TestCheckpointer(self.tmp, enabled=True)
        checkpointer.on_test_start(FakeTrainer(), resumed)
        self.assertEqual(resumed.skip_test_batches, 0)

    def test_a_nonsense_cadence_is_rejected(self):
        with self.assertRaises(ValueError):
            TestCheckpointer(self.tmp, every_n_batches=0)

    def test_the_write_is_atomic(self):
        # Renamed into place, so a kill mid-write cannot leave a truncated file
        # that a later run would trust.
        model = module()
        populate(model)
        checkpointer = TestCheckpointer(self.tmp, enabled=True)
        checkpointer.save(model, batches=4)
        self.assertFalse(checkpointer.state_path.with_suffix(".tmp").exists())


class TestDualThresholdReporting(unittest.TestCase):
    """Both the universal and the tuned operating point must be reported.

    The universal point is fitted once on the training corpora's pooled
    validation split and applied everywhere -- the deployable number, and the
    only one a corpus with no validation split can have. The tuned point is that
    corpus's own best. The gap between them is the cost of tuning, and hiding
    either would misrepresent the benchmark.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def report_with_recordings(self) -> dict:
        model = module()
        starts = [0, 2, 4]
        blink = torch.zeros(len(starts), TIME_DIM)
        blink[0, 1:3] = 1.0
        for name in model.target_names:
            model.test_metrics.update(
                name,
                blink.clone(),
                blink.clone(),
                torch.ones(len(starts), TIME_DIM, dtype=torch.bool),
            )
        model.test_sample_ids = [f"talking|{s:06d}|left" for s in starts]
        model.test_datasets = ["talkingface"] * len(starts)

        report = EventReport(self.tmp, fps=25.0)
        report.on_test_epoch_end(FakeTrainer(), model)
        return json.loads((self.tmp / "test_events.json").read_text())

    def test_the_universal_threshold_is_recorded(self):
        corpus = self.report_with_recordings()["_corpus"]
        self.assertIn("event/threshold", corpus)

    def test_a_tuned_score_is_reported_per_criterion(self):
        corpus = self.report_with_recordings()["_corpus"]
        tuned = [k for k in corpus if k.startswith("tuned/") and k.endswith("/f1")]
        self.assertTrue(tuned, "expected a tuned F1 per criterion")

    def test_the_tuned_point_is_never_worse_than_the_universal_one(self):
        # It is read off the same curve by argmax, so by construction it cannot
        # lose. If it ever does, the curve and the applied threshold disagree.
        corpus = self.report_with_recordings()["_corpus"]
        for criterion in CRITERIA:
            tuned = corpus.get(f"tuned/{criterion}/f1")
            applied = corpus.get(f"event/{criterion}/f1")
            if tuned is not None and applied is not None:
                self.assertGreaterEqual(round(tuned, 6), round(applied, 6), criterion)

    def test_the_tuned_threshold_is_a_probability(self):
        corpus = self.report_with_recordings()["_corpus"]
        for key, value in corpus.items():
            if key.startswith("tuned/") and key.endswith("/threshold"):
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)


class TestBoundaryHitWarning(unittest.TestCase):
    """A threshold fitted at the edge of the sweep must announce itself.

    Landing on the first or last swept value means the search ran out of range
    or the objective is flat -- not that an optimum was found. Measured on RN15
    the fitted point was the sweep floor at a validation F1 of 0.14, and test
    lost 7 points of event F1 to the miscalibration. It was silent.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def fit_on(self, model: BlinkLightningModule) -> EventReport:
        report = EventReport(self.tmp, fps=25.0)
        report.on_validation_epoch_end(FakeTrainer(), model)
        return report

    def model_with_validation(self) -> BlinkLightningModule:
        model = module()
        starts = [0, 2, 4]
        blink = torch.zeros(len(starts), TIME_DIM)
        blink[0, 1:3] = 1.0
        for name in model.target_names:
            model.valid_metrics.update(
                name,
                blink.clone(),
                blink.clone(),
                torch.ones(len(starts), TIME_DIM, dtype=torch.bool),
            )
        model.valid_sample_ids = [f"talking|{s:06d}|left" for s in starts]
        return model

    def test_the_sweep_is_fine_enough_to_tell_them_apart(self):
        # A 0.05-step sweep cannot distinguish "the optimum is 0.03" from "the
        # search ran out of range at 0.05". This is what makes the warning
        # meaningful rather than noise.
        self.assertLessEqual(float(np.diff(DEFAULT_THRESHOLDS).max()), 0.011)

    def test_the_sweep_reaches_below_the_old_floor(self):
        self.assertLess(float(DEFAULT_THRESHOLDS.min()), 0.05)

    def test_a_fit_at_the_floor_warns(self):
        # Driven through the real fitting path with a curve whose argmax is the
        # first swept point, which is exactly the RN15/RN30 failure.
        report = EventReport(self.tmp, fps=25.0)
        thresholds = DEFAULT_THRESHOLDS
        f1 = np.linspace(0.9, 0.1, thresholds.size)  # best at index 0
        with self.assertLogs("blinklinmult.train.callbacks", level="WARNING") as captured:
            report._warn_on_boundary({"thresholds": thresholds, "f1": f1}, int(f1.argmax()))
        self.assertTrue(any("boundary hit" in line for line in captured.output))

    def test_a_fit_at_the_ceiling_warns(self):
        report = EventReport(self.tmp, fps=25.0)
        thresholds = DEFAULT_THRESHOLDS
        f1 = np.linspace(0.1, 0.9, thresholds.size)  # best at the last index
        with self.assertLogs("blinklinmult.train.callbacks", level="WARNING") as captured:
            report._warn_on_boundary({"thresholds": thresholds, "f1": f1}, int(f1.argmax()))
        self.assertTrue(any("boundary hit" in line for line in captured.output))

    def test_an_interior_fit_is_silent(self):
        # The other half of the claim: it must not cry wolf on a genuine fit.
        report = EventReport(self.tmp, fps=25.0)
        thresholds = DEFAULT_THRESHOLDS
        f1 = np.concatenate(
            [
                np.linspace(0.1, 0.9, thresholds.size // 2),
                np.linspace(0.9, 0.1, thresholds.size - thresholds.size // 2),
            ]
        )
        with self.assertNoLogs("blinklinmult.train.callbacks", level="WARNING"):
            report._warn_on_boundary({"thresholds": thresholds, "f1": f1}, int(f1.argmax()))


class TestThresholdSelection(unittest.TestCase):
    """The operating point is chosen from a smoothed curve, not a raw argmax.

    Validation yields very few events after rasterising -- a few dozen on RN15
    -- so a plain argmax lands wherever the noise peaks. Measured on the
    frame-wise benchmark it selected 0.01 and 0.06 while the same corpora's test
    optima were 0.51 and 0.65, costing 10 and 20 points of event F1.
    """

    def curve(self, f1) -> dict:
        f1 = np.asarray(f1, dtype=np.float64)
        return {"f1": f1, "thresholds": np.linspace(0.01, 0.99, f1.size)}

    def test_a_broad_maximum_beats_a_narrow_spike(self):
        # The whole point: a lone tall threshold is noise; a wide plateau is a
        # region that will survive the move to test.
        f1 = np.full(40, 0.1)
        f1[2] = 0.95  # a single lucky threshold
        f1[20:30] = 0.7  # a broad, genuine optimum
        chosen = EventReport._select_threshold(self.curve(f1))
        self.assertGreaterEqual(chosen, 20)
        self.assertLess(chosen, 30)

    def test_a_clean_peak_is_still_found(self):
        # It must not smear away a real optimum.
        f1 = np.concatenate([np.linspace(0.1, 0.9, 20), np.linspace(0.9, 0.1, 20)])
        chosen = EventReport._select_threshold(self.curve(f1))
        self.assertGreater(chosen, 14)
        self.assertLess(chosen, 26)

    def test_a_short_curve_falls_back_to_argmax(self):
        f1 = np.asarray([0.1, 0.9, 0.2])
        self.assertEqual(EventReport._select_threshold(self.curve(f1)), 1)

    def test_it_returns_a_valid_index(self):
        f1 = np.random.default_rng(0).random(50)
        chosen = EventReport._select_threshold(self.curve(f1))
        self.assertGreaterEqual(chosen, 0)
        self.assertLess(chosen, f1.size)

    def test_a_flat_curve_does_not_crash(self):
        self.assertIsInstance(EventReport._select_threshold(self.curve(np.full(30, 0.4))), int)


class TestSharedThreshold(unittest.TestCase):
    """One universal operating point must be shareable across separate runs.

    A per-corpus evaluation runs each corpus in its own process. Without a
    shared cache each fits privately and the corpora end up scored on different
    scales -- measured at 0.01, 0.06, and two silent 0.50 fallbacks in a single
    split run. Pointing every run at one directory is what makes the "universal"
    column actually universal.
    """

    def setUp(self):
        self.shared = Path(tempfile.mkdtemp())
        self.run_a = Path(tempfile.mkdtemp())
        self.run_b = Path(tempfile.mkdtemp())

    def test_the_cache_lands_in_the_shared_directory(self):
        report = EventReport(self.run_a, fps=25.0, shared_threshold_dir=self.shared)
        report.fitted_threshold = 0.42
        report._save_threshold(0.8)
        self.assertTrue((self.shared / "event_threshold.json").is_file())
        self.assertFalse((self.run_a / "event_threshold.json").exists())

    def test_a_second_run_reads_the_first_ones_value(self):
        first = EventReport(self.run_a, fps=25.0, shared_threshold_dir=self.shared)
        first.fitted_threshold = 0.42
        first._save_threshold(0.8)

        second = EventReport(self.run_b, fps=25.0, shared_threshold_dir=self.shared)
        self.assertEqual(second.load_cached_threshold(), 0.42)
        self.assertEqual(second.operating_point, 0.42)

    def test_every_corpus_lands_on_the_same_point(self):
        # The property that matters: without this the corpora are not
        # comparable, whatever the report calls the column.
        source = EventReport(self.run_a, fps=25.0, shared_threshold_dir=self.shared)
        source.fitted_threshold = 0.37
        source._save_threshold(0.7)

        points = []
        for _ in range(3):
            corpus = EventReport(
                Path(tempfile.mkdtemp()), fps=25.0, shared_threshold_dir=self.shared
            )
            corpus.load_cached_threshold()
            points.append(corpus.operating_point)
        self.assertEqual(len(set(points)), 1)

    def test_without_a_shared_directory_the_cache_stays_private(self):
        report = EventReport(self.run_a, fps=25.0)
        report.fitted_threshold = 0.42
        report._save_threshold(0.8)
        self.assertTrue((self.run_a / "event_threshold.json").is_file())
        self.assertFalse((self.shared / "event_threshold.json").exists())

    def test_a_corpus_with_no_cache_keeps_the_fallback(self):
        # An empty shared directory must not silently look like a fitted point.
        corpus = EventReport(self.run_b, fps=25.0, shared_threshold_dir=self.shared)
        self.assertIsNone(corpus.load_cached_threshold())
        self.assertEqual(corpus.operating_point, 0.5)

    def test_a_criterion_mismatch_is_still_refused(self):
        # Sharing must not weaken the guard: a threshold fitted for another
        # criterion describes a different notion of "detected".
        source = EventReport(
            self.run_a, fps=25.0, tuning_criterion="iou50", shared_threshold_dir=self.shared
        )
        source.fitted_threshold = 0.42
        source._save_threshold(0.8)

        other = EventReport(
            self.run_b, fps=25.0, tuning_criterion="any", shared_threshold_dir=self.shared
        )
        self.assertIsNone(other.load_cached_threshold())


class TestLoadSignals(unittest.TestCase):
    """The signal archive names its target, and the reader enforces it.

    The bug this guards against is silent rather than loud: the frame-wise
    event report scores ``blink_presence``, whose intervals are 3-4x wider than
    closure, so reading its ``truth`` arrays as ``eye_state`` fabricates false
    positives and reports a plausible wrong number.
    """

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def _write(self, target, **arrays):
        path = self.directory / "test_signals.npz"
        payload = dict(arrays)
        if target is not None:
            payload[SIGNALS_TARGET_KEY] = np.asarray(target)
        np.savez_compressed(path, **payload)
        return path

    def test_it_reads_back_recordings_for_the_matching_target(self):
        path = self._write(
            BLINK_PRESENCE,
            **{
                "rec_a/signal": np.array([0.1, 0.9], dtype=np.float32),
                "rec_a/truth": np.array([0.0, 1.0], dtype=np.float32),
                "rec_a/mask": np.array([True, True]),
            },
        )
        recordings = load_signals(path, BLINK_PRESENCE)
        self.assertEqual(set(recordings), {"rec_a"})
        self.assertEqual(set(recordings["rec_a"]), {"signal", "truth", "mask"})
        np.testing.assert_allclose(recordings["rec_a"]["signal"], [0.1, 0.9], rtol=1e-6)

    def test_it_refuses_a_different_target(self):
        path = self._write(
            BLINK_PRESENCE,
            **{
                "rec_a/signal": np.array([0.5], dtype=np.float32),
                "rec_a/truth": np.array([1.0], dtype=np.float32),
                "rec_a/mask": np.array([True]),
            },
        )
        with self.assertRaises(ValueError) as caught:
            load_signals(path, EYE_STATE)
        self.assertIn(BLINK_PRESENCE, str(caught.exception))

    def test_it_refuses_an_archive_with_no_recorded_target(self):
        path = self._write(
            None,
            **{
                "rec_a/signal": np.array([0.5], dtype=np.float32),
                "rec_a/truth": np.array([1.0], dtype=np.float32),
                "rec_a/mask": np.array([True]),
            },
        )
        with self.assertRaises(ValueError) as caught:
            load_signals(path, EYE_STATE)
        self.assertIn(SIGNALS_TARGET_KEY, str(caught.exception))


class TestHysteresisFitting(unittest.TestCase):
    """The operating point is a (high, low) pair, fitted and cached as one."""

    def setUp(self):
        self.directory = Path(tempfile.mkdtemp())

    def test_the_default_search_includes_the_single_threshold(self):
        """`None` must stay in the search so hysteresis has to earn its place."""
        report = EventReport(self.directory, fps=25.0)
        self.assertIn(None, report.low_ratios)
        self.assertEqual(report.low_ratios[0], None)

    def test_no_fitted_ratio_means_no_low_threshold(self):
        report = EventReport(self.directory, fps=25.0)
        report.fitted_threshold = 0.4
        report.fitted_low_ratio = None
        self.assertIsNone(report.low_operating_point)

    def test_the_low_point_tracks_the_high_one(self):
        report = EventReport(self.directory, fps=25.0)
        report.fitted_threshold = 0.75
        report.fitted_low_ratio = 1.0 / 3.0
        self.assertAlmostEqual(report.low_operating_point, 0.25)

    def test_the_ratio_survives_a_cache_round_trip(self):
        report = EventReport(self.directory, fps=25.0)
        report.fitted_threshold = 0.75
        report.fitted_low_ratio = 0.25
        report._save_threshold(0.61)

        restored = EventReport(self.directory, fps=25.0)
        self.assertEqual(restored.load_cached_threshold(), 0.75)
        self.assertEqual(restored.fitted_low_ratio, 0.25)
        self.assertAlmostEqual(restored.low_operating_point, 0.1875)

    def test_the_cache_records_the_absolute_low_threshold(self):
        report = EventReport(self.directory, fps=25.0)
        report.fitted_threshold = 0.75
        report.fitted_low_ratio = 1.0 / 3.0
        report._save_threshold(0.59)
        cached = json.loads((self.directory / "event_threshold.json").read_text())
        self.assertAlmostEqual(cached["low_threshold"], 0.25)

    def test_a_single_threshold_caches_a_null_low_point(self):
        report = EventReport(self.directory, fps=25.0)
        report.fitted_threshold = 0.5
        report.fitted_low_ratio = None
        report._save_threshold(0.4)
        cached = json.loads((self.directory / "event_threshold.json").read_text())
        self.assertIsNone(cached["low_ratio"])
        self.assertIsNone(cached["low_threshold"])


class TestCarrierCorpusIsDropped(unittest.TestCase):
    """An eval-only corpus is scored through a carrier, which must not be scored.

    Evaluating a corpus that supplies no trained target still needs a `datasets`
    entry to build the prediction head, and the Makefile uses CEW. Those stills
    were being counted as *recordings* in the event report: 366 of them
    contributed 192 false positives and 0 true positives to every eval-only run,
    dragging TalkingFace from a true F1 of 0.9573 down to a reported 0.3625.
    """

    def setUp(self):
        # `carrier_only` is off by default: dropping the minority is right only
        # when a carrier supplies the head, and wrong on a joint split where the
        # largest corpus is simply the largest.
        self.report = EventReport(Path(tempfile.mkdtemp()), fps=25.0, carrier_only=True)
        self.joint = EventReport(Path(tempfile.mkdtemp()), fps=25.0)

    def _ids(self, spec):
        ids, datasets = [], []
        for name, video, count in spec:
            for index in range(count):
                ids.append(f"{video}|{index * 4:06d}|left")
                datasets.append(name)
        return ids, datasets

    def test_the_minority_corpus_is_dropped(self):
        ids, datasets = self._ids([("talkingface", "talking", 8), ("cew", "still", 3)])
        values = torch.rand(len(ids) * 4)
        recordings = self.report._by_recording(ids, values, values, 4, datasets)
        self.assertEqual(set(recordings), {"talking"})

    def test_everything_is_kept_when_one_corpus_is_present(self):
        ids, datasets = self._ids([("rn30", "test_1", 5)])
        values = torch.rand(len(ids) * 4)
        recordings = self.report._by_recording(ids, values, values, 4, datasets)
        self.assertEqual(set(recordings), {"test_1"})

    def test_a_joint_split_keeps_every_corpus(self):
        """Measured: taking the majority of a joint split discarded RN,
        HUST-LEBW and TalkingFace because MPEblink held 51 083 of 64 027
        samples, and the report then scored a corpus annotating no closure."""
        ids, datasets = self._ids([("mpeblink", "mpe", 9), ("rn30", "test_1", 3)])
        values = torch.rand(len(ids) * 4)
        recordings = self.joint._by_recording(ids, values, values, 4, datasets)
        self.assertEqual(set(recordings), {"mpe", "test_1"})

    def test_none_keeps_every_sample(self):
        """The old behaviour, for callers that score a single corpus."""
        ids, datasets = self._ids([("talkingface", "talking", 8), ("cew", "still", 3)])
        values = torch.rand(len(ids) * 4)
        recordings = self.report._by_recording(ids, values, values, 4, None)
        self.assertEqual(set(recordings), {"talking", "still"})

    def test_a_length_mismatch_is_ignored_rather_than_misaligning(self):
        """A wrong-length list would drop the wrong samples silently."""
        ids, datasets = self._ids([("talkingface", "talking", 4), ("cew", "still", 2)])
        values = torch.rand(len(ids) * 4)
        recordings = self.report._by_recording(ids, values, values, 4, datasets[:2])
        self.assertEqual(set(recordings), {"talking", "still"})


class TestCacheLimiterCoversTraining(unittest.TestCase):
    """The MPS pool grows during *training* too, not only evaluation.

    Measured 2026-08-30: a sequence-model run reached 20 GB of wired memory,
    climbing ~8 GB per 20 s, because the limiter ran only on validation and test
    batches. Wired memory cannot be paged out, so the machine reached 0.06 GB
    free while `ps` showed the process holding 0.2 GB.
    """

    def test_it_hooks_training_batches(self):
        self.assertTrue(hasattr(AcceleratorCacheLimiter, "on_train_batch_end"))

    def test_every_stage_shares_one_interval(self):
        limiter = AcceleratorCacheLimiter(every_n_batches=7)
        self.assertEqual(limiter.every_n_batches, 7)

    def test_it_drops_on_the_interval_and_not_between(self):
        calls: list[int] = []
        limiter = AcceleratorCacheLimiter(every_n_batches=5)
        limiter._drop = lambda: calls.append(1)  # noqa: SLF001
        for index in range(11):
            limiter.on_train_batch_end(None, None, None, None, index)
        # 5 and 10 only -- batch 0 is skipped so a run does not drop immediately.
        self.assertEqual(len(calls), 2)

    def test_it_rejects_a_zero_interval(self):
        with self.assertRaises(ValueError):
            AcceleratorCacheLimiter(every_n_batches=0)
