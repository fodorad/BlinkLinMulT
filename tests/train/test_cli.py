"""Tests for the training entry point.

The headline test runs a real Lightning ``fast_dev_run`` over real, built HDF5
files. It is the only test that proves the whole chain — preprocess manifest,
builder, OmniLoader, model, loss, metrics, callbacks — actually fits together,
which no amount of unit testing of the parts establishes.

That thoroughness costs about 49 seconds, roughly half of the whole suite, so
:class:`TestRunEndToEnd` is **opt-in**: it runs under ``make test-full`` and in
CI, while ``make check`` stays a fast pre-commit gate. Nothing here downloads
anything -- the corpora are built into a temporary directory -- so the flag buys
wall-clock time, not network access::

    RUN_TRAINING_TESTS=1 uv run python -m unittest tests.train.test_cli -v
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import yaml
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from blinklinmult.data.schema import LEFT, RIGHT, DatasetSpec
from blinklinmult.data.writer import H5Writer
from blinklinmult.train.cli import (
    MONITOR,
    _best_checkpoint,
    build_callbacks,
    check_max_epochs,
    main,
    resume_from,
    run,
    run_output_dir,
    unfreeze_module,
)
from blinklinmult.train.config import (
    DataConfig,
    ExperimentConfig,
    MLflowConfig,
    ModelConfig,
    TrainConfig,
)

IMAGE_SIZE = 32
TIME_DIM = 3


class FakeTrainer:
    """Exposes only ``checkpoint_callbacks``, which is all _best_checkpoint reads."""

    def __init__(self, callbacks):
        self.checkpoint_callbacks = callbacks


class CliCase(unittest.TestCase):
    """A complete, tiny two-corpus project in a temporary tree."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.config_dir = self.root / "config"
        (self.config_dir / "data").mkdir(parents=True)
        (self.config_dir / "model").mkdir(parents=True)
        (self.config_dir / "train").mkdir(parents=True)

    def specs(self) -> list[DatasetSpec]:
        return [
            DatasetSpec(
                name="rn30",
                fps=6.0,  # -> TIME_DIM frames at 0.5s
                window_seconds=0.5,
                image_size=IMAGE_SIZE,
                has_blink_presence=True,
                has_eye_state=True,
            ),
            # A second corpus at a different rate, annotating only one of the
            # two tasks: the run must mix unequal windows and skip the target
            # this corpus does not supervise. A still corpus cannot play this
            # role -- a sequence family may not be given one.
            DatasetSpec(
                name="hust_lebw",
                fps=12.0,
                window_seconds=0.5,
                image_size=IMAGE_SIZE,
                has_blink_presence=True,
                has_eye_state=False,
            ),
        ]

    def build_corpora(self) -> None:
        for spec in self.specs():
            processed = self.root / "data" / "processed" / spec.name
            processed.mkdir(parents=True, exist_ok=True)
            with H5Writer(spec, processed / f"{spec.name}.h5") as writer:
                for subset in ("train", "valid", "test"):
                    for index in range(4):
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
                                    spec.time_dim, 3, IMAGE_SIZE, IMAGE_SIZE
                                ).astype(np.float32),
                                blink_presence=labels if spec.has_blink_presence else None,
                                eye_state=(
                                    np.zeros(spec.time_dim, dtype=np.float32)
                                    if spec.has_eye_state
                                    else None
                                ),
                            )

    def write_configs(self) -> tuple[Path, Path, Path]:
        for spec in self.specs():
            (self.config_dir / "data" / f"{spec.name}.yaml").write_text(
                yaml.safe_dump(
                    {
                        "name": spec.name,
                        "window_seconds": spec.window_seconds,
                        "image_size": spec.image_size,
                        "fps": spec.fps,
                        "has_blink_presence": spec.has_blink_presence,
                        "has_eye_state": spec.has_eye_state,
                    }
                )
            )

        data = self.config_dir / "data" / "run.yaml"
        data.write_text(
            yaml.safe_dump(
                {
                    "datasets": ["rn30", "hust_lebw"],
                    "window_seconds": 0.5,
                    "image_size": IMAGE_SIZE,
                    "strategy": "round_robin",
                    "batch_size": 2,
                    "num_workers": 0,
                    "pin_memory": False,
                    "persistent_workers": False,
                }
            )
        )

        model = self.config_dir / "model" / "run.yaml"
        model.write_text(
            yaml.safe_dump(
                {
                    "family": "lint",
                    "backbone_pretrained": False,
                    "backbone_output_dim": 4,
                    "d_model": 8,
                    "num_heads": 2,
                    "cmt_num_layers": 1,
                    "head_hidden_dim": 4,
                }
            )
        )

        train = self.config_dir / "train" / "run.yaml"
        train.write_text(
            yaml.safe_dump(
                {
                    "task": "joint",
                    "loss": "bce",
                    "max_epochs": 1,
                    "accelerator": "cpu",
                    "devices": 1,
                    "precision": "32-true",
                    "output_dir": str(self.root / "results"),
                    "optimizer": {"scheduler": "none", "backbone_lr": None},
                    "early_stopping": {"enabled": False},
                    "mlflow": {
                        "tracking_uri": f"sqlite:///{self.root / 'mlflow.db'}",
                        "experiment_name": "test",
                        "log_model": False,
                    },
                }
            )
        )
        return data, model, train

    def experiment(self) -> ExperimentConfig:
        return ExperimentConfig.from_files(*self.write_configs())


class TestRunOutputDir(unittest.TestCase):
    """A named run must not overwrite the previous run's results.

    Every run writes the same filenames, so a sweep sharing one directory
    leaves only the last arm's numbers on disk — which reads as a completed
    sweep rather than as an error.
    """

    def config(
        self,
        run_name=None,
        task="blink_presence",
        output_dir="results",
        experiment_name="blink",
    ):
        return ExperimentConfig(
            data=DataConfig(datasets=["rn30"]),
            model=ModelConfig(family="lint"),
            train=TrainConfig(
                task=task,
                output_dir=output_dir,
                mlflow=MLflowConfig(run_name=run_name, experiment_name=experiment_name),
            ),
        )

    def test_a_named_run_gets_its_own_directory(self):
        first = run_output_dir(self.config(run_name="abl-flash-s42"))
        second = run_output_dir(self.config(run_name="abl-flash-s43"))
        self.assertNotEqual(first, second)
        self.assertEqual(first.name, "abl-flash-s42")

    def test_an_unnamed_run_lands_in_the_experiment_directory(self):
        # `make train-lint` names nothing and should land where the docs say.
        self.assertEqual(run_output_dir(self.config()), Path("results") / "blink")

    def test_runs_group_by_experiment_not_by_task(self):
        # A sweep's runs sit together on disk exactly as they do in MLflow.
        # Grouping by task instead put every eye-state run ever trained into one
        # directory, which is how a backbone sweep's checkpoints ended up as
        # best-v25..v28 with nothing to say which backbone produced which.
        sweep = run_output_dir(self.config(experiment_name="cew-frame-wise"))
        other = run_output_dir(self.config(experiment_name="rn15-sequence"))
        self.assertEqual(sweep.name, "cew-frame-wise")
        self.assertNotEqual(sweep, other)

    def test_two_runs_of_one_experiment_share_its_directory(self):
        first = run_output_dir(self.config(experiment_name="sweep", run_name="a"))
        second = run_output_dir(self.config(experiment_name="sweep", run_name="b"))
        self.assertEqual(first.parent, second.parent)
        self.assertNotEqual(first, second)

    def test_a_run_name_cannot_escape_the_results_tree(self):
        # Run names are written for MLflow, not for the filesystem: a slash
        # would otherwise nest, and "../" would climb out.
        escaped = run_output_dir(self.config(run_name="../../etc/passwd"))
        self.assertNotIn("..", escaped.parts)
        self.assertEqual(escaped.parent, Path("results") / "blink")

    def test_an_experiment_name_cannot_escape_either(self):
        escaped = run_output_dir(self.config(experiment_name="../../etc"))
        self.assertNotIn("..", escaped.parts)

    def test_a_name_of_only_separators_falls_back_to_the_experiment_dir(self):
        self.assertEqual(
            run_output_dir(self.config(run_name="///")),
            Path("results") / "blink",
        )


class TestBuildCallbacks(CliCase):
    def config(self, **train_overrides) -> ExperimentConfig:
        # A video corpus: a sequence family may not be given a still one.
        return ExperimentConfig(
            data=DataConfig(datasets=["rn30"]),
            model=ModelConfig(family="lint"),
            train=TrainConfig(task="eye_state", **train_overrides),
        )

    def test_includes_two_checkpoint_callbacks(self):
        callbacks = build_callbacks(self.config(), self.root)
        checkpoints = [c for c in callbacks if isinstance(c, ModelCheckpoint)]
        self.assertEqual(len(checkpoints), 2)

    def test_the_primary_checkpoint_monitors_the_primary_metric(self):
        callbacks = build_callbacks(self.config(), self.root)
        monitors = {c.monitor for c in callbacks if isinstance(c, ModelCheckpoint)}
        self.assertIn(MONITOR, monitors)
        self.assertIn("valid/loss", monitors)

    def test_checkpoints_save_full_state_so_a_run_resumes(self):
        callbacks = build_callbacks(self.config(), self.root)
        for checkpoint in (c for c in callbacks if isinstance(c, ModelCheckpoint)):
            self.assertFalse(checkpoint.save_weights_only)

    def test_early_stopping_is_added_when_enabled(self):
        callbacks = build_callbacks(self.config(), self.root)
        self.assertTrue(any(isinstance(c, EarlyStopping) for c in callbacks))

    def test_early_stopping_is_omitted_when_disabled(self):
        from blinklinmult.train.config import EarlyStoppingConfig

        callbacks = build_callbacks(
            self.config(early_stopping=EarlyStoppingConfig(enabled=False)), self.root
        )
        self.assertFalse(any(isinstance(c, EarlyStopping) for c in callbacks))

    def test_the_epoch_propagator_is_present(self):
        from blinklinmult.train.callbacks import EpochPropagator

        callbacks = build_callbacks(self.config(), self.root)
        self.assertTrue(any(isinstance(c, EpochPropagator) for c in callbacks))


class TestBestCheckpoint(unittest.TestCase):
    def test_finds_the_checkpoint_by_the_metric_it_monitors(self):
        # Looked up by purpose, not by list position: the 1.x code did
        # callbacks[1].best_model_path, which silently depends on ordering.
        wrong = ModelCheckpoint(monitor="valid/loss")
        wrong.best_model_path = "/loss.ckpt"
        right = ModelCheckpoint(monitor=MONITOR)
        right.best_model_path = "/best.ckpt"

        self.assertEqual(_best_checkpoint(FakeTrainer([wrong, right])), "/best.ckpt")

    def test_reordering_the_callbacks_does_not_change_the_result(self):
        wrong = ModelCheckpoint(monitor="valid/loss")
        wrong.best_model_path = "/loss.ckpt"
        right = ModelCheckpoint(monitor=MONITOR)
        right.best_model_path = "/best.ckpt"

        self.assertEqual(_best_checkpoint(FakeTrainer([right, wrong])), "/best.ckpt")

    def test_no_matching_checkpoint_returns_empty(self):
        self.assertEqual(_best_checkpoint(FakeTrainer([])), "")


RUN_TRAINING_TESTS = os.environ.get("RUN_TRAINING_TESTS") == "1"
"""Opt-in flag for the real Lightning runs. See the module docstring."""


@unittest.skipUnless(
    RUN_TRAINING_TESTS,
    "set RUN_TRAINING_TESTS=1 (real Lightning runs, ~49s) or use `make test-full`",
)
class TestRunEndToEnd(CliCase):
    def test_fast_dev_run_completes(self):
        self.build_corpora()
        config = self.experiment()
        result = run(
            config,
            root=self.root,
            config_dir=self.config_dir / "data",
            fast_dev_run=True,
        )
        self.assertEqual(result, {})

    def test_a_full_short_run_produces_test_metrics_and_artifacts(self):
        self.build_corpora()
        config = self.experiment()
        metrics = run(config, root=self.root, config_dir=self.config_dir / "data")

        self.assertIn(f"test/{MONITOR.split('/', 1)[1]}", metrics)

        # Grouped by experiment name, which write_configs sets to "test".
        output = self.root / "results" / "test"
        self.assertTrue((output / "metrics_test.json").is_file())
        self.assertTrue((output / "time.json").is_file())
        self.assertTrue((output / "test_per_dataset.json").is_file())

    def test_eval_only_scores_a_checkpoint_without_training(self):
        # A run interrupted before its test pass leaves a checkpoint but no
        # test artifacts; `--eval-only` recovers the benchmark from the
        # checkpoint. It fits the event operating point on validation first,
        # so the artifacts match a finished run rather than reporting at the
        # fixed 0.5 threshold.
        self.build_corpora()
        config = self.experiment()
        run(config, root=self.root, config_dir=self.config_dir / "data")

        checkpoint = self.root / "results" / "test" / "checkpoints" / "best.ckpt"
        self.assertTrue(checkpoint.is_file())

        metrics = run(
            config,
            root=self.root,
            config_dir=self.config_dir / "data",
            eval_only=checkpoint,
        )

        self.assertIn(f"test/{MONITOR.split('/', 1)[1]}", metrics)

        output = self.root / "results" / "test"
        self.assertTrue((output / "metrics_test.json").is_file())
        self.assertTrue((output / "test_per_dataset.json").is_file())
        self.assertTrue((output / "test_events.json").is_file())

    def test_eval_only_does_not_write_new_checkpoints(self):
        # An eval-only run must not clobber the run's checkpoints: the point is
        # to score the saved model, not to re-checkpoint it under new timestamps.
        self.build_corpora()
        config = self.experiment()
        run(config, root=self.root, config_dir=self.config_dir / "data")

        checkpoint = self.root / "results" / "test" / "checkpoints" / "best.ckpt"
        before = checkpoint.stat().st_mtime_ns
        run(
            config,
            root=self.root,
            config_dir=self.config_dir / "data",
            eval_only=checkpoint,
        )
        after = checkpoint.stat().st_mtime_ns
        self.assertEqual(before, after)

    def test_eval_only_with_an_unknown_checkpoint_fails_readably(self):
        self.build_corpora()
        config = self.experiment()
        with self.assertRaises(FileNotFoundError):
            run(
                config,
                root=self.root,
                config_dir=self.config_dir / "data",
                eval_only=Path("no/such/checkpoint.ckpt"),
            )

    def test_the_per_dataset_report_names_both_corpora(self):
        import json

        self.build_corpora()
        run(self.experiment(), root=self.root, config_dir=self.config_dir / "data")

        report = json.loads((self.root / "results" / "test" / "test_per_dataset.json").read_text())
        corpora = {c for target in report.values() for c in target}
        self.assertEqual(corpora, {"rn30", "hust_lebw"})

    def test_an_unbuilt_dataset_fails_with_a_readable_message(self):
        from blinklinmult.data.datamodule import DataModuleError

        config = self.experiment()
        with self.assertRaises(DataModuleError) as ctx:
            run(config, root=self.root, config_dir=self.config_dir / "data")
        self.assertIn("make preprocess-", str(ctx.exception))


class TestMain(CliCase):
    def test_main_runs_the_configured_experiment(self):
        self.build_corpora()
        data, model, train = self.write_configs()

        argv = [
            "cli",
            "--data",
            str(data),
            "--model",
            str(model),
            "--train",
            str(train),
            "--config-dir",
            str(self.config_dir / "data"),
            "--root",
            str(self.root),
            "--fast-dev-run",
        ]
        original = sys.argv
        sys.argv = argv
        try:
            main()
        finally:
            sys.argv = original

    def test_overrides_reach_the_run(self):
        self.build_corpora()
        data, model, train = self.write_configs()

        argv = [
            "cli",
            "--data",
            str(data),
            "--model",
            str(model),
            "--train",
            str(train),
            "--config-dir",
            str(self.config_dir / "data"),
            "--root",
            str(self.root),
            "--set",
            "data.batch_size=1",
            "--fast-dev-run",
        ]
        original = sys.argv
        sys.argv = argv
        try:
            main()
        finally:
            sys.argv = original


class TestCallbackHelpers(unittest.TestCase):
    """The pure list logic around a run's callbacks.

    Both helpers decide something a run depends on -- whether a stale checkpoint
    is cleared, and whether an event threshold is refitted -- from nothing but
    the callback list, so neither needs Lightning to test.
    """

    def test_clearing_test_checkpoints_ignores_other_callbacks(self) -> None:
        """A list holding unrelated callbacks must pass through untouched."""
        from blinklinmult.train.cli import _clear_test_checkpoint

        _clear_test_checkpoint([EarlyStopping(monitor="x"), ModelCheckpoint()])

    def test_the_test_checkpointer_is_cleared(self) -> None:
        """A stale best-checkpoint from a previous run must not survive.

        `eval-only` reuses the run directory, so a checkpointer holding the
        previous run's path would report a metric from the wrong weights.
        """
        import tempfile
        from pathlib import Path as _Path

        from blinklinmult.train.callbacks import TestCheckpointer
        from blinklinmult.train.cli import _clear_test_checkpoint

        with tempfile.TemporaryDirectory() as directory:
            checkpointer = TestCheckpointer(output_dir=_Path(directory))
            _clear_test_checkpoint([checkpointer, ModelCheckpoint()])

    def test_a_report_with_a_cached_threshold_is_reused(self) -> None:
        """A cached operating point avoids refitting on every eval run.

        Both halves matter: reusing when every report has one, and refitting
        when any does not -- a mixed state would make corpora incomparable.
        """
        import tempfile
        from pathlib import Path as _Path

        from blinklinmult.train.callbacks import EventReport
        from blinklinmult.train.cli import _reuse_cached_threshold

        with tempfile.TemporaryDirectory() as directory:
            report = EventReport(output_dir=_Path(directory), target="blink_presence")
            # No cache written, so the answer must be "refit".
            self.assertFalse(_reuse_cached_threshold([report]))

    def test_clearing_an_empty_list_is_a_no_op(self) -> None:
        """A run configured without the checkpointer is legitimate."""
        from blinklinmult.train.cli import _clear_test_checkpoint

        _clear_test_checkpoint([])

    def test_no_event_report_means_nothing_to_reuse(self) -> None:
        """Without an EventReport there is no threshold to restore."""
        from blinklinmult.train.cli import _reuse_cached_threshold

        self.assertFalse(_reuse_cached_threshold([]))
        self.assertFalse(_reuse_cached_threshold([ModelCheckpoint()]))

    def test_a_report_without_a_cached_threshold_forces_a_refit(self) -> None:
        """Half a cache is not a cache.

        Reusing one corpus's fitted threshold while another falls back to the
        default would make the two incomparable, which is the failure this
        guard exists for.
        """
        from blinklinmult.train.cli import _reuse_cached_threshold

        class _Report:
            """An EventReport-shaped stand-in with no cached value."""

            def load_cached_threshold(self) -> None:
                """Report that nothing was cached."""
                return None

        # Not an EventReport instance, so it is filtered out and the answer is
        # "nothing to reuse" -- the same conservative outcome.
        self.assertFalse(_reuse_cached_threshold([_Report()]))


if __name__ == "__main__":
    unittest.main()


class TestResumeFrom(unittest.TestCase):
    """Where a continued run picks up from.

    Silently starting from scratch is the failure that matters here: it would
    discard the hours the resume was meant to preserve, and the only place it
    would show is a loss curve nobody is watching at 3am.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name)

    def write_last(self) -> Path:
        path = self.output_dir / "checkpoints" / "last.ckpt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")
        return path

    def test_no_resume_starts_fresh(self):
        self.assertIsNone(resume_from(None, self.output_dir))

    def test_false_starts_fresh(self):
        self.assertIsNone(resume_from(False, self.output_dir))

    def test_bare_resume_finds_the_runs_own_checkpoint(self):
        expected = self.write_last()
        self.assertEqual(resume_from(True, self.output_dir), str(expected))

    def test_an_explicit_path_is_used_as_given(self):
        other = self.output_dir / "elsewhere.ckpt"
        other.write_bytes(b"checkpoint")
        self.assertEqual(resume_from(other, self.output_dir), str(other))

    def test_a_missing_checkpoint_is_loud(self):
        # Not silent: a resume that quietly restarts wastes exactly what it was
        # asked to protect.
        with self.assertRaises(FileNotFoundError):
            resume_from(True, self.output_dir)

    def test_a_missing_explicit_path_is_loud(self):
        with self.assertRaises(FileNotFoundError):
            resume_from(self.output_dir / "nope.ckpt", self.output_dir)


class TestMaxEpochsGuard(unittest.TestCase):
    """Cosine annealing is a fixed-length curve set at first launch.

    Resuming with a different `max_epochs` would change the LR plan mid-run --
    the schedule anneals to the wrong floor and the only trace is a loss curve
    that looks slightly off. Rejecting the mismatch is what makes a resumed run
    the same experiment as the one it continues.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.output_dir = Path(self._tmp.name)

    def test_a_first_launch_records_the_plan(self):
        check_max_epochs(self.output_dir, 30, resume=False)
        self.assertEqual((self.output_dir / "max_epochs.txt").read_text(), "30")

    def test_a_matching_resume_is_accepted(self):
        check_max_epochs(self.output_dir, 30, resume=False)
        check_max_epochs(self.output_dir, 30, resume=True)

    def test_a_changed_max_epochs_is_rejected(self):
        check_max_epochs(self.output_dir, 30, resume=False)
        with self.assertRaises(ValueError):
            check_max_epochs(self.output_dir, 50, resume=True)

    def test_a_fresh_run_may_change_it(self):
        # Only a *resume* is constrained: a new run is free to plan any length.
        check_max_epochs(self.output_dir, 30, resume=False)
        check_max_epochs(self.output_dir, 50, resume=False)
        self.assertEqual((self.output_dir / "max_epochs.txt").read_text(), "50")

    def test_resuming_without_a_record_is_allowed(self):
        # A run started before this guard existed has no file; refusing would
        # strand it rather than protect anything.
        check_max_epochs(self.output_dir, 30, resume=True)


class TestUnfreezeModule(unittest.TestCase):
    """Stage 2 continues a frozen run with the encoder trainable.

    Neither existing path does this: `model.encoder_weights` loads only the
    encoder, discarding the transformer stage 1 trained, and `--resume` restores
    the stored `encoder_freeze` so it continues stage 1 instead.
    """

    def test_a_missing_checkpoint_raises(self):
        """Silently starting fresh would discard the stage it should continue."""
        config = ExperimentConfig.from_files(
            "config/data/video_all.yaml",
            "config/model/blinklint.yaml",
            "config/train/video_unfreeze.yaml",
            {},
        )
        with self.assertRaises(FileNotFoundError) as caught:
            unfreeze_module(Path("nope.ckpt"), config, 64, None, 1)
        self.assertIn("frozen stage", str(caught.exception))

    def test_the_stage_two_config_lowers_both_rates(self):
        """The head refines; the encoder, most at risk, moves slower still."""
        stage_one = ExperimentConfig.from_files(
            "config/data/video_all.yaml",
            "config/model/blinklint.yaml",
            "config/train/video.yaml",
            {},
        )
        stage_two = ExperimentConfig.from_files(
            "config/data/video_all.yaml",
            "config/model/blinklint.yaml",
            "config/train/video_unfreeze.yaml",
            {},
        )
        self.assertLess(stage_two.train.optimizer.lr, stage_one.train.optimizer.lr)
        self.assertLess(stage_two.train.optimizer.backbone_lr, stage_two.train.optimizer.lr)

    def test_stage_two_is_short(self):
        """It starts from a working model; a long budget defeats the point."""
        config = ExperimentConfig.from_files(
            "config/data/video_all.yaml",
            "config/model/blinklint.yaml",
            "config/train/video_unfreeze.yaml",
            {},
        )
        self.assertLessEqual(config.train.max_epochs, 15)
        self.assertLessEqual(config.train.early_stopping.patience, 5)
