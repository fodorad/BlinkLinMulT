"""Tests for the typed run configuration."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

from blinklinmult.data.schema import BLINK_PRESENCE, DATASETS, EYE_STATE, DatasetSpec
from blinklinmult.train.config import (
    ConfigError,
    DataConfig,
    EarlyStoppingConfig,
    ExperimentConfig,
    MLflowConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    apply_override,
    load_yaml,
    parse_override,
)

REPO_CONFIG = Path(__file__).resolve().parents[2] / "config"


class TestDataConfig(unittest.TestCase):
    def test_minimal_config_is_valid(self):
        config = DataConfig(datasets=["cew"])
        self.assertEqual(config.datasets, ["cew"])
        self.assertEqual(config.batch_size, 32)

    def test_empty_dataset_list_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            DataConfig(datasets=[])
        self.assertIn("at least one", str(ctx.exception))

    def test_duplicate_datasets_raise(self):
        with self.assertRaises(ConfigError) as ctx:
            DataConfig(datasets=["cew", "cew"])
        self.assertIn("duplicates", str(ctx.exception))

    def test_unknown_strategy_raises(self):
        with self.assertRaises(ConfigError):
            DataConfig(datasets=["cew"], strategy="magic")

    def test_out_of_range_values_raise(self):
        for kwargs in (
            {"batch_size": 0},
            {"num_workers": -1},
            {"time_dim": 0},
            {"window_seconds": 0.0},
            {"image_size": 0},
            {"cache_size": -1},
            {"still_stride": 0},
            {"still_open_to_closed": 0.0},
            # Still mode serves one frame per sample, so any other shared
            # window length is a contradiction rather than an override.
            {"stills": True, "time_dim": 4},
        ):
            with self.subTest(**kwargs), self.assertRaises(ConfigError):
                DataConfig(datasets=["cew"], **kwargs)

    def test_still_mode_accepts_a_time_dim_of_one(self):
        self.assertTrue(DataConfig(datasets=["cew"], stills=True, time_dim=1).stills)

    def test_unknown_key_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            DataConfig.from_dict({"datasets": ["cew"], "batch_sise": 4})
        self.assertIn("batch_sise", str(ctx.exception))

    def test_resolve_specs_reads_the_repository_declarations(self):
        config = DataConfig(datasets=["rn30", "cew"])
        specs = config.resolve_specs(REPO_CONFIG / "data")
        self.assertEqual([s.name for s in specs], ["rn30", "cew"])
        self.assertTrue(specs[0].has_blink_presence)
        self.assertFalse(specs[1].has_blink_presence)

    def test_resolve_specs_missing_declaration_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            DataConfig(datasets=["mrl"]).resolve_specs(Path("/nonexistent"))
        self.assertIn("mrl", str(ctx.exception))


class TestOptimConfig(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertEqual(OptimConfig().name, "adamw")

    def test_backbone_lr_may_be_none(self):
        self.assertIsNone(OptimConfig(backbone_lr=None).backbone_lr)

    def test_unknown_optimizer_raises(self):
        with self.assertRaises(ConfigError):
            OptimConfig(name="lbfgs")

    def test_unknown_scheduler_raises(self):
        with self.assertRaises(ConfigError):
            OptimConfig(scheduler="exponential")

    def test_non_positive_lr_raises(self):
        with self.assertRaises(ConfigError):
            OptimConfig(lr=0.0)

    def test_non_positive_backbone_lr_raises(self):
        with self.assertRaises(ConfigError):
            OptimConfig(backbone_lr=-1e-4)

    def test_warmup_ratio_of_one_raises(self):
        with self.assertRaises(ConfigError):
            OptimConfig(warmup_ratio=1.0)


class TestModelConfig(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = ModelConfig()
        self.assertEqual(config.family, "linmult")
        self.assertEqual(config.attention_type, "linear")

    def test_unknown_family_raises(self):
        with self.assertRaises(ConfigError):
            ModelConfig(family="rnn")

    def test_unknown_backbone_raises(self):
        with self.assertRaises(ConfigError):
            ModelConfig(backbone="vgg16")

    def test_d_model_not_divisible_by_heads_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            ModelConfig(d_model=30, num_heads=8)
        self.assertIn("divisible", str(ctx.exception))

    def test_freezing_an_untrained_backbone_raises(self):
        # A frozen randomly-initialised backbone can never learn.
        with self.assertRaises(ConfigError) as ctx:
            ModelConfig(backbone_pretrained=False, backbone_freeze=True)
        self.assertIn("never learn", str(ctx.exception))

    def test_legacy_keys_raise_with_a_rename_hint(self):
        legacy = {
            "projected_modality_dim": "d_model",
            "number_of_layers": "cmt_num_layers",
            "n_heads": "num_heads",
            "add_projection_fusion": "add_module_ffn_fusion",
        }
        for old, new in legacy.items():
            with self.subTest(key=old):
                with self.assertRaises(ConfigError) as ctx:
                    ModelConfig.from_dict({old: 4})
                message = str(ctx.exception)
                self.assertIn(old, message)
                self.assertIn(new, message)

    def test_derived_keys_are_rejected_rather_than_accepted(self):
        # These are read from the data, so accepting them would let a config
        # disagree with the dataset it trains on.
        for key in ("input_modality_channels", "input_dim", "output_dim"):
            with self.subTest(key=key), self.assertRaises(ConfigError):
                ModelConfig.from_dict({key: 160})

    def test_unknown_key_raises(self):
        with self.assertRaises(ConfigError):
            ModelConfig.from_dict({"d_modell": 32})


class TestMLflowConfig(unittest.TestCase):
    def test_database_uri_is_accepted(self):
        self.assertEqual(MLflowConfig().tracking_uri, "sqlite:///mlflow.db")

    def test_filesystem_store_raises(self):
        # MLflow 3.x has retired the file store and raises on it.
        with self.assertRaises(ConfigError) as ctx:
            MLflowConfig(tracking_uri="./mlruns")
        self.assertIn("database URI", str(ctx.exception))


class TestEarlyStoppingConfig(unittest.TestCase):
    def test_defaults_monitor_the_primary_metric(self):
        self.assertEqual(EarlyStoppingConfig().monitor, "valid/mean_f1")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ConfigError):
            EarlyStoppingConfig(mode="highest")


class TestTrainConfig(unittest.TestCase):
    def test_targets_default_to_the_task(self):
        self.assertEqual(TrainConfig(task="joint").targets, [BLINK_PRESENCE, EYE_STATE])
        self.assertEqual(TrainConfig(task="eye_state").targets, [EYE_STATE])
        self.assertEqual(TrainConfig(task="blink_presence").targets, [BLINK_PRESENCE])

    def test_unknown_task_raises(self):
        with self.assertRaises(ConfigError):
            TrainConfig(task="regression")

    def test_unknown_loss_raises(self):
        with self.assertRaises(ConfigError):
            TrainConfig(loss="mse")

    def test_unknown_precision_raises(self):
        with self.assertRaises(ConfigError):
            TrainConfig(precision="8-bit")

    def test_targets_contradicting_the_task_raise(self):
        with self.assertRaises(ConfigError) as ctx:
            TrainConfig(task="eye_state", targets=[BLINK_PRESENCE])
        self.assertIn("supervises", str(ctx.exception))

    def test_unknown_target_raises(self):
        with self.assertRaises(ConfigError):
            TrainConfig(task="joint", targets=[BLINK_PRESENCE, "gaze"])

    def test_task_weight_defaults_to_one(self):
        self.assertEqual(TrainConfig(task="joint").weight_for(BLINK_PRESENCE), 1.0)

    def test_task_weights_are_honoured(self):
        config = TrainConfig(task="joint", task_weights={BLINK_PRESENCE: 2.0})
        self.assertEqual(config.weight_for(BLINK_PRESENCE), 2.0)
        self.assertEqual(config.weight_for(EYE_STATE), 1.0)

    def test_weight_for_an_unsupervised_target_raises(self):
        with self.assertRaises(ConfigError) as ctx:
            TrainConfig(task="eye_state", task_weights={BLINK_PRESENCE: 1.0})
        self.assertIn("does not supervise", str(ctx.exception))

    def test_negative_weight_raises(self):
        with self.assertRaises(ConfigError):
            TrainConfig(task="joint", task_weights={EYE_STATE: -1.0})

    def test_nested_sections_are_typed(self):
        config = TrainConfig.from_dict(
            {
                "task": "joint",
                "optimizer": {"lr": 0.01},
                "mlflow": {"experiment_name": "x"},
                "early_stopping": {"patience": 3},
            }
        )
        self.assertIsInstance(config.optimizer, OptimConfig)
        self.assertEqual(config.optimizer.lr, 0.01)
        self.assertEqual(config.mlflow.experiment_name, "x")
        self.assertEqual(config.early_stopping.patience, 3)

    def test_nested_validation_still_applies(self):
        with self.assertRaises(ConfigError):
            TrainConfig.from_dict({"task": "joint", "optimizer": {"name": "lbfgs"}})


class TestExperimentConfig(unittest.TestCase):
    def build(self, **overrides) -> ExperimentConfig:
        defaults = {
            "data": DataConfig(datasets=["rn30"]),
            "model": ModelConfig(),
            "train": TrainConfig(task="joint"),
        }
        return ExperimentConfig(**{**defaults, **overrides})

    def test_valid_combination(self):
        self.assertEqual(self.build().train.task, "joint")

    def test_cnn_cannot_predict_blink_presence(self):
        # The CNN family has no sequence model, so the combination is rejected
        # rather than trained on a task the architecture cannot express.
        with self.assertRaises(ConfigError) as ctx:
            self.build(model=ModelConfig(family="cnn"))
        self.assertIn("no sequence model", str(ctx.exception))

    def test_cnn_with_eye_state_is_allowed(self):
        config = self.build(model=ModelConfig(family="cnn"), train=TrainConfig(task="eye_state"))
        self.assertEqual(config.model.family, "cnn")

    def test_a_sequence_family_rejects_a_still_corpus(self):
        # The dual of the guard above. A still image has no temporal axis, so a
        # sequence model given one would be modelling a degenerate one-frame
        # window -- which runs, looks healthy, and learns nothing temporal.
        for family in ("lint", "linmult"):
            with self.subTest(family=family):
                with self.assertRaises(ConfigError) as ctx:
                    self.build(
                        data=DataConfig(datasets=["rn30", "cew"]),
                        model=ModelConfig(family=family),
                    )
                message = str(ctx.exception)
                self.assertIn("cew", message)
                self.assertIn("stills.yaml", message)

    def test_the_frame_wise_family_accepts_a_still_corpus(self):
        config = self.build(
            data=DataConfig(datasets=["cew", "mrl"]),
            model=ModelConfig(family="cnn"),
            train=TrainConfig(task="eye_state"),
        )
        self.assertEqual(config.data.datasets, ["cew", "mrl"])

    def test_still_mode_lets_a_sequence_config_name_a_still_corpus(self):
        # stills.yaml names every corpus that labels a single crop, including
        # the still ones; `stills: true` is what makes that coherent.
        config = self.build(
            data=DataConfig(datasets=["cew", "rn30"], stills=True),
            model=ModelConfig(family="cnn"),
            train=TrainConfig(task="eye_state"),
        )
        self.assertTrue(config.data.stills)

    def test_to_flat_dict_uses_dotted_keys(self):
        flat = self.build().to_flat_dict()
        self.assertIn("data.batch_size", flat)
        self.assertIn("train.optimizer.lr", flat)
        self.assertIn("model.d_model", flat)

    def test_to_flat_dict_joins_lists(self):
        flat = self.build().to_flat_dict()
        self.assertEqual(flat["data.datasets"], "rn30")


class TestFromFiles(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, name: str, data: dict) -> Path:
        path = self.tmp / name
        path.write_text(yaml.safe_dump(data))
        return path

    def paths(self) -> tuple[Path, Path, Path]:
        return (
            self.write("data.yaml", {"datasets": ["rn30"], "batch_size": 4}),
            self.write("model.yaml", {"family": "lint", "d_model": 32}),
            self.write("train.yaml", {"task": "joint", "max_epochs": 3}),
        )

    def test_assembles_the_three_files(self):
        config = ExperimentConfig.from_files(*self.paths())
        self.assertEqual(config.data.batch_size, 4)
        self.assertEqual(config.model.family, "lint")
        self.assertEqual(config.train.max_epochs, 3)

    def test_overrides_are_applied(self):
        config = ExperimentConfig.from_files(
            *self.paths(), overrides={"data.batch_size": 16, "train.max_epochs": 9}
        )
        self.assertEqual(config.data.batch_size, 16)
        self.assertEqual(config.train.max_epochs, 9)

    def test_nested_overrides_are_applied(self):
        config = ExperimentConfig.from_files(*self.paths(), overrides={"train.optimizer.lr": 0.5})
        self.assertEqual(config.train.optimizer.lr, 0.5)

    def test_missing_file_raises(self):
        data, model, _ = self.paths()
        with self.assertRaises(ConfigError):
            ExperimentConfig.from_files(data, model, self.tmp / "nope.yaml")


class TestRepositoryConfigs(unittest.TestCase):
    """The shipped configs must actually load; a broken one breaks every Make target."""

    # Discovered from disk rather than listed here. A hardcoded list turns
    # *deleting* a config -- a legitimate act -- into a test failure that names
    # the wrong culprit, and it silently skips a config someone adds.
    def configs(self, kind: str) -> list:
        found = sorted(p for p in (REPO_CONFIG / kind).glob("*.yaml"))
        self.assertTrue(found, f"no {kind} configs found")
        return found

    def test_every_run_data_config_loads(self):
        # `config/data/` holds two kinds: run configs, which name the corpora a
        # run mixes, and per-corpus declarations, which describe one corpus's
        # shape. Only the former has a `datasets` key, and only the former is a
        # DataConfig.
        for path in self.configs("data"):
            if "datasets" not in load_yaml(path):
                continue
            with self.subTest(config=path.stem):
                config = DataConfig.from_dict(load_yaml(path))
                self.assertTrue(config.datasets)

    def test_every_corpus_declaration_loads(self):
        for path in self.configs("data"):
            raw = load_yaml(path)
            if "datasets" in raw:
                continue
            with self.subTest(config=path.stem):
                # The builder owns where files live; the spec does not.
                for key in ("processed_dir", "h5_path"):
                    raw.pop(key, None)
                self.assertTrue(DatasetSpec(**raw).name)

    def test_every_model_config_loads(self):
        for path in self.configs("model"):
            with self.subTest(config=path.stem):
                ModelConfig.from_dict(load_yaml(path))

    def test_every_train_config_loads(self):
        for path in self.configs("train"):
            with self.subTest(config=path.stem):
                TrainConfig.from_dict(load_yaml(path))

    def test_the_shipped_make_targets_assemble(self):
        # Mirrors the Makefile's train-* recipes.
        combinations = [
            ("all", "blinklinmult", "joint"),
            ("all", "blinklint", "joint"),
            ("all", "blinklinmult", "blink_presence"),
            ("all", "blinklint", "blink_presence"),
            # stills.yaml pairs only with the frame-wise model; a sequence
            # family given a still corpus is rejected -- see the guard test.
            ("stills", "blinkcnn", "eye_state"),
            ("rn", "blinklint", "blink_presence"),
            ("smoke", "blinklint", "smoke"),
        ]
        for data, model, train in combinations:
            with self.subTest(data=data, model=model, train=train):
                ExperimentConfig.from_files(
                    REPO_CONFIG / "data" / f"{data}.yaml",
                    REPO_CONFIG / "model" / f"{model}.yaml",
                    REPO_CONFIG / "train" / f"{train}.yaml",
                )

    def test_every_dataset_declaration_resolves(self):
        # Every corpus the schema declares must have a config, and vice versa.
        specs = DataConfig(datasets=list(DATASETS)).resolve_specs(REPO_CONFIG / "data")
        self.assertEqual(len(specs), len(DATASETS))

    def test_no_run_config_is_mistaken_for_a_declaration(self):
        # config/data holds both per-corpus declarations and run configs; only
        # the former are named after a corpus.
        stems = {p.stem for p in (REPO_CONFIG / "data").glob("*.yaml")}
        self.assertTrue(set(DATASETS) <= stems)
        run_configs = stems - set(DATASETS)
        for name in run_configs:
            with self.subTest(config=name):
                DataConfig.from_dict(load_yaml(REPO_CONFIG / "data" / f"{name}.yaml"))

    def test_the_union_config_names_only_temporal_corpora(self):
        # all.yaml feeds the sequence models, which need a window to model. A
        # still corpus here would be trained on degenerate one-frame windows.
        config = DataConfig.from_dict(load_yaml(REPO_CONFIG / "data" / "all.yaml"))
        for spec in config.resolve_specs(REPO_CONFIG / "data"):
            self.assertTrue(spec.is_video, spec.name)
            self.assertTrue(spec.has_blink_presence, spec.name)

    def test_the_stills_config_names_only_eye_state_corpora(self):
        config = DataConfig.from_dict(load_yaml(REPO_CONFIG / "data" / "stills.yaml"))
        for spec in config.resolve_specs(REPO_CONFIG / "data"):
            self.assertTrue(spec.has_eye_state, spec.name)


class TestOverrideParsing(unittest.TestCase):
    def test_parses_an_integer(self):
        self.assertEqual(parse_override("data.batch_size=16"), ("data.batch_size", 16))

    def test_parses_a_float(self):
        self.assertEqual(parse_override("train.optimizer.lr=0.01")[1], 0.01)

    def test_parses_scientific_notation(self):
        # PyYAML 1.1 parses "1e-4" as a string; a learning rate arriving as a
        # string only surfaces deep inside the optimizer.
        key, value = parse_override("train.optimizer.lr=1e-4")
        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, 0.0001)

    def test_parses_a_boolean(self):
        self.assertIs(parse_override("data.pin_memory=false")[1], False)

    def test_parses_null(self):
        self.assertIsNone(parse_override("data.time_dim=null")[1])

    def test_keeps_a_genuine_string(self):
        self.assertEqual(parse_override("train.task=joint")[1], "joint")

    def test_missing_equals_raises(self):
        with self.assertRaises(ConfigError):
            parse_override("data.batch_size")

    def test_apply_override_creates_missing_levels(self):
        raw: dict = {}
        apply_override(raw, "train.optimizer.lr", 0.5)
        self.assertEqual(raw, {"train": {"optimizer": {"lr": 0.5}}})

    def test_apply_override_replaces_an_existing_value(self):
        raw = {"data": {"batch_size": 4}}
        apply_override(raw, "data.batch_size", 32)
        self.assertEqual(raw["data"]["batch_size"], 32)


class TestLoadYaml(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def test_reads_a_mapping(self):
        path = self.tmp / "c.yaml"
        path.write_text("a: 1\n")
        self.assertEqual(load_yaml(path), {"a": 1})

    def test_missing_file_raises(self):
        with self.assertRaises(ConfigError):
            load_yaml(self.tmp / "nope.yaml")

    def test_non_mapping_raises(self):
        path = self.tmp / "c.yaml"
        path.write_text("- a\n- b\n")
        with self.assertRaises(ConfigError):
            load_yaml(path)


if __name__ == "__main__":
    unittest.main()


class TestLimitFitBatches(unittest.TestCase):
    """Cheap diagnostic sweeps must still be scored on the full test split."""

    def test_it_defaults_to_none(self):
        self.assertIsNone(TrainConfig().limit_fit_batches)

    def test_a_fraction_is_accepted(self):
        self.assertEqual(TrainConfig(limit_fit_batches=0.5).limit_fit_batches, 0.5)

    def test_it_is_independent_of_limit_batches(self):
        # The two exist to do different things: one narrows the whole run, the
        # other narrows only what the model learns from.
        config = TrainConfig(limit_fit_batches=0.5)
        self.assertIsNone(config.limit_batches)


class TestEvalDatasets(unittest.TestCase):
    """Corpora evaluated on but never trained on."""

    def test_it_defaults_to_empty(self):
        self.assertEqual(DataConfig(datasets=["rn30"]).eval_datasets, [])

    def test_eval_only_corpora_are_resolved_too(self):
        # They need a spec even though they never reach the training split.
        config = DataConfig(datasets=["rn30"], eval_datasets=["talkingface"])
        self.assertEqual(config.eval_datasets, ["talkingface"])

    def test_a_corpus_in_both_lists_is_not_duplicated(self):
        # Two specs for one corpus would open its h5 twice and double-count it
        # in the test breakdown.
        config = DataConfig(datasets=["rn30"], eval_datasets=["rn30", "talkingface"])
        ordered = list(dict.fromkeys([*config.datasets, *config.eval_datasets]))
        self.assertEqual(ordered, ["rn30", "talkingface"])

    def test_training_order_is_preserved(self):
        # The mixing strategy indexes the training corpora by position, so
        # eval-only corpora must be appended, never interleaved.
        config = DataConfig(datasets=["rn15", "rn30"], eval_datasets=["hust_lebw"])
        ordered = list(dict.fromkeys([*config.datasets, *config.eval_datasets]))
        self.assertEqual(ordered[:2], ["rn15", "rn30"])


class TestEvalTargets(unittest.TestCase):
    """Targets scored but never trained."""

    def test_it_defaults_to_empty(self):
        self.assertEqual(TrainConfig().eval_targets, [])

    def test_a_cnn_may_evaluate_blink_presence(self):
        # The frame-wise model predicts closure and is benchmarked on events.
        # Training a blink head on it is still rejected; scoring is not.
        config = ExperimentConfig.from_files(
            Path("config/data/stills_all.yaml"),
            Path("config/model/blinkcnn.yaml"),
            Path("config/train/frame_wise.yaml"),
        )
        self.assertEqual(config.train.targets, ["eye_state"])
        self.assertIn("blink_presence", config.train.eval_targets)

    def test_a_cnn_still_cannot_train_blink_presence(self):
        # `task: blink_presence` is what actually puts a blink head on the
        # model; the guard must still refuse it for a frame-wise family.
        with self.assertRaises(ConfigError) as caught:
            ExperimentConfig.from_files(
                Path("config/data/stills_all.yaml"),
                Path("config/model/blinkcnn.yaml"),
                Path("config/train/frame_wise.yaml"),
                {"train.task": "blink_presence", "train.targets": ["blink_presence"]},
            )
        # A message that only says "no" leaves the reader stuck.
        self.assertIn("eval_targets", str(caught.exception))


class TestNestedOverrides(unittest.TestCase):
    """Dotted keys are the safe way to set a nested value from the CLI.

    A YAML flow mapping works too, but only with a space after the colon:
    `{gamma: 2.0}` is a mapping while `{gamma:2.0}` is a single key whose value
    is null. Make and the shell both eat unquoted spaces, so a Makefile target
    written the second way fails deep inside a dataclass constructor rather than
    at parse time. These pin the form the Makefile actually uses.
    """

    def test_a_dotted_key_sets_a_nested_value(self):
        key, value = parse_override("train.loss_kwargs.gamma=2.0")
        self.assertEqual(key, "train.loss_kwargs.gamma")
        self.assertEqual(value, 2.0)

    def test_a_flow_mapping_needs_a_space_after_the_colon(self):
        # The trap: without the space this is one key, not a mapping.
        _, without = parse_override("data.augment={strength:1.0}")
        _, with_space = parse_override("data.augment={strength: 1.0}")
        self.assertEqual(without, {"strength:1.0": None})
        self.assertEqual(with_space, {"strength": 1.0})

    def test_scientific_notation_survives(self):
        # PyYAML implements YAML 1.1, where "1e-4" is a string; a learning rate
        # arriving as a string only surfaces inside the optimizer.
        _, value = parse_override("train.optimizer.lr=1e-4")
        self.assertIsInstance(value, float)
        self.assertAlmostEqual(value, 0.0001)

    def test_an_override_without_equals_is_rejected(self):
        with self.assertRaises(ConfigError):
            parse_override("train.loss")


class TestVideoConfigs(unittest.TestCase):
    """The shipped video-benchmark configs must load and route targets correctly."""

    DATA = "config/data/video_all.yaml"
    TRAIN = "config/train/video.yaml"

    def _build(self, model: str, overrides: dict | None = None):
        return ExperimentConfig.from_files(self.DATA, model, self.TRAIN, overrides or {})

    def test_the_lint_arm_loads(self):
        config = self._build("config/model/blinklint.yaml")
        self.assertEqual(config.model.family, "lint")

    def test_the_linmult_arm_reads_the_feature_stream(self):
        """The whole point of BlinkLinMulT over BlinkLinT."""
        config = self._build("config/model/blinklinmult.yaml")
        self.assertEqual(config.model.family, "linmult")
        self.assertTrue(config.data.model_reads_features)

    def test_it_supervises_both_targets(self):
        """mpeblink can only contribute through blink_presence."""
        config = self._build("config/model/blinklint.yaml")
        self.assertEqual(set(config.train.targets), {"eye_state", "blink_presence"})

    def test_talkingface_is_held_out(self):
        """No threshold, no calibration, nothing may be fitted on it."""
        config = self._build("config/model/blinklint.yaml")
        self.assertNotIn("talkingface", config.data.datasets)
        self.assertIn("talkingface", config.data.eval_datasets)

    def test_it_keeps_the_temporal_axis(self):
        """A sequence model on expanded stills would train on one-frame windows."""
        config = self._build("config/model/blinklint.yaml")
        self.assertFalse(config.data.stills)

    def test_mpeblink_is_in_the_training_mix(self):
        """It is 10x RN15+RN30 combined -- the largest asset in the benchmark."""
        config = self._build("config/model/blinklint.yaml")
        self.assertIn("mpeblink", config.data.datasets)

    def test_occlusion_masking_is_on(self):
        config = self._build("config/model/blinklint.yaml")
        self.assertEqual(config.train.occlusion_yaw, 45.0)

    def test_the_frozen_arm_loads(self):
        config = self._build(
            "config/model/blinklint.yaml",
            {"model": {"encoder_weights": "some/path.ckpt", "encoder_freeze": True}},
        )
        self.assertTrue(config.model.encoder_freeze)
