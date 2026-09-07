"""Tests for the MLflow parameter collection and artifact logging."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import h5py

from blinklinmult.data.schema import DatasetSpec
from blinklinmult.train.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TrainConfig,
)
from blinklinmult.train.mlflow_utils import (
    MAX_PARAM_LENGTH,
    SKIP_ARTIFACT_DIRS,
    dataset_params,
    environment_params,
    file_md5,
    git_sha,
    truncate_params,
)


class RecordingLogger:
    """Captures the artifact uploads a real MLFlowLogger would perform."""

    class _Experiment:
        def __init__(self):
            self.uploads: list[tuple[str, str | None]] = []

        def log_artifact(self, run_id, local_path, artifact_path=None):  # noqa: ARG002
            self.uploads.append((local_path, artifact_path))

    def __init__(self):
        self.run_id = "run"
        self.experiment = self._Experiment()


class TempDirCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)


class TestGitSha(unittest.TestCase):
    def test_returns_a_string(self):
        self.assertIsInstance(git_sha(), str)

    def test_outside_a_checkout_returns_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(git_sha(Path(tmp)), "unknown")


class TestFileMd5(TempDirCase):
    def test_hashes_a_file(self):
        path = self.tmp / "a.bin"
        path.write_bytes(b"hello")
        self.assertEqual(len(file_md5(path)), 32)

    def test_identical_contents_hash_identically(self):
        (self.tmp / "a").write_bytes(b"x" * 1000)
        (self.tmp / "b").write_bytes(b"x" * 1000)
        self.assertEqual(file_md5(self.tmp / "a"), file_md5(self.tmp / "b"))

    def test_different_contents_hash_differently(self):
        (self.tmp / "a").write_bytes(b"x")
        (self.tmp / "b").write_bytes(b"y")
        self.assertNotEqual(file_md5(self.tmp / "a"), file_md5(self.tmp / "b"))

    def test_a_missing_file_is_reported_not_raised(self):
        self.assertEqual(file_md5(self.tmp / "nope"), "missing")

    def test_reads_across_chunk_boundaries(self):
        path = self.tmp / "big.bin"
        path.write_bytes(b"a" * 4096)
        self.assertEqual(file_md5(path, chunk_size=64), file_md5(path))


class TestDatasetParams(TempDirCase):
    def spec(self, name: str = "cew") -> DatasetSpec:
        return DatasetSpec(name=name, has_eye_state=True)

    def write_h5(self, name: str, **attrs) -> Path:
        path = self.tmp / "data" / "processed" / name / f"{name}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as handle:
            for key, value in attrs.items():
                handle.attrs[key] = value
        return path

    def test_records_hash_and_size_per_corpus(self):
        self.write_h5("cew", schema_version=1, git_sha="abc", created_utc="now")
        params = dataset_params([self.spec()], self.tmp)
        self.assertNotEqual(params["data.cew.md5"], "missing")
        self.assertGreater(params["data.cew.bytes"], 0)

    def test_reads_the_builder_provenance_out_of_the_file(self):
        # Two runs over "the same corpus" are not comparable if one was rebuilt;
        # nothing else in the config would reveal that.
        self.write_h5("cew", schema_version=1, git_sha="deadbeef", created_utc="t")
        params = dataset_params([self.spec()], self.tmp)
        self.assertEqual(params["data.cew.builder_git_sha"], "deadbeef")
        self.assertEqual(params["data.cew.schema_version"], 1)

    def test_an_unbuilt_corpus_is_reported_not_fatal(self):
        params = dataset_params([self.spec()], self.tmp)
        self.assertEqual(params["data.cew.md5"], "missing")
        self.assertEqual(params["data.cew.bytes"], 0)

    def test_every_corpus_gets_its_own_keys(self):
        self.write_h5("cew", schema_version=1)
        self.write_h5("mrl", schema_version=1)
        params = dataset_params([self.spec("cew"), self.spec("mrl")], self.tmp)
        self.assertIn("data.cew.md5", params)
        self.assertIn("data.mrl.md5", params)


class TestEnvironmentParams(TempDirCase):
    def config(self) -> ExperimentConfig:
        # A video corpus: a sequence family may not be given a still one.
        return ExperimentConfig(
            data=DataConfig(datasets=["rn30"]),
            model=ModelConfig(family="lint"),
            train=TrainConfig(task="eye_state"),
        )

    def test_records_the_library_versions(self):
        params = environment_params(self.config(), [], self.tmp)
        for key in ("env.torch", "env.lightning", "env.linmult", "env.omniloader"):
            with self.subTest(key=key):
                self.assertIn(key, params)
                self.assertNotEqual(params[key], "")

    def test_records_the_run_shape(self):
        params = environment_params(self.config(), [], self.tmp)
        self.assertEqual(params["run.task"], "eye_state")
        self.assertEqual(params["run.model_family"], "lint")
        # Against the config rather than a literal: this asserts the backbone is
        # *recorded*, which is what makes a run reproducible. Pinning the name
        # here would fail every time the default backbone changes, which is a
        # config decision and not a regression in the logging.
        self.assertEqual(params["run.backbone"], self.config().model.backbone)

    def test_includes_the_dataset_identities(self):
        spec = DatasetSpec(name="cew", has_eye_state=True)
        params = environment_params(self.config(), [spec], self.tmp)
        self.assertIn("data.cew.md5", params)


class TestTruncateParams(unittest.TestCase):
    def test_short_values_are_unchanged(self):
        self.assertEqual(truncate_params({"a": "x"}), {"a": "x"})

    def test_long_values_are_clipped(self):
        result = truncate_params({"a": "x" * (MAX_PARAM_LENGTH + 100)})
        self.assertEqual(len(result["a"]), MAX_PARAM_LENGTH)
        self.assertTrue(result["a"].endswith("..."))

    def test_non_string_values_survive_their_type(self):
        result = truncate_params({"a": 1, "b": 2.5, "c": True})
        self.assertEqual(result["a"], 1)
        self.assertEqual(result["b"], 2.5)
        self.assertIs(result["c"], True)


class TestLogArtifacts(TempDirCase):
    def build_outputs(self) -> Path:
        output = self.tmp / "results"
        (output / "checkpoints").mkdir(parents=True)
        (output / "plots").mkdir(parents=True)
        (output / "metrics.json").write_text("{}")
        (output / "plots" / "curve.png").write_bytes(b"x")
        (output / "checkpoints" / "best.ckpt").write_bytes(b"x" * 100)
        return output

    def test_uploads_the_files(self):
        from blinklinmult.train.mlflow_utils import log_artifacts

        logger = RecordingLogger()
        log_artifacts(logger, self.build_outputs())
        names = [Path(p).name for p, _ in logger.experiment.uploads]
        self.assertIn("metrics.json", names)
        self.assertIn("curve.png", names)

    def test_checkpoints_are_excluded(self):
        # Hundreds of MB each; the chosen one is logged deliberately at the end.
        from blinklinmult.train.mlflow_utils import log_artifacts

        logger = RecordingLogger()
        log_artifacts(logger, self.build_outputs())
        names = [Path(p).name for p, _ in logger.experiment.uploads]
        self.assertNotIn("best.ckpt", names)

    def test_root_level_files_use_no_artifact_path(self):
        # MLflow rejects "." as an artifact path.
        from blinklinmult.train.mlflow_utils import log_artifacts

        logger = RecordingLogger()
        log_artifacts(logger, self.build_outputs())
        root = next(path for path, _ in logger.experiment.uploads if path.endswith("metrics.json"))
        artifact_path = dict(logger.experiment.uploads)[root]
        self.assertIsNone(artifact_path)

    def test_nested_files_keep_their_directory(self):
        from blinklinmult.train.mlflow_utils import log_artifacts

        logger = RecordingLogger()
        log_artifacts(logger, self.build_outputs())
        nested = next(
            artifact for path, artifact in logger.experiment.uploads if path.endswith("curve.png")
        )
        self.assertEqual(nested, "plots")

    def test_a_missing_output_directory_is_a_no_op(self):
        from blinklinmult.train.mlflow_utils import log_artifacts

        logger = RecordingLogger()
        log_artifacts(logger, self.tmp / "nope")
        self.assertEqual(logger.experiment.uploads, [])

    def test_the_skip_list_names_the_heavy_directories(self):
        self.assertIn("checkpoints", SKIP_ARTIFACT_DIRS)


if __name__ == "__main__":
    unittest.main()
