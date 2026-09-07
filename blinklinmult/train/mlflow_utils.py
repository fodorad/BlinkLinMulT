"""MLflow wiring for blink training runs.

Logging is explicit rather than via ``mlflow.pytorch.autolog()``: autolog picks
its own cadence and parameter names, which would fight the flattened dataclass
params and make two runs hard to compare. Everything logged here is chosen.

What makes a run reproducible is not just its hyperparameters but the exact data
it saw, and this project's data is *several* files — one per corpus.
:func:`environment_params` therefore records each built dataset's hash, size,
and embedded builder SHA individually, so two runs over "the same six corpora"
can be told apart when one of them was rebuilt.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from blinklinmult import PROJECT_ROOT

if TYPE_CHECKING:
    from blinklinmult.data.schema import DatasetSpec
    from blinklinmult.train.config import ExperimentConfig

logger = logging.getLogger(__name__)
"""Module-level logger."""

MAX_PARAM_LENGTH = 500
"""MLflow rejects parameter values longer than this."""


def git_sha(root: Path = PROJECT_ROOT) -> str:
    """Return the current commit SHA.

    Args:
        root (Path): Repository root.

    Returns:
        str: The SHA, or ``"unknown"`` outside a git checkout.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"
    return result.stdout.strip()


def file_md5(path: Path, chunk_size: int = 1 << 20) -> str:
    """Hash a file's contents.

    Args:
        path (Path): File to hash.
        chunk_size (int): Read size in bytes.

    Returns:
        str: Hex digest, or ``"missing"`` if the file does not exist.
    """
    if not path.is_file():
        return "missing"
    digest = hashlib.md5()  # noqa: S324 -- artifact identity, not security
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_params(specs: list[DatasetSpec], root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Record the identity of every dataset file the run reads.

    Two runs with identical hyperparameters over differently-built corpora are
    not comparable, and nothing else in the config would reveal that.

    Args:
        specs (list[DatasetSpec]): Corpora the run uses.
        root (Path): Repository root.

    Returns:
        dict: Per-corpus hash, size, and the builder SHA embedded in the file.
    """
    import h5py

    params: dict[str, Any] = {}
    for spec in specs:
        path = root / "data" / "processed" / spec.name / f"{spec.name}.h5"
        prefix = f"data.{spec.name}"
        params[f"{prefix}.md5"] = file_md5(path)
        params[f"{prefix}.bytes"] = path.stat().st_size if path.is_file() else 0

        if path.is_file():
            try:
                with h5py.File(path, "r") as handle:
                    params[f"{prefix}.builder_git_sha"] = str(
                        handle.attrs.get("git_sha", "unknown")
                    )
                    params[f"{prefix}.created_utc"] = str(
                        handle.attrs.get("created_utc", "unknown")
                    )
                    params[f"{prefix}.schema_version"] = int(handle.attrs.get("schema_version", -1))
            except OSError:
                logger.warning(f"{path}: could not read attributes for MLflow params.")

    return params


def environment_params(
    config: ExperimentConfig, specs: list[DatasetSpec], root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """Collect the facts that make a run reproducible.

    Args:
        config (ExperimentConfig): The run's configuration.
        specs (list[DatasetSpec]): Corpora the run uses.
        root (Path): Repository root.

    Returns:
        dict: Library versions, the commit, and every dataset's identity.
    """
    import lightning
    import linmult
    import omniloader
    import torch

    params: dict[str, Any] = {
        "env.git_sha": git_sha(root),
        "env.torch": torch.__version__,
        "env.lightning": lightning.__version__,
        "env.linmult": getattr(linmult, "__version__", "unknown"),
        "env.omniloader": getattr(omniloader, "__version__", "unknown"),
        "run.task": config.train.task,
        "run.model_family": config.model.family,
        "run.backbone": config.model.backbone,
    }
    params.update(dataset_params(specs, root))
    return params


def truncate_params(params: dict[str, Any]) -> dict[str, Any]:
    """Clip over-long values so MLflow accepts every parameter.

    Args:
        params (dict): Parameters to log.

    Returns:
        dict: The same parameters, with long values truncated.
    """
    clipped: dict[str, Any] = {}
    for key, value in params.items():
        text = str(value)
        clipped[key] = (
            text[: MAX_PARAM_LENGTH - 3] + "..." if len(text) > MAX_PARAM_LENGTH else value
        )
    return clipped


RUN_ID_FILE = "mlflow_run_id.txt"
"""Name of the file in a run's output directory holding its MLflow run id.

Written on first launch and read back on resume, so a continued run **appends to
the same MLflow run** instead of minting a second, disjoint one. Without it a
resumed run's metric curve starts over at the resumed epoch and the two halves
cannot be read as one training history.
"""


def stored_run_id(output_dir: Path) -> str | None:
    """The MLflow run id a previous launch recorded, if any.

    Args:
        output_dir (Path): The run's output directory.

    Returns:
        str | None: The id, or ``None`` when this is a first launch.
    """
    path = output_dir / RUN_ID_FILE
    return path.read_text().strip() if path.is_file() else None


def remember_run_id(output_dir: Path, mlflow_logger) -> None:
    """Record a run's MLflow id so a later resume can continue it.

    Reading ``mlflow_logger.run_id`` also materialises the run, which is why
    this is called once at launch rather than lazily.

    Args:
        output_dir (Path): The run's output directory.
        mlflow_logger: The run's ``MLFlowLogger``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / RUN_ID_FILE).write_text(str(mlflow_logger.run_id))


def build_logger(config: ExperimentConfig, run_name: str | None = None, run_id: str | None = None):
    """Construct the MLflow logger for a run.

    Args:
        config (ExperimentConfig): The run's configuration.
        run_name (str | None): Overrides the configured run name.
        run_id (str | None): An existing run to append to, for a resumed run.
            ``None`` starts a new one.

    Returns:
        MLFlowLogger: The configured logger.
    """
    from lightning.pytorch.loggers import MLFlowLogger

    return MLFlowLogger(
        experiment_name=config.train.mlflow.experiment_name,
        tracking_uri=config.train.mlflow.tracking_uri,
        run_name=run_name or config.train.mlflow.run_name,
        run_id=run_id,
        # Checkpoints are logged deliberately at the end of a run rather than on
        # every improvement, which would copy hundreds of MB repeatedly.
        log_model=False,
    )


SKIP_ARTIFACT_DIRS: frozenset[str] = frozenset({"checkpoints", "lightning_logs"})
"""Directories excluded from the bulk artifact upload.

Checkpoints are hundreds of MB and the chosen one is logged deliberately at the
end of a run; copying every one of them into the tracking store would bloat it
for no benefit.
"""


def log_artifacts(mlflow_logger, output_dir: Path) -> None:
    """Upload a run's output directory as artifacts.

    Args:
        mlflow_logger: The run's ``MLFlowLogger``.
        output_dir (Path): Directory of run outputs.
    """
    if not output_dir.is_dir():
        return

    count = 0
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        if SKIP_ARTIFACT_DIRS & set(path.relative_to(output_dir).parts):
            continue
        # MLflow rejects "." as an artifact path, which is what relative_to
        # yields for files sitting at the root of the output directory.
        relative = path.parent.relative_to(output_dir)
        artifact_path = None if relative == Path() else str(relative)
        mlflow_logger.experiment.log_artifact(
            mlflow_logger.run_id, str(path), artifact_path=artifact_path
        )
        count += 1

    logger.info(f"Logged {count} artifacts from {output_dir} to MLflow run {mlflow_logger.run_id}")
