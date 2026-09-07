"""Summarise the LinT diagnostic runs from MLflow.

The diagnostics are short, single-variable runs on RN30, and what matters is the
*ordering* between arms rather than any absolute score. This prints them as one
table so the comparison is direct, and reports the metrics that answer the
question each arm was run to settle.

Example:
    ``uv run python experiments/diagnose_report.py --experiment lint-diagnose``
"""

from __future__ import annotations

import argparse
import logging

import mlflow

logger = logging.getLogger(__name__)
"""Module-level logger."""

METRICS: tuple[str, ...] = (
    "test/mean_f1",
    "test/blink_presence/frame_max/precision",
    "test/blink_presence/frame_max/recall",
    "test/event/iou50/f1",
    "test/event/iou50/best_f1",
    "test/event/blink_ap",
)
"""What each diagnostic is judged on.

Precision and recall are shown beside F1 deliberately: the alpha sweep is
expected to *trade* them rather than move F1 much, and a table of F1 alone
would hide exactly the effect it was run to measure.

``best_f1`` is the event F1 at the *best* threshold of the sweep, against
``f1`` at the fixed 0.5 operating point. On a short run the two diverge widely:
an undertrained model can rank blinks well while placing none of its
probability mass above 0.5, which reads as a flat zero at the fixed threshold
and a real score at the best one. Judging these arms on ``f1`` alone would
conclude "nothing works" from a thresholding artefact.
"""

SHORT: dict[str, str] = {
    "test/mean_f1": "frame F1",
    "test/blink_presence/frame_max/precision": "precision",
    "test/blink_presence/frame_max/recall": "recall",
    "test/event/iou50/f1": "event F1",
    "test/event/iou50/best_f1": "best evF1",
    "test/event/blink_ap": "Blink-AP",
}
"""Column headings, kept short enough for one terminal line."""


def collect(tracking_uri: str, experiment: str) -> list[tuple[str, dict[str, float]]]:
    """Read every finished run of one experiment.

    Args:
        tracking_uri (str): MLflow tracking URI.
        experiment (str): Experiment name.

    Returns:
        list[tuple[str, dict[str, float]]]: ``(run_name, metrics)``, in the
        order the runs were started.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.MlflowClient()
    found = client.get_experiment_by_name(experiment)
    if found is None:
        return []

    rows = []
    for run in reversed(client.search_runs([found.experiment_id], max_results=200)):
        name = run.data.tags.get("mlflow.runName", run.info.run_id[:8])
        rows.append((name, dict(run.data.metrics)))
    return rows


def report(rows: list[tuple[str, dict[str, float]]]) -> None:
    """Print the runs as one comparison table.

    Args:
        rows (list[tuple[str, dict[str, float]]]): What :func:`collect` returned.
    """
    if not rows:
        print("No runs found. Has the experiment been run yet?")
        return

    header = f"{'run':22s}" + "".join(f"{SHORT[m]:>11s}" for m in METRICS)
    print(header)
    print("-" * len(header))
    for name, metrics in rows:
        cells = "".join(
            f"{metrics[m] * 100:10.2f}%" if m in metrics else f"{'--':>11s}" for m in METRICS
        )
        print(f"{name:22s}{cells}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--experiment", default="lint-diagnose")
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    report(collect(args.tracking_uri, args.experiment))


if __name__ == "__main__":
    main()
