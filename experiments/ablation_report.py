"""Summarise the sequence-model ablation from MLflow.

Reads the runs `make ablation-stage<N>` produced and prints one row per
configuration with mean ± standard deviation across seeds. Frame-level and
event-level metrics sit side by side deliberately: a model can win on frames and
lose on events, and the event numbers are the ones the literature reports.

Nothing here recomputes a metric. Every value is read back from MLflow exactly
as the run logged it, so the table cannot disagree with the runs it summarises.

Example:
    ``uv run python experiments/ablation_report.py``
    ``uv run python experiments/ablation_report.py --experiment blink-presence``
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from typing import TYPE_CHECKING, Any, NamedTuple

from blinklinmult.train.events import CRITERIA
from blinklinmult.train.metrics import PRIMARY_METRIC

if TYPE_CHECKING:
    from collections.abc import Iterable

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"
"""Where the runs are recorded, matching ``config/train/*.yaml``."""

DEFAULT_EXPERIMENT = "blink-presence"
"""Experiment the blink-presence ablation logs under."""

RUN_PREFIX = "abl-"
"""Prefix `make ablation-stage<N>` gives every run it starts."""

FRAME_METRIC = f"test/{PRIMARY_METRIC}"
"""Frame-level headline: the metric that also drives checkpoint selection."""

EVENT_CRITERION = "iou50"
"""Event criterion reported in the table; MPEblink's Blink-AP operating point."""

AXES = (
    "model.attention_type",
    "model.add_module_tcn",
    "model.d_model",
    "model.cmt_num_layers",
    "model.num_heads",
)
"""Parameters that identify a configuration, shown as the table's left columns."""


class Arm(NamedTuple):
    """One configuration's results across seeds.

    Args:
        name (str): Arm name, from the run name without prefix or seed.
        axes (tuple[str, ...]): Values of :data:`AXES`, in order.
        metrics (dict[str, list[float]]): Metric name to one value per seed.
    """

    name: str
    axes: tuple[str, ...]
    metrics: dict[str, list[float]]


def spread(values: list[float]) -> str:
    """Format a metric as ``mean ±sd`` in percent.

    Args:
        values (list[float]): One value per seed.

    Returns:
        str: Formatted cell, or ``--`` when the metric was never logged.
    """
    if not values:
        return "--"
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{100 * statistics.mean(values):6.2f} ±{100 * deviation:4.2f}"


def arm_name(run_name: str) -> str:
    """Strip the prefix and seed suffix from a run name.

    ``abl-flash-s42`` and ``abl-flash-s43`` are two seeds of one arm.

    Args:
        run_name (str): The MLflow run name.

    Returns:
        str: The arm name.
    """
    stem = run_name.removeprefix(RUN_PREFIX)
    head, separator, tail = stem.rpartition("-s")
    return head if separator and tail.isdigit() else stem


def collect(runs: Iterable[Any], wanted: list[str]) -> list[Arm]:
    """Group finished runs into arms.

    Args:
        runs (Iterable[Any]): MLflow ``Run`` objects.
        wanted (list[str]): Metric names to gather.

    Returns:
        list[Arm]: One entry per configuration, ordered by mean frame metric.
    """
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    axes: dict[str, tuple[str, ...]] = {}

    for run in runs:
        name = run.data.tags.get("mlflow.runName", "")
        if not name.startswith(RUN_PREFIX):
            continue
        arm = arm_name(name)
        axes.setdefault(arm, tuple(run.data.params.get(a, "?") for a in AXES))
        for metric in wanted:
            if metric in run.data.metrics:
                grouped[arm][metric].append(run.data.metrics[metric])

    collected = [Arm(arm, axes[arm], dict(values)) for arm, values in grouped.items()]
    collected.sort(key=lambda a: statistics.mean(a.metrics.get(FRAME_METRIC, [0.0])), reverse=True)
    return collected


def report(arms: list[Arm], event_metrics: dict[str, str]) -> None:
    """Print the comparison table.

    Args:
        arms (list[Arm]): Collected arms, best first.
        event_metrics (dict[str, str]): Column label to metric name.
    """
    if not arms:
        print(f"No runs named {RUN_PREFIX}* found. Run `make ablation-stage0` first.")
        return

    columns = ["frame F1", *event_metrics]
    header = (
        f"{'arm':14s} {'attn':8s} {'tcn':6s} {'d':>4s} {'L':>3s} {'h':>3s} "
        + " ".join(f"{c:>14s}" for c in columns)
        + "   n"
    )
    print(f"\nSequence-model ablation, {EVENT_CRITERION} event criterion\n")
    print(header)
    print("-" * len(header))

    for arm in arms:
        attn, tcn, d_model, layers, heads = arm.axes
        cells = [spread(arm.metrics.get(FRAME_METRIC, []))]
        cells += [spread(arm.metrics.get(m, [])) for m in event_metrics.values()]
        seeds = len(arm.metrics.get(FRAME_METRIC, []))
        print(
            f"{arm.name:14s} {attn:8s} {tcn:6s} {d_model:>4s} {layers:>3s} {heads:>3s} "
            + " ".join(f"{c:>14s}" for c in cells)
            + f"   {seeds}"
        )

    print(
        "\nGreedy search: each stage fixes the winner before the next runs, so "
        "interactions between axes are not measured."
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tracking-uri", default=DEFAULT_TRACKING_URI, help="MLflow store.")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT, help="Experiment name.")
    parser.add_argument(
        "--criterion",
        default=EVENT_CRITERION,
        choices=sorted(CRITERIA),
        help="Event matching criterion to report.",
    )
    args = parser.parse_args()

    import mlflow

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(args.experiment)
    if experiment is None:
        names = [e.name for e in client.search_experiments()]
        print(f"No experiment {args.experiment!r}. Found: {names}")
        return

    event_metrics = {
        "event F1": f"test/event/{args.criterion}/f1",
        "event AP": f"test/event/{args.criterion}/average_precision",
        "FA/min": f"test/event/{args.criterion}/fa_per_min",
    }
    wanted = [FRAME_METRIC, *event_metrics.values()]

    runs = client.search_runs([experiment.experiment_id], max_results=1000)
    report(collect(runs, wanted), event_metrics)


if __name__ == "__main__":
    main()
