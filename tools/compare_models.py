"""Compare models on one corpus, with confidence intervals and p-values.

Reads the per-recording signal archives an evaluation run already wrote, so no
model is re-run: :class:`~blinklinmult.train.callbacks.EventReport` dumps
``test_signals.npz`` precisely so "threshold tuning, per-corpus analysis, and
significance testing run later without re-running the model."

What comes out is frequently "these models are not distinguishable on this
corpus", and that is the intended output rather than a failure of the analysis.
RN30 has 7312 test windows but only **35 recordings**, and 35 independent
observations cannot resolve a two-point difference in average precision. Saying
so is more useful than a rank order that would not survive a different 35
subjects.

Two guards worth knowing about:

* **Fitting must not touch test.** ``--fit-signals`` and ``--test-signals`` are
  separate arguments and the script refuses when they resolve to one path.
* **The target is checked, not assumed.** Archives are read through
  :func:`~blinklinmult.train.callbacks.load_signals`, which rejects a mismatch.
  Reading a ``blink_presence`` archive as ``eye_state`` once produced 200 045
  phantom false positives -- a plausible, entirely wrong number.

Run::

    uv run python tools/compare_models.py \
        --signals results/frame-wise/fw-bce-hyst-rn30/test_signals.npz=fw-bce \
        --signals results/frame-wise/fw-focal-hyst-rn30/test_signals.npz=fw-focal \
        --corpus rn30 --out results/comparison/rn30.json
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from blinklinmult.compare.bootstrap import (
    DEFAULT_ROUNDS,
    DEFAULT_SEED,
    MIN_CLUSTERS_FOR_INTERVAL,
    Interval,
)
from blinklinmult.compare.protocol import CorpusReport, compare_average_precision

logger = logging.getLogger(__name__)
"""Module-level logger."""

SCHEMA_VERSION = 1
"""Output schema version, so a reader can tell what shape to expect."""

DEFAULT_TARGET = "blink_presence"
"""Target the archives are expected to hold.

The frame-wise event runs dump ``blink_presence`` -- blink *intervals*, which
cover 3-4x more frames than actual closure. ``eye_state`` archives are a
different quantity and scoring one against the other measures an annotation
convention rather than a model.
"""


def _git_sha() -> str:
    """Current commit, so a result traces back to the code that made it.

    Returns:
        str: The short SHA, or ``"unknown"`` outside a repository.
    """
    try:
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _interval_json(interval: Interval) -> dict:
    """Render an interval for the output document.

    Args:
        interval (Interval): The estimate.

    Returns:
        dict: Its fields, with ``null`` bounds when none could be produced.
    """
    return {
        "point": interval.point,
        "ci_low": interval.low,
        "ci_high": interval.high,
        "n_clusters": interval.n_clusters,
        "reason": interval.reason or None,
    }


def _report_json(report: CorpusReport, rounds: int, seed: int) -> dict:
    """Render a whole corpus report.

    Args:
        report (CorpusReport): The result.
        rounds (int): Replicates used.
        seed (int): Seed used.

    Returns:
        dict: The serialisable document.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(UTC).isoformat(),
        "git_sha": _git_sha(),
        "protocol": {
            "statistic": "average_precision",
            "bootstrap": {
                "resamples": rounds,
                "seed": seed,
                "method": "percentile",
                "unit": "recording",
                "paired": True,
                "min_clusters_for_interval": MIN_CLUSTERS_FOR_INTERVAL,
            },
            "correction": {"method": "holm", "family": "all pairs within this corpus"},
            "note": (
                "The resampling unit is the recording, not the window. Windows "
                "from one recording share a subject, a camera and a blink rate; "
                "resampling them independently shrank the interval 5.9x on this "
                "repo's own data and reversed the conclusion."
            ),
        },
        "corpus": {
            "name": report.corpus,
            "n_clusters": report.n_clusters,
            "n_frames": report.n_frames,
            "positive_rate": report.positive_rate,
            "resolvable": report.resolvable,
        },
        "scores": {model: _interval_json(interval) for model, interval in report.scores.items()},
        "comparisons": [
            {
                "first": comparison.first,
                "second": comparison.second,
                **_interval_json(comparison.interval),
                "p_raw": comparison.p_raw,
                "p_holm": comparison.p_holm,
                "conclusive": comparison.conclusive,
            }
            for comparison in report.comparisons
        ],
    }


def _print_report(report: CorpusReport) -> None:
    """Log the tables a reader actually looks at.

    Args:
        report (CorpusReport): The result.
    """
    logger.info(
        f"\n{report.corpus}: {report.n_clusters} recordings, {report.n_frames} valid frames, "
        f"{report.positive_rate:.2%} positive"
    )
    if not report.resolvable:
        logger.info(
            f"  Too few recordings for an interval (need {MIN_CLUSTERS_FOR_INTERVAL}); "
            "point estimates only."
        )

    logger.info("\n  Average precision (threshold-free)")
    for model, interval in sorted(report.scores.items(), key=lambda kv: -kv[1].point):
        logger.info(f"    {model:24s} {interval.describe()}")

    logger.info("\n  Pairwise differences, Holm-corrected")
    for comparison in report.comparisons:
        logger.info(f"    {comparison.describe()}")

    conclusive = sum(comparison.conclusive for comparison in report.comparisons)
    logger.info(f"\n  {conclusive} of {len(report.comparisons)} pairs separate after correction.")
    if not conclusive and report.resolvable:
        logger.info(
            f"  With {report.n_clusters} recordings the corpus cannot resolve differences "
            "this small. That is a fact about the sample size, not a defect in the models."
        )


def _parse_signals(entries: list[str]) -> dict[str, Path]:
    """Parse ``path=label`` arguments.

    Args:
        entries (list[str]): Raw ``--signals`` values.

    Returns:
        dict[str, Path]: Label to archive path.

    Raises:
        SystemExit: If an entry is malformed or a label repeats.
    """
    parsed: dict[str, Path] = {}
    for entry in entries:
        path, separator, label = entry.partition("=")
        if not separator:
            raise SystemExit(f"--signals expects PATH=LABEL, got {entry!r}")
        if label in parsed:
            raise SystemExit(f"duplicate label {label!r}")
        parsed[label] = Path(path)
    return parsed


def main() -> None:
    """Compare the given signal archives."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signals",
        action="append",
        required=True,
        metavar="PATH=LABEL",
        help="A test signal archive and the name to report it under; repeatable.",
    )
    parser.add_argument("--corpus", required=True, help="Corpus name, for the report.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target the archives must hold.")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="Bootstrap replicates.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Resampling seed.")
    parser.add_argument("--out", type=Path, default=None, help="Where to write the JSON.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from blinklinmult.train.callbacks import load_signals

    paths = _parse_signals(args.signals)
    if len({path.resolve() for path in paths.values()}) != len(paths):
        raise SystemExit("two labels point at the same archive; that is a copy, not a comparison")

    stores = {label: load_signals(path, args.target) for label, path in paths.items()}
    report = compare_average_precision(stores, args.corpus, rounds=args.rounds, seed=args.seed)
    _print_report(report)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        document = _report_json(report, args.rounds, args.seed)
        args.out.write_text(json.dumps(document, indent=2) + "\n")
        logger.info(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
