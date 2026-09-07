"""Append a finished run to the experiment log in ``tmp/overview.md``.

Reads the run's own artifacts rather than taking numbers on the command line, so
a logged result cannot drift from what the run actually produced. Each entry
records the command that started it and the MLflow run name, so any row in the
document can be traced back to a run in the UI.

Run::

    uv run python tools/log_experiment.py vid-lint-s1-mpe10 \
      --command "bash experiments/run_lint_search.sh step1" \
      --title "Stage A -- redefined baseline" \
      --note "First arm under the split-head definitions."
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

OVERVIEW = Path("tmp/overview.md")
"""The single reference document."""

RESULTS = Path("results")
"""Root of the run outputs."""

ESR_CORPORA = ("rn15", "rn30", "talkingface")
"""Corpora annotating per-frame closure."""

EVENT_CORPORA = ("rn15", "rn30", "talkingface", "hust_lebw", "mpeblink")
"""Corpora with a temporal axis."""


def read_json(path: Path) -> dict | None:
    """Read a JSON artifact if present.

    Args:
        path (Path): The artifact.

    Returns:
        dict | None: Its contents, or ``None`` when absent or unreadable.
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def run_dir(run: str) -> Path | None:
    """Locate a run's output directory under any experiment.

    Args:
        run (str): MLflow run name, which is also the directory name.

    Returns:
        Path | None: The directory, or ``None``.
    """
    for experiment in sorted(RESULTS.glob("*")):
        candidate = experiment / run
        if candidate.is_dir():
            return candidate
    return None


def corpus_scores(directory: Path, target: str, corpora: tuple[str, ...]) -> dict[str, dict]:
    """Per-corpus scores for one target.

    Args:
        directory (Path): The run directory.
        target (str): ``eye_state`` or ``blink_presence``.
        corpora (tuple[str, ...]): Which corpora to keep.

    Returns:
        dict[str, dict]: Metrics per corpus, only where the target was supervised.
    """
    payload = read_json(directory / "test_per_dataset.json") or {}
    return {
        corpus: values
        for corpus, values in payload.get(target, {}).items()
        if corpus in corpora and values.get("valid_positions", 0) > 0
    }


def render_table(title: str, scores: dict[str, dict]) -> list[str]:
    """Render one metric table.

    Args:
        title (str): Table heading.
        scores (dict[str, dict]): Per-corpus metrics.

    Returns:
        list[str]: Markdown lines; empty when nothing was scored.
    """
    if not scores:
        return []
    lines = [f"**{title}**", "", "| corpus | F1 | precision | recall |", "|---|---:|---:|---:|"]
    for corpus in sorted(scores):
        row = scores[corpus]
        lines.append(
            f"| {corpus} | {row.get('f1', 0):.4f} | "
            f"{row.get('precision', 0):.4f} | {row.get('recall', 0):.4f} |"
        )
    lines.append("")
    return lines


def operating_point(directory: Path) -> str:
    """The fitted operating point, as a readable string.

    Args:
        directory (Path): The run directory.

    Returns:
        str: The threshold, plus the hysteresis low value when one was fitted.
    """
    threshold = read_json(directory / "event_threshold.json") or {}
    high = threshold.get("threshold", float("nan"))
    ratio = threshold.get("low_ratio")
    if ratio is None:
        return f"{high:.2f} (single threshold)"
    return f"{high:.2f} / {high * ratio:.3f} (hysteresis)"


def build_entry(run: str, command: str, title: str, note: str | None) -> str:
    """Build the log entry for one run.

    Args:
        run (str): MLflow run name.
        command (str): The command that started it.
        title (str): Section heading.
        note (str | None): Optional interpretation line.

    Returns:
        str: Markdown to append.

    Raises:
        FileNotFoundError: If the run directory does not exist.
    """
    directory = run_dir(run)
    if directory is None:
        raise FileNotFoundError(
            f"No run directory for {run!r} under {RESULTS}/. Has it finished training?"
        )

    identifier = directory / "mlflow_run_id.txt"
    run_id = identifier.read_text().strip() if identifier.is_file() else "unknown"

    timing = read_json(directory / "time.json") or {}
    epochs = len(timing.get("epochs", []))
    minutes = timing.get("total_seconds", 0) / 60

    lines = [
        "",
        f"### {title}",
        "",
        f"**Run:** `{run}` -- MLflow experiment **`{directory.parent.name}`**, run id `{run_id}`.",
        "",
        "```bash",
        command,
        "```",
        "",
        f"{epochs} epochs, {minutes:.0f} min. "
        f"Fitted operating point: {operating_point(directory)}.",
        "",
    ]
    if note:
        lines += [note, ""]

    lines += render_table(
        "ESR -- per-frame closure", corpus_scores(directory, "eye_state", ESR_CORPORA)
    )
    lines += render_table(
        "BPD -- per-frame event signal",
        corpus_scores(directory, "blink_presence", EVENT_CORPORA),
    )
    lines.append(f"*Logged {datetime.now(UTC):%Y-%m-%d}.*")
    return "\n".join(lines)


def main() -> None:
    """Append one entry to the overview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", help="MLflow run name, also the output directory name.")
    parser.add_argument("--command", required=True, help="The command that started the run.")
    parser.add_argument("--title", required=True, help="Section heading for the entry.")
    parser.add_argument("--note", default=None, help="Optional one-line interpretation.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    text = build_entry(args.run, args.command, args.title, args.note)
    with OVERVIEW.open("a") as handle:
        handle.write(text + "\n")
    logger.info(f"Appended {args.title!r} to {OVERVIEW}.")


if __name__ == "__main__":
    main()
