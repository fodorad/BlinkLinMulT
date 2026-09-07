"""Benchmark table for the frame-wise model, across every corpus.

One model, scored three ways, because the corpora annotate three different
things:

* **Still corpora** (CEW, MRL-Eye) annotate per-frame closure, which is exactly
  what the model predicts. Frame accuracy and F1, nothing to reconstruct.
* **Video corpora** (RN15, RN30, TalkingFace, MPEblink) annotate blink
  *events*. The per-frame closure signal is reassembled onto each recording's
  timeline, thresholded into intervals, and matched against the annotation.
* **HUST-LEBW** annotates one label per *clip* -- its per-clip positive
  fraction is 1.0 or 0.0, never partial -- so the clip is the unit and
  per-frame scoring there is meaningless.

**Read the event columns against the oracle ceiling, not against 1.0.** The
model predicts closure while the benchmark annotates events, and closure is a
strict subset: measured on every corpus carrying both labels,
``P(blink | closed) = 100%`` but ``P(closed | blink) = 19-34%``. A *perfect*
closure detector -- ground-truth closure used as its own prediction -- scores
0.9833 at ``any`` on TalkingFace but 0.0333 at ``iou50``, because a 2-frame
closure inside an 8-frame event has IoU 0.25. The IoU columns therefore measure
an annotation-convention mismatch, not the model.

Example:
    ``uv run python experiments/frame_wise_report.py --experiment frame-wise``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

STILL_CORPORA = ("cew", "mrl")
"""Corpora that annotate per-frame closure directly."""

CLIP_CORPORA = ("hust_lebw",)
"""Corpora annotated once per clip rather than per frame."""

ORACLE_ANY_F1 = 0.9833
"""Event F1 a *perfect* closure detector reaches at ``any`` on TalkingFace.

Measured by using ground-truth closure as the prediction against ground-truth
events: 59 recovered closure runs against 61 annotated blinks. Quoted beside
the model's number so a reader can see what was achievable.
"""


def read(path: Path) -> dict:
    """Load a JSON artifact, tolerating its absence.

    Args:
        path (Path): The file.

    Returns:
        dict: Its contents, or empty when the run did not produce it.
    """
    if not path.is_file():
        return {}
    return json.loads(path.read_text())


def frame_table(per_dataset: dict, target: str, title: str) -> None:
    """Per-corpus scores for one target.

    ``test_per_dataset.json`` is keyed ``target -> corpus -> metrics``. A
    corpus that does not annotate the target reports ``valid_positions: 0`` and
    is skipped rather than shown as a zero, which would read as a failure
    rather than an absence.

    Args:
        per_dataset (dict): Contents of ``test_per_dataset.json``.
        target (str): Which target to tabulate.
        title (str): Heading for the block.
    """
    corpora = per_dataset.get(target, {})
    if not corpora:
        return

    print(f"\n{title}")
    print(f"  {'corpus':12s} {'F1':>8s} {'precision':>10s} {'recall':>8s} {'n':>10s}")
    print("  " + "-" * 52)
    for corpus, row in sorted(corpora.items()):
        if not row.get("valid_positions"):
            continue
        cells = "".join(f"{100 * row[k]:9.2f}%" for k in ("f1", "precision", "recall"))
        print(f"  {corpus:12s}{cells}{int(row['valid_positions']):11d}")


def event_table(corpus_row: dict) -> None:
    """Event-level scores, with the oracle ceiling for context.

    Args:
        corpus_row (dict): The ``_corpus`` row of ``test_events.json``.
    """
    print("\nEVENT-WISE (closure signal -> intervals -> matched against blinks)")
    print(f"  {'criterion':12s} {'F1':>8s} {'precision':>10s} {'recall':>8s}")
    print("  " + "-" * 42)
    for criterion in ("any", "iou20", "iou50", "iou75"):
        cells = "".join(
            f"{100 * corpus_row[f'event/{criterion}/{k}']:9.2f}%"
            if f"event/{criterion}/{k}" in corpus_row
            else f"{'--':>10s}"
            for k in ("f1", "precision", "recall")
        )
        note = ""
        if criterion == "any":
            note = f"   <- headline (oracle ceiling {100 * ORACLE_ANY_F1:.2f}%)"
        elif criterion in ("iou50", "iou75"):
            note = "   <- convention mismatch, not model failure"
        print(f"  {criterion:12s}{cells}{note}")

    for key in ("blink_ap", "blink_ap50"):
        if f"event/{key}" in corpus_row:
            print(f"  {key:12s}{100 * corpus_row[f'event/{key}']:9.2f}%")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("results/frame-wise/frame-wise-r1"),
        help="The run's output directory.",
    )
    args = parser.parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    per_dataset = read(args.run_dir / "test_per_dataset.json")
    events = read(args.run_dir / "test_events.json")

    if not per_dataset and not events:
        print(f"No artifacts under {args.run_dir}. Has the run finished?")
        return

    print(f"=== {args.run_dir.name} ===")
    if per_dataset:
        frame_table(per_dataset, "eye_state", "FRAME-WISE eye state (the trained task)")
        frame_table(
            per_dataset,
            "blink_presence",
            "FRAME-WISE blink presence (scored, never trained)",
        )
    if events.get("_corpus"):
        event_table(events["_corpus"])

    print(
        "\nCaveats: the model is trained on closure and scored on events; the "
        "operating point is fitted on validation and applied to test; MPEblink's "
        "Blink-AP is oracle-instance and uses interval-peak as its confidence."
    )


if __name__ == "__main__":
    main()
