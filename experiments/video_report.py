"""Assemble the video-benchmark table from finished runs.

Reads the per-corpus artifacts written by ``experiments/eval_video.sh`` and prints
one row per (arm, corpus). **Per dataset, never pooled** -- the corpora sit at
different points on the difficulty curve and a pooled mean hides all of it:

* **RN15 / RN30** -- ordinary notebook webcams under varying lighting, and until
  recently one of the most widely used benchmarks for eye-state recognition and
  blink-presence detection. This is the headline number, directly comparable to
  the literature.
* **TalkingFace** -- one held-out video. The target is precision 1.0 with zero
  false positives and recall 1.0; anything less is a countable defect rather
  than a statistical shortfall.
* **MPEblink** -- in-the-wild movie footage with instance-level annotation, new
  and largely unsolved. Strong numbers are the goal, parity with RN is not
  expected, and falloff here is an honest result rather than a hidden failure.
* **HUST-LEBW** -- clip-wise annotation, so ``any`` is the only comparable
  criterion; IoU-based numbers would measure the annotation convention.

The frame-wise best arm is printed as the first row of the event table, so the
temporal contribution is visible in one place rather than across two documents.

Run::

    uv run python experiments/video_report.py
    uv run python experiments/video_report.py --frame-wise-only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

RESULTS = Path("results")
"""Root of the run outputs."""

VIDEO_ARMS = (
    "vid-lint-scratch",
    "vid-lint-frozen",
    "vid-lint-unfreeze",
    "vid-lint-ft",
    "vid-mult-scratch",
    "vid-mult-frozen",
    "vid-mult-unfreeze",
    "vid-mult-ft",
)
"""The video arms, in the order the matrix defines them.

``unfreeze`` follows its ``frozen`` stage directly, because it *is* that run
continued with the encoder trainable: the two rows next to each other are the
whole question -- did unfreezing beat the checkpoint it started from?
"""

FRAME_WISE_ARMS = ("fw-bce", "fw-bce-augment", "fw-focal", "fw-focal-augment")
"""Frame-wise arms, for the baseline rows."""

EVENT_CORPORA = ("rn15", "rn30", "talkingface", "mpeblink", "hust_lebw")
"""Corpora with a temporal axis, in report order."""

ESR_CORPORA = ("rn15", "rn30", "talkingface")
"""Corpora annotating per-frame closure.

MPEblink and HUST-LEBW are absent by construction: they annotate blink
*intervals* and carry no per-frame closure label, so a frame-level row for them
would measure the closure-versus-event convention gap rather than the model.
"""


def _corpus_row(path: Path) -> dict[str, float] | None:
    """Read one run's corpus-level event scores.

    Args:
        path (Path): The run's ``test_events.json``.

    Returns:
        dict[str, float] | None: The ``_corpus`` row, or ``None`` if absent.
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text()).get("_corpus")
    except (OSError, ValueError):
        logger.warning(f"Could not read {path}.")
        return None


def event_table(arms: tuple[str, ...], template: str) -> list[tuple]:
    """Collect event-level rows for a set of arms.

    Args:
        arms (tuple[str, ...]): Run names.
        template (str): Directory template taking ``arm`` and ``corpus``.

    Returns:
        list[tuple]: ``(arm, corpus, f1, precision, recall, fa_per_min, tp, fp, fn)``.
    """
    rows = []
    for arm in arms:
        for corpus in EVENT_CORPORA:
            run = RESULTS / template.format(arm=arm, corpus=corpus)
            row = _corpus_row(run / "test_events.json")
            if row is None:
                continue
            rows.append(
                (
                    arm,
                    corpus,
                    row.get("event/any/f1", 0.0),
                    row.get("event/any/precision", 0.0),
                    row.get("event/any/recall", 0.0),
                    row.get("event/any/fa_per_min", 0.0),
                    int(row.get("event/any/tp", 0)),
                    int(row.get("event/any/fp", 0)),
                    int(row.get("event/any/fn", 0)),
                )
            )
    return rows


def esr_table(arms: tuple[str, ...], template: str) -> list[tuple]:
    """Collect frame-level rows for a set of arms.

    Args:
        arms (tuple[str, ...]): Run names.
        template (str): Directory template taking ``arm`` and ``corpus``.

    Returns:
        list[tuple]: ``(arm, corpus, n, f1, precision, recall)``.
    """
    rows = []
    for arm in arms:
        for corpus in ESR_CORPORA:
            path = RESULTS / template.format(arm=arm, corpus=corpus) / "test_per_dataset.json"
            if not path.is_file():
                continue
            try:
                per_target = json.loads(path.read_text()).get("eye_state", {})
            except (OSError, ValueError):
                continue
            scores = per_target.get(corpus)
            if not scores or int(scores.get("valid_positions", 0)) == 0:
                continue
            rows.append(
                (
                    arm,
                    corpus,
                    int(scores["valid_positions"]),
                    scores.get("f1", 0.0),
                    scores.get("precision", 0.0),
                    scores.get("recall", 0.0),
                )
            )
    return rows


def main() -> None:
    """Print the benchmark tables."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--frame-wise-only",
        action="store_true",
        help="Report only the frame-wise baseline, before any video arm has run.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    sources = [(FRAME_WISE_ARMS, "frame-wise/{arm}-hyst-{corpus}", "FRAME-WISE (baseline)")]
    if not args.frame_wise_only:
        sources.append((VIDEO_ARMS, "blink-video/{arm}-eval-{corpus}", "VIDEO (sequence models)"))

    logger.info("EVENT LEVEL -- `any` criterion, per dataset")
    logger.info(
        f"{'arm':20s} {'corpus':12s} {'F1':>7s} {'prec':>7s} {'rec':>7s} "
        f"{'FA/min':>7s} {'TP':>6s} {'FP':>6s} {'FN':>6s}"
    )
    for arms, template, label in sources:
        rows = event_table(arms, template)
        if not rows:
            logger.info(f"  -- {label}: nothing scored yet --")
            continue
        logger.info(f"  -- {label} --")
        for arm, corpus, f1, precision, recall, rate, tp, fp, fn in rows:
            logger.info(
                f"{arm:20s} {corpus:12s} {f1:7.4f} {precision:7.4f} {recall:7.4f} "
                f"{rate:7.2f} {tp:6d} {fp:6d} {fn:6d}"
            )

    logger.info("")
    logger.info("FRAME LEVEL (ESR) -- per dataset")
    logger.info(f"{'arm':20s} {'corpus':12s} {'n':>8s} {'F1':>7s} {'prec':>7s} {'rec':>7s}")
    for arms, template, label in sources:
        rows = esr_table(arms, template)
        if not rows:
            continue
        logger.info(f"  -- {label} --")
        for arm, corpus, count, f1, precision, recall in rows:
            logger.info(
                f"{arm:20s} {corpus:12s} {count:8d} {f1:7.4f} {precision:7.4f} {recall:7.4f}"
            )

    logger.info("")
    logger.info(
        "Report per dataset, never pooled: RN15/RN30 are the established webcam\n"
        "benchmark, TalkingFace is a one-video correctness check targeting 1.0,\n"
        "MPEblink is new and largely unsolved, and HUST-LEBW is clip-wise so only\n"
        "`any` is comparable there."
    )


if __name__ == "__main__":
    main()
