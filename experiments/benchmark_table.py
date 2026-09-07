"""The benchmark table: one row per model, one column per corpus.

Reports **RN as the mean of RN15 and RN30**, since they are the same recording
setup at two frame rates and a single number is what the literature quotes.
HUST-LEBW is included from its clip-level evaluation, which is the only level it
annotates.

Two tables, never merged:

* **Frame-level ESR** -- per-frame closure, on the corpora that annotate it.
* **Event-level** -- blink intervals, on every corpus with a temporal axis.

Merging them would compare different quantities: a blink interval covers 3-4x
more frames than actual closure (measured P(closed | inside a blink) = 0.237 /
0.310 / 0.344 on RN15 / RN30 / TalkingFace).

Run::

    uv run python experiments/benchmark_table.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

RESULTS = Path("results")
"""Root of the run outputs."""

RN = ("rn15", "rn30")
"""The Researcher's Night corpora.

**Reported separately, not averaged.** They are the same webcam setup at two
frame rates, but the rates change the task: at 15 fps a blink's closed apex is a
median of 1.22 frames against 2 at 30, so RN15 is measurably harder and an
average hides which rate a deficit belongs to.
"""

ESR_COLUMNS = ("cew", "mrl", "rn15", "rn30", "talkingface")
"""Corpora annotating per-frame closure.

The still corpora come first: they are single-frame, so they measure the encoder
alone, and a deficit there would explain a sequence-model deficit rather than
being explained by it. They are also where the model is strongest.

MPEblink and HUST-LEBW annotate intervals only, so a frame-level number for them
would measure the closure-versus-event convention gap rather than the model.
"""

STILL_CORPORA = ("cew", "mrl")
"""Single-frame corpora, scored only for ESR.

Read from the arm's own run directory rather than a per-corpus evaluation: the
frame-wise training run already scores them, and no separate pass was ever run.
"""

EVENT_COLUMNS = ("rn15", "rn30", "talkingface", "hust_lebw", "mpeblink")
"""Corpora with a temporal axis."""

FRAME_WISE_ARMS = ("fw-bce", "fw-bce-augment", "fw-focal", "fw-focal-augment")
"""The frame-wise arms, which are the current baseline."""


def _read(path: Path) -> dict | None:
    """Read a JSON artifact if it exists.

    Args:
        path (Path): The artifact.

    Returns:
        dict | None: Its contents, or ``None``.
    """
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        logger.warning(f"Could not read {path}.")
        return None


def _mean(values: list[float]) -> float | None:
    """Unweighted mean, or ``None`` when nothing was measured.

    Args:
        values (list[float]): Per-corpus scores.

    Returns:
        float | None: Their mean.
    """
    return sum(values) / len(values) if values else None


def frame_wise_esr(arm: str) -> dict[str, float]:
    """Per-column frame-level F1 for one frame-wise arm.

    Args:
        arm (str): Run name.

    Returns:
        dict[str, float]: Keyed by :data:`ESR_COLUMNS`.
    """
    scores: dict[str, float] = {}

    for corpus in RN:
        payload = _read(RESULTS / "frame-wise" / f"{arm}-hyst-{corpus}" / "test_per_dataset.json")
        entry = (payload or {}).get("eye_state", {}).get(corpus, {})
        if entry.get("valid_positions", 0):
            scores[corpus] = entry["f1"]

    payload = _read(RESULTS / "frame-wise" / f"{arm}-hyst-talkingface" / "test_per_dataset.json")
    entry = (payload or {}).get("eye_state", {}).get("talkingface", {})
    if entry.get("valid_positions", 0):
        scores["talkingface"] = entry["f1"]

    # The stills have no per-corpus run: the training run scored them directly.
    main = _read(RESULTS / "frame-wise" / arm / "test_per_dataset.json") or {}
    for corpus in STILL_CORPORA:
        entry = main.get("eye_state", {}).get(corpus, {})
        if entry.get("valid_positions", 0):
            scores[corpus] = entry["f1"]
    return scores


def frame_wise_events(arm: str) -> dict[str, float]:
    """Per-column event-level F1 for one frame-wise arm.

    HUST-LEBW comes from ``experiments/eval_hust_lebw.py`` rather than the event
    report: it annotates clips, not intervals, so the usual protocol does not
    apply to it. Absent unless that script has been run.

    Args:
        arm (str): Run name.

    Returns:
        dict[str, float]: Keyed by :data:`EVENT_COLUMNS`.
    """
    scores: dict[str, float] = {}

    for corpus in RN:
        payload = _read(RESULTS / "frame-wise" / f"{arm}-hyst-{corpus}" / "test_events.json")
        if payload:
            scores[corpus] = payload["_corpus"]["event/any/f1"]

    for corpus in ("talkingface", "mpeblink"):
        payload = _read(RESULTS / "frame-wise" / f"{arm}-hyst-{corpus}" / "test_events.json")
        if payload:
            scores[corpus] = payload["_corpus"]["event/any/f1"]

    clips = _read(RESULTS / "frame-wise" / arm / "hust_lebw_clips.json")
    if clips:
        scores["hust_lebw"] = clips.get("f1", 0.0)
    return scores


def _table(title: str, columns: tuple[str, ...], rows: list[tuple[str, dict[str, float]]]) -> None:
    """Print one table.

    Args:
        title (str): Heading.
        columns (tuple[str, ...]): Column order.
        rows (list[tuple[str, dict[str, float]]]): ``(label, scores)`` pairs.
    """
    logger.info(title)
    logger.info(f"{'model':26s}" + "".join(f"{c:>13s}" for c in columns))
    for label, scores in rows:
        cells = "".join(f"{scores[c]:13.4f}" if c in scores else f"{'--':>13s}" for c in columns)
        logger.info(f"{label:26s}{cells}")


def main() -> None:
    """Print both tables."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    _table(
        "FRAME-LEVEL ESR (eye_state) F1",
        ESR_COLUMNS,
        [(arm, frame_wise_esr(arm)) for arm in FRAME_WISE_ARMS],
    )
    logger.info("")
    _table(
        "EVENT-LEVEL (any) F1",
        EVENT_COLUMNS,
        [(arm, frame_wise_events(arm)) for arm in FRAME_WISE_ARMS],
    )
    logger.info(
        "\nrn15 and rn30 are reported separately: the frame rate changes the task,"
        "\nsince a blink's closed apex is a median 1.22 frames at 15 fps against 2"
        "\nat 30. hust_lebw is clip-level, from experiments/eval_hust_lebw.py."
    )


if __name__ == "__main__":
    main()
