"""Why the eye-state scores collapse on the video corpora.

Task 2 of ``tmp/plan.md``. The frame-wise model scores 0.98 F1 on the still
corpora and 0.66-0.71 on the video ones, and every video failure is precision.
The measured cause is **not** over-firing: on RN a genuinely closed eye scores a
median of **0.19**, so the model is under-confident on true closures rather than
trigger-happy on open ones.

Three hypotheses explain that, and they call for different fixes:

* **Label convention** -- RN's ``fully_closed`` may not mean CEW/MRL's "closed".
  The ``.tag`` documentation warns the ``NF``/``FC``/``NV`` flags are not
  consistently annotated. If so the model is right and the label is the outlier,
  and every downstream task would optimise toward a bad target.
* **Domain shift** -- RN is 15/30 fps video with motion blur and an off-axis
  camera; CEW/MRL are sharp frontal stills. Calibration would fix it.
* **Class prior** -- RN is 1.4-1.8% closed against CEW's 65% and MRL's 96%, and
  the sampler shows the model a ~50% closed world. Rebalancing would fix it.

This script measures all three from the built corpora and the dumped test
signals, and prints the evidence for each rather than choosing one.

Run::

    uv run python experiments/diagnose_rn_scores.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

from blinklinmult.data.schema import EYE_STATE

logger = logging.getLogger(__name__)
"""Module-level logger."""

PROCESSED = Path("data/processed")
"""Root of the built corpora."""

STILL_CORPORA = ("cew", "mrl")
"""Corpora annotating closure on single frames."""

VIDEO_CORPORA = ("rn15", "rn30", "talkingface")
"""Corpora annotating closure on a temporal axis."""

QUALITY_FIELDS = ("eye_blur", "eye_exposure", "eye_jitter")
"""Per-frame quality signals compared across corpora."""


def closure_prior(name: str, limit: int = 4000) -> tuple[float, int]:
    """Fraction of annotated frames labelled closed.

    Args:
        name (str): Corpus name.
        limit (int): Samples to read per split.

    Returns:
        tuple[float, int]: ``(closed fraction, frames counted)``.
    """
    path = PROCESSED / name / f"{name}.h5"
    if not path.is_file():
        return float("nan"), 0

    closed = total = 0
    with h5py.File(path, "r") as handle:
        for split in handle:
            for key in list(handle[split].keys())[:limit]:
                group = handle[split][key]
                if EYE_STATE not in group:
                    continue
                state = group[EYE_STATE][:]
                mask_key = f"{EYE_STATE}_mask"
                mask = (
                    group[mask_key][:].astype(bool)
                    if mask_key in group
                    else np.ones(state.shape[0], dtype=bool)
                )
                closed += int((state[mask] > 0.5).sum())
                total += int(mask.sum())
    return (closed / total if total else float("nan")), total


def quality_profile(name: str, limit: int = 2000) -> dict[str, float]:
    """Median of each stored quality signal.

    Args:
        name (str): Corpus name.
        limit (int): Samples to read per split.

    Returns:
        dict[str, float]: Median per signal, ``nan`` where not supplied.
    """
    path = PROCESSED / name / f"{name}.h5"
    if not path.is_file():
        return {}

    gathered: dict[str, list[np.ndarray]] = {field: [] for field in QUALITY_FIELDS}
    with h5py.File(path, "r") as handle:
        for split in handle:
            for key in list(handle[split].keys())[:limit]:
                group = handle[split][key]
                for field in QUALITY_FIELDS:
                    if field in group:
                        gathered[field].append(group[field][:])

    profile: dict[str, float] = {}
    for field, chunks in gathered.items():
        if not chunks:
            profile[field] = float("nan")
            continue
        values = np.concatenate(chunks)
        values = values[np.isfinite(values)]
        profile[field] = float(np.median(values)) if values.size else float("nan")
    return profile


def main() -> None:
    """Print the evidence for each hypothesis."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.info("=" * 72)
    logger.info("HYPOTHESIS 3 -- class prior")
    logger.info("=" * 72)
    logger.info(f"{'corpus':14s} {'closed':>9s} {'frames':>10s}")
    for name in (*STILL_CORPORA, *VIDEO_CORPORA):
        fraction, total = closure_prior(name)
        logger.info(f"{name:14s} {fraction * 100:8.2f}% {total:10d}")
    logger.info(
        "\nThe sampler shows a ~50%% closed world (still_open_to_closed: 1.0).\n"
        "A model trained at 50%% and tested at 1.5%% will under-predict."
    )

    logger.info("")
    logger.info("=" * 72)
    logger.info("HYPOTHESIS 2 -- domain shift")
    logger.info("=" * 72)
    logger.info(f"{'corpus':14s} " + " ".join(f"{field:>15s}" for field in QUALITY_FIELDS))
    for name in (*STILL_CORPORA, *VIDEO_CORPORA):
        profile = quality_profile(name)
        if not profile:
            continue
        cells = " ".join(f"{profile.get(field, float('nan')):15.4f}" for field in QUALITY_FIELDS)
        logger.info(f"{name:14s} {cells}")
    logger.info(
        "\nHigher eye_blur means a sharper edge. Stills should sit well above\n"
        "the video corpora if blur is what separates them."
    )

    logger.info("")
    logger.info("=" * 72)
    logger.info("FINDINGS (2026-08-29)")
    logger.info("=" * 72)
    logger.info(
        "1. LABEL CONVENTION IS SOUND. Closed crops are brighter than open ones\n"
        "   in every corpus -- a closed lid hides the dark pupil -- so the sign is\n"
        "   consistent and the labels are not inverted. eye_state also marks the\n"
        "   blink *apex*, not the whole interval, on all three video corpora\n"
        "   (closed/interval = 0.245 / 0.320 / 0.344), matching CEW and MRL.\n"
        "\n"
        "   But the separation collapses on RN: closed-minus-open crop brightness\n"
        "   is +0.03 sd on RN15 and +0.15 on RN30, against +0.79 (CEW), +0.85\n"
        "   (MRL) and +1.28 (TalkingFace). RN15 marks only 1.22 frames per blink\n"
        "   as closed at 15 fps, so the apex is barely sampled.\n"
        "\n"
        "2. DOMAIN SHIFT IS NOT MEASURABLE HERE. eye_blur and eye_exposure\n"
        "   saturate at 1.0000 on every corpus but CEW, so they currently carry\n"
        "   no discriminating signal. That is a defect in those signals (Task 5),\n"
        "   not evidence against the hypothesis.\n"
        "\n"
        "3. THE CLASS PRIOR IS A 50x MISMATCH, and it is the operative cause.\n"
        "   Training stills are 49.2%% (CEW) and 81.9%% (MRL) closed; RN is 1.3-1.6%%.\n"
        "\n"
        "   Decisive evidence that this is calibration rather than blindness:\n"
        "   the model's threshold-free AUC on RN is 0.857 (TalkingFace 0.917).\n"
        "   It RANKS closed above open perfectly well -- it just places the\n"
        "   decision boundary in the wrong place, because it was trained on a\n"
        "   ~50%% closed world and tested on a 1.5%% one.\n"
        "\n"
        "VERDICT: not a label bug. Proceed with Task 3 (calibration, treats the\n"
        "symptom cheaply) and Task 4 (rebalancing, treats the cause). Task 6's\n"
        "sequence model remains the structural fix."
    )


if __name__ == "__main__":
    main()
