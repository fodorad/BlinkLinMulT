"""Score a frame-wise model on HUST-LEBW, at the level the corpus annotates.

HUST-LEBW is the one corpus in the benchmark that cannot be scored the usual
way, and it is worth being precise about why.

**What the corpus is.** 225 test clips of 13 frames each -- 127 labelled "blink"
and 98 "unblink" -- stored as 444 samples, since a clip contributes one sample
per eye (219 clips supply both, 6 only one). Measured on the built corpus, a
blink clip carries 7-13 annotated blink frames (median **13**, i.e. the whole
clip) and an unblink clip carries exactly **0**. So ``blink_presence`` here is
not a temporal boundary at all -- it is a clip-level class label rasterised
across the window.

**Why interval matching is the wrong tool.** The usual event protocol extracts
intervals from the predicted signal and matches them against annotated ones by
overlap or IoU. Against an annotation that spans the entire clip, IoU measures
how closely the prediction reproduces a label that was never localised, so a
*correct* detection of a short real blink inside a 13-frame clip scores badly
for being short. That is the closure-versus-event convention gap in another
guise.

**What this scores instead.** The question HUST-LEBW actually asks: *did a blink
occur in this clip?* The pipeline answers it end to end and unchanged --

    eye crops -> ESR score per frame -> hysteresis extraction -> events

-- and a clip is predicted positive when the extractor returns **at least one
event**. That uses the same model, the same fitted operating point, and the same
extractor as every other corpus; only the final reduction differs, because only
the annotation differs.

Both eyes of a clip are combined with ``max``: the corpus labels a clip as a
blink if a blink is visible, and a wink or a partly occluded eye should not mask
one that is plainly closing.

Reported: precision, recall, F1 and accuracy over clips, plus the confusion
counts. **Not** IoU-based event metrics, which are not defined here.

Run::

    uv run python experiments/eval_hust_lebw.py --arm fw-focal-augment
    uv run python experiments/eval_hust_lebw.py --arm fw-focal-augment --split valid
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from blinklinmult.train.events import to_intervals

logger = logging.getLogger(__name__)
"""Module-level logger."""

CORPUS = Path("data/processed/hust_lebw/hust_lebw.h5")
"""The built corpus."""

RESULTS = Path("results/frame-wise")
"""Where the frame-wise arms wrote their checkpoints and fitted thresholds."""

BATCH = 64
"""Clips per forward pass. One clip is 13 frames x 2 eyes."""


def load_threshold(arm: str) -> tuple[float, float | None]:
    """Read an arm's validation-fitted operating point.

    Args:
        arm (str): Run name, e.g. ``fw-focal-augment``.

    Returns:
        tuple[float, float | None]: ``(high, low)``; ``low`` is ``None`` when the
        arm fitted a single threshold.

    Raises:
        FileNotFoundError: If the arm has no cached threshold.
    """
    path = RESULTS / arm / "hysteresis" / "event_threshold.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist. Run experiments/refit_arms.sh first so the "
            "operating point is fitted on validation rather than guessed."
        )
    cached = json.loads(path.read_text())
    high = float(cached["threshold"])
    ratio = cached.get("low_ratio")
    return high, (None if ratio is None else high * float(ratio))


def clip_scores(arm: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    """Run the model over one split and return per-clip signals and labels.

    The two eyes of a clip are reduced with ``max`` at each frame, so a clip's
    signal is ``(13,)`` however many eyes contributed.

    Args:
        arm (str): Run name whose ``best.ckpt`` is scored.
        split (str): ``train``, ``valid`` or ``test``.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(signals, labels)`` of shape
        ``(clips, frames)`` and ``(clips,)``.
    """
    import h5py

    from blinklinmult.train.module import BlinkLightningModule

    checkpoint = RESULTS / arm / "checkpoints" / "best.ckpt"
    module = BlinkLightningModule.load_from_checkpoint(checkpoint, map_location="cpu")
    module.eval()

    per_clip: dict[str, list[np.ndarray]] = {}
    labels: dict[str, float] = {}

    with h5py.File(CORPUS, "r") as handle:
        keys = sorted(handle[split].keys())
        for start in range(0, len(keys), BATCH):
            chunk = keys[start : start + BATCH]
            images, masks, clips = [], [], []
            for key in chunk:
                group = handle[split][key]
                images.append(np.asarray(group["eye_image"][:], dtype=np.float32))
                masks.append(np.asarray(group["eye_image_mask"][:], dtype=bool))
                clip = key.split("|")[0]
                clips.append(clip)
                labels[clip] = float(np.max(group["blink_presence"][:]) > 0.5)

            batch_images = torch.from_numpy(np.stack(images))
            batch_masks = torch.from_numpy(np.stack(masks))
            with torch.no_grad():
                output = module.model(batch_images, batch_masks)
            # The eye-state head: the ESR score the whole architecture is built
            # around. `blink_presence` has no head on a frame-wise model.
            scores = torch.sigmoid(output["eye_state"]).squeeze(-1).cpu().numpy()

            for index, clip in enumerate(clips):
                per_clip.setdefault(clip, []).append(scores[index])

    names = sorted(per_clip)
    # `max` over the eyes: the corpus labels a clip a blink if a blink is
    # visible, so one plainly closing eye is enough.
    signals = np.stack([np.max(np.stack(per_clip[name]), axis=0) for name in names])
    return signals, np.asarray([labels[name] for name in names], dtype=np.float32)


def score_clips(
    signals: np.ndarray,
    labels: np.ndarray,
    high: float,
    low: float | None,
) -> dict[str, float]:
    """Reduce per-frame signals to a clip decision and score it.

    A clip is positive when the extractor returns at least one event -- the same
    extractor, at the same operating point, as every other corpus.

    Args:
        signals (np.ndarray): ``(clips, frames)`` ESR scores.
        labels (np.ndarray): ``(clips,)`` binary clip labels.
        high (float): Hysteresis high threshold.
        low (float | None): Hysteresis low threshold, or ``None``.

    Returns:
        dict[str, float]: Precision, recall, F1, accuracy and confusion counts.
    """
    mask = np.ones(signals.shape[1], dtype=bool)
    predicted = np.asarray(
        [len(to_intervals(signal, mask, high, low)) > 0 for signal in signals],
        dtype=bool,
    )
    positive = labels > 0.5

    tp = float(np.sum(predicted & positive))
    fp = float(np.sum(predicted & ~positive))
    fn = float(np.sum(~predicted & positive))
    tn = float(np.sum(~predicted & ~positive))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1.0),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> None:
    """Score one or more arms on HUST-LEBW."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        action="append",
        default=None,
        help="Run name to score; repeatable. Defaults to all four frame-wise arms.",
    )
    parser.add_argument("--split", default="test", help="Split to score (default: test).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    arms = args.arm or ["fw-bce", "fw-bce-augment", "fw-focal", "fw-focal-augment"]

    logger.info(f"HUST-LEBW, {args.split} split -- CLIP-LEVEL blink detection")
    logger.info(
        "A clip is positive when the extractor returns at least one event, at "
        "the arm's own\nvalidation-fitted operating point. IoU metrics are not "
        "reported: the annotation spans\nthe whole clip, so it carries no "
        "temporal boundary to match against."
    )
    logger.info("")
    header = (
        f"{'arm':20s} {'high':>5s} {'low':>6s} {'F1':>7s} {'prec':>7s} "
        f"{'rec':>7s} {'acc':>7s} {'TP':>5s} {'FP':>5s} {'FN':>5s} {'TN':>5s}"
    )
    logger.info(header)

    for arm in arms:
        try:
            high, low = load_threshold(arm)
            signals, labels = clip_scores(arm, args.split)
        except (FileNotFoundError, OSError) as error:
            logger.warning(f"{arm}: skipped ({error})")
            continue
        scores = score_clips(signals, labels, high, low)

        # Written into the arm's own run directory so `benchmark_table.py` can
        # read it. Without this the clip-level number exists only in this
        # script's stdout, and the benchmark table shows `--` for a corpus that
        # was in fact evaluated.
        destination = RESULTS / arm / "hust_lebw_clips.json"
        destination.write_text(
            json.dumps(
                {
                    **{k: float(v) for k, v in scores.items()},
                    "split": args.split,
                    "threshold": high,
                    "low_threshold": low,
                    "protocol": "clip-level",
                },
                indent=2,
            )
        )

        shown = f"{low:6.3f}" if low is not None else "     -"
        logger.info(
            f"{arm:20s} {high:5.2f} {shown} {scores['f1']:7.4f} {scores['precision']:7.4f} "
            f"{scores['recall']:7.4f} {scores['accuracy']:7.4f} "
            f"{int(scores['tp']):5d} {int(scores['fp']):5d} "
            f"{int(scores['fn']):5d} {int(scores['tn']):5d}"
        )


if __name__ == "__main__":
    main()
