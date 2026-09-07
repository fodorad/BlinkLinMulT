"""Score the shipped models on a corpus, so they can be compared with each other.

The four published models had never been evaluated side by side. Everything in
``results/`` is a *training arm* -- ``fw-bce``, ``fw-focal-augment`` and so on --
which answers "did this training change help?" and not "which of the models I
ship is best?". This script answers the second.

It writes one archive per model in the same shape
:class:`~blinklinmult.train.callbacks.EventReport` writes, so
``tools/compare_models.py`` consumes it unchanged and the statistics never
re-run a model.

**Scored on eye state, not blink presence.** Per-frame closure is what all four
models actually predict. The 1.x models were published as eye-state recognisers;
turning their output into blink *events* needs an operating point, and three of
the four carry ``threshold=0.5`` as an unmeasured placeholder (see
``registry.py``). Comparing events would measure calibration effort, not the
models. Average precision over the closure signal needs no threshold at all.

**Both eyes are scored and kept separate**, then pooled per recording, matching
how the corpora store them.

Run::

    uv run python tools/score_models.py --corpus rn30 --split test
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)
"""Module-level logger."""

SIGNALS_TARGET_KEY = "_target"
"""Provenance key naming which target ``truth`` holds.

Written so :func:`~blinklinmult.train.callbacks.load_signals` can refuse a
mismatch: reading a ``blink_presence`` archive as ``eye_state`` once fabricated
200 045 phantom false positives.
"""

TARGET = "eye_state"
"""The target scored here. See the module docstring for why not events."""


def _feature_stats(directory: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Corpus feature statistics for the two-stream model.

    Args:
        directory (Path): Where the artifacts live.

    Returns:
        tuple[np.ndarray, np.ndarray] | None: Mean and standard deviation, or
        ``None`` when absent.
    """
    path = directory / "rn30_feature_stats.json"
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return (
        np.asarray(payload["mean"], dtype="float32"),
        np.asarray(payload["std"], dtype="float32"),
    )


def score_corpus(
    model_id: str,
    corpus_path: Path,
    split: str,
    weights_dir: Path,
    stats: tuple[np.ndarray, np.ndarray] | None,
    limit: int | None = None,
    stride: int = 1,
) -> dict[str, dict[str, np.ndarray]]:
    """Score every window of a split, pooled per recording.

    Args:
        model_id (str): Registry id.
        corpus_path (Path): The corpus ``.h5``.
        split (str): Which split to score.
        weights_dir (Path): Where local weights live.
        stats (tuple[np.ndarray, np.ndarray] | None): Feature statistics.
        limit (int | None): Score at most this many windows, for a smoke run.
        stride (int): Score every ``stride``-th window. Because the corpus keys
            sort by recording, a stride still touches every recording -- at 6 it
            covers all 35 rn30 recordings with 8-67 windows each -- so the
            cluster bootstrap keeps its full sample size while the scoring pass
            costs a sixth as much. Use 1 for the definitive run.

    Returns:
        dict[str, dict[str, np.ndarray]]: ``{recording: {signal, truth, mask}}``.

    Raises:
        SystemExit: If the two-stream model is asked for without statistics.
    """
    import h5py

    from blinklinmult import BlinkDetector
    from blinklinmult.registry import spec

    model_spec = spec(model_id)
    if model_spec.needs_features and stats is None:
        raise SystemExit(f"{model_id} needs feature statistics; none found in {weights_dir}")

    detector = BlinkDetector.from_pretrained(model_id, weights=weights_dir / model_spec.filename)

    pooled: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    with h5py.File(corpus_path, "r") as handle:
        group = handle[split]
        keys = sorted(group.keys())[::stride]
        if limit is not None:
            keys = keys[:limit]

        for index, key in enumerate(keys):
            record = group[key]
            crops = np.asarray(record["eye_image"], dtype="float32")
            truth = np.asarray(record["eye_state"], dtype="float32")
            mask = np.asarray(record["eye_state_mask"], dtype=bool)

            kwargs: dict = {}
            if model_spec.needs_features:
                features = np.asarray(record["eye_feature"], dtype="float32")
                kwargs["features"] = detector.prepare_features(
                    features, mean=stats[0], std=stats[1]
                )

            # The 1.x sequence models were trained on 15-frame windows and lose
            # a lot of accuracy on longer ones (92% -> 72% on blinklint), so
            # `score_long` slides their native window and averages the overlap.
            # The frame-wise models take the whole window in one call.
            signal = (
                detector.score_long(crops, **kwargs)
                if model_spec.window
                else detector.score(crops, **kwargs)
            )

            recording = record["video_id"][()]
            recording = recording.decode() if isinstance(recording, bytes) else str(recording)
            pooled[recording].append((np.asarray(signal, dtype="float32"), truth, mask))

            if index and index % 500 == 0:
                logger.info(f"  {model_id}: {index}/{len(keys)} windows")

    return {
        recording: {
            "signal": np.concatenate([part[0] for part in parts]),
            "truth": np.concatenate([part[1] for part in parts]),
            "mask": np.concatenate([part[2] for part in parts]),
        }
        for recording, parts in pooled.items()
    }


def main() -> None:
    """Score every shipped model on one corpus split."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="rn30", help="Corpus name.")
    parser.add_argument("--split", default="test", help="Split to score.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--weights-dir", type=Path, default=Path("artifacts/onnx"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/shipped"))
    parser.add_argument(
        "--limit", type=int, default=None, help="Windows to score, for a smoke run."
    )
    parser.add_argument("--stride", type=int, default=1, help="Score every Nth window.")
    parser.add_argument("--model", action="append", default=None, help="Model id; repeatable.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from blinklinmult.registry import MODELS

    corpus_path = args.data_dir / args.corpus / f"{args.corpus}.h5"
    if not corpus_path.is_file():
        raise SystemExit(f"no corpus at {corpus_path}; run `make pull-{args.corpus}`")

    stats = _feature_stats(args.weights_dir)
    model_ids = args.model or list(MODELS)

    for model_id in model_ids:
        weights = args.weights_dir / MODELS[model_id].filename
        if not weights.is_file():
            logger.info(f"{model_id}: no local weights at {weights}, skipping")
            continue

        logger.info(f"scoring {model_id} on {args.corpus}/{args.split}")
        signals = score_corpus(
            model_id, corpus_path, args.split, args.weights_dir, stats, args.limit, args.stride
        )

        destination = args.out_dir / f"{model_id}-{args.corpus}"
        destination.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {SIGNALS_TARGET_KEY: np.array(TARGET)}
        for recording, arrays in signals.items():
            for field, values in arrays.items():
                payload[f"{recording}/{field}"] = values
        np.savez_compressed(destination / "test_signals.npz", **payload)

        frames = sum(int(arrays["mask"].sum()) for arrays in signals.values())
        logger.info(f"  {len(signals)} recordings, {frames} valid frames -> {destination}")


if __name__ == "__main__":
    main()
