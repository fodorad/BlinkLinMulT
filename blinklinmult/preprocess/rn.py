"""Preprocess the Researcher's Night (RN) corpus.

RN is recorded at two frame rates — ``rn15`` and ``rn30`` — and ships its own
train/val/test division as directories::

    data/raw/RN/
        train/rn30/<id>/<id>.avi, .tag, .txt
        val/rn30/...
        test/rn15/...

**The two rates are separate corpora.** They are reported separately in the
benchmark, and they cannot share a declaration because the analysis window is
declared in seconds and each rate derives a different frame count from it — 8
frames at 15 fps against 15 at 30 fps for the same half-second.
``config/data/rn.yaml`` lists both for the aggregate run.

**The corpus's own splits are honoured** rather than re-derived. They divide by
participant, which is the property that matters, and re-splitting would make
published RN numbers incomparable with this project's.

Example:
    ``uv run python -m blinklinmult.preprocess.rn --rate 30``
"""

from __future__ import annotations

import argparse
import logging
from functools import partial
from typing import TYPE_CHECKING

from blinklinmult import PROJECT_ROOT
from blinklinmult.data.schema import DEFAULT_WINDOW_SECONDS
from blinklinmult.preprocess.common import PreprocessError
from blinklinmult.preprocess.video_corpus import (
    VideoCorpusLayout,
    add_cli_arguments,
    process,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

RATES: dict[int, str] = {15: "rn15", 30: "rn30"}
"""Maps a recording rate to this project's corpus name for it."""

SPLIT_DIRS: dict[str, str] = {"train": "train", "val": "valid", "test": "test"}
"""Maps the corpus's split directory names onto this project's split names."""

NOT_A_RECORDING = frozenset({"18trainvalLeftRight_MergedDoubleBlinks_eval.txt"})
"""Files that sit alongside the recording directories but are not recordings.

The corpus ships an evaluation list in ``test/rn30``; globbing for ``.tag``
files skips it naturally, but it is named here so its presence is documented
rather than rediscovered.
"""


def layout(rate: int, root: Path = PROJECT_ROOT, **overrides) -> VideoCorpusLayout:
    """Build the layout for one of RN's two rates.

    Args:
        rate (int): ``15`` or ``30``.
        root (Path): Repository root.
        **overrides: Fields to override on the layout.

    Returns:
        VideoCorpusLayout: The layout, whose ``tag_glob`` spans every split of
        this rate only.

    Raises:
        PreprocessError: If the rate is not one RN was recorded at.
    """
    if rate not in RATES:
        raise PreprocessError(f"RN has no {rate} fps recordings; expected {sorted(RATES)}.")

    name = RATES[rate]
    defaults = {
        "name": name,
        "raw_dir": root / "data" / "raw" / "RN",
        "processed_dir": root / "data" / "processed" / name,
        "fps": float(rate),
        # Every split of this rate, and no other rate's.
        "tag_glob": f"*/{name}/*/*.tag",
        "window_seconds": DEFAULT_WINDOW_SECONDS,
        # A continuous sweep at half the window, so evaluation sees the
        # recording as deployed rather than as a balanced sample.
        "eval_stride": max(1, round(rate * DEFAULT_WINDOW_SECONDS / 2)),
    }
    return VideoCorpusLayout(**{**defaults, **overrides})


def split_of_path(tag_path: Path, raw_dir: Path) -> str:
    """Read a recording's split off its path.

    Args:
        tag_path (Path): The recording's ``.tag`` file.
        raw_dir (Path): The corpus root.

    Returns:
        str: This project's split name.

    Raises:
        PreprocessError: If the path does not have the expected
            ``<split>/<rate>/<id>/`` shape.
    """
    try:
        relative = tag_path.relative_to(raw_dir).parts
    except ValueError as error:
        raise PreprocessError(f"{tag_path} is not inside {raw_dir}.") from error

    if len(relative) < 3:
        raise PreprocessError(
            f"{tag_path}: expected <split>/<rate>/<id>/<file>.tag under {raw_dir}."
        )

    split_dir = relative[0]
    if split_dir not in SPLIT_DIRS:
        raise PreprocessError(
            f"{tag_path}: unknown split directory {split_dir!r}; expected one of "
            f"{sorted(SPLIT_DIRS)}."
        )
    return SPLIT_DIRS[split_dir]


def recording_name(tag_path: Path, raw_dir: Path) -> str:
    """Name one RN recording, uniquely across the whole corpus.

    RN restarts its numbering inside every split directory, so ``train/rn15/1``,
    ``val/rn15/1``, and ``test/rn15/1`` are three different recordings that all
    sit in a directory called ``1``. Naming them by that directory alone gives
    all three the same id, and their sample keys then collide inside the HDF5 --
    which is what the writer refuses to do.

    Args:
        tag_path (Path): The recording's ``.tag`` file.
        raw_dir (Path): The corpus root.

    Returns:
        str: ``<split>_<number>``, e.g. ``train_1``.
    """
    return f"{split_of_path(tag_path, raw_dir)}_{tag_path.parent.name}"


def process_rn(
    corpus: VideoCorpusLayout,
    limit: int | None = None,
    device_id: int | None = None,
) -> Path:
    """Build one RN rate, honouring the corpus's own splits.

    The only thing RN does differently is where its splits come from: they are
    given as directories rather than derived, and re-deriving them would make
    published RN numbers incomparable with this project's. Everything else is
    the shared pipeline.

    Args:
        corpus (VideoCorpusLayout): The corpus layout.
        limit (int | None): Process only the first N recordings.
        device_id (int | None): GPU index for the extractors; ``None`` is CPU.

    Returns:
        Path: The written HDF5 file.

    Raises:
        PreprocessError: If the corpus cannot be processed.
    """
    return process(
        corpus,
        limit=limit,
        device_id=device_id,
        split_of=partial(split_of_path, raw_dir=corpus.raw_dir),
        name_of=partial(recording_name, raw_dir=corpus.raw_dir),
    )


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_cli_arguments(parser)
    parser.add_argument(
        "--rate",
        type=int,
        choices=sorted(RATES),
        default=None,
        help="Process only this rate; both are processed when omitted.",
    )
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    for rate in [args.rate] if args.rate else sorted(RATES):
        process_rn(
            layout(
                rate,
                args.root,
                image_size=args.image_size,
                window_seconds=args.window_seconds,
                eval_stride=args.stride,
                with_features=not args.no_features,
            ),
            limit=args.limit,
            device_id=args.device,
        )


if __name__ == "__main__":
    main()
