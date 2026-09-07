"""Preprocess the TalkingFace corpus into its HDF5 file.

A single ~200-second recording of one subject talking to camera, with a ``.tag``
annotation giving per-frame blink ids, eye visibility, and both eyes' corners.

**Held out entirely as a test set.** One recording of one subject cannot be
split: a train/test division within continuous footage of the same face
measures memorisation, not generalisation. TalkingFace is the standard
cross-corpus check in the blink literature for exactly that reason, so every
sample goes to ``test`` and the corpus never contributes a gradient.

Expected layout — flat, unlike the other corpora::

    data/raw/TalkingFace/
        talking.avi
        talking.tag        frame_id:blink_id:...:eye corners
        talking.txt        frame_id timestamp

Everything after locating those files is
:mod:`~blinklinmult.preprocess.video_corpus`: one pass decodes, locates both
eyes from the annotation, describes them with exordium, crops, and writes
``data/processed/talkingface/talkingface.h5``.

Example:
    ``uv run python -m blinklinmult.preprocess.talkingface``
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from blinklinmult import PROJECT_ROOT
from blinklinmult.preprocess.video_corpus import VideoCorpusLayout, run_cli

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

NAME = "talkingface"
"""Corpus name, matching its config and processed directory."""

FPS = 25.0
"""Native frame rate of the recording.

Not the 30 the ``.avi`` container advertises. The corpus README states "5000
frames ... about 200 seconds", and ``talking.txt`` agrees: its timestamps step
by 0.04 s. The container's metadata is what disagrees, and trusting it would
give this corpus 0.6-second windows while every other corpus got 0.5.
"""

HELD_OUT_SPLIT = {"train": 0.0, "valid": 0.0, "test": 1.0}
"""Every recording goes to test.

This corpus exists in the benchmark to answer "does a model trained elsewhere
work here?", which requires that none of it was ever trained on.
"""


def layout(root: Path = PROJECT_ROOT, **overrides) -> VideoCorpusLayout:
    """Build the corpus layout.

    Args:
        root (Path): Repository root.
        **overrides: Fields to override on the layout.

    Returns:
        VideoCorpusLayout: The layout.
    """
    defaults = {
        "name": NAME,
        "raw_dir": root / "data" / "raw" / "TalkingFace",
        "processed_dir": root / "data" / "processed" / NAME,
        "fps": FPS,
        # The corpus keeps its three files loose in its root rather than in a
        # per-recording directory, so the recording is named after the .tag
        # file; the parent here is the corpus itself.
        "tag_glob": "*.tag",
        "video_id_from_stem": True,
        "split_ratios": HELD_OUT_SPLIT,
    }
    return VideoCorpusLayout(**{**defaults, **overrides})


def main() -> None:
    """CLI entry point."""
    run_cli(layout, __doc__.splitlines()[0])


if __name__ == "__main__":
    main()
