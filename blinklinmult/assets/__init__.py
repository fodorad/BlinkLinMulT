"""Small files that ship with the package.

Only the demo clip lives here, and only because a demo with nothing to run is
not a demo: a fresh ``pip install`` should be able to show the pipeline working
without the user first obtaining a corpus.

The clip is the **first 10 seconds** of TalkingFace, 1.6 MB rather than the
22 MB source, carrying three annotated blinks (frames 168, 225 and 274) and its
matching ``.tag`` slice so a prediction can be read against the annotation.

Nothing else belongs here. Model weights are downloaded from the Hub on first
use and cached; the corpora are gigabytes and stay out of the wheel entirely.
"""

from __future__ import annotations

from pathlib import Path

ASSETS_DIR = Path(__file__).resolve().parent
"""Directory this module lives in, and the files sit beside it."""

EXAMPLE_VIDEO = ASSETS_DIR / "talkingface_10s.mp4"
"""The bundled demo clip: TalkingFace, first 10 s at 30 fps."""

EXAMPLE_TAG = ASSETS_DIR / "talkingface_10s.tag"
"""Its annotation, sliced to the same 300 frames.

Frame ids are unchanged from the source, so a prediction on this clip lines up
with the published TalkingFace numbers.
"""

__all__ = ["ASSETS_DIR", "EXAMPLE_TAG", "EXAMPLE_VIDEO"]
