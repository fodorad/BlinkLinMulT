"""Researcher's Night at 30 fps.

A swept window over a continuous recording, at twice the frame rate of
RN15, so a window covers the same 1.5 s in 45 frames rather than 23.

Move the slider to step through samples. Each row shows the window's frames with
the sample key above them; a **red** border marks a frame the model never sees.
"""

# %%
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd().parents[1]
sys.path.insert(0, str(ROOT / "notebooks" / "datasets"))

from _common import slider  # noqa: E402
from _h5_browser import browse  # noqa: E402

CORPUS = "rn30"
"""This notebook's corpus. One file per dataset, so a session shows only it."""

SPLIT = None
"""Which split to browse, or ``None`` for the first populated one."""

H5 = ROOT / "data" / "processed" / CORPUS / f"{CORPUS}.h5"


# %%
show, total = browse(H5, SPLIT)
slider(show, total)
