"""MRL-Eye: pre-cropped eye stills, no surrounding face.

Single frames of a bare eye: no face, so no head pose and no
occlusion rule. Only the two pixel signals apply.

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

CORPUS = "mrl"
"""This notebook's corpus. One file per dataset, so a session shows only it."""

SPLIT = None
"""Which split to browse, or ``None`` for the first populated one."""

H5 = ROOT / "data" / "processed" / CORPUS / f"{CORPUS}.h5"


# %%
show, total = browse(H5, SPLIT)
slider(show, total)
