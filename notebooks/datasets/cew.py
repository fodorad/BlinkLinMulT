"""CEW: closed eyes in the wild, still photographs.

Single frames, so the window strip is one panel wide. CEW ships full
head photographs, so head pose is available even though no handcrafted
descriptor is extracted.

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

CORPUS = "cew"
"""This notebook's corpus. One file per dataset, so a session shows only it."""

SPLIT = None
"""Which split to browse, or ``None`` for the first populated one."""

H5 = ROOT / "data" / "processed" / CORPUS / f"{CORPUS}.h5"


# %%
show, total = browse(H5, SPLIT)
slider(show, total)
