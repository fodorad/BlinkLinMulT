"""A slider browser over any built corpus, shared by the per-dataset notebooks.

What it shows per sample, in one row:

* the **eye crop** the model reads, at the window's mid frame;
* the **window strip** -- every frame of the window, so a blink is visible as
  motion rather than inferred from one still;
* the **signals** that decide whether the sample is used: head pose in degrees,
  the quality scores, and the annotation.

Every row is titled with the sample key, so anything suspicious can be traced
back to the recording it came from.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from blinklinmult.data.inspect import populated_splits, read_field, sample_keys

if TYPE_CHECKING:
    import h5py

STRIP_LIMIT = 12
"""Most frames to draw in the window strip, so a 45-frame window stays legible."""


def summarise(handle: h5py.File, split: str, key: str) -> str:
    """One line of the signals that decide whether a sample is used.

    Args:
        handle (h5py.File): Open corpus.
        split (str): Which split.
        key (str): Sample key.

    Returns:
        str: Pose, quality signals, and annotation, as far as the corpus has them.
    """
    parts = []
    pose = read_field(handle, split, key, "head_pose")
    if pose is not None:
        angles = np.asarray(pose).reshape(-1, 3).mean(axis=0)
        parts.append(f"ypr {angles[0]:.0f},{angles[1]:.0f},{angles[2]:.0f}")

    for name in ("eye_blur", "eye_exposure", "eye_contour_fit", "eye_jitter"):
        values = read_field(handle, split, key, name)
        if values is not None:
            parts.append(f"{name[4:8]} {float(np.asarray(values).mean()):.2f}")

    for target in ("blink_presence", "eye_state"):
        values = read_field(handle, split, key, target)
        if values is not None:
            positive = int((np.asarray(values) > 0.5).sum())
            parts.append(f"{target[:5]} {positive}/{np.asarray(values).size}")

    return "   ".join(parts)


def show_sample(handle: h5py.File, split: str, key: str) -> None:
    """Draw one sample: its mid frame, its window strip, and its signals.

    Args:
        handle (h5py.File): Open corpus.
        split (str): Which split.
        key (str): Sample key.
    """
    images = np.asarray(read_field(handle, split, key, "eye_image"))
    mask = read_field(handle, split, key, "eye_image_mask")
    valid = np.asarray(mask).astype(bool) if mask is not None else np.ones(len(images), bool)

    steps = min(len(images), STRIP_LIMIT)
    picks = np.linspace(0, len(images) - 1, steps).astype(int)

    figure, axes = plt.subplots(1, steps, figsize=(1.5 * steps, 2.4))
    for axis, index in zip(np.atleast_1d(axes), picks, strict=True):
        axis.imshow(np.clip(images[index].transpose(1, 2, 0), 0.0, 1.0))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"t{index}", fontsize=7)
        # A masked frame is one the model never sees; mark it rather than
        # leaving it indistinguishable from a frame that simply looks dark.
        for spine in axis.spines.values():
            spine.set_edgecolor("seagreen" if valid[index] else "crimson")
            spine.set_linewidth(2.0)

    figure.suptitle(f"{split}/{key}\n{summarise(handle, split, key)}", fontsize=9)
    figure.tight_layout()
    plt.show()


def browse(path, split: str | None = None, count: int = 60):
    """Build a slider over one corpus.

    Args:
        path: The corpus's ``.h5``.
        split (str | None): Which split, or ``None`` for the first populated one.
        count (int): How many samples to make browsable.

    Returns:
        tuple: ``(show, total)`` -- a draw function taking an index, and how many
        samples it covers.
    """
    import h5py

    handle = h5py.File(path)
    chosen = split or populated_splits(handle)[0]
    keys = sample_keys(handle, chosen, min(count, len(handle[chosen])))

    def show(position: int) -> None:
        show_sample(handle, chosen, keys[position])

    print(f"{path.name}: browsing {len(keys)} samples of '{chosen}'")
    return show, len(keys)
