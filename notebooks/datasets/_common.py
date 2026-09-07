"""Shared drawing helpers for the per-dataset inspection notebooks.

Each corpus gets its own notebook under ``notebooks/datasets/`` so that focusing
on one dataset shows only that dataset's samples. What they share is how a
sample is *drawn*: the frame with its boxes, the face crop the pose came from,
and the eye crops beside it, every panel titled with the path it came from.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

FACE_COLOUR = (0, 200, 0)
"""Box colour for the face the pipeline selected."""

OTHER_FACE_COLOUR = (140, 140, 140)
"""Box colour for detections the pipeline ignored."""

EYE_COLOUR = (255, 200, 0)
"""Box colour for an eye the model will see."""

MASKED_EYE_COLOUR = (220, 40, 40)
"""Box colour for an eye a rule excluded."""


def draw_box(
    image: np.ndarray, box: tuple[int, int, int, int], colour: tuple[int, int, int], width: int = 3
) -> np.ndarray:
    """Draw one rectangle, without touching the caller's image.

    Args:
        image (np.ndarray): ``(H, W, 3)`` RGB.
        box (tuple[int, int, int, int]): ``(x1, y1, x2, y2)``.
        colour (tuple[int, int, int]): RGB.
        width (int): Line thickness.

    Returns:
        np.ndarray: A copy with the rectangle drawn.
    """
    import cv2

    drawn = image.copy()
    cv2.rectangle(drawn, (box[0], box[1]), (box[2], box[3]), colour, width)
    return drawn


def panel_row(panels: list[tuple[np.ndarray, str]], title: str, frame_first: bool = True) -> None:
    """Draw one sample as a row of titled panels.

    Args:
        panels (list[tuple[np.ndarray, str]]): ``(image, title)`` per panel.
        title (str): Row title -- the sample id or its path.
        frame_first (bool): Give the first panel extra width, for a wide frame
            beside square crops. Squeezed into an equal share a 1920-px frame is
            too small to check anything against.
    """
    ratios = [2.4] + [1.0] * (len(panels) - 1) if frame_first else None
    figure, axes = plt.subplots(
        1,
        len(panels),
        figsize=(4.6 * (len(panels) + (1.4 if frame_first else 0)), 4.6),
        gridspec_kw={"width_ratios": ratios} if ratios else None,
    )
    for axis, (image, panel_title) in zip(np.atleast_1d(axes), panels, strict=True):
        axis.imshow(np.clip(image, 0, 255) if image.dtype != np.float32 else np.clip(image, 0, 1))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(panel_title, fontsize=8)
    figure.suptitle(title, fontsize=9)
    figure.tight_layout()
    plt.show()


def slider(show, count: int) -> None:
    """Attach a slider to a per-sample draw function.

    Falls back to drawing the first sample when ``ipywidgets`` is absent, so the
    file still runs top to bottom outside Jupyter.

    Args:
        show: Callable taking a sample index.
        count (int): How many samples there are.
    """
    if count < 1:
        print("nothing to show")
        return
    try:
        from ipywidgets import IntSlider, interact

        interact(show, position=IntSlider(min=0, max=count - 1, step=1, value=0))
    except ImportError:
        print("ipywidgets not installed (uv sync --extra notebooks); showing the first sample")
        show(0)


def crop_to(image: np.ndarray, box: tuple[int, int, int, int] | None) -> np.ndarray:
    """Cut a box out of an image, or return a placeholder.

    Args:
        image (np.ndarray): ``(H, W, 3)``.
        box (tuple[int, int, int, int] | None): ``(x1, y1, x2, y2)``.

    Returns:
        np.ndarray: The crop, or an 8x8 black square when there is no box.
    """
    if box is None:
        return np.zeros((8, 8, 3), dtype=np.uint8)
    cut = image[max(box[1], 0) : box[3], max(box[0], 0) : box[2]]
    return cut if cut.size else np.zeros((8, 8, 3), dtype=np.uint8)


def frame_from_h5(images: np.ndarray, index: int) -> np.ndarray:
    """One frame of a stored window, as an image.

    Args:
        images (np.ndarray): ``(T, C, H, W)`` normalised crops.
        index (int): Which timestep.

    Returns:
        np.ndarray: ``(H, W, C)`` in ``[0, 1]``.
    """
    return np.clip(np.asarray(images)[index].transpose(1, 2, 0), 0.0, 1.0)
