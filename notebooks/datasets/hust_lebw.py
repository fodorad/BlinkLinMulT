"""HUST-LEBW: verify every eye crop against the frame it came from.

The corpus ships full 1920x800 frames **and** per-eye crops cut from them. This
notebook checks that the pipeline recovers the crop's position correctly, by
putting four things in one row per sample:

1. the **original frame**, with the detected face box and the located eye boxes
   drawn on it;
2. the **face crop** the head pose is estimated from;
3. the **shipped eye crop** — the corpus's own, the ground truth for position;
4. **our re-crop** at :data:`RECROP_SCALE` times that box, which is what the
   descriptor actually reads.

If (3) and the corresponding region of (4) show the same eye, the localisation
is right. Every subplot is titled with the sample's path, so anything that looks
wrong can be opened directly.

Move the slider to step through samples.
"""

# %%
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from blinklinmult.preprocess.hust_crops import (
    EYE_DIRECTORIES,
    RECROP_SCALE,
    containing_box,
    crop_files,
    locate,
    recrop,
)

ROOT = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd().parents[1]
RAW = ROOT / "data" / "raw" / "HUST-LEBW"

SPLIT = "train"
"""Which split to browse: ``train`` or ``test``."""

LABEL = "blink"
"""Which class: ``blink`` or ``unblink``."""


def clips(split: str, label: str) -> list[Path]:
    """Every clip directory of one class, in numeric order.

    Args:
        split (str): ``train`` or ``test``.
        label (str): ``blink`` or ``unblink``.

    Returns:
        list[Path]: Clip directories.
    """
    root = RAW / split / label
    if not root.is_dir():
        return []
    found = [
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("._") and path.name.isdigit()
    ]
    return sorted(found, key=lambda path: int(path.name))


def takes(clip: Path) -> list[Path]:
    """The per-length takes inside one clip.

    A clip ships the same footage cut to several lengths (``10``, ``13``); each
    is its own directory of frames with its own eye crops.

    Args:
        clip (Path): A clip directory.

    Returns:
        list[Path]: Take directories, in numeric order.
    """
    found = [
        path
        for path in clip.iterdir()
        if path.is_dir() and not path.name.startswith("._") and path.name.isdigit()
    ]
    return sorted(found, key=lambda path: int(path.name))


def frames_of(take: Path) -> list[Path]:
    """The full frames of one take, in order.

    Args:
        take (Path): A take directory.

    Returns:
        list[Path]: Frame paths.
    """
    return sorted(
        path for path in take.iterdir() if path.suffix == ".bmp" and not path.name.startswith("._")
    )


# %%
def face_boxes(frame: np.ndarray, detector) -> list[tuple[int, int, int, int]]:
    """Every detected face in a frame.

    Args:
        frame (np.ndarray): ``(H, W, 3)`` RGB.
        detector: The face detector, or ``None`` to skip detection.

    Returns:
        list[tuple[int, int, int, int]]: ``(x1, y1, x2, y2)`` per face.
    """
    if detector is None:
        return []
    import torch

    try:
        detections = detector.detect_image(
            torch.from_numpy(np.ascontiguousarray(frame.transpose(2, 0, 1)))
        )
    except Exception:  # noqa: BLE001 - inspection must survive a bad frame
        return []
    boxes = []
    for detection in detections:
        values = np.asarray(detection.bb_xyxy).reshape(-1)[:4]
        boxes.append(tuple(int(round(float(value))) for value in values))
    return boxes


def sample_rows(split: str, label: str, limit: int = 40) -> list[dict]:
    """One row per (take, frame, eye), with everything needed to draw it.

    Args:
        split (str): ``train`` or ``test``.
        label (str): ``blink`` or ``unblink``.
        limit (int): How many rows to collect.

    Returns:
        list[dict]: Each holding the frame path, the eye side, the shipped crop
        path, and the crop's index.
    """
    rows: list[dict] = []
    for clip in clips(split, label):
        for take in takes(clip):
            frames = frames_of(take)
            for side, directory in EYE_DIRECTORIES.items():
                for index, crop_path in crop_files(take / directory):
                    if index - 1 >= len(frames):
                        continue
                    rows.append(
                        {
                            "frame": frames[index - 1],
                            "crop": crop_path,
                            "side": side,
                            "index": index,
                            "sample": f"{split}/{label}/{clip.name}/{take.name}",
                        }
                    )
                    if len(rows) >= limit:
                        return rows
    return rows


ROWS = sample_rows(SPLIT, LABEL)
print(f"{len(ROWS)} rows from {SPLIT}/{LABEL}")

# The detector is optional: without it the face box panel is blank but the crop
# localisation — the thing being checked — still shows.
try:
    from exordium.video.face.detector.yolo11 import YoloFace11Detector

    DETECTOR = YoloFace11Detector()
    print("YOLO face detector loaded")
except Exception as error:  # noqa: BLE001 - inspection works without it
    DETECTOR = None
    print(f"no detector ({error}); face boxes will be skipped")


# %%
def show(position: int) -> None:
    """Draw one sample: frame, face crop, shipped eye crop, our re-crop.

    Args:
        position (int): Index into :data:`ROWS`.
    """
    row = ROWS[position]
    frame = cv2.cvtColor(cv2.imread(str(row["frame"])), cv2.COLOR_BGR2RGB)
    crop = cv2.cvtColor(cv2.imread(str(row["crop"])), cv2.COLOR_BGR2RGB)

    x, y, score = locate(frame, crop)
    height, width = crop.shape[:2]
    wide = recrop(frame, x, y, width, height)

    boxes = face_boxes(frame, DETECTOR)
    face = containing_box(boxes, x, y, width, height)

    drawn = frame.copy()
    for box in boxes:
        # Every detection in grey; the subject's — the one containing the eye —
        # in green, since that is the only one the pipeline uses.
        colour = (0, 200, 0) if box == face else (140, 140, 140)
        thickness = 4 if box == face else 2
        cv2.rectangle(drawn, (box[0], box[1]), (box[2], box[3]), colour, thickness)
    cv2.rectangle(drawn, (x, y), (x + width, y + height), (255, 200, 0), 3)
    half = int(width * (RECROP_SCALE - 1) / 2)
    cv2.rectangle(
        drawn, (x - half, y - half), (x + width + half, y + height + half), (0, 120, 255), 3
    )

    # The frame is 1920x800 and the rest are square, so give it more width --
    # squeezed into an equal quarter it is too small to check anything against.
    figure, axes = plt.subplots(
        1, 4, figsize=(20, 4.6), gridspec_kw={"width_ratios": [2.4, 1, 1, 1]}
    )
    panels = [
        (drawn, f"frame  {row['frame'].name}\ngold: shipped crop, blue: our re-crop"),
        (
            frame[face[1] : face[3], face[0] : face[2]] if face else np.zeros((8, 8, 3), np.uint8),
            "face crop (head pose)" if face else "no face containing the eye",
        ),
        (crop, f"shipped {row['side']} crop  {crop.shape[1]}x{crop.shape[0]}\nmatch {score:.4f}"),
        (wide, f"our re-crop  {wide.shape[1]}x{wide.shape[0]}  ({RECROP_SCALE}x)"),
    ]
    for axis, (image, title) in zip(axes, panels, strict=True):
        axis.imshow(image)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(title, fontsize=8)

    figure.suptitle(
        f"{row['sample']}  frame {row['index']}  ({row['side']})\n{row['crop']}", fontsize=9
    )
    figure.tight_layout()
    plt.show()


# Slider in Jupyter; a plain call elsewhere, so the file still runs top to bottom.
try:
    from ipywidgets import IntSlider, interact

    interact(show, position=IntSlider(min=0, max=max(len(ROWS) - 1, 0), step=1, value=0))
except ImportError:
    print("ipywidgets not installed; showing the first sample")
    if ROWS:
        show(0)
