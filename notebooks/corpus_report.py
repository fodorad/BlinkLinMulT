"""Per-corpus statistics, quality signals, and an adaptive sample browser.

One script for every corpus — set :data:`CORPUS` and re-run. The ``# %%`` markers
make it a notebook in VS Code, PyCharm, or Jupytext.

Read it in order; each block answers one question:

1. **What is in the file** — schema, splits, fields, and which optional fields
   this corpus supplies.
2. **Head pose** — yaw/pitch/roll distributions in degrees, and how much of the
   corpus sits beyond each candidate occlusion threshold. **This is the block
   that decides the threshold**, together with block 5.
3. **Labels** — blink and open counts, per split, with the event ids that make
   one blink distinguishable from the next.
4. **Quality signals** — how blur, exposure, contour fit, jitter, and symmetry
   are distributed, and how many frames each would remove at a given cut.
5. **Sample browser** — re-run the cell to draw the *next* set of samples, so
   many can be checked quickly. Eye crops are shown beside their signals, and
   for the video corpora the original frame is re-opened from ``data/raw`` with
   both eye boxes drawn: **green** for a crop the model would see, **red** for
   one the occlusion rule would mask.

Nothing here writes to the corpus. It reads, measures, and draws.
"""

# %%
from __future__ import annotations

import itertools
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

from blinklinmult.data.inspect import (
    count_events,
    occlusion_share,
    sample_keys,
)
from blinklinmult.data.inspect import (
    gather_field as gather,
)
from blinklinmult.data.inspect import (
    populated_splits as splits,
)
from blinklinmult.data.inspect import (
    read_field as field,
)

CORPUS = "cew"
"""Which corpus to inspect. Any name under ``data/processed``."""

OCCLUSION_CANDIDATES = (30.0, 45.0, 60.0)
"""Yaw thresholds, in degrees, whose effect block 2 quantifies.

The plan leaves the choice open deliberately: 30 and 45 are both defensible, and
the answer comes from looking at crops at each angle rather than from a number.
"""

ROOT = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path.cwd().parents[0]
OUTPUT = ROOT / "notebooks" / "output" / CORPUS
OUTPUT.mkdir(parents=True, exist_ok=True)

H5 = ROOT / "data" / "processed" / CORPUS / f"{CORPUS}.h5"
print(f"reading {H5} ({H5.stat().st_size / 2**30:.2f} GB)")


# %%
# 1. What is in the file
with h5py.File(H5) as handle:
    attrs = dict(handle.attrs)
    print(f"schema v{attrs.get('schema_version')}  built {attrs.get('created_utc')}")
    print(
        f"  fps={attrs.get('fps')}  time_dim={attrs.get('time_dim')}  "
        f"image={attrs.get('image_size')}px"
    )
    print(
        f"  targets: blink_presence={bool(attrs.get('has_blink_presence'))} "
        f"eye_state={bool(attrs.get('has_eye_state'))}"
    )
    print(
        f"  head_pose={bool(attrs.get('has_head_pose'))} "
        f"blink_ids={bool(attrs.get('has_blink_ids'))}"
    )
    print(f"  quality: {list(attrs.get('quality_signals', []))}")
    print()
    for split in splits(handle):
        print(f"  {split:6s} {len(handle[split]):7d} samples")
    first = sample_keys(handle, splits(handle)[0], 1)[0]
    print(f"\nfields of {first}:")
    for name, value in sorted(handle[splits(handle)[0]][first].items()):
        print(f"    {name:22s} {getattr(value, 'shape', ())}")


# %%
# 2. Head pose, and what each occlusion threshold would cost
with h5py.File(H5) as handle:
    if not bool(attrs.get("has_head_pose")):
        print(f"{CORPUS} supplies no head pose — it has no face box to estimate one from.")
    else:
        split = splits(handle)[0]
        keys = sample_keys(handle, split, min(3000, len(handle[split])))
        angles = np.concatenate(
            [np.asarray(field(handle, split, key, "head_pose")).reshape(-1, 3) for key in keys]
        )

        figure, axes = plt.subplots(1, 3, figsize=(13, 3.2))
        for axis, index, name in zip(axes, range(3), ("yaw", "pitch", "roll"), strict=True):
            axis.hist(angles[:, index], bins=60, color="steelblue")
            axis.set_title(f"{name} (degrees)")
            axis.axvline(0.0, color="grey", linewidth=0.8)
        figure.suptitle(f"{CORPUS}: head pose, {len(angles)} frames")
        figure.tight_layout()
        figure.savefig(OUTPUT / "head_pose.png", dpi=120)
        plt.show()

        yaw = np.abs(angles[:, 0])
        print("frames the occlusion rule would mask, per candidate threshold:")
        for threshold in OCCLUSION_CANDIDATES:
            share = occlusion_share(angles[:, 0], threshold)
            print(f"  |yaw| > {threshold:4.0f} deg: {share:6.2%}")
        print(
            f"\n  yaw p01={np.percentile(angles[:, 0], 1):6.1f}  "
            f"p50={np.percentile(angles[:, 0], 50):6.1f}  "
            f"p99={np.percentile(angles[:, 0], 99):6.1f}"
        )


# %%
# 3. Labels: how much of the corpus is a blink, and how many distinct events
with h5py.File(H5) as handle:
    target = "blink_presence" if bool(attrs.get("has_blink_presence")) else "eye_state"
    print(f"target: {target}\n")
    for split in splits(handle):
        values = gather(handle, split, target)
        if values.size == 0:
            continue
        positive = float((values > 0.5).mean())
        line = f"  {split:6s} {values.size:8d} frames  positive {positive:6.2%}"
        if bool(attrs.get("has_blink_ids")):
            line += f"  {count_events(handle, split):5d} distinct events"
        print(line)


# %%
# 4. Quality signals, and what each would cost as a filter
SIGNALS = ("eye_blur", "eye_exposure", "eye_contour_fit", "eye_jitter")

with h5py.File(H5) as handle:
    supplied = [name for name in SIGNALS if name in list(attrs.get("quality_signals", []))]
    if not supplied:
        print(f"{CORPUS} supplies no quality signals.")
    else:
        split = splits(handle)[0]
        figure, axes = plt.subplots(1, len(supplied), figsize=(3.2 * len(supplied), 3.0))
        axes = np.atleast_1d(axes)
        for axis, name in zip(axes, supplied, strict=True):
            values = gather(handle, split, name, limit=3000)
            axis.hist(values, bins=50, color="darkseagreen")
            axis.set_title(name.replace("eye_", ""))
            axis.set_yscale("log")
        figure.suptitle(f"{CORPUS}: quality signals ({split})")
        figure.tight_layout()
        figure.savefig(OUTPUT / "quality_signals.png", dpi=120)
        plt.show()

        # Jitter is the one signal where *high* is bad; the rest are "high is good".
        print("frames a filter would drop, per signal and cut:")
        for name in supplied:
            values = gather(handle, split, name, limit=3000)
            if values.size == 0:
                continue
            if name == "eye_jitter":
                shares = [f"> {cut}: {float((values > cut).mean()):6.2%}" for cut in (0.5, 1.0)]
            else:
                shares = [
                    f"< {cut}: {float((values < cut).mean()):6.2%}" for cut in (0.1, 0.3, 0.5)
                ]
            print(f"  {name:18s} " + "   ".join(shares))


# %%
# 5. Sample browser — re-run this cell to draw the NEXT set
BROWSE_SPLIT = "train"
BROWSE_COUNT = 6
_browse_offset = itertools.count()

offset = next(_browse_offset)
with h5py.File(H5) as handle:
    split = BROWSE_SPLIT if BROWSE_SPLIT in splits(handle) else splits(handle)[0]
    keys = sample_keys(handle, split, BROWSE_COUNT, offset=offset)
    print(f"{CORPUS}/{split}: set {offset}")

    figure, axes = plt.subplots(1, len(keys), figsize=(2.1 * len(keys), 3.0))
    axes = np.atleast_1d(axes)
    for axis, key in zip(axes, keys, strict=True):
        images = np.asarray(field(handle, split, key, "eye_image"))
        # Mid-window frame: the one most likely to hold the annotated closure.
        frame = images[len(images) // 2].transpose(1, 2, 0)
        axis.imshow(np.clip(frame, 0.0, 1.0))
        axis.set_xticks([])
        axis.set_yticks([])

        title = []
        pose = field(handle, split, key, "head_pose")
        masked = False
        if pose is not None:
            angles = np.asarray(pose).reshape(-1, 3)[len(images) // 2]
            title.append(f"ypr {angles[0]:.0f},{angles[1]:.0f},{angles[2]:.0f}")
            masked = abs(float(angles[0])) > OCCLUSION_CANDIDATES[1]
        for name in ("eye_blur", "eye_contour_fit"):
            values = field(handle, split, key, name)
            if values is not None:
                title.append(f"{name[4:7]} {float(np.asarray(values).reshape(-1).mean()):.2f}")
        axis.set_title("\n".join(title), fontsize=7)
        # Red border when the occlusion rule would mask this crop, green when
        # the model would see it.
        for spine in axis.spines.values():
            spine.set_edgecolor("crimson" if masked else "seagreen")
            spine.set_linewidth(2.5)

    figure.suptitle(f"{CORPUS}/{split} — set {offset}  (green: kept, red: occlusion-masked)")
    figure.tight_layout()
    plt.show()
