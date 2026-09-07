"""Augmentation for eye crops, calibrated against the corpora it trains on.

Every constant here is measured rather than chosen. Rotation matches the roll
the corpora actually contain; brightness and contrast match the gap a shared
encoder has to absorb between them. Augmenting outside those ranges would teach
invariance to variation that never occurs, at the cost of the capacity to
separate the variation that does.

**Measured across the built corpora**

===========  ==========  ==========  ==========
corpus       roll p05    roll p95    mean level
===========  ==========  ==========  ==========
CEW              -12.2         9.8        0.556
RN15              -8.9        13.4        0.413
RN30              -9.3        11.4        0.403
TalkingFace       -9.7         3.7        0.521
MRL-Eye              -           -        0.346
===========  ==========  ==========  ==========

So roll spans roughly ±13 degrees, and corpus mean level spans 0.346 to 0.556 --
a 60% gap, which is what a model transferring between them must absorb.

**One draw per sample, applied to every frame of it.** A sample is one eye's
whole window, and geometry and lighting are continuous in time: a head does not
jump orientation between adjacent frames, and a light source does not flicker.
Sampling per frame would manufacture exactly the chatter that
:func:`~blinklinmult.preprocess.quality.box_jitter` measures and that
:func:`~blinklinmult.train.losses.temporal_smoothness` penalises. The frame-wise
path gets per-frame variation for free, because there a sample *is* one frame.

**No horizontal flip.** Flipping a left eye produces something that looks like a
right eye, which is the usual argument for it -- but here samples are eye-wise
precisely so that winks and per-eye patterns are detectable, and the eye side is
part of the label. Flipping without swapping the side teaches that side does not
matter; flipping with it is a separate decision, deliberately not taken here.

**No vertical flip**, ever: eyelids close downward and that asymmetry is the
signal.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import torch.nn.functional as F
from omniloader.transforms.base import Transform

from blinklinmult.data.schema import EYE_IMAGE

# Softened 2026-08-30 after the first four frame-wise arms.
#
# **Augmentation cost recall on every pairing it was measured against**, and
# recall is the binding constraint on this task -- precision is already
# 0.87-1.00 across the corpora. On TalkingFace, where every arm reaches perfect
# precision with zero false positives, the *only* thing separating F1 from 1.0
# is missed blinks:
#
#     ============  =======  =======  =======
#     arm           misses   recall   F1
#     ============  =======  =======  =======
#     fw-bce             3   0.9508   0.9748
#     fw-bce-augment     9   0.8525   0.9204
#     fw-focal           4   0.9344   0.9661
#     fw-focal-augment   5   0.9180   0.9573
#     ============  =======  =======  =======
#
# BCE lost 6 extra blinks to augmentation, focal 1. A blink is a *brief, subtle*
# appearance change -- a few frames of lid movement -- so a transform strong
# enough to help a coarse classifier can bury the very signal being detected.
#
# The geometric ranges are roughly halved and blur is cut hardest, since blur
# attacks the lid edge the task actually reads. The photometric ranges are
# reduced least: they were calibrated against the measured gap between corpus
# means, so they describe a real domain difference rather than an arbitrary
# perturbation.

MAX_ROTATION_DEGREES = 6.0
"""Rotation range. Halved from 12.0.

The corpora's own roll spans ~13 degrees p05-p95, so 12 covered the full
observed range -- but sampling the extremes of a distribution as *routine*
training noise is stronger than reproducing it. Half the range still covers the
bulk of real head roll.
"""

MAX_BRIGHTNESS_DELTA = 0.15
"""Additive brightness range. Reduced from 0.20.

Kept near the measured gap between corpus means (0.346 to 0.556): this one
describes a genuine domain difference the model must survive, so it is trimmed
rather than halved.
"""

MAX_CONTRAST_DELTA = 0.15
"""Multiplicative contrast range around 1.0. Reduced from 0.20, as brightness."""

MAX_TRANSLATE_FRACTION = 0.03
"""Shift range as a fraction of the crop. Reduced from 0.05.

Localisation jitter is real, but a 64 px crop shifted 5% moves the lid edge by
3 px -- comparable to the closure it is meant to detect.
"""

MAX_SCALE_DELTA = 0.06
"""Zoom range around 1.0, for box-size error. Reduced from 0.10."""

BLUR_PROBABILITY = 0.1
"""How often to blur. Halved from 0.2 -- blur destroys the lid edge the task reads."""

MAX_BLUR_SIGMA = 0.5
"""Gaussian sigma at full strength, in pixels of a 64 px crop.

Cut hardest of all (from 0.8). Motion blur is the one corruption that removes
the *evidence* rather than transforming it: a blurred closing lid and a blurred
open eye look alike, so a strong blur trains the model on frames whose label it
cannot see.
"""


class AugmentError(ValueError):
    """Raised when an augmentation is configured out of range."""


@dataclass(frozen=True)
class AugmentConfig:
    """How strongly to augment, as a fraction of each measured range.

    Args:
        strength (float): Scales every range. ``0`` disables augmentation
            entirely; ``1`` uses the measured ranges above.
        rotate (bool): Rotate within :data:`MAX_ROTATION_DEGREES`.
        photometric (bool): Jitter brightness and contrast.
        translate (bool): Shift and zoom, for localisation error.
        blur (bool): Occasionally blur.
    """

    strength: float = 1.0
    rotate: bool = True
    photometric: bool = True
    translate: bool = True
    blur: bool = True

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            AugmentError: If the strength is negative or above one.
        """
        if not 0.0 <= self.strength <= 1.0:
            raise AugmentError(f"strength must be in [0, 1], got {self.strength}.")

    @property
    def geometric(self) -> bool:
        """Whether this configuration moves pixels around.

        Returns:
            bool: ``True`` when rotation or translation is on. These are the
            transforms that invalidate a handcrafted descriptor, because 152 of
            its 160 dimensions are pixel-space landmark coordinates describing
            the *unrotated* crop.
        """
        return self.strength > 0.0 and (self.rotate or self.translate)

    def without_geometry(self) -> AugmentConfig:
        """This configuration with the geometry-changing transforms removed.

        Brightness, contrast and blur leave every landmark where it was, so they
        are safe alongside a handcrafted descriptor; rotation, translation and
        scale are not.

        Returns:
            AugmentConfig: A copy with ``rotate`` and ``translate`` off.
        """
        return replace(self, rotate=False, translate=False)

    @property
    def enabled(self) -> bool:
        """Whether this configuration changes anything.

        Returns:
            bool: ``False`` when the strength is zero or every transform is off.
        """
        return self.strength > 0.0 and any(
            (self.rotate, self.photometric, self.translate, self.blur)
        )


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    """One draw from a uniform range, on the sample's own generator.

    Args:
        generator (torch.Generator): The sample's seeded generator.
        low (float): Lower bound.
        high (float): Upper bound.

    Returns:
        float: The draw.
    """
    return float(torch.empty(1).uniform_(low, high, generator=generator).item())


def _affine_grid(
    frames: int, height: int, width: int, degrees: float, shift: tuple[float, float], scale: float
) -> torch.Tensor:
    """A sampling grid rotating, shifting and zooming every frame identically.

    Args:
        frames (int): Timesteps in the window.
        height (int): Crop height.
        width (int): Crop width.
        degrees (float): Rotation, positive anticlockwise.
        shift (tuple[float, float]): ``(x, y)`` shift in normalised units.
        scale (float): Zoom factor; above one zooms in.

    Returns:
        torch.Tensor: ``(T, H, W, 2)`` grid for :func:`torch.nn.functional.grid_sample`.
    """
    radians = torch.deg2rad(torch.tensor(degrees))
    cos, sin = torch.cos(radians) / scale, torch.sin(radians) / scale
    # The inverse map: grid_sample reads *from* the source, so the matrix takes
    # output coordinates back to input ones.
    matrix = torch.tensor(
        [[cos, -sin, shift[0]], [sin, cos, shift[1]]], dtype=torch.float32
    ).unsqueeze(0)
    return F.affine_grid(
        matrix.expand(frames, -1, -1), [frames, 1, height, width], align_corners=False
    )


def _blur(images: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur over a window.

    Args:
        images (torch.Tensor): ``(T, C, H, W)``.
        sigma (float): Standard deviation in pixels.

    Returns:
        torch.Tensor: The blurred window, same shape.
    """
    if sigma <= 0.0:
        return images
    radius = max(1, int(round(2.0 * sigma)))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(offsets**2) / (2.0 * sigma**2))
    kernel = kernel / kernel.sum()

    channels = images.shape[1]
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    blurred = F.conv2d(images, horizontal, padding=(0, radius), groups=channels)
    return F.conv2d(blurred, vertical, padding=(radius, 0), groups=channels)


def augment_window(
    images: torch.Tensor, config: AugmentConfig, generator: torch.Generator
) -> torch.Tensor:
    """Augment one eye window, every frame identically.

    Args:
        images (torch.Tensor): ``(T, C, H, W)`` crops in ``[0, 1]``.
        config (AugmentConfig): How strongly to augment.
        generator (torch.Generator): The sample's seeded generator, so the same
            sample augments the same way given the same epoch.

    Returns:
        torch.Tensor: The augmented window, same shape, clamped to ``[0, 1]``.

    Raises:
        AugmentError: If the window is not ``(T, C, H, W)``.
    """
    if images.ndim != 4:
        raise AugmentError(f"expected (T, C, H, W) crops, got {tuple(images.shape)}.")
    if not config.enabled:
        return images

    frames, _, height, width = images.shape
    strength = config.strength
    out = images

    # Geometry: one draw for the whole window, because a head does not jump
    # orientation between adjacent frames.
    degrees = (
        _uniform(generator, -MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES) * strength
        if config.rotate
        else 0.0
    )
    shift = (0.0, 0.0)
    scale = 1.0
    if config.translate:
        limit = MAX_TRANSLATE_FRACTION * strength
        shift = (_uniform(generator, -limit, limit), _uniform(generator, -limit, limit))
        span = MAX_SCALE_DELTA * strength
        scale = 1.0 + _uniform(generator, -span, span)

    if degrees or shift != (0.0, 0.0) or scale != 1.0:
        grid = _affine_grid(frames, height, width, degrees, shift, scale)
        # Border padding, not zeros: a black wedge at the corner is a feature
        # the model can learn to key on, and it is not a thing that happens to
        # real crops.
        out = F.grid_sample(out, grid, mode="bilinear", padding_mode="border", align_corners=False)

    if config.photometric:
        brightness = _uniform(generator, -MAX_BRIGHTNESS_DELTA, MAX_BRIGHTNESS_DELTA) * strength
        contrast = 1.0 + _uniform(generator, -MAX_CONTRAST_DELTA, MAX_CONTRAST_DELTA) * strength
        # Around the window's own mean, so contrast pivots on the crop's level
        # rather than on an arbitrary 0.5.
        level = out.mean()
        out = (out - level) * contrast + level + brightness

    if config.blur and _uniform(generator, 0.0, 1.0) < BLUR_PROBABILITY:
        out = _blur(out, _uniform(generator, 0.1, MAX_BLUR_SIGMA) * strength)

    return out.clamp(0.0, 1.0)


class EyeAugmentation(Transform):
    """OmniLoader transform applying :func:`augment_window` to a sample.

    Subclasses OmniLoader's :class:`~omniloader.transforms.base.Transform` so
    the train/eval gate is the loader's own: ``train_only`` means a validation
    or test loader sees the crops exactly as built, with no flag to forget.

    The generator OmniLoader passes is seeded from ``(seed, epoch, index)``, so
    a sample augments identically given the same epoch and differently across
    epochs -- which is what
    :class:`~blinklinmult.train.callbacks.EpochPropagator` exists to advance.

    Args:
        config (AugmentConfig): How strongly to augment.
    """

    train_only = True

    def __init__(self, config: AugmentConfig | None = None):
        self.config = config or AugmentConfig()

    def apply(self, sample: dict, generator: torch.Generator | None) -> dict:
        """Augment one sample's eye crops.

        Args:
            sample (dict): A unified sample.
            generator (torch.Generator | None): The sample's seeded generator.
                ``None`` leaves the sample untouched rather than drawing from
                global state, which would make a run unreproducible.

        Returns:
            dict: The sample, with :data:`~blinklinmult.data.schema.EYE_IMAGE`
            augmented. A shallow copy, so the caller's dict is not mutated.
        """
        if generator is None or not self.config.enabled or EYE_IMAGE not in sample:
            return sample

        images = sample[EYE_IMAGE]
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            return sample

        updated = dict(sample)
        updated[EYE_IMAGE] = augment_window(images, self.config, generator)
        return updated
