"""The batch-to-model boundary.

OmniLoader hands back a batch in which every declared key is paired with a
``<name>_mask``. The eye crops arrive in native ``(B, T, C, H, W)`` form — since
omniloader 1.1 a spec declares a structured trailing ``shape``, so nothing is
flattened on the way in and nothing is reshaped here.

What remains is folding the time axis into the batch so a 2-D CNN can embed
every crop in one pass, and assembling the ``(inputs, masks)`` the sequence
model expects. Batching itself is :func:`omniloader.unified_collate`, used
unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from blinklinmult.data.schema import (
    BLINK_PRESENCE,
    EYE_IMAGE,
    EYE_STATE,
    HEAD_POSE,
    IMAGE_CHANNELS,
    LEFT,
    RIGHT,
    SAMPLE_KEY,
    UNKNOWN_EYE,
    SchemaError,
    parse_sample_id,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class BatchError(KeyError):
    """Raised when a batch does not carry what the model was configured for."""


def check_eye_images(images: torch.Tensor, image_size: int) -> torch.Tensor:
    """Verify a batch's eye crops match the configured crop size.

    The crops arrive already in image form, so this validates rather than
    reshapes — but it validates at the model boundary, where a mismatch between
    the run's config and the built dataset would otherwise surface as an
    inscrutable convolution error.

    Args:
        images (torch.Tensor): ``(B, T, C, H, W)`` as loaded.
        image_size (int): Side length ``H = W`` the run is configured for.

    Returns:
        torch.Tensor: The batch, unchanged.

    Raises:
        BatchError: If the rank or the trailing shape contradicts the config.
    """
    expected = (IMAGE_CHANNELS, image_size, image_size)
    if images.ndim != 5 or tuple(images.shape[2:]) != expected:
        raise BatchError(
            f"eye_image has shape {tuple(images.shape)}, expected (B, T, "
            f"{', '.join(str(d) for d in expected)}). The config and the built "
            "dataset disagree; rebuild or fix data.image_size."
        )
    return images


def fold_time_into_batch(images: torch.Tensor) -> torch.Tensor:
    """Flatten the batch and time axes so a 2-D CNN can consume the crops.

    A per-frame image backbone has no notion of time; folding both leading axes
    into the batch lets it process every crop in one pass instead of looping
    over ``T``, which is what the 1.x code did (a Python loop over the time
    dimension per forward call).

    Args:
        images (torch.Tensor): ``(B, T, 3, H, W)``.

    Returns:
        torch.Tensor: ``(B*T, 3, H, W)``.
    """
    batch, time, channels, height, width = images.shape
    return images.reshape(batch * time, channels, height, width)


def unfold_time_from_batch(flat: torch.Tensor, batch: int, time: int) -> torch.Tensor:
    """Undo :func:`fold_time_into_batch` on the backbone's output.

    Args:
        flat (torch.Tensor): ``(B*T, D)`` per-crop embeddings.
        batch (int): Original batch size.
        time (int): Original window length.

    Returns:
        torch.Tensor: ``(B, T, D)`` — one embedding per timestep.
    """
    return flat.reshape(batch, time, flat.shape[-1])


OCCLUSION_YAW = 45.0
"""Absolute head yaw, in degrees, beyond which the far eye is self-occluded.

Past roughly this angle the far eye is behind the nose and its crop shows
something that is not an eye — while every geometric check still passes, because
the landmarks track a face that is genuinely there.

**A dataloader rule, not a build-time one.** Head pose is stored in degrees, so
this threshold stays a hyperparameter: sweeping it costs a config change rather
than re-preprocessing tens of gigabytes. Measured across the corpora, only
MPEblink is materially affected (35.7% of frames beyond 45 degrees); RN15, RN30
and TalkingFace are near-frontal and lose essentially nothing.

The value is deliberately provisional. 30 degrees is equally defensible and the
choice belongs to inspection — see the head-pose block of
``notebooks/corpus_report.py``, which shows crops binned by yaw beside the
original frame.
"""

YAW_INDEX = 0
"""Column of :data:`~blinklinmult.data.schema.HEAD_POSE` holding yaw."""


def occluded_eyes(
    head_pose: torch.Tensor, eye_side: Sequence[str], threshold: float = OCCLUSION_YAW
) -> torch.Tensor:
    """Which positions show an eye hidden by the head's own rotation.

    Yaw's **sign** says which eye: turning one way hides the left eye, the other
    way the right. A frame is only occluded for the eye on the far side, so the
    near eye of the same frame stays fully usable — which is why this returns a
    per-sample-per-timestep mask rather than dropping the frame.

    Args:
        head_pose (torch.Tensor): ``(B, T, 3)`` ``[yaw, pitch, roll]`` in degrees.
        eye_side (Sequence[str]): ``B`` eye sides, as the batch records them.
        threshold (float): Absolute yaw beyond which the far eye is occluded.

    Returns:
        torch.Tensor: ``(B, T)`` bool, ``True`` where the eye is occluded.

    Raises:
        BatchError: If the pose is not ``(B, T, 3)`` or the sides do not match
            the batch.
    """
    if head_pose.ndim != 3 or head_pose.shape[-1] < 1:
        raise BatchError(f"head_pose must be (B, T, 3), got {tuple(head_pose.shape)}.")
    if len(eye_side) != head_pose.shape[0]:
        raise BatchError(
            f"{len(eye_side)} eye sides for {head_pose.shape[0]} samples; they must agree."
        )

    yaw = head_pose[..., YAW_INDEX]
    beyond = yaw.abs() > threshold

    # Positive yaw hides one side, negative the other. An unknown side (MRL
    # records no side) is never masked: without knowing which eye it is, the
    # rule cannot say whether this one is the far one.
    turned_positive = yaw > 0
    is_left = torch.tensor([side == LEFT for side in eye_side], device=head_pose.device).unsqueeze(
        -1
    )
    is_right = torch.tensor(
        [side == RIGHT for side in eye_side], device=head_pose.device
    ).unsqueeze(-1)

    far_side = (is_left & turned_positive) | (is_right & ~turned_positive)
    return beyond & far_side


def _sides_from_keys(sample_ids: Sequence[str]) -> list[str]:
    """Eye side of each sample, read out of its key.

    Args:
        sample_ids (Sequence[str]): One ``video|frame_group|eye_side`` key per
            sample.

    Returns:
        list[str]: One side per sample; an unparseable key yields
        :data:`~blinklinmult.data.schema.UNKNOWN_EYE`, which is never occluded.
    """
    sides = []
    for sample_id in sample_ids:
        try:
            sides.append(parse_sample_id(str(sample_id))[2])
        except SchemaError:
            sides.append(UNKNOWN_EYE)
    return sides


def apply_occlusion(batch: dict[str, Any], threshold: float = OCCLUSION_YAW) -> dict[str, Any]:
    """Mask the eye-image validity of self-occluded positions.

    Returns a **shallow copy** with the image and target masks replaced, so the
    crops themselves are untouched and the original batch is not mutated — a
    caller inspecting the same batch afterwards sees what the loader produced.

    A batch without head pose is returned unchanged: the still corpora have no
    face to estimate pose from, so there is nothing to rule on.

    Args:
        batch (dict): A collated OmniLoader batch.
        threshold (float): Absolute yaw beyond which the far eye is occluded.

    Returns:
        dict: The batch, with occluded positions marked invalid.
    """
    pose = batch.get(HEAD_POSE)
    mask_key = f"{EYE_IMAGE}_mask"
    if pose is None or mask_key not in batch:
        return batch

    # From the **sample key**, not from an `eye_side` field: OmniLoader forwards
    # only `key`, `dataset` and `subset` as metadata (see `METADATA_KEYS`), so
    # `eye_side` never reaches a batch and reading it would make this rule a
    # silent no-op in production.
    sides = _sides_from_keys(batch.get(SAMPLE_KEY, []))
    if len(sides) != pose.shape[0]:
        return batch

    occluded = occluded_eyes(pose, sides, threshold)
    updated = dict(batch)
    updated[mask_key] = batch[mask_key].bool() & ~occluded

    # The **target** masks too, not only the image mask. The two are read
    # independently by design -- a usable crop may still yield no reliable
    # descriptor -- so masking the image alone would leave the loss supervising
    # an eye the model was never shown. An occluded eye must reach neither.
    for target in (EYE_STATE, BLINK_PRESENCE):
        target_mask = f"{target}_mask"
        if target_mask in batch:
            updated[target_mask] = batch[target_mask].bool() & ~occluded

    return updated


def unpack_batch(
    batch: dict[str, Any], feature_names: Sequence[str]
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split a batch into the ``(inputs, masks)`` the model consumes.

    Each feature keeps its **own** mask. A frame whose eye crop is usable may
    still yield no reliable handcrafted descriptors, so the image and feature
    masks are read independently and never assumed to agree.

    Args:
        batch (dict): A collated OmniLoader batch.
        feature_names (Sequence[str]): Features in model input order.

    Returns:
        tuple: ``inputs`` of ``(B, T, F)`` and ``masks`` of ``(B, T)`` with
        ``True`` = valid.

    Raises:
        BatchError: If a named feature or its mask is absent.
    """
    inputs: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []

    for name in feature_names:
        if name not in batch:
            raise BatchError(f"Batch has no feature {name!r}. Present: {sorted(batch)}.")
        mask_key = f"{name}_mask"
        if mask_key not in batch:
            raise BatchError(
                f"Batch has no mask for {name!r}. OmniLoader pairs every declared key "
                "with a mask; a missing one means the schema and the data disagree."
            )
        inputs.append(batch[name])
        masks.append(batch[mask_key])

    return inputs, masks


def target_and_mask(batch: dict[str, Any], name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Read one target and its validity mask.

    Args:
        batch (dict): A collated OmniLoader batch.
        name (str): Target key.

    Returns:
        tuple: ``(target, mask)``, both ``(B, T)``. The mask is ``True`` where
        the corpus that produced the sample actually annotates this target, and
        ``False`` for samples drawn from a corpus that does not.

    Raises:
        BatchError: If the target or its mask is absent.
    """
    if name not in batch:
        raise BatchError(f"Batch has no target {name!r}. Present: {sorted(batch)}.")
    mask_key = f"{name}_mask"
    if mask_key not in batch:
        raise BatchError(f"Batch has no mask for target {name!r}.")

    return batch[name], batch[mask_key]
