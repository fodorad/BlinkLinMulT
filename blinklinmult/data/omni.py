"""Bridge from a :class:`~blinklinmult.data.schema.DatasetSpec` to OmniLoader.

OmniLoader unifies several datasets that annotate different things into one
masked, batchable stream. To do that it needs, per dataset, a
:class:`omniloader.DatasetSchema` declaring which keys that dataset provides and
what shape they have. This module generates those declarations from the same
:class:`~blinklinmult.data.schema.DatasetSpec` objects the builder wrote the
files with, so the schema a corpus is *read* under can never drift from the one
it was *written* under.

**Shape mapping.** OmniLoader describes values structurally: ``feature_dim``
declares a flat trailing axis, ``shape`` declares a structured one. One eye's
crops are ``(T, 3, H, W)``, so they are declared with ``shape=(3, H, W)`` and
travel in native image form the whole way — stored, padded, masked, mixed and
batched without a reshape.

That needs **omniloader >= 1.1**, which added structured trailing shapes. Before
it, images had to be flattened into a wide ``feature_dim`` and reshaped back at
the model boundary; the schema then described a 12288-wide feature rather than
an image, and nothing validated that the width factorised correctly.

**Cross-dataset agreement.** OmniLoader requires that specs sharing a name agree
on shape and dtype across datasets. The corpora differ in frame rate and
therefore in native window length, so :func:`unified_specs` resolves one shared
``time_dim`` for the whole run — the longest any participating corpus needs —
and every shorter corpus is padded to it with an all-``False`` mask. That is how
a 15 fps and a 30 fps corpus train together while still covering the same real
duration.
"""

from __future__ import annotations

import logging

import torch
from omniloader import DatasetSchema, TensorSpec

from blinklinmult.data.schema import (
    BLINK_IDS,
    BLINK_PRESENCE,
    EYE_EMBEDDING,
    EYE_FEATURE,
    EYE_IMAGE,
    EYE_STATE,
    HEAD_POSE,
    HEAD_POSE_DIM,
    IMAGE_CHANNELS,
    TARGET_PLACEHOLDER,
    DatasetSpec,
    SchemaError,
)

logger = logging.getLogger(__name__)
"""Module-level logger."""


def eye_image_shape(image_size: int) -> tuple[int, int, int]:
    """Trailing shape of one frame's eye crop.

    Args:
        image_size (int): Side length of a square eye crop.

    Returns:
        tuple[int, int, int]: ``(C, H, W)``.
    """
    return (IMAGE_CHANNELS, image_size, image_size)


def dataset_schema(
    spec: DatasetSpec,
    time_dim: int,
    image_size: int,
    feature_dim: int | None,
    embedding_dim: int | None = None,
) -> DatasetSchema:
    """Build the OmniLoader schema for one corpus.

    Args:
        spec (DatasetSpec): The corpus.
        time_dim (int): Shared window length for the run, in frames. Every
            dataset in a joint run declares the same ``T``; OmniLoader pads each
            corpus's native window to it and masks the difference.
        image_size (int): Shared eye-crop size for the run.
        feature_dim (int | None): Shared handcrafted-descriptor width, or
            ``None`` when no corpus in the run supplies them.
        embedding_dim (int | None): Width of a precomputed eye embedding, when
            the run reads one from a cache, else ``None``. Declaring it is what
            lets the field survive collation; see the comment at its
            :class:`~omniloader.TensorSpec`.

    Returns:
        DatasetSchema: Features and targets this corpus provides. Keys it lacks
        are simply absent, which is how OmniLoader knows to mask them.

    Raises:
        SchemaError: If the corpus declares a feature width that disagrees with
            the run's shared width.
    """
    features = [
        TensorSpec(
            name=EYE_IMAGE,
            # A structured trailing shape, not a flat width: the schema says
            # "image" and OmniLoader keeps it in that form end to end.
            shape=eye_image_shape(image_size),
            time_dim=time_dim,
            dtype=torch.float32,
        )
    ]

    if embedding_dim is not None:
        # **Declared, or the collate drops it.** OmniLoader forwards only the
        # keys a schema names, so an undeclared field is attached to the sample
        # and then silently discarded before the batch reaches the model -- the
        # write-only trap this module's docstring warns about. Measured: the
        # datamodule logged "reading cached embeddings" while every batch
        # arrived without one, so the encoder ran anyway and the cache bought
        # nothing.
        features.append(
            TensorSpec(
                name=EYE_EMBEDDING,
                feature_dim=embedding_dim,
                time_dim=time_dim,
                dtype=torch.float32,
            )
        )

    if spec.has_eye_feature:
        if feature_dim is None or spec.feature_dim != feature_dim:
            raise SchemaError(
                f"{spec.name}: supplies {spec.feature_dim}-d eye features but the run "
                f"resolved a shared width of {feature_dim}. Handcrafted descriptors must "
                "be extracted identically across corpora to be comparable."
            )
        features.append(
            TensorSpec(
                name=EYE_FEATURE,
                feature_dim=feature_dim,
                time_dim=time_dim,
                dtype=torch.float32,
            )
        )

    # Both targets are scalar sequences: one value per timestep for this
    # sample's single eye. The eye axis lives in the sample identity, not the
    # tensor shape.
    targets = []
    if spec.has_blink_presence:
        targets.append(
            TensorSpec(
                name=BLINK_PRESENCE,
                time_dim=time_dim,
                dtype=torch.float32,
                placeholder=TARGET_PLACEHOLDER,
            )
        )
    if spec.has_eye_state:
        targets.append(
            TensorSpec(
                name=EYE_STATE,
                time_dim=time_dim,
                dtype=torch.float32,
                placeholder=TARGET_PLACEHOLDER,
            )
        )

    # Declared as *features*, not targets: nothing is supervised against them.
    # They condition what the model is shown -- head pose decides which eye is
    # self-occluded, confidence orders a curriculum -- and blink ids individuate
    # the events the report groups by.
    #
    # **A field that is written but not declared here never reaches training.**
    # The quality group has always been written this way and has always been
    # invisible, which is why these are features rather than quality extras.
    if spec.has_head_pose:
        features.append(
            TensorSpec(
                name=HEAD_POSE,
                feature_dim=HEAD_POSE_DIM,
                time_dim=time_dim,
                dtype=torch.float32,
            )
        )
    # One spec per signal the corpus actually supplies, never a pre-combined
    # score: how to weigh them is decided in the dataloader once the
    # distributions have been seen. Per-signal because a still corpus has no
    # jitter and MRL-Eye, which does not say which eye a sample is, has no
    # symmetry.
    features.extend(
        TensorSpec(name=signal, time_dim=time_dim, dtype=torch.float32)
        for signal in spec.quality_signals
    )
    if spec.has_blink_ids:
        features.append(
            TensorSpec(
                name=BLINK_IDS,
                time_dim=time_dim,
                # Long, not float: these are identities. A float32 id silently
                # loses precision past 2**24, and the ids are compared for
                # equality rather than magnitude.
                dtype=torch.long,
            )
        )

    return DatasetSchema(features=features, targets=targets)


def unified_specs(
    specs: list[DatasetSpec],
    time_dim: int | None = None,
    image_size: int | None = None,
    stills: bool = False,
) -> tuple[int, int, int | None]:
    """Resolve the shared shape a joint run brings every corpus to.

    Args:
        specs (list[DatasetSpec]): Corpora participating in the run, already
            carrying the run's window length (see
            :meth:`~blinklinmult.data.schema.DatasetSpec.with_window`).
        time_dim (int | None): Window length in frames to use. ``None`` takes
            the maximum any corpus derives from its own rate, so no corpus's
            window is truncated by default.
        image_size (int | None): Eye-crop size. ``None`` requires every corpus
            to agree, since resizing a crop after extraction would resample an
            already-resampled image.
        stills (bool): Whether the run serves single frames. Suppresses the
            truncation warning: in still mode the frames are chosen before
            OmniLoader sees them (see :mod:`blinklinmult.data.stills`), so
            nothing is cropped and warning about it would be false.

    Returns:
        tuple[int, int, int | None]: ``(time_dim, image_size, feature_dim)``.

    Raises:
        SchemaError: If the list is empty, crop sizes disagree with no explicit
            choice, or the corpora supply descriptors of differing widths.
    """
    if not specs:
        raise SchemaError("A run needs at least one dataset.")

    resolved_time = time_dim if time_dim is not None else max(s.time_dim for s in specs)
    if resolved_time < 1:
        raise SchemaError(f"time_dim must be >= 1, got {resolved_time}.")

    sizes = sorted({s.image_size for s in specs})
    if image_size is not None:
        resolved_size = image_size
    elif len(sizes) == 1:
        resolved_size = sizes[0]
    else:
        raise SchemaError(
            f"Datasets declare different eye-crop sizes {sizes}. Set data.image_size "
            "explicitly and re-extract the crops at that size; resizing them here "
            "would resample an already-resampled image."
        )

    # `has_eye_feature` already implies feature_dim is not None, but the checker
    # cannot see through the property, so the narrowing is written out.
    widths = sorted({s.feature_dim for s in specs if s.feature_dim is not None})
    if len(widths) > 1:
        raise SchemaError(
            f"Datasets supply handcrafted eye features of differing widths {widths}. "
            "They must be extracted identically across corpora to be comparable."
        )
    resolved_feature = widths[0] if widths else None

    truncated = [] if stills else [s.name for s in specs if s.time_dim > resolved_time]
    if truncated:
        logger.warning(
            f"time_dim={resolved_time} is shorter than the native window of "
            f"{truncated}; those corpora will be cropped on read."
        )

    padded = {s.name: s.time_dim for s in specs if s.time_dim < resolved_time}
    if padded:
        # Expected whenever rates differ -- a 15 fps corpus covers the same
        # duration in half the frames -- but worth surfacing, since it is also
        # what a misdeclared fps looks like.
        logger.info(
            f"Padding to T={resolved_time} with an all-False mask: "
            + ", ".join(f"{k} (T={v})" for k, v in sorted(padded.items()))
        )

    logger.info(
        f"Unified shape: T={resolved_time}, image={resolved_size}px, "
        f"eye_feature_dim={resolved_feature}, datasets={[s.name for s in specs]}"
    )
    return resolved_time, resolved_size, resolved_feature


def build_schemas(
    specs: list[DatasetSpec],
    time_dim: int | None = None,
    image_size: int | None = None,
    stills: bool = False,
    embedding_dim: int | None = None,
) -> tuple[list[DatasetSchema], int, int, int | None]:
    """Resolve the run's shared shape and build every corpus's schema.

    Args:
        specs (list[DatasetSpec]): Corpora participating in the run.
        time_dim (int | None): Shared window length in frames, or ``None`` to
            infer from the corpora's rates.
        image_size (int | None): Shared crop size, or ``None`` to infer.
        stills (bool): Whether the run serves single frames.
        embedding_dim (int | None): Width of a cached eye embedding, or ``None``
            when the run encodes live.

    Returns:
        tuple: ``(schemas, time_dim, image_size, feature_dim)``, with schemas in
        the same order as ``specs``.
    """
    resolved_time, resolved_size, resolved_feature = unified_specs(
        specs, time_dim=time_dim, image_size=image_size, stills=stills
    )
    schemas = [
        dataset_schema(spec, resolved_time, resolved_size, resolved_feature, embedding_dim)
        for spec in specs
    ]
    return schemas, resolved_time, resolved_size, resolved_feature
