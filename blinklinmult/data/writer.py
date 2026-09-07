"""Write samples straight into a corpus's HDF5 file, as they are produced.

The 1.x pipeline staged every sample as a ``.npy`` file plus a manifest, then a
second pass read them back and assembled the HDF5. Measured on MRL-Eye, the
staging cost **884 MB against an 11 MB h5** — 80x the artifact, for a copy that
existed only to be read once and deleted.

This writer removes that layer. A preprocessing run opens one file, appends
each sample as it is cut, and closes it: one command, one artifact, and no
intermediate that can drift out of step with either end.

**The file is complete or it is absent.** Samples go to ``<name>.h5.tmp``,
renamed only on a clean close. An interrupted run leaves nothing behind rather
than a partial file that a later run would mistake for a finished one.

The layout, dtypes, compression, and root attributes are exactly those
:mod:`blinklinmult.data.builder` writes — that contract is what OmniLoader
reads, and it does not change.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import h5py
import numpy as np

from blinklinmult.data.schema import (
    BLINK_ID,
    BLINK_IDS,
    BLINK_PRESENCE,
    EYE_FEATURE,
    EYE_IMAGE,
    EYE_SIDE,
    EYE_STATE,
    FRAME_GROUP,
    HEAD_POSE,
    HEAD_POSE_DIM,
    IMAGE_CHANNELS,
    NO_BLINK,
    SAMPLE_KEY,
    SCHEMA_VERSION,
    SOURCE_KEY,
    SUBSETS,
    BuildStats,
    DatasetSpec,
    build_sample_id,
)

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)
"""Module-level logger."""

COMPRESSION = "gzip"
"""Compression filter, paired with the shuffle filter for float data."""

COMPRESSION_LEVEL = 4
"""Compression level: a middle ground between file size and read CPU."""

IMAGE_DTYPE = np.float16
"""On-disk dtype for eye crops and descriptors.

The crops dominate the file — a 15x3x64x64 window is 184k values against 15
labels — and fp16 halves both the file and the read bandwidth at a precision
finer than 8-bit pixel data carries in the first place.
"""

LABEL_DTYPE = np.float32
"""On-disk dtype for targets. Labels are tiny; precision costs nothing."""


class WriterError(RuntimeError):
    """Raised when a corpus cannot be written as configured."""


class H5Writer:
    """Streams one corpus's samples into its HDF5 file.

    Use as a context manager: the file is renamed into place on a clean exit
    and discarded on an exception.

    Args:
        spec (DatasetSpec): The corpus declaration, which fixes the shapes and
            which targets are written.
        h5_path (Path): Destination file.
        config_yaml (str): The corpus config, embedded for traceability.
        git_sha (str): Commit the build was made from.
        extra_attrs (dict | None): Further root attributes describing how this
            corpus was sampled — the evaluation protocol and its stride, say.
            A file read out of context should still be able to say what it is.

    Raises:
        WriterError: If a sample disagrees with the declaration.
    """

    def __init__(
        self,
        spec: DatasetSpec,
        h5_path: Path,
        config_yaml: str = "",
        git_sha: str = "unknown",
        extra_attrs: dict[str, Any] | None = None,
    ):
        self.spec = spec
        self.h5_path = h5_path
        self.config_yaml = config_yaml
        self.git_sha = git_sha
        self.extra_attrs = dict(extra_attrs or {})
        self.stats = BuildStats(dataset=spec.name)

        self._tmp_path = h5_path.with_suffix(h5_path.suffix + ".tmp")
        self._handle: h5py.File | None = None
        self._groups: dict[str, h5py.Group] = {}
        self._seen: set[str] = set()
        self._positives: dict[str, int] = dict.fromkeys(SUBSETS, 0)

    def __enter__(self) -> H5Writer:
        """Open the temporary file and write the root attributes.

        Returns:
            H5Writer: This writer.
        """
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = h5py.File(self._tmp_path, "w")
        self._write_root_attrs(self._handle)
        for subset in SUBSETS:
            self._groups[subset] = self._handle.create_group(subset)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Close the file, renaming it into place only on success.

        Args:
            exc_type: Exception type, if one is propagating.
            exc_value: The exception.
            traceback: Its traceback.
        """
        if self._handle is not None:
            self._handle.close()
            self._handle = None

        if exc_type is not None:
            # A half-written corpus must not look like a finished one.
            self._tmp_path.unlink(missing_ok=True)
            logger.warning(f"{self.spec.name}: build failed; {self._tmp_path} discarded.")
            return

        self._tmp_path.replace(self.h5_path)
        self.stats.bytes_written = self.h5_path.stat().st_size
        self.stats.positive_fraction = {
            subset: (self._positives[subset] / count if count else 0.0)
            for subset, count in self.stats.per_subset.items()
        }
        logger.info(
            f"Wrote {self.h5_path} ({self.stats.bytes_written / 1e6:.1f} MB, "
            f"{self.stats.n_samples} samples: "
            + ", ".join(f"{k}={v}" for k, v in self.stats.per_subset.items())
            + ")"
        )

    def _write_root_attrs(self, handle: h5py.File) -> None:
        """Record what this file is and what produced it.

        Args:
            handle (h5py.File): The open file.
        """
        spec = self.spec
        handle.attrs["schema_version"] = SCHEMA_VERSION
        handle.attrs["dataset"] = spec.name
        handle.attrs["created_utc"] = datetime.now(UTC).isoformat()
        handle.attrs["git_sha"] = self.git_sha
        handle.attrs["builder_config_yaml"] = self.config_yaml
        handle.attrs["time_dim"] = spec.time_dim
        handle.attrs["image_size"] = spec.image_size
        handle.attrs["image_channels"] = IMAGE_CHANNELS
        handle.attrs["feature_dim"] = -1 if spec.feature_dim is None else spec.feature_dim
        handle.attrs["fps"] = -1.0 if spec.fps is None else spec.fps
        # The window is declared in seconds and time_dim derived from the rate,
        # so both are recorded: a frame count alone does not say what duration
        # it covers.
        handle.attrs["window_seconds"] = (
            -1.0 if spec.window_seconds is None else spec.window_seconds
        )
        handle.attrs["has_blink_presence"] = spec.has_blink_presence
        handle.attrs["has_eye_state"] = spec.has_eye_state
        handle.attrs["has_head_pose"] = spec.has_head_pose
        handle.attrs["has_blink_ids"] = spec.has_blink_ids
        handle.attrs["quality_signals"] = list(spec.quality_signals)

        for key, value in self.extra_attrs.items():
            handle.attrs[key] = value

    def add(
        self,
        subset: str,
        video_id: str,
        frame_group: str,
        eye_side: str,
        eye_images: np.ndarray,
        eye_image_mask: np.ndarray | None = None,
        blink_presence: np.ndarray | None = None,
        blink_presence_mask: np.ndarray | None = None,
        eye_state: np.ndarray | None = None,
        eye_state_mask: np.ndarray | None = None,
        eye_features: np.ndarray | None = None,
        eye_feature_mask: np.ndarray | None = None,
        blink_id: int = NO_BLINK,
        head_pose: np.ndarray | None = None,
        quality_signals: dict[str, np.ndarray] | None = None,
        blink_ids: np.ndarray | None = None,
        quality: dict[str, np.ndarray | float | str] | None = None,
    ) -> str:
        """Write one eye-wise sample.

        Args:
            subset (str): Split this sample belongs to.
            video_id (str): Source recording or subject.
            frame_group (str): Identifier shared by both eyes of one window,
                which is what lets frame-level scoring recombine them.
            eye_side (str): Which eye.
            eye_images (np.ndarray): ``(T, C, H, W)`` normalised crops.
            eye_image_mask (np.ndarray | None): ``(T,)`` validity; ``None``
                means every frame is real.
            blink_presence (np.ndarray | None): ``(T,)`` target.
            blink_presence_mask (np.ndarray | None): ``(T,)`` validity.
            eye_state (np.ndarray | None): ``(T,)`` target.
            eye_state_mask (np.ndarray | None): ``(T,)`` validity.
            eye_features (np.ndarray | None): ``(T, F)`` descriptors.
            eye_feature_mask (np.ndarray | None): ``(T,)`` validity.
            blink_id (int): The blink this window contains, or
                :data:`~blinklinmult.data.schema.NO_BLINK`.
            head_pose (np.ndarray | None): ``(T, 3)`` ``[yaw, pitch, roll]`` in
                degrees. Required when the corpus declares
                :attr:`~blinklinmult.data.schema.DatasetSpec.has_head_pose`.
            quality_signals (dict[str, np.ndarray] | None): Each of
                :data:`~blinklinmult.data.schema.QUALITY_SIGNALS` as a ``(T,)``
                array in ``[0, 1]``. Stored separately rather than combined, so
                the weighting stays a dataloader decision.
            blink_ids (np.ndarray | None): ``(T,)`` which annotated blink each
                frame belongs to, :data:`~blinklinmult.data.schema.NO_BLINK`
                outside every event.
            quality (dict | None): Optional diagnostics stored verbatim
                alongside the sample -- per-frame ``(T,)`` arrays such as
                :data:`~blinklinmult.data.schema.EYE_ON_SCREEN`, the
                :data:`~blinklinmult.data.schema.FACE_BOX`, or scalars like
                :data:`~blinklinmult.data.schema.CONFIDENCE`. Nothing here is
                read during training; it exists so bad data can be found and
                looked at. Per-frame arrays are fitted to the window length
                exactly as the images are.

        Returns:
            str: The sample key.

        Raises:
            WriterError: If the writer is closed, the subset is unknown, the
                key collides, or an array disagrees with the declaration.
        """
        if self._handle is None:
            raise WriterError(f"{self.spec.name}: writer is not open. Use it as a context manager.")
        if subset not in SUBSETS:
            raise WriterError(f"{self.spec.name}: unknown subset {subset!r}; expected {SUBSETS}.")

        sample_id = build_sample_id(video_id, frame_group, eye_side)
        if sample_id in self._seen:
            raise WriterError(
                f"{self.spec.name}: duplicate sample key {sample_id!r}. Keys must be "
                "unique — an HDF5 group name collision would silently drop a sample."
            )
        self._seen.add(sample_id)

        time_dim = self.spec.time_dim
        images = self._check_images(eye_images, sample_id)
        group = self._groups[subset].create_group(sample_id)

        self._create(group, EYE_IMAGE, self._fit(images, time_dim).astype(IMAGE_DTYPE))
        self._create(
            group,
            f"{EYE_IMAGE}_mask",
            self._fit_mask(eye_image_mask, images.shape[0], time_dim),
        )

        if self.spec.has_eye_feature:
            self._write_features(group, eye_features, eye_feature_mask, sample_id, time_dim)

        has_blink = self._write_targets(
            group,
            time_dim,
            blink_presence=blink_presence,
            blink_presence_mask=blink_presence_mask,
            eye_state=eye_state,
            eye_state_mask=eye_state_mask,
        )

        if quality:
            self._write_quality(group, quality, time_dim)

        group.create_dataset(BLINK_ID, data=np.int64(blink_id))
        self._write_frame_fields(group, time_dim, head_pose, quality_signals, blink_ids)
        for key, value in (
            (SAMPLE_KEY, sample_id),
            (SOURCE_KEY, self.spec.name),
            ("video_id", video_id),
            (FRAME_GROUP, frame_group),
            (EYE_SIDE, eye_side),
        ):
            group.create_dataset(key, data=value)

        self.stats.per_subset[subset] = self.stats.per_subset.get(subset, 0) + 1
        self._positives[subset] += int(has_blink)
        return sample_id

    def _write_quality(
        self,
        group: h5py.Group,
        quality: dict[str, np.ndarray | float | str],
        time_dim: int,
    ) -> None:
        """Store per-sample diagnostics beside the data.

        Written verbatim and never read back by training: these exist so a
        marginal crop can be *found*, having learnt the hard way that automated
        checks miss what looking catches. Per-frame arrays are padded or
        truncated to the window length so they stay aligned with the images;
        scalars and strings are stored as-is.

        Args:
            group (h5py.Group): The sample's group.
            quality (dict): Diagnostic name to value.
            time_dim (int): Window length to fit per-frame arrays to.
        """
        for name, value in quality.items():
            if isinstance(value, str):
                group.create_dataset(name, data=value)
                continue
            array = np.asarray(value)
            if array.ndim == 0:
                group.create_dataset(name, data=array)
                continue
            fitted = self._fit(array, time_dim)
            if array.dtype == np.bool_:
                dtype = np.bool_
            elif np.issubdtype(array.dtype, np.integer):
                # Coordinates stay integral: float16 is exact only to 2048, and
                # MPEblink frames are 2560 px wide, so a face box stored as
                # float16 would come back displaced by pixels.
                dtype = np.int32
            else:
                # A span or a ratio needs nothing like float32's range.
                dtype = IMAGE_DTYPE
            self._create(group, name, fitted.astype(dtype))

    def _write_features(
        self,
        group: h5py.Group,
        features: np.ndarray | None,
        mask: np.ndarray | None,
        sample_id: str,
        time_dim: int,
    ) -> None:
        """Write the handcrafted descriptor stream.

        Args:
            group (h5py.Group): The sample's group.
            features (np.ndarray | None): ``(T, F)`` descriptors.
            mask (np.ndarray | None): ``(T,)`` validity.
            sample_id (str): For error messages.
            time_dim (int): Window length to fit to.

        Raises:
            WriterError: If the corpus declares features but none were supplied,
                or their width disagrees with the declaration.
        """
        if features is None:
            raise WriterError(
                f"{self.spec.name}/{sample_id}: the corpus declares "
                f"feature_dim={self.spec.feature_dim} but no eye features were supplied."
            )
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.spec.feature_dim:
            raise WriterError(
                f"{self.spec.name}/{sample_id}: eye features have shape {values.shape}, "
                f"expected (T, {self.spec.feature_dim})."
            )
        self._create(group, EYE_FEATURE, self._fit(values, time_dim).astype(IMAGE_DTYPE))
        self._create(group, f"{EYE_FEATURE}_mask", self._fit_mask(mask, values.shape[0], time_dim))

    def _write_frame_fields(
        self,
        group: h5py.Group,
        time_dim: int,
        head_pose: np.ndarray | None,
        quality_signals: dict[str, np.ndarray] | None,
        blink_ids: np.ndarray | None,
    ) -> None:
        """Write the per-frame conditioning fields this corpus declares.

        These are declared as *features* in the OmniLoader schema rather than
        stored in the quality group, because a quality field is written but
        never loaded -- see :func:`~blinklinmult.data.omni.dataset_schema`.

        Args:
            group (h5py.Group): The sample's group.
            time_dim (int): Window length to fit to.
            head_pose (np.ndarray | None): ``(T, 3)`` degrees.
            quality_signals (dict[str, np.ndarray] | None): Per-signal ``(T,)``.
            blink_ids (np.ndarray | None): ``(T,)`` event ids.

        Raises:
            WriterError: If a declared field was not supplied, or head pose does
                not have three columns.
        """
        if self.spec.has_head_pose:
            if head_pose is None:
                raise WriterError(
                    f"{self.spec.name}: the corpus declares {HEAD_POSE} but the sample "
                    "supplied none."
                )
            angles = np.asarray(head_pose, dtype=LABEL_DTYPE).reshape(-1, HEAD_POSE_DIM)
            self._create(group, HEAD_POSE, self._fit(angles, time_dim))

        supplied = quality_signals or {}
        missing = [name for name in self.spec.quality_signals if name not in supplied]
        if missing:
            raise WriterError(
                f"{self.spec.name}: the corpus declares quality signals but the "
                f"sample supplied none for {missing}."
            )
        for name in self.spec.quality_signals:
            values = np.asarray(supplied[name], dtype=LABEL_DTYPE).reshape(-1)
            self._create(group, name, self._fit(values, time_dim))

        if self.spec.has_blink_ids:
            if blink_ids is None:
                raise WriterError(
                    f"{self.spec.name}: the corpus declares {BLINK_IDS} but the sample "
                    "supplied none."
                )
            # int32, and padded with NO_BLINK rather than zero: zero is a valid
            # event id, so zero-padding would invent an event on every short
            # window.
            ids = np.asarray(blink_ids, dtype=np.int32).reshape(-1)
            fitted = np.full(time_dim, NO_BLINK, dtype=np.int32)
            keep = min(ids.size, time_dim)
            fitted[:keep] = ids[:keep]
            self._create(group, BLINK_IDS, fitted)

    def _write_targets(self, group: h5py.Group, time_dim: int, **arrays) -> bool:
        """Write whichever targets this corpus annotates.

        Args:
            group (h5py.Group): The sample's group.
            time_dim (int): Window length to fit to.
            **arrays: ``<target>`` and ``<target>_mask`` keyword arrays.

        Returns:
            bool: Whether this sample contains a blink.

        Raises:
            WriterError: If a declared target was not supplied.
        """
        has_blink = False
        for key, declared in (
            (BLINK_PRESENCE, self.spec.has_blink_presence),
            (EYE_STATE, self.spec.has_eye_state),
        ):
            if not declared:
                continue
            values = arrays.get(key)
            if values is None:
                raise WriterError(
                    f"{self.spec.name}: the corpus declares {key} but the sample supplied none."
                )
            array = np.asarray(values, dtype=LABEL_DTYPE).reshape(-1)
            fitted = self._fit(array, time_dim)
            self._create(group, key, fitted)
            self._create(
                group,
                f"{key}_mask",
                self._fit_mask(arrays.get(f"{key}_mask"), array.size, time_dim),
            )
            if key == BLINK_PRESENCE:
                has_blink = bool((fitted > 0.5).any())
        return has_blink

    def _check_images(self, images: np.ndarray, sample_id: str) -> np.ndarray:
        """Validate a sample's crops against the declaration.

        Args:
            images (np.ndarray): ``(T, C, H, W)`` crops.
            sample_id (str): For error messages.

        Returns:
            np.ndarray: The crops, unchanged.

        Raises:
            WriterError: If the shape disagrees with the declaration.
        """
        array = np.asarray(images, dtype=np.float32)
        expected = (IMAGE_CHANNELS, self.spec.image_size, self.spec.image_size)
        if array.ndim != 4 or tuple(array.shape[1:]) != expected:
            raise WriterError(
                f"{self.spec.name}/{sample_id}: eye images have shape {array.shape}, "
                f"expected (T, {', '.join(str(v) for v in expected)})."
            )
        return array

    @staticmethod
    def _fit(array: np.ndarray, time_dim: int) -> np.ndarray:
        """Pad or truncate an array's leading axis to the window length.

        Args:
            array (np.ndarray): The array.
            time_dim (int): Target length.

        Returns:
            np.ndarray: The fitted array.
        """
        if array.shape[0] == time_dim:
            return array
        if array.shape[0] > time_dim:
            return array[:time_dim]
        padding = [(0, time_dim - array.shape[0])] + [(0, 0)] * (array.ndim - 1)
        return np.pad(array, padding)

    @classmethod
    def _fit_mask(cls, mask: np.ndarray | None, length: int, time_dim: int) -> np.ndarray:
        """Fit a validity mask, marking padding invalid whatever it said.

        Args:
            mask (np.ndarray | None): The source mask, or ``None`` for all-valid.
            length (int): Number of real timesteps.
            time_dim (int): Target length.

        Returns:
            np.ndarray: ``(time_dim,)`` bool.
        """
        source = np.ones(length, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
        fitted = cls._fit(source.reshape(-1), time_dim)
        fitted[min(length, time_dim) :] = False
        return fitted

    @staticmethod
    def _create(group: h5py.Group, name: str, data: np.ndarray) -> None:
        """Create one compressed dataset.

        Args:
            group (h5py.Group): Destination group.
            name (str): Dataset name.
            data (np.ndarray): The array.
        """
        group.create_dataset(
            name,
            data=data,
            compression=COMPRESSION,
            compression_opts=COMPRESSION_LEVEL,
            shuffle=True,
        )
