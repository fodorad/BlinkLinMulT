"""Derive single frames from windowed corpora, for the frame-wise model.

BlinkCNN classifies one eye crop at a time, so it can train on any
annotated frame — the still corpora, and equally the video corpora with their
time dimension ignored. Rather than build a second set of HDF5 files, this
module reduces an already-built window to its individual frames at load time:
the same bytes serve both the sequence models and the frame-wise one.

**Balancing.** Sampled as they fall, video frames are overwhelmingly open-eyed —
a blink occupies a handful of frames per second of recording. Training on that
distribution teaches "predict open", which scores well on accuracy and detects
nothing. Every closed-eye frame is therefore kept and the open-eye frames are
subsampled to :attr:`StillConfig.open_to_closed`, giving a set comparable to
CEW's natural 1193/1232.

**Determinism.** The subsample is drawn once, at construction, from a seeded
generator. Two runs with the same seed see the same frames, and a frame's
identity does not change between epochs.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from blinklinmult.data.schema import (
    EYE_IMAGE,
    EYE_STATE,
    SAMPLE_KEY,
    build_sample_id,
    parse_sample_id,
)

if TYPE_CHECKING:
    from omniloader.loader import SizedDataset

logger = logging.getLogger(__name__)
"""Module-level logger."""

MASK_SUFFIX = "_mask"
"""Suffix OmniLoader appends to a key to name its validity mask."""

CLOSED_THRESHOLD = 0.5
"""Eye-state value above which a frame counts as closed.

The targets are written as floats so a corpus may express uncertainty, but the
balancing only needs the two classes.
"""


class StillsError(RuntimeError):
    """Raised when a windowed corpus cannot be reduced to frames."""


@dataclass(frozen=True)
class StillConfig:
    """How to draw single frames from windowed samples.

    Args:
        stride (int): Keep every Nth frame of a window. Consecutive frames of a
            recording are near-identical, so taking all of them inflates the
            corpus without adding information. Closed-eye frames ignore the
            stride — they are too scarce to thin.
        open_to_closed (float): Open-eye frames to keep per closed-eye frame.
            ``1.0`` balances the classes. Raise it to keep more negatives.
        seed (int): Seed for the open-eye subsample.
    """

    stride: int = 6
    open_to_closed: float = 1.0
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the configuration.

        Raises:
            StillsError: If a value is out of range.
        """
        if self.stride < 1:
            raise StillsError(f"stills.stride must be >= 1, got {self.stride}.")
        if self.open_to_closed <= 0:
            raise StillsError(f"stills.open_to_closed must be positive, got {self.open_to_closed}.")


class StillFramesDataset:
    """A windowed corpus served one frame at a time.

    Each item is a window collapsed to a single timestep: every sequence-valued
    key is indexed at the chosen frame and re-wrapped with a length-1 time axis,
    so the sample satisfies the same schema as a natively still corpus.

    Args:
        dataset (SizedDataset): The corpus's split, yielding windowed samples.
        config (StillConfig): Sampling and balancing settings.
        name (str): Corpus name, for log lines and error messages.
        balance (bool): Subsample open-eye frames. Left off for evaluation
            splits, which must be scored on the real class distribution.

    Raises:
        StillsError: If the corpus does not annotate eye state, or no frame
            survives selection.
    """

    def __init__(
        self,
        dataset: SizedDataset,
        config: StillConfig,
        name: str,
        balance: bool = True,
    ):
        self.dataset = dataset
        self.config = config
        self.name = name
        self.index = self._select(balance)

    def _select(self, balance: bool) -> list[tuple[int, int]]:
        """Choose the ``(sample, frame)`` pairs this dataset serves.

        Args:
            balance (bool): Whether to subsample open-eye frames.

        Returns:
            list[tuple[int, int]]: Sample index and frame offset, in order.

        Raises:
            StillsError: If eye state is missing or nothing survives selection.
        """
        closed: list[tuple[int, int]] = []
        opened: list[tuple[int, int]] = []

        for position in range(len(self.dataset)):
            sample = self.dataset[position]
            if EYE_STATE not in sample:
                if balance:
                    raise StillsError(
                        f"{self.name}: still mode needs per-frame eye state, which this "
                        "corpus does not annotate. Train the frame-wise model on the "
                        "corpora that do -- see config/data/stills_all.yaml."
                    )
                # Evaluation split: a corpus can be *scored* without a closure
                # label, because the benchmark scores it as blink events rather
                # than per-frame state. MPEblink and HUST-LEBW annotate events
                # only, and refusing to serve their frames would put them out of
                # the benchmark entirely. Every valid frame is served, strided.
                length = self._length(sample)
                usable = self._valid_mask(sample, length)
                opened.extend(
                    (position, frame)
                    for frame in range(length)
                    if usable[frame] and frame % self.config.stride == 0
                )
                continue

            state = np.asarray(sample[EYE_STATE]).reshape(-1)
            valid = self._valid_mask(sample, state.shape[0])

            for frame in range(state.shape[0]):
                if not valid[frame]:
                    continue
                if state[frame] > CLOSED_THRESHOLD:
                    # Every closed frame is kept: they are the scarce class, and
                    # striding them away is what makes a model predict "open".
                    closed.append((position, frame))
                elif frame % self.config.stride == 0:
                    opened.append((position, frame))

        if balance:
            opened = self._subsample(opened, len(closed))

        index = sorted(closed + opened)
        if not index:
            raise StillsError(
                f"{self.name}: no annotated frame survived still-mode selection. "
                f"Check the corpus was built with valid eye-state targets."
            )

        logger.info(
            f"{self.name}: {len(index)} still frames "
            f"({len(closed)} closed, {len(index) - len(closed)} open)"
        )
        return index

    @staticmethod
    def _length(sample: dict) -> int:
        """Timesteps in a window, read from its images.

        Args:
            sample (dict): One window.

        Returns:
            int: The window length, or ``0`` when it carries no images.
        """
        images = sample.get(EYE_IMAGE)
        return 0 if images is None else int(np.asarray(images).shape[0])

    def _valid_mask(self, sample: dict, length: int) -> np.ndarray:
        """Positions of the window that carry real annotation.

        A window shorter than the run's shared ``T`` is padded, and a corpus may
        annotate only part of one; either way the padding must not be trained on.

        Args:
            sample (dict): One windowed sample.
            length (int): Number of timesteps in the eye-state target.

        Returns:
            np.ndarray: Boolean array of length ``length``.
        """
        masks = [
            np.asarray(sample[key]).reshape(-1)
            for key in (f"{EYE_STATE}{MASK_SUFFIX}", f"{EYE_IMAGE}{MASK_SUFFIX}")
            if key in sample
        ]
        valid = np.ones(length, dtype=bool)
        for mask in masks:
            valid &= mask[:length].astype(bool)
        return valid

    def _subsample(self, opened: list[tuple[int, int]], closed: int) -> list[tuple[int, int]]:
        """Thin the open-eye frames to the configured ratio.

        Args:
            opened (list[tuple[int, int]]): Candidate open-eye frames.
            closed (int): Number of closed-eye frames kept.

        Returns:
            list[tuple[int, int]]: The retained open-eye frames.
        """
        keep = int(round(closed * self.config.open_to_closed))
        if keep >= len(opened):
            return opened

        # Seeded once at construction, so the drawn set is a property of the
        # config rather than of when the dataset happened to be built.
        generator = np.random.default_rng(self.config.seed)
        chosen = generator.choice(len(opened), size=keep, replace=False)
        return [opened[i] for i in sorted(chosen)]

    def __len__(self) -> int:
        """Number of frames served.

        Returns:
            int: Frame count.
        """
        return len(self.index)

    def __getitem__(self, index: int) -> dict:
        """Return one frame as a single-timestep sample.

        Args:
            index (int): Position within the selected frames.

        Returns:
            dict: The window's keys, each reduced to its chosen frame and given
            a length-1 time axis.
        """
        position, frame = self.index[index]
        window = self.dataset[position]
        return {key: self._reduce(key, value, frame) for key, value in window.items()}

    def _reduce(self, key: str, value: object, frame: int) -> object:
        """Take one timestep of a windowed value, leaving scalars alone.

        Args:
            key (str): The value's key.
            value (object): The stored value.
            frame (int): Timestep to keep.

        Returns:
            object: A length-1 sequence for time-varying values, the sample
            key with its frame appended, and everything else unchanged.
        """
        if key == SAMPLE_KEY:
            # Rewrite the frame group to this frame's **absolute** id, rather
            # than appending a suffix. `parse_sample_id` splits on "|" and does
            # not validate the eye-side field, so a suffixed key parses without
            # error and the suffix is silently dropped -- every frame of a
            # window would collapse onto one frame group. That corrupts
            # `frame_group_ids` (and so `valid/mean_f1`, which selects the
            # checkpoint) and `EventReport._by_recording`, both without a
            # warning. An absolute id keeps every consumer correct unchanged:
            # `_by_recording` does `range(start, start + 1)` here, which is
            # exactly this frame.
            video_id, frame_group, eye_side = parse_sample_id(str(value))
            try:
                start = int(frame_group)
            except ValueError as error:
                raise StillsError(
                    f"{self.name}: sample key {value!r} has a non-numeric frame "
                    "group, so its absolute frame id cannot be derived."
                ) from error
            return build_sample_id(video_id, f"{start + frame:06d}", eye_side)

        # A length-1 time axis rather than a bare scalar: the sample must
        # satisfy the same schema as a natively still corpus.
        if isinstance(value, torch.Tensor):
            return value[frame : frame + 1] if value.ndim >= 1 else value
        if isinstance(value, np.ndarray):
            return np.asarray(value)[frame : frame + 1] if value.ndim >= 1 else value
        return value
