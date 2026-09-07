"""Cutting fixed-length windows out of an annotated recording.

Blink presence detection is a windowed task: the model sees ``T`` consecutive
frames and decides whether they contain a blink. This module turns a
:class:`~blinklinmult.preprocess.annotation.TagFile` into those windows.

**One sampling rule, for every split.** :func:`sliding_windows` sweeps the whole
recording at a fixed stride — 50% overlap by default — and that is what train,
validation and test all get. There is deliberately no second policy.

1.x sampled training differently: one window *centred* on each blink, plus an
equal number of randomly drawn blink-free ones. That was removed, for two
measured reasons:

* it fit the model at a **50% blink prior** and evaluated it at ~10%, because
  the sweep a real recording produces is overwhelmingly blink-free;
* centring every training window on its event meant the model never saw a blink
  *partially observed at a window edge* — which is the case a deployed sweep
  produces constantly.

Training and evaluation are now the same process, and the natural class prior
reaches the loss, which is what ``focal`` is configured for.

**Label convention.** A window's frame-level label is
:attr:`~blinklinmult.preprocess.annotation.TagFile.blink_presence` restricted to
the window; its clip-level label is whether *any* frame in it is a blink frame.
The sequence label is what the model is trained against, and the clip label is
what the "blink presence" metric scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from blinklinmult.data.schema import LEFT, RIGHT, SchemaError, frames_for
from blinklinmult.preprocess.annotation import NO_BLINK

if TYPE_CHECKING:
    import numpy as np

    from blinklinmult.preprocess.annotation import TagFile

DEFAULT_WINDOW = 15
"""Default window length in frames, for callers that do not derive one.

At 30 fps this is the 0.5 s window the published work used. Corpora derive
their own frame count from their rate via
:func:`window_frames`, so that a 15 fps and a 30 fps recording span the same
real duration; this constant is only the fallback.
"""


def window_frames(fps: float, window_seconds: float) -> int:
    """Frames spanning a given duration at a given rate.

    Args:
        fps (float): Frame rate of the recording.
        window_seconds (float): Desired window duration.

    Returns:
        int: Frame count, at least 1.

    Raises:
        WindowError: If either argument is not positive.
    """
    try:
        return frames_for(fps, window_seconds)
    except SchemaError as error:
        raise WindowError(str(error)) from error


class WindowError(ValueError):
    """Raised when windows cannot be cut as requested."""


@dataclass(frozen=True)
class Window:
    """One fixed-length window of a recording.

    Args:
        video_id (str): Recording the window came from.
        start_index (int): First record index, inclusive.
        length (int): Window length in frames.
        frame_ids (np.ndarray): ``(T,)`` int64 frame ids, for locating images.
        labels (np.ndarray): ``(T,)`` float32 per-frame blink-presence labels.
        eye_state (np.ndarray): ``(T, 2)`` float32 per-eye closed labels.
        validity (np.ndarray): ``(T, 2)`` bool per-eye visibility. ``False``
            where the annotator could not see the eye, so the frame carries no
            usable eye-state supervision.
        _blink_ids (np.ndarray): ``(T,)`` int64 blink id per frame, behind
            :attr:`blink_ids`.
    """

    video_id: str
    start_index: int
    length: int
    frame_ids: np.ndarray
    labels: np.ndarray
    eye_state: np.ndarray
    validity: np.ndarray
    _blink_ids: np.ndarray

    @property
    def has_blink(self) -> bool:
        """Whether this window contains at least one blink frame.

        Returns:
            bool: The clip-level blink-presence label.
        """
        return bool(self.labels.any())

    @property
    def frame_group(self) -> str:
        """Identifier of the frames this window covers.

        Built from the first frame id rather than from a counter, so a rebuild
        produces the same value, and so the two eye-wise samples cut from this
        window share it — which is what lets the frame-level evaluation
        recombine them.

        Returns:
            str: e.g. ``"000881"``.
        """
        return f"{int(self.frame_ids[0]):06d}"

    def eye_state_for(self, eye_side: str) -> np.ndarray:
        """Per-frame closed label for one eye.

        Args:
            eye_side (str): :data:`~blinklinmult.data.schema.LEFT` or
                :data:`~blinklinmult.data.schema.RIGHT`.

        Returns:
            np.ndarray: ``(T,)`` float32.

        Raises:
            WindowError: If the side is not one of the two annotated eyes.
        """
        if eye_side == LEFT:
            return self.eye_state[:, 0]
        if eye_side == RIGHT:
            return self.eye_state[:, 1]
        raise WindowError(
            f"eye_state is annotated per side; {eye_side!r} is not one of {LEFT!r}/{RIGHT!r}."
        )

    @property
    def blink_ids(self) -> np.ndarray:
        """Blink id of every frame in the window.

        Returns:
            np.ndarray: ``(T,)`` int64;
            :data:`~blinklinmult.preprocess.annotation.NO_BLINK` where the eye
            is open.
        """
        return self._blink_ids

    @property
    def blink_id(self) -> int:
        """The single blink this window was cut around.

        For the **balanced training** windows, where the protocol requires that
        every annotated event is countable in exactly one sample. A continuous
        evaluation sweep cannot honour that — two blinks half a second apart
        genuinely fall in one window — so it reads :attr:`first_blink_id`
        instead.

        Returns:
            int: The event's id, or
            :data:`~blinklinmult.preprocess.annotation.NO_BLINK` for a
            blink-free window.

        Raises:
            WindowError: If the window contains frames from more than one blink.
        """
        present = self._present_blinks()
        if len(present) > 1:
            raise WindowError(
                f"{self.video_id}@{self.frame_group}: window spans blinks "
                f"{sorted(present)}. A window must contain at most one blink, so "
                "that every annotated event is countable in exactly one sample. "
                "Use first_blink_id for a continuous evaluation sweep, where "
                "overlapping events are expected."
            )
        return present.pop() if present else NO_BLINK

    @property
    def first_blink_id(self) -> int:
        """The earliest blink in this window, whatever else it contains.

        The evaluation counterpart of :attr:`blink_id`. A sliding sweep at a
        stride shorter than the window will sometimes span two blinks — on
        TalkingFace, 9 of 713 windows do, which is natural double-blinking
        rather than an annotation fault. This records one of them for
        provenance rather than refusing to write the sample.

        Scoring is unaffected: blink presence detection asks whether a window
        contains *a* blink, and the stored id is not read back for that.

        Returns:
            int: The first event's id, or
            :data:`~blinklinmult.preprocess.annotation.NO_BLINK`.
        """
        for value in self._blink_ids:
            if int(value) != NO_BLINK:
                return int(value)
        return NO_BLINK

    def _present_blinks(self) -> set[int]:
        """Distinct blink ids appearing in this window.

        Returns:
            set[int]: The ids, excluding
            :data:`~blinklinmult.preprocess.annotation.NO_BLINK`.
        """
        return {int(value) for value in self._blink_ids if int(value) != NO_BLINK}

    def validity_for(self, eye_side: str) -> np.ndarray:
        """Per-frame visibility for one eye.

        Args:
            eye_side (str): :data:`~blinklinmult.data.schema.LEFT` or
                :data:`~blinklinmult.data.schema.RIGHT`.

        Returns:
            np.ndarray: ``(T,)`` bool, ``True`` where the eye is visible.

        Raises:
            WindowError: If the side is not one of the two annotated eyes.
        """
        if eye_side == LEFT:
            return self.validity[:, 0]
        if eye_side == RIGHT:
            return self.validity[:, 1]
        raise WindowError(
            f"validity is annotated per side; {eye_side!r} is not one of {LEFT!r}/{RIGHT!r}."
        )


def _cut(tag: TagFile, start: int, window: int) -> Window:
    """Build a :class:`Window` from a validated start index.

    Args:
        tag (TagFile): Source annotation.
        start (int): First record index.
        window (int): Window length.

    Returns:
        Window: The cut window.
    """
    stop = start + window
    return Window(
        video_id=tag.video_id,
        start_index=start,
        length=window,
        frame_ids=tag.frame_ids[start:stop],
        labels=tag.blink_presence[start:stop],
        eye_state=tag.eye_state[start:stop],
        validity=tag.validity[start:stop],
        _blink_ids=tag.blink_ids[start:stop],
    )


def _check_window(tag: TagFile, window: int) -> None:
    """Verify a window of this length can be cut from this recording.

    Args:
        tag (TagFile): Source annotation.
        window (int): Requested window length.

    Raises:
        WindowError: If the length is not positive or exceeds the recording.
    """
    if window < 1:
        raise WindowError(f"window must be >= 1, got {window}.")
    if window > len(tag):
        raise WindowError(
            f"{tag.video_id}: cannot cut a {window}-frame window from a recording "
            f"with only {len(tag)} annotated frames."
        )


def sliding_windows(
    tag: TagFile, window: int = DEFAULT_WINDOW, stride: int | None = None
) -> list[Window]:
    """Cut windows at a fixed stride across the whole recording.

    **The only sampling policy**, used for training as well as evaluation. A
    recording swept end to end is what a deployed model sees — no oracle says
    where the blinks are — so training on anything else fits a prior the model
    will never meet. See this module's docstring for what was removed and why.

    Args:
        tag (TagFile): Source annotation.
        window (int): Window length in frames.
        stride (int | None): Frames between window starts. Defaults to
            ``window`` (non-overlapping); callers pass ``window // 2`` for the
            50%-overlapping sweep the evaluation protocol expects.

    Returns:
        list[Window]: Windows covering the recording, in order.

    Raises:
        WindowError: If the window does not fit, or the stride is not positive.
    """
    _check_window(tag, window)
    stride = window if stride is None else stride
    if stride < 1:
        raise WindowError(f"stride must be >= 1, got {stride}.")

    starts = range(0, len(tag) - window + 1, stride)
    return [_cut(tag, start, window) for start in starts]
