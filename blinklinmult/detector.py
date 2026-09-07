"""One interface over every published model: crops in, blinks out.

The four shipped models were trained years apart, in two codebases, with
different input conventions and different output shapes. Comparing them fairly
means holding everything but the weights constant -- the same crops, the same
normalisation rule applied per model, the same event extractor, the same
metrics. That is what this class is for::

    from blinklinmult import BlinkDetector

    v1 = BlinkDetector.from_pretrained("densenet121-union")
    v2 = BlinkDetector.from_pretrained("blinkcnn")

    v1.detect(crops)   # [(12, 15), (48, 52)]
    v2.detect(crops)   # same crops, same extractor -- only the weights differ

Two methods, and the split is deliberate:

:meth:`BlinkDetector.score`
    Per-frame closure probability. A continuous physiological signal -- how shut
    the eye is -- which carries blink *shape*: depth, duration, the asymmetry
    between a fast closing ramp and a slower reopening.
:meth:`BlinkDetector.detect`
    Discrete ``(start, end)`` intervals, by running
    :func:`~blinklinmult.train.events.to_intervals` over that signal at the
    model's own fitted operating point.

``T=1`` is a single frame and any larger ``T`` is video, so one object serves
both. The sequence models need real temporal context to be worth anything, but
they accept ``T=1`` rather than failing.

**Normalisation travels with the model, never with the caller.** The 1.x networks
were trained on ImageNet-standardised crops and ``BlinkCNN`` on plain ``/255``;
applying the wrong convention costs accuracy and raises nothing. Pass raw
``[0, 1]`` crops and let :data:`~blinklinmult.registry.MODELS` decide.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from blinklinmult.registry import V1_FEATURE_ORDER, ModelSpec, spec
from blinklinmult.train.events import Interval, to_intervals

if TYPE_CHECKING:
    import torch
    import torch.nn as nn

    from blinklinmult.fit import OperatingPoint
    from blinklinmult.paper.session import OnnxModel

logger = logging.getLogger(__name__)
"""Module-level logger."""


class DetectorError(Exception):
    """Raised when a detector is built or called with something it cannot use."""


def _as_window(crops: np.ndarray | torch.Tensor) -> np.ndarray:
    """Coerce any accepted crop layout into a batched window.

    Accepts a single crop ``(3, H, W)``, a sequence ``(T, 3, H, W)`` or an
    already-batched window ``(B, T, 3, H, W)``.

    **numpy, not torch.** The shared scoring path -- reshaping, normalising,
    the sigmoid -- is array arithmetic that numpy does perfectly well, and
    keeping it torch-free is what lets an ONNX-only install skip a 469 MB
    dependency. A torch tensor is still accepted and converted.

    Args:
        crops (np.ndarray | torch.Tensor): Crops in ``[0, 1]``, channel-first.

    Returns:
        np.ndarray: ``(B, T, 3, H, W)`` float32.

    Raises:
        DetectorError: If the rank is not 3, 4 or 5.
    """
    array = _as_array(crops)
    if array.ndim == 3:
        return array[None, None]
    if array.ndim == 4:
        return array[None]
    if array.ndim == 5:
        return array
    raise DetectorError(
        "Expected crops of rank 3 (C,H,W), 4 (T,C,H,W) or 5 (B,T,C,H,W); "
        f"got shape {tuple(array.shape)}."
    )


def _as_array(values: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert anything array-like to a float32 numpy array.

    Torch tensors are handled without importing torch: every tensor exposes
    ``detach``/``cpu``/``numpy``, so duck-typing keeps this module importable in
    an environment that has no torch at all.

    Args:
        values (np.ndarray | torch.Tensor): The input.

    Returns:
        np.ndarray: float32, on the host.
    """
    if isinstance(values, np.ndarray):
        return values.astype(np.float32, copy=False)

    # A torch tensor. `detach` is needed before `numpy` when the tensor carries
    # a gradient -- numpy refuses otherwise -- and torch is imported here rather
    # than at module scope so an ONNX-only install never loads it. Reaching this
    # branch means the caller already has torch, so the import is free.
    detach = getattr(values, "detach", None)
    if callable(detach):
        import torch

        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.asarray(values, dtype=np.float32)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function.

    The naive ``1 / (1 + exp(-x))`` overflows on large negative logits, which
    the 1.x two-stream model produces routinely -- its unstandardised inputs
    saturate around -9.9.

    Args:
        logits (np.ndarray): Raw model output.

    Returns:
        np.ndarray: Probabilities in ``[0, 1]``.
    """
    positive = logits >= 0
    result = np.empty_like(logits, dtype=np.float32)
    result[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exponential = np.exp(logits[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _published_path(model_spec: ModelSpec) -> Path:
    """Resolve a model's weights, downloading them once if needed.

    Args:
        model_spec (ModelSpec): The model to locate.

    Returns:
        Path: A local file.

    Raises:
        DetectorError: If the weights are absent and cannot be fetched.
    """
    from blinklinmult import WEIGHTS_DIR
    from blinklinmult.registry import HF_MODEL_REPO

    cached = WEIGHTS_DIR / model_spec.filename
    if cached.is_file():
        return cached

    from huggingface_hub import hf_hub_download

    try:
        return Path(
            hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=model_spec.filename,
                cache_dir=str(WEIGHTS_DIR),
            )
        )
    except Exception as error:  # noqa: BLE001 - network, auth, or a missing file
        raise DetectorError(
            f"{model_spec.model_id!r} weights are not at {cached} and the "
            f"download from {HF_MODEL_REPO} failed ({error}). Pass weights=<path>."
        ) from error


class BlinkDetector:
    """A published model plus the preprocessing and extraction around it.

    Build one with :meth:`from_pretrained` rather than calling the constructor,
    unless wiring a model you loaded yourself.

    Args:
        model (nn.Module | OnnxModel): A loaded model -- a PyTorch module for the
            v2 generation, an ONNX session for the frozen 1.x graphs.
        model_spec (ModelSpec): Its entry from :data:`~blinklinmult.registry.MODELS`,
            carrying the normalisation and the fitted operating point.
    """

    def __init__(self, model: nn.Module | OnnxModel, model_spec: ModelSpec):
        self.model = model
        self.spec = model_spec
        self._mean = np.asarray(model_spec.mean, dtype=np.float32).reshape(1, 1, 3, 1, 1)
        self._std = np.asarray(model_spec.std, dtype=np.float32).reshape(1, 1, 3, 1, 1)

    @classmethod
    def from_pretrained(cls, model_id: str, weights: str | Path | None = None) -> BlinkDetector:
        """Load a published model by id.

        Args:
            model_id (str): A key of :data:`~blinklinmult.registry.MODELS` --
                ``densenet121-union``, ``blinklint-union``, ``blinklinmult-union``
                or ``blinkcnn``.
            weights (str | Path | None): A local checkpoint, instead of the
                published one.

        Returns:
            BlinkDetector: Ready to score.

        Raises:
            DetectorError: If the id is unknown or its weights cannot be loaded.
        """
        try:
            model_spec = spec(model_id)
        except KeyError as error:
            raise DetectorError(str(error)) from error

        if model_spec.runtime == "onnx":
            from blinklinmult.paper import SessionError, load_onnx

            path = weights if weights is not None else _published_path(model_spec)
            try:
                model = load_onnx(path)
            except SessionError as error:
                raise DetectorError(str(error)) from error
        else:
            from blinklinmult.models import ModelLoadError, load_checkpoint

            path = weights if weights is not None else _published_path(model_spec)
            try:
                model = load_checkpoint(path)
            except ModelLoadError as error:
                raise DetectorError(str(error)) from error

        return cls(model, model_spec)

    def _normalise(self, window: np.ndarray) -> np.ndarray:
        """Apply this model's own normalisation.

        Args:
            window (np.ndarray): ``(B, T, 3, H, W)`` crops in ``[0, 1]``.

        Returns:
            np.ndarray: Normalised the way this model was trained.
        """
        return (window - self._mean) / self._std

    def _forward(self, window: np.ndarray, features: np.ndarray | None) -> np.ndarray:
        """Run the wrapped model and return per-frame logits.

        Args:
            window (np.ndarray): ``(B, T, 3, H, W)``, already normalised.
            features (np.ndarray | None): ``(B, T, 160)`` for the two-stream
                model.

        Returns:
            np.ndarray: ``(B, T, 1)`` logits.

        Raises:
            DetectorError: If a required feature stream is missing.
        """
        if self.spec.needs_features and features is None:
            raise DetectorError(
                f"{self.spec.model_id!r} needs the 160-d feature stream; pass features=..."
            )

        if self.spec.runtime == "onnx":
            # `sequence` is the per-frame head; the two-stream graph's pooled
            # `clip` output answers a different question and is not what `score`
            # reports.
            from blinklinmult.paper import SessionError

            graph = cast("OnnxModel", self.model)
            try:
                outputs = graph(window, features)
            except SessionError as error:
                raise DetectorError(str(error)) from error
            return np.asarray(outputs["sequence"], dtype=np.float32)

        # The torch path, and the only place torch is needed. Imported here so
        # an ONNX-only install never pays for it -- `pip install
        # blinklinmult[onnx]` is ~100 MB against ~600 MB with the training stack.
        import torch

        tensor = torch.from_numpy(np.ascontiguousarray(window))
        mask = torch.ones(tensor.shape[:2], dtype=torch.bool)
        module = cast("nn.Module", self.model)
        with torch.no_grad():
            logits = module(tensor, mask)["eye_state"]
        return logits.detach().cpu().numpy().astype(np.float32)

    def prepare_features(
        self,
        features: np.ndarray,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
    ) -> np.ndarray:
        """Put a v2 feature vector into the layout this model expects.

        Only ``blinklinmult-union`` needs this, and it needs two things the v2
        schema does not provide:

        * **head pose first.** 1.x ordered the blocks
          ``[pose, landmarks, iris, diameters, eyelid, ear]``; v2 puts pose last.
        * **corpus-wide standardisation.** 1.x z-scored each descriptor against
          dataset statistics. Fed raw values the model's logits sit near -9.9 and
          it never fires -- silently, with no error.

        Args:
            features (np.ndarray): ``(T, 160)`` in the v2 layout.
            mean (np.ndarray | None): Per-dimension corpus mean, in the v2
                layout. Required when the model standardises.
            std (np.ndarray | None): Per-dimension corpus standard deviation.

        Returns:
            np.ndarray: ``(T, 160)`` ready to pass to :meth:`score`.

        Raises:
            DetectorError: If standardisation is required but no statistics were
                given.
        """
        prepared = np.asarray(features, dtype=np.float64)

        if self.spec.standardise_features:
            if mean is None or std is None:
                raise DetectorError(
                    f"{self.spec.model_id!r} needs corpus-wide feature statistics; "
                    "pass mean= and std= computed over the corpus you are scoring."
                )
            prepared = (prepared - np.asarray(mean)) / np.asarray(std)

        # Reorder last, so the statistics are indexed in the v2 layout the
        # caller computed them in.
        return np.concatenate(
            [prepared[:, start:stop] for start, stop in V1_FEATURE_ORDER], axis=-1
        ).astype(np.float32)

    def calibrate(self, point: OperatingPoint) -> None:
        """Adopt a fitted operating point in place of the registered default.

        The 1.x models ship with a placeholder 0.5 threshold, never a fitted one.
        Once :func:`~blinklinmult.fit.fit_operating_point` has measured a better
        pair on **validation** data, this applies it so :meth:`detect` uses it.

        Args:
            point (OperatingPoint): The fitted point, from
                :func:`~blinklinmult.fit.fit_operating_point`.
        """
        self.spec = point.applied_to(self.spec)
        logger.info(
            f"{self.spec.model_id!r}: operating point set to "
            f"threshold={self.spec.threshold:.2f} low_ratio={self.spec.low_ratio}."
        )

    def score(
        self,
        crops: np.ndarray | torch.Tensor,
        features: np.ndarray | torch.Tensor | None = None,
    ) -> np.ndarray:
        """Per-frame closure probability.

        Args:
            crops (np.ndarray | torch.Tensor): Crops in ``[0, 1]``, channel-first,
                as ``(3, H, W)``, ``(T, 3, H, W)`` or ``(B, T, 3, H, W)``.
            features (np.ndarray | torch.Tensor | None): ``(T, 160)`` or
                ``(B, T, 160)`` descriptors, required only by ``blinklinmult-union``.

        Returns:
            np.ndarray: ``(T,)`` for one window, else ``(B, T)``. Probabilities
                in ``[0, 1]``, not logits.

        Raises:
            DetectorError: If the crop rank is unusable, or a needed feature
                stream is absent.
        """
        window = _as_window(crops)
        batched = _as_array(crops).ndim == 5

        native = self.spec.window
        if native is not None and window.shape[1] != native:
            # Not an error: the graph accepts any length, and a caller may know
            # what they are doing. But the 1.x sequence models lose a lot of
            # accuracy off their native window -- measured 72% -> 92% and
            # 52% -> 84% on rn30 -- and that loss is otherwise invisible.
            logger.warning(
                f"{self.spec.model_id!r} was trained on {native}-frame windows but "
                f"was given {window.shape[1]}. Slide a {native}-frame window instead; "
                "longer windows measurably reduce accuracy."
            )

        stream = None
        if features is not None:
            stream = _as_array(features)
            if stream.ndim == 2:
                stream = stream[None]

        logits = self._forward(self._normalise(window), stream)
        probabilities = _sigmoid(np.squeeze(logits, axis=-1))
        return probabilities if batched else probabilities[0]

    def score_long(
        self,
        crops: np.ndarray | torch.Tensor,
        features: np.ndarray | torch.Tensor | None = None,
        stride: int = 1,
    ) -> np.ndarray:
        """Score a recording longer than the model's native window.

        The 1.x sequence models were trained on 15-frame clips and lose accuracy
        on longer input, so a long recording is scored by sliding that window
        across it and **averaging** the overlapping predictions per frame.

        **Averaging, not taking the maximum.** A maximum can only push a frame's
        score up, so a single badly-positioned window -- one where the blink sits
        at the very edge, an input distribution the model never trained on --
        raises that frame permanently and errors accumulate instead of
        cancelling. Measured over 200 rn30 windows carrying an annotated blink:

        ==========  =========  ========  =========  ======
        reduction   precision  recall    F1         stride
        ==========  =========  ========  =========  ======
        maximum     0.236      0.954     0.379      1
        **mean**    **0.346**  0.899     **0.500**  1
        mean        0.322      0.876     0.471      15
        ==========  =========  ========  =========  ======

        The maximum's near-perfect recall against 0.24 precision is the
        signature of over-firing: it fires almost everywhere, so it never misses.
        Dense striding also beats non-overlapping windows, so ``stride=1`` is the
        default despite costing more forward passes.

        A frame-wise model has no native window and is scored in one pass.

        Args:
            crops (np.ndarray | torch.Tensor): ``(T, 3, H, W)`` in ``[0, 1]``.
            features (np.ndarray | torch.Tensor | None): ``(T, 160)`` descriptors,
                required only by ``blinklinmult-union``.
            stride (int): Frames between successive windows.

        Returns:
            np.ndarray: ``(T,)`` closure probability per frame.

        Raises:
            DetectorError: If ``stride`` is not positive.
        """
        if stride < 1:
            raise DetectorError(f"stride must be at least 1, got {stride}.")

        window = _as_window(crops)
        total = int(window.shape[1])
        native = self.spec.window

        if native is None or native >= total:
            return self.score(crops, features)

        stream = None if features is None else np.asarray(features)
        summed = np.zeros(total, dtype=np.float64)
        counts = np.zeros(total, dtype=np.float64)

        for start in range(0, total - native + 1, stride):
            stop = start + native
            piece = self.score(
                window[0, start:stop],
                None if stream is None else stream[start:stop],
            )
            summed[start:stop] += piece
            counts[start:stop] += 1.0

        # A tail shorter than one window when stride does not divide the
        # remainder: score the last full window so no frame is left uncovered.
        if counts.min() == 0:
            piece = self.score(
                window[0, total - native : total],
                None if stream is None else stream[total - native : total],
            )
            summed[total - native :] += piece
            counts[total - native :] += 1.0

        return (summed / counts).astype(np.float32)

    def detect(
        self,
        crops: np.ndarray | torch.Tensor,
        features: np.ndarray | torch.Tensor | None = None,
    ) -> list[Interval]:
        """Find blink intervals in a single window.

        Scores every frame, then merges runs with
        :func:`~blinklinmult.train.events.to_intervals` at this model's fitted
        operating point. Where a low threshold is registered the extraction is
        *hysteretic*: a run must peak above the high threshold to count, but
        extends while above the low one, which recovers the shallow onset and
        offset a single cut clips away.

        Args:
            crops (np.ndarray | torch.Tensor): One window; see :meth:`score`.
            features (np.ndarray | torch.Tensor | None): Descriptors, if needed.

        Returns:
            list[Interval]: Inclusive ``(start, end)`` frame indices.

        Raises:
            DetectorError: If more than one window is passed, since intervals are
                indices into a single timeline.
        """
        signal = self.score(crops, features)
        if signal.ndim != 1:
            raise DetectorError(
                f"detect() scores one window at a time; got a batch of {signal.shape[0]}. "
                "Call it per window, or use score() for the batched signal."
            )

        low = None if self.spec.low_ratio is None else self.spec.threshold * self.spec.low_ratio
        mask = np.ones(signal.shape[0], dtype=bool)
        return to_intervals(signal, mask, self.spec.threshold, low)
