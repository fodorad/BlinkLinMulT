"""Running a frozen ONNX graph, with numpy and ``onnxruntime`` alone.

No torch here, and none reachable from here. That is the whole point of shipping
the 1.x models as graphs: their inference path cannot be broken by a PyTorch
upgrade, a ``linmult`` release, or anything else this repository does later.

The class is deliberately thin. It holds a session, remembers what the graph's
inputs are called, and returns numpy arrays --
:class:`~blinklinmult.detector.BlinkDetector` supplies the normalisation and the
event extraction on top, exactly as it does for the PyTorch-backed v2 model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

logger = logging.getLogger(__name__)
"""Module-level logger."""

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
"""Per-channel mean the 1.x models were trained against.

Torchvision's standard ImageNet normalisation, applied after a bicubic resize to
64x64. The v2 ``BlinkCNN`` uses plain ``/255`` instead; the two are **not**
interchangeable, and applying the wrong one degrades accuracy without erroring.
"""

IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
"""Per-channel standard deviation. See :data:`IMAGENET_MEAN`."""

CROPS_INPUT = "crops"
"""Name of the eye-crop input in every exported graph."""

FEATURES_INPUT = "features"
"""Name of the descriptor input, present only in the two-stream graph."""

SEQUENCE_OUTPUT = "sequence"
"""Name of the per-frame output in every exported graph."""

CLIP_OUTPUT = "clip"
"""Name of the pooled output, present only in the two-stream graph."""


class SessionError(Exception):
    """Raised when a graph cannot be loaded or is fed the wrong inputs."""


class OnnxModel:
    """A frozen 1.x model, backed by an ``onnxruntime`` session.

    Args:
        session (ort.InferenceSession): The loaded graph.
        metadata (dict[str, Any]): Its sidecar, if one was found beside it.
    """

    def __init__(self, session: ort.InferenceSession, metadata: dict[str, Any] | None = None):
        self.session = session
        self.metadata = metadata or {}
        self._inputs = {item.name for item in session.get_inputs()}
        self._outputs = [item.name for item in session.get_outputs()]

    @property
    def needs_features(self) -> bool:
        """Whether this graph takes the 160-d descriptor stream."""
        return FEATURES_INPUT in self._inputs

    def __call__(
        self, crops: np.ndarray, features: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        """Run the graph.

        Args:
            crops (np.ndarray): ``(B, T, 3, H, W)`` float32, already normalised.
            features (np.ndarray | None): ``(B, T, 160)`` float32, for the
                two-stream graph.

        Returns:
            dict[str, np.ndarray]: Output name to array. Always carries
                ``"sequence"`` ``(B, T, 1)``; the two-stream graph also carries
                ``"clip"`` ``(B, 1)``.

        Raises:
            SessionError: If a required input is missing.
        """
        feed: dict[str, np.ndarray] = {CROPS_INPUT: np.ascontiguousarray(crops, dtype=np.float32)}
        if self.needs_features:
            if features is None:
                raise SessionError(
                    f"This graph needs the {FEATURES_INPUT!r} input; pass features=..."
                )
            feed[FEATURES_INPUT] = np.ascontiguousarray(features, dtype=np.float32)

        # onnxruntime types `run` as returning sparse tensors and dicts too;
        # these graphs only ever emit dense arrays, so the results are narrowed
        # rather than propagating an untrue union to every caller.
        results = [np.asarray(value) for value in self.session.run(None, feed)]
        return dict(zip(self._outputs, results, strict=True))


def load_onnx(path: str | Path, providers: list[str] | None = None) -> OnnxModel:
    """Open an exported graph.

    Args:
        path (str | Path): The ``.onnx`` file. A ``.onnx.json`` sidecar beside it
            is read if present.
        providers (list[str] | None): Execution providers, in preference order.
            Defaults to ``onnxruntime``'s own choice.

    Returns:
        OnnxModel: Ready to run.

    Raises:
        SessionError: If the file is missing, or ``onnxruntime`` is not installed.
    """
    try:
        import onnxruntime as ort
    except ImportError as error:  # pragma: no cover - depends on the install
        raise SessionError(
            "onnxruntime is required to run the 1.x models: `pip install blinklinmult[onnx]`."
        ) from error

    location = Path(path)
    if not location.is_file():
        raise SessionError(f"No ONNX graph at {location}.")

    options = ort.SessionOptions()
    # Left to the caller's environment rather than pinned: oversubscribing a
    # shared machine is a real cost, and onnxruntime's default is sensible for a
    # dedicated one.
    session = ort.InferenceSession(str(location), sess_options=options, providers=providers)

    sidecar = location.with_suffix(".onnx.json")
    metadata = None
    if sidecar.is_file():
        import json

        try:
            metadata = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            logger.warning(f"Could not read {sidecar}; continuing without its metadata.")

    logger.info(f"Loaded ONNX graph {location.name}.")
    return OnnxModel(session, metadata)
