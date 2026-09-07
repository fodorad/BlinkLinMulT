r"""Freeze the v2 frame-wise model into an ONNX graph.

The checkpoint and the graph both ship. The checkpoint is the *trainable*
artifact -- ``blinkcnn`` is the one model this repo still fine-tunes -- and the
graph is the *deployable* one. Keeping both is what makes the speed claim
measurable rather than asserted: ``blinkcnn`` and ``blinkcnn-onnx`` are the same
weights behind the same interface, so a benchmark across them isolates the
runtime.

Measured on a 15-frame window, 4 threads, warm: **194 ms in PyTorch against
41.5 ms through onnxruntime, a 4.7x speedup** at 13.07 -> 2.77 ms per frame. That
gap is not architectural. ConvNeXt-Femto is a *smaller* network than the
DenseNet121 behind ``densenet121-union``, which was already a frozen graph and
already ran at ~2.8 ms/frame; ``blinkcnn`` was slower only because it ran eager.

**The parity gate is the deliverable, not the export.** A graph that runs is
worth nothing; a graph proven to agree with its checkpoint is a drop-in
replacement. Three probe distributions at three window lengths, plus an
end-to-end check through :class:`~blinklinmult.detector.BlinkDetector`:

* **Three lengths** (:data:`PARITY_LENGTHS`) because a graph with ``T``
  accidentally unrolled still passes at the length it was traced with.
* **Three distributions** (:data:`PARITY_PROBES`) because Gaussian noise is not
  what this model ever sees. Crops arrive as uniform ``[0, 1]``; a graph checked
  only outside its input domain is checked on inputs it will never meet. The
  constant probe additionally pins that normalisation survived the trace -- a
  graph that dropped it returns a *different* answer on a flat image.
* **End-to-end**, because graph-level parity cannot see a wrong ``mean``/``std``
  in the registry entry. That is the likeliest way this breaks, and the only
  check that catches it is one that goes through the detector.

Run::

    uv run --extra export python tools/export_blinkcnn_onnx.py \\
        --checkpoint artifacts/onnx/blinkcnn.pt --out artifacts/onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import warnings
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)
"""Module-level logger."""

MODEL_ID = "blinkcnn-onnx"
"""Registry id of the exported graph."""

SOURCE_ID = "blinkcnn"
"""Registry id of the checkpoint it is exported from."""

IMAGE_SIZE = 64
"""Crop side the model was trained on."""

OPSET = 17
"""ONNX opset, matching the 1.x graphs so both run under one runtime floor."""

PARITY_LENGTHS = (1, 15, 30)
"""Window lengths the exported graph is checked at.

More than one, deliberately: a graph with ``T`` unrolled still passes at the
length it was traced with, and only fails at a different one.
"""

PARITY_PROBES = ("uniform", "gaussian", "constant")
"""Input distributions the graph is checked against.

``uniform`` is the real domain -- crops arrive as ``[0, 1]``. ``gaussian``
matches what the 1.x export script uses, kept so both gates are comparable.
``constant`` is a flat mid-grey image: cheap, and a graph that lost its internal
normalisation answers a flat image differently from one that kept it.
"""

RTOL = 1e-3
"""Relative tolerance for torch-vs-onnxruntime agreement."""

ATOL = 1e-5
"""Absolute tolerance for torch-vs-onnxruntime agreement."""

MIN_CONTRAST_SPREAD = 0.5
"""Smallest black-vs-white logit gap a correctly normalised graph shows.

Measured: the shipped graph spreads **1.280**, and a build with the ImageNet
standardisation removed spreads **0.117** -- an 11x collapse, because the raw
``[0, 1]`` range is ~4.4x narrower than the standardised one and the backbone's
response narrows with it. This floor sits between the two, far enough from both
that ordinary weight drift will not trip it.

**Parity alone cannot catch this.** Parity compares the graph against the
checkpoint it was traced from, so a checkpoint that lost its normalisation
exports to a graph that faithfully reproduces the loss -- both sides agree, and
every probe passes. Verified by sabotage: removing the transform from
``EyeEncoder.forward`` and re-exporting left all nine parity probes green. This
threshold is the check that fails.
"""


class ExportError(Exception):
    """Raised when the model cannot be exported or fails its parity check."""


class _CnnStream(torch.nn.Module):
    """Pins the traced signature to one tensor in, one tensor out.

    :meth:`~blinklinmult.train.model.BlinkModel.forward` takes a mask and returns
    a dict, neither of which ONNX wants. The mask is safe to drop: for the
    ``cnn`` family the forward short-circuits before reading it, which is
    verified by :func:`_mask_is_ignored` rather than assumed.

    The output is named ``sequence`` to match the 1.x graphs, so
    :meth:`~blinklinmult.detector.BlinkDetector._forward` reads both through the
    same key.
    """

    def __init__(self, model: torch.nn.Module) -> None:
        """Wrap a loaded model.

        Args:
            model (torch.nn.Module): The rebuilt checkpoint.
        """
        super().__init__()
        self.model = model

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """Score a window of crops.

        Args:
            crops (torch.Tensor): ``(B, T, 3, H, W)`` in ``[0, 1]``.

        Returns:
            torch.Tensor: ``(B, T, 1)`` logits.
        """
        mask = torch.ones(crops.shape[:2], dtype=torch.bool, device=crops.device)
        return self.model(crops, mask)["eye_state"]


def _digest(path: Path) -> str:
    """SHA-256 of a file, so the graph traces back to its checkpoint.

    Args:
        path (Path): The file to hash.

    Returns:
        str: Hex digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _probe(kind: str, length: int, generator: torch.Generator) -> torch.Tensor:
    """Build one parity probe.

    Args:
        kind (str): One of :data:`PARITY_PROBES`.
        length (int): Frames in the window.
        generator (torch.Generator): Seeded, so a failure is reproducible.

    Returns:
        torch.Tensor: ``(1, length, 3, 64, 64)``.

    Raises:
        ExportError: If the probe kind is unknown.
    """
    shape = (1, length, 3, IMAGE_SIZE, IMAGE_SIZE)
    if kind == "uniform":
        return torch.rand(shape, generator=generator)
    if kind == "gaussian":
        return torch.randn(shape, generator=generator)
    if kind == "constant":
        return torch.full(shape, 0.5)
    raise ExportError(f"Unknown parity probe {kind!r}; expected one of {list(PARITY_PROBES)}.")


def _mask_is_ignored(model: torch.nn.Module) -> bool:
    """Check the mask really is unused before dropping it from the graph.

    The ``cnn`` family short-circuits before reading the mask, which is why
    :class:`_CnnStream` can synthesise one. That is a property of the current
    architecture, not a guarantee, so it is measured here: if a future family
    starts reading the mask, the export must stop rather than silently freeze
    an all-ones assumption into the graph.

    Args:
        model (torch.nn.Module): The rebuilt checkpoint.

    Returns:
        bool: Whether an all-zeros mask gives the same answer as an all-ones one.
    """
    crops = torch.rand(1, 4, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        ones = model(crops, torch.ones(1, 4, dtype=torch.bool))["eye_state"]
        zeros = model(crops, torch.zeros(1, 4, dtype=torch.bool))["eye_state"]
    return bool(torch.allclose(ones, zeros))


def _check_parity(wrapper: torch.nn.Module, destination: Path) -> dict[str, float]:
    """Assert the graph agrees with the checkpoint it came from.

    Args:
        wrapper (torch.nn.Module): The traced module, in eval mode.
        destination (Path): The written ``.onnx``.

    Returns:
        dict[str, float]: Worst absolute deviation per probe kind.

    Raises:
        ExportError: If any probe at any length exceeds tolerance.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    generator = torch.Generator().manual_seed(0)
    worst: dict[str, float] = {}

    for kind in PARITY_PROBES:
        for length in PARITY_LENGTHS:
            crops = _probe(kind, length, generator)
            with torch.no_grad():
                expected = wrapper(crops).numpy()
            actual = session.run(None, {"crops": crops.numpy()})[0]
            deviation = float(np.abs(expected - actual).max())
            worst[kind] = max(worst.get(kind, 0.0), deviation)
            try:
                np.testing.assert_allclose(expected, actual, rtol=RTOL, atol=ATOL)
            except AssertionError as error:
                raise ExportError(
                    f"parity failed for probe {kind!r} at T={length}: {error}"
                ) from error
            logger.info(f"  parity ok: {kind:9s} T={length:<3d} max|diff|={deviation:.2e}")

    _check_normalisation_survived(session)
    return worst


def _check_normalisation_survived(session: object) -> None:
    """Assert the graph still standardises its input.

    ``EyeEncoder`` normalises with ImageNet statistics from buffers registered
    ``persistent=False``, so they live outside the state dict and exist only
    because ``forward`` builds them. They should be constant-folded into the
    graph -- but "should" is the word that precedes a silent accuracy loss, so
    it is checked.

    A graph that kept the transform maps a black image and a white image to
    clearly different logits. One that dropped it maps them closer together,
    because the raw ``[0, 1]`` range is ~4.4x narrower than the standardised one.

    This is the **only** check here that catches a checkpoint which itself lost
    the transform, since parity would compare that checkpoint against a graph
    that faithfully reproduces the same loss. See :data:`MIN_CONTRAST_SPREAD`.

    Args:
        session (object): A live ``onnxruntime.InferenceSession``.

    Raises:
        ExportError: If the two constant images land closer together than
            :data:`MIN_CONTRAST_SPREAD`, which means the graph is not
            standardising its input.
    """
    shape = (1, 2, 3, IMAGE_SIZE, IMAGE_SIZE)
    dark = np.zeros(shape, dtype=np.float32)
    bright = np.ones(shape, dtype=np.float32)
    run = session.run  # type: ignore[attr-defined]
    spread = float(np.abs(run(None, {"crops": dark})[0] - run(None, {"crops": bright})[0]).max())
    if spread < MIN_CONTRAST_SPREAD:
        raise ExportError(
            f"black-vs-white logit spread is {spread:.3f}, below the "
            f"{MIN_CONTRAST_SPREAD} floor, so this graph is not standardising its "
            "input. Check that EyeEncoder.forward still applies pixel_mean/pixel_std "
            "before the backbone -- note that parity cannot catch this, because a "
            "checkpoint missing the transform exports to a graph that matches it."
        )
    logger.info(f"  normalisation present: black-vs-white logit spread {spread:.3f}")


def _check_detector_parity(destination: Path, checkpoint: Path) -> float:
    """Assert the two registry entries agree end to end.

    Graph parity proves the trace is faithful. It cannot prove the *registry*
    is: ``blinkcnn-onnx`` declares its own normalisation, and declaring ImageNet
    there would standardise twice -- once outside the graph and once within it.
    The model would still run and still look plausible. Only a check that goes
    through :class:`~blinklinmult.detector.BlinkDetector` sees that.

    Args:
        destination (Path): The written ``.onnx``.
        checkpoint (Path): The ``.pt`` it came from.

    Returns:
        float: Worst absolute deviation between the two detectors.

    Raises:
        ExportError: If the two disagree beyond tolerance.
    """
    from blinklinmult import BlinkDetector

    crops = np.random.default_rng(0).random((15, 3, IMAGE_SIZE, IMAGE_SIZE)).astype("float32")
    torch_scores = BlinkDetector.from_pretrained(SOURCE_ID, weights=checkpoint).score(crops)
    onnx_scores = BlinkDetector.from_pretrained(MODEL_ID, weights=destination).score(crops)
    deviation = float(np.abs(np.asarray(torch_scores) - np.asarray(onnx_scores)).max())
    if deviation > ATOL + RTOL:
        raise ExportError(
            f"the two detectors disagree by {deviation:.2e}. The graph matches its "
            f"checkpoint, so suspect the {MODEL_ID!r} registry entry -- most likely "
            "its normalisation, which must stay UNIT because the graph standardises "
            "internally."
        )
    logger.info(f"  detector parity ok: max|diff|={deviation:.2e}")
    return deviation


def export(checkpoint: Path, out_dir: Path) -> Path:
    """Export the checkpoint to ONNX and gate it on parity.

    Args:
        checkpoint (Path): The stripped ``.pt`` from ``export_blinkcnn.py``.
        out_dir (Path): Directory for the graph and its sidecar.

    Returns:
        Path: The written ``.onnx``.

    Raises:
        ExportError: If the checkpoint will not load, the mask turns out to be
            load-bearing, or any parity check fails.
    """
    from blinklinmult.models import ModelLoadError, load_checkpoint

    try:
        model = load_checkpoint(checkpoint)
    except ModelLoadError as error:
        raise ExportError(str(error)) from error

    if not _mask_is_ignored(model):
        raise ExportError(
            "this model reads eye_image_mask, so the exported graph cannot "
            "synthesise an all-ones mask without changing its answer. The wrapper "
            "needs a mask input."
        )

    wrapper = _CnnStream(model).eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{MODEL_ID.replace('-onnx', '')}.onnx"

    with warnings.catch_warnings():
        # `check_eye_images` validates rank and shape in Python, so the tracer
        # folds it away and warns that it did. That is correct and expected --
        # the guard is a developer aid, not part of the computation -- but the
        # warning reads like a defect, so it is suppressed here and stated in
        # the sidecar instead.
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            wrapper,
            (torch.rand(1, PARITY_LENGTHS[1], 3, IMAGE_SIZE, IMAGE_SIZE),),
            str(destination),
            input_names=["crops"],
            output_names=["sequence"],
            dynamic_axes={"crops": {0: "batch", 1: "time"}, "sequence": {0: "batch", 1: "time"}},
            opset_version=OPSET,
            do_constant_folding=True,
            dynamo=False,
        )

    worst = _check_parity(wrapper, destination)
    detector_deviation = _check_detector_parity(destination, checkpoint)

    sidecar = {
        "model_id": MODEL_ID,
        "generation": "v2",
        "runtime": "onnx",
        "opset": OPSET,
        "inputs": ["crops"],
        "outputs": ["sequence"],
        "image_size": IMAGE_SIZE,
        "feature_dim": None,
        "normalisation": {"mean": [0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0]},
        "normalisation_note": (
            "ImageNet standardisation lives INSIDE the graph (EyeEncoder's "
            "pixel_mean/pixel_std buffers). Callers pass raw [0, 1] crops. "
            "Applying ImageNet externally as well normalises twice: the model "
            "still runs and still returns plausible probabilities, silently "
            "degraded."
        ),
        "operating_point": {"threshold": 0.53, "low_ratio": 0.25},
        "source_checkpoint": checkpoint.name,
        "source_sha256": _digest(checkpoint),
        "parity": {
            "lengths": list(PARITY_LENGTHS),
            "probes": list(PARITY_PROBES),
            "rtol": RTOL,
            "atol": ATOL,
            "max_deviation": worst,
            "detector_max_deviation": detector_deviation,
        },
        "traced_guard_dropped": (
            "collate.check_eye_images validates rank and shape in Python; the "
            "tracer folds it away, so the graph does not re-check its input."
        ),
    }
    (out_dir / f"{destination.name}.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    size = destination.stat().st_size / 1048576
    logger.info(f"{MODEL_ID}: wrote {destination} ({size:.1f} MB)")
    return destination


def main() -> None:
    """Export the graph from a stripped checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/onnx/blinkcnn.pt"),
        help="Stripped inference checkpoint to export.",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/onnx"), help="Output directory."
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger.info(f"exporting {SOURCE_ID} -> {MODEL_ID}")
    export(args.checkpoint, args.out)


if __name__ == "__main__":
    main()
