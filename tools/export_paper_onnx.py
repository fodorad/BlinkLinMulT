"""How the published 1.x ONNX graphs were produced (already run; kept for the record).

This script needed two things that are **no longer in the repository**: the 1.x
PyTorch model definitions (``blinklinmult/paper/_models.py``) and the vendored
``linmult`` 1.1.0 layers (``blinklinmult/paper/_linmult/``). Both were deleted
once the graphs existed, which was the point of exporting them -- the published
models now run on ``onnxruntime`` alone, with no torch, no linmult, and nothing
to fine-tune.

It is kept so the provenance of the shipped graphs is auditable rather than
folklore: this is exactly what produced them, including the parity gate. To run
it again, recover those two paths from git history
(``git show <rev>:blinklinmult/paper/_models.py``) into a scratch checkout with
``pip install linmult==1.1.0``.

That is the point. The 1.x networks are frozen artefacts: their numbers are
published and their training code is not being shared. Freezing the *graph*
matches that, and removes 1,029 lines of vendored third-party code from the
repository along with the version clash that made it necessary.

Run it from a checkout that still has ``blinklinmult/paper/_models.py``::

    uv run --extra export python tools/export_paper_onnx.py --out artifacts/onnx

Each model produces ``<id>.onnx`` and a ``<id>.onnx.json`` sidecar recording the
input contract, the normalisation it expects, and the source weights' SHA-256.

**Dynamic time.** The 1.x forward pass looped over ``T`` in Python, which would
unroll into ``T`` copies of DenseNet121 with the sequence length baked in.
:mod:`blinklinmult.paper._models` already replaced that with a fold/unfold
reshape, so ``T`` is exported as a genuine dynamic axis -- and the parity check
below verifies it at several lengths rather than trusting the flag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch
from blinklinmult.paper._models import FEATURE_DIM, IMAGENET_MEAN, IMAGENET_STD

from blinklinmult.paper import MODEL_CLASSES, WEIGHT_FILES, load_pretrained

logger = logging.getLogger(__name__)
"""Module-level logger."""

IMAGE_SIZE = 64
"""Crop side the 1.x models were trained on."""

OPSET = 17
"""ONNX opset. 17 covers everything these graphs use and is widely supported."""

PARITY_LENGTHS = (1, 15, 30)
"""Window lengths the exported graph is checked at.

More than one, deliberately: a graph with ``T`` unrolled still passes at the
length it was traced with, and only fails at a different one.
"""

RTOL = 1e-3
"""Relative tolerance for torch-vs-onnxruntime agreement."""

ATOL = 1e-5
"""Absolute tolerance for torch-vs-onnxruntime agreement."""


class ExportError(Exception):
    """Raised when a model cannot be exported or fails its parity check."""


def _digest(path: Path) -> str:
    """SHA-256 of a file, for provenance in the sidecar.

    Args:
        path (Path): File to hash.

    Returns:
        str: Hex digest.
    """
    engine = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            engine.update(block)
    return engine.hexdigest()


def _int_shapes_for_tracing() -> object:
    """Force ``FullMask`` to see a Python int, as it does in eager mode.

    The vendored attention builds its causal mask with ``FullMask(L, ...)`` where
    ``L = x.shape[1]``. In eager mode that is an ``int`` and ``FullMask`` takes
    its integer branch. Under ``torch.onnx.export`` the tracer makes ``x.shape[1]``
    a *tensor*, so ``FullMask`` takes its tensor branch instead and rejects it for
    not being bool.

    The fix belongs here rather than in the vendored file, which is third-party
    code kept byte-for-byte so the 1.x checkpoints keep loading. Wrapping the
    constructor to coerce that first argument reproduces eager behaviour exactly
    -- the mask is all-ones either way -- and the parity check downstream is what
    proves the graph still matches PyTorch.

    Returns:
        object: A context manager restoring the original constructor on exit.
    """
    import contextlib

    from blinklinmult.paper._linmult import masking

    original = masking.FullMask.__init__

    def patched(self, mask=None, N=None, M=None, device="cpu"):  # noqa: ANN001, ANN202
        if isinstance(mask, torch.Tensor) and mask.dtype != torch.bool and mask.numel() == 1:
            mask = int(mask.item())
        original(self, mask, N, M, device)

    @contextlib.contextmanager
    def _scope():  # noqa: ANN202
        masking.FullMask.__init__ = patched
        try:
            yield
        finally:
            masking.FullMask.__init__ = original

    return _scope()


class _SingleStream(torch.nn.Module):
    """Export wrapper for the models taking crops alone.

    ``torch.onnx.export`` traces a ``forward``; wrapping keeps the exported
    signature to exactly one input and one output, whatever the underlying
    model's Python signature looks like.

    Args:
        model (torch.nn.Module): The loaded 1.x model.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """Score a window.

        Args:
            crops (torch.Tensor): ``(B, T, 3, H, W)``.

        Returns:
            torch.Tensor: ``(B, T, 1)`` logits.
        """
        return self.model(crops)


class _TwoStream(torch.nn.Module):
    """Export wrapper for ``BlinkLinMulT``.

    Keeps **both** heads: 1.x built the model with ``aggregation='meanpooling'``,
    so it returns a pooled clip logit alongside the per-frame sequence. Dropping
    either would silently narrow what the published model can answer.

    Args:
        model (torch.nn.Module): The loaded 1.x model.
    """

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(
        self, crops: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score a window from crops and descriptors.

        Args:
            crops (torch.Tensor): ``(B, T, 3, H, W)``.
            features (torch.Tensor): ``(B, T, 160)``.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: ``(clip, sequence)`` logits.
        """
        clip, sequence = self.model(crops, features)
        return clip, sequence


def _sample(model_id: str, time: int) -> tuple[torch.Tensor, ...]:
    """Build a random input of the right shape for one model.

    Args:
        model_id (str): Which model.
        time (int): Window length.

    Returns:
        tuple[torch.Tensor, ...]: Positional inputs for its forward pass.
    """
    crops = torch.randn(1, time, 3, IMAGE_SIZE, IMAGE_SIZE)
    if model_id == "blinklinmult-union":
        return crops, torch.randn(1, time, FEATURE_DIM)
    return (crops,)


def export(model_id: str, out_dir: Path) -> Path:
    """Export one 1.x model and verify the graph against PyTorch.

    Args:
        model_id (str): A key of :data:`~blinklinmult.paper.MODEL_CLASSES`.
        out_dir (Path): Directory to write the graph and sidecar into.

    Returns:
        Path: The written ``.onnx`` file.

    Raises:
        ExportError: If export fails, or the graph disagrees with PyTorch at any
            of :data:`PARITY_LENGTHS`.
    """
    import onnxruntime as ort

    torch_model = load_pretrained(model_id)
    two_stream = model_id == "blinklinmult-union"
    wrapper = (_TwoStream if two_stream else _SingleStream)(torch_model).eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{model_id}.onnx"

    inputs = ["crops", "features"] if two_stream else ["crops"]
    outputs = ["clip", "sequence"] if two_stream else ["sequence"]
    dynamic: dict[str, dict[int, str]] = {name: {0: "batch", 1: "time"} for name in inputs}
    dynamic["sequence"] = {0: "batch", 1: "time"}
    if two_stream:
        # The clip head is mean-pooled over time, so it has no time axis.
        dynamic["clip"] = {0: "batch"}

    sample = _sample(model_id, PARITY_LENGTHS[1])
    try:
        with _int_shapes_for_tracing():
            torch.onnx.export(
                wrapper,
                sample,
                str(destination),
                input_names=inputs,
                output_names=outputs,
                dynamic_axes=dynamic,
                opset_version=OPSET,
                do_constant_folding=True,
                dynamo=False,
            )
    except Exception as error:  # noqa: BLE001 - torch raises many types here
        raise ExportError(f"{model_id}: export failed: {error}") from error

    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    for time in PARITY_LENGTHS:
        probe = _sample(model_id, time)
        with torch.no_grad():
            expected = wrapper(*probe)
        expected_list = list(expected) if isinstance(expected, tuple) else [expected]

        feed = {name: tensor.numpy() for name, tensor in zip(inputs, probe, strict=True)}
        actual = session.run(None, feed)

        for name, want, got in zip(outputs, expected_list, actual, strict=True):
            try:
                np.testing.assert_allclose(want.numpy(), got, rtol=RTOL, atol=ATOL)
            except AssertionError as error:
                raise ExportError(
                    f"{model_id}: ONNX output {name!r} disagrees with PyTorch at T={time}. "
                    "If this only fails at some lengths, the time axis did not stay dynamic."
                ) from error
        logger.info(f"  T={time}: parity OK")

    from blinklinmult import WEIGHTS_DIR

    source = WEIGHTS_DIR / WEIGHT_FILES[model_id]
    sidecar = {
        "model_id": model_id,
        "generation": "paper",
        "opset": OPSET,
        "inputs": inputs,
        "outputs": outputs,
        "image_size": IMAGE_SIZE,
        "feature_dim": FEATURE_DIM if two_stream else None,
        # The 1.x models were trained on ImageNet-standardised crops; BlinkCNN
        # takes plain /255. Recording it here is what stops a caller applying the
        # wrong one, which costs accuracy and raises nothing.
        "normalisation": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "source_weights": WEIGHT_FILES[model_id],
        "source_sha256": _digest(source) if source.is_file() else None,
        "parity": {"lengths": list(PARITY_LENGTHS), "rtol": RTOL, "atol": ATOL},
    }
    (out_dir / f"{model_id}.onnx.json").write_text(json.dumps(sidecar, indent=2) + "\n")

    size = destination.stat().st_size / 1048576
    logger.info(f"{model_id}: wrote {destination} ({size:.1f} MB)")
    return destination


def main() -> None:
    """Export every 1.x model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("artifacts/onnx"), help="Output directory."
    )
    parser.add_argument("--model", action="append", default=None, help="Model id; repeatable.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    for model_id in args.model or list(MODEL_CLASSES):
        logger.info(f"exporting {model_id}")
        export(model_id, args.out)


if __name__ == "__main__":
    main()
