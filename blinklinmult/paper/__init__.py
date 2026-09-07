"""The 1.x paper models, frozen as ONNX graphs.

The three networks published with *BlinkLinMulT: Transformer-Based Eye Blink
Detection* (`PMC10607707 <https://pmc.ncbi.nlm.nih.gov/articles/PMC10607707/>`_):

``densenet121-union``
    Frame-wise eye state. One crop in, one closure logit out -- the 1.x
    counterpart to v2's ``BlinkCNN``.
``blinklint-union``
    The same embeddings read as a sequence by a linear-attention transformer, so
    each frame is scored in the context of its neighbours.
``blinklinmult-union``
    Adds a 160-d handcrafted iris/eyelid stream cross-attended with the crops.
    The paper's headline model, and the only one needing that second input.

**These ship as graphs, not as PyTorch modules, and that is deliberate.** They are
finished artefacts: their numbers are published, their training code is not, and
nothing here is meant to be fine-tuned. Freezing the graph makes that structural
rather than a matter of documentation -- and it means running them needs
``onnxruntime`` alone, with **no torch and no linmult**, so they cannot be broken
by a future dependency upgrade.

Reach them through :class:`~blinklinmult.detector.BlinkDetector`, which pairs each
with the normalisation it expects and turns scores into blink intervals::

    from blinklinmult import BlinkDetector

    model = BlinkDetector.from_pretrained("blinklint-union")
    model.detect(crops)

Two properties of these models to know:

* They expect **ImageNet-normalised** crops, where ``BlinkCNN`` expects plain
  ``/255``. :data:`~blinklinmult.registry.MODELS` carries the constants so a
  caller never has to choose; passing the wrong ones costs accuracy silently.
* ``blinklinmult-union`` produces **two** outputs, a pooled clip logit and the
  per-frame sequence, because 1.x built it with ``aggregation='meanpooling'``.

The graphs were produced by ``scripts/export_paper_onnx.py``, which verifies each
against the original PyTorch model at several window lengths before writing it.
"""

from __future__ import annotations

import logging

from blinklinmult.paper.session import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    OnnxModel,
    SessionError,
    load_onnx,
)

logger = logging.getLogger(__name__)
"""Module-level logger."""

RELEASE_URL = "https://github.com/fodorad/LinMulT/releases/download/v1.0.0"
"""Where the original 1.x PyTorch weights were published.

Kept for provenance: each exported graph's sidecar records which file it came
from and that file's SHA-256, so a published graph can be traced back to the
weights the paper reports on.
"""

GRAPH_FILES: dict[str, str] = {
    "densenet121-union": "densenet121-union.onnx",
    "blinklint-union": "blinklint-union.onnx",
    "blinklinmult-union": "blinklinmult-union.onnx",
}
"""Model id to its ONNX graph filename in the published model repository.

``-union`` marks the checkpoint trained on the union of the 1.x corpora, which is
the variant the paper reports and the only one released.
"""

FEATURE_DIM = 160
"""Width of ``blinklinmult-union``'s second input stream.

71 eye-region landmarks (142) + 5 iris landmarks (10) + 2 iris diameters + 2
eyelid-pupil distances + EAR (1) + head pose (3).
"""

__all__ = [
    "FEATURE_DIM",
    "GRAPH_FILES",
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "RELEASE_URL",
    "OnnxModel",
    "SessionError",
    "load_onnx",
]
