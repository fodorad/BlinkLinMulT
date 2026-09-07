"""Training-data layer: the per-corpus HDF5 artifacts and their loaders.

This package owns everything between the corpus-specific scripts in
:mod:`blinklinmult.preprocess` and the training loop in
:mod:`blinklinmult.train`:

* :mod:`~blinklinmult.data.schema` — the typed per-corpus contract.
* :mod:`~blinklinmult.data.builder` — the config-driven HDF5 writer.
* :mod:`~blinklinmult.data.omni` — the bridge to OmniLoader's schema.
* :mod:`~blinklinmult.data.collate` — the batch-to-model boundary.
* :mod:`~blinklinmult.data.datamodule` — the Lightning ``DataModule`` that mixes
  every corpus into one stream.

Each corpus's HDF5 file is a reproducibility boundary: training reads those
files and nothing else. Each is published to a public Hugging Face dataset repo
(``make push-<name>``) so a run reproduces without re-running extraction, and
each is rebuilt from source by ``make preprocess-<name>``.
"""

from blinklinmult.data.schema import (
    BLINK_PRESENCE,
    DATASETS,
    EYE_FEATURE,
    EYE_IMAGE,
    EYE_STATE,
    SUBSETS,
    DatasetSpec,
    SchemaError,
)

__all__ = [
    "BLINK_PRESENCE",
    "DATASETS",
    "EYE_FEATURE",
    "EYE_IMAGE",
    "EYE_STATE",
    "SUBSETS",
    "DatasetSpec",
    "SchemaError",
]
