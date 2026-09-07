"""BlinkLinMulT: transformer-based eye blink detection.

Version 2 is a full rewrite of the published 1.x code. It keeps the two tasks
of the paper — **eye state recognition** (is this eye closed?) and **blink
presence detection** (does this window contain a blink?) — and replaces
everything around them:

* :mod:`blinklinmult.preprocess` turns each raw corpus into a uniform
  ``data/processed/<db>`` tree, one dataset at a time.
* :mod:`blinklinmult.data` collects those trees into per-dataset HDF5 files in
  the layout `OmniLoader <https://github.com/fodorad/OmniLoader>`_ reads, and
  declares each dataset's OmniLoader schema.
* :mod:`blinklinmult.train` trains the current
  `LinMulT <https://github.com/fodorad/LinMulT>`_ over those datasets jointly,
  driven entirely by YAML.

The 1.x public API (``DenseNet121``, ``BlinkLinT``, ``BlinkLinMulT`` importable
from ``blinklinmult.models``, with weights auto-downloaded on construction) is
gone; see ``docs/migration.md``.
"""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("blinklinmult")
except PackageNotFoundError:  # pragma: no cover - only when running uninstalled
    __version__ = "2.0.1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
"""Repository root. Every path in a config is resolved relative to this."""

WEIGHTS_DIR = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "blinklinmult"
"""Where ``from_pretrained`` caches downloaded weights.

Passed as ``huggingface_hub.hf_hub_download``'s ``cache_dir``, so asking for the
same model twice reuses the local copy instead of re-downloading. Under the
user's home rather than the repository, so an installed package works the same
way as a checkout.
"""

__all__ = ["PROJECT_ROOT", "WEIGHTS_DIR", "BlinkDetector", "__version__"]


def __getattr__(name: str) -> object:
    """Expose :class:`~blinklinmult.detector.BlinkDetector` without importing torch.

    Resolved on first access rather than at import: ``blinklinmult`` is imported
    by the preprocessing CLIs and the config layer, and pulling ``torch`` in for
    every one of them would make a cheap import expensive.

    Args:
        name (str): Attribute being looked up.

    Returns:
        object: The requested attribute.

    Raises:
        AttributeError: If the package has no such attribute.
    """
    if name == "BlinkDetector":
        from blinklinmult.detector import BlinkDetector

        return BlinkDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
