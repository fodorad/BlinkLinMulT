"""Training layer: configs, model, losses, metrics, and the Lightning loop.

* :mod:`~blinklinmult.train.config` — typed YAML configuration for a run.
* :mod:`~blinklinmult.train.model` — the eye-crop CNN plus LinMulT architecture.
* :mod:`~blinklinmult.train.losses` — masked binary losses for partially
  annotated supervision.
* :mod:`~blinklinmult.train.metrics` — masked detection metrics.
* :mod:`~blinklinmult.train.module` — the Lightning module.
* :mod:`~blinklinmult.train.cli` — the entry point ``make train-*`` calls.
"""
