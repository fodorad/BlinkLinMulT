"""Corpus-specific preprocessing: raw data to a uniform processed tree.

One module per corpus, each producing ``data/processed/<name>/`` containing
per-sample ``.npy`` arrays and a ``manifest.jsonl`` that
:mod:`blinklinmult.data.builder` turns into the HDF5 file training reads. All
the corpus-specific knowledge — where the images are, how the labels are
encoded, what counts as one recording — is confined to these modules; everything
downstream sees the same format.

* :mod:`~blinklinmult.preprocess.annotation` — the shared ``.tag`` parser.
* :mod:`~blinklinmult.preprocess.windows` — cutting fixed-length windows.
* :mod:`~blinklinmult.preprocess.common` — cropping, splits, manifest writing.

Video corpora (``.tag`` annotation over continuous recordings):

* :mod:`~blinklinmult.preprocess.talkingface`
* :mod:`~blinklinmult.preprocess.rn`

Still-image corpora (one labelled image per sample):

* :mod:`~blinklinmult.preprocess.cew`
* :mod:`~blinklinmult.preprocess.mrl`

Pre-cut clip corpus:

* :mod:`~blinklinmult.preprocess.hust_lebw`

These are standalone research CLIs, not library API: they are excluded from the
published wheel and from the API docs, and they need the optional
``preprocess`` extra.
"""
