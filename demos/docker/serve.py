"""A REST server for the four published blink models.

Runs the same code the library and the Gradio demo run; only the transport
differs. Two shapes of request, matching the split the library already makes:

* ``/invocations`` and ``/score`` take **eye crops** and return the per-frame
  eye-state curve. Cheap: no face detection, no torch for the 1.x models.
* ``/detect`` takes a **video** and runs the seven-stage pipeline. It needs the
  extraction stack, so it is only available in the image built with the
  ``preprocess`` extra -- asking for it without one returns a 501 naming the
  reason rather than a stack trace.

**Errors are deliberately generic.** ffmpeg, onnxruntime and exordium all embed
the failing file's path in their messages, and that path is this server's own
``NamedTemporaryFile`` location. Returning it verbatim to an unauthenticated
caller leaks the container's filesystem layout for no benefit to them; the real
exception is logged server-side instead. This mirrors
``PersonalityLinMulT/demos/docker/serve.py``, for the same reason.

**Reproducibility is a feature of the response, not a promise.** Every scoring
reply carries the model id, the SHA-256 of the weights that produced it, and the
extraction rule applied. A blink count without its threshold is not a
measurement anyone can repeat -- and the shipped default was fitted on RN
validation data, so on other footage it is a transfer.

Run locally::

    make serve
"""

from __future__ import annotations

import atexit
import hashlib
import logging
import os
import tempfile
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

logger = logging.getLogger(__name__)
"""Module-level logger. The only place a real exception is ever written."""

DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "blinkcnn")
"""Model loaded at startup. Any registered id; the others load on demand."""

MAX_SCORE_FRAMES = int(os.environ.get("MAX_SCORE_FRAMES", 1000))
"""Most frames one ``/score`` request may carry.

The upload endpoint is capped by :data:`MAX_UPLOAD_BYTES`, but a JSON body is
parsed and materialised before any of this module's code runs, so the cap has to
be applied to the decoded list. A ``T=5000`` request is ~1.2 GB of JSON and
~234 MB as float32 -- enough to exhaust the 2 GiB container this image targets.

1000 frames is 40 seconds at 25 fps, well past what a single request should
carry: longer recordings belong on ``/detect``, which streams from a file.
"""

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 200 * 1024 * 1024))
"""Largest accepted upload, 200 MB by default.

Enforced **while reading the stream**, not from ``Content-Length``: a client that
lies about its size would otherwise exhaust the container's memory, and this
service is meant to be reachable without authentication.
"""

GENERIC_FAILURE = "Could not process the request. See the server log for details."
"""What a caller is told when anything fails. See the module docstring."""

_models: dict[str, Any] = {}
"""Loaded detectors, kept for the process's life.

Each holds a native onnxruntime session. **They are released at exit** by
:func:`_release_models`: Python tears down extension modules while the sessions
are still referenced, and freeing one afterwards can abort the process with
``libc++abi: recursive_mutex lock failed``. Observed roughly once in ten runs
when this module's tests ran alongside the detector and streaming suites.
"""


@atexit.register
def _release_models() -> None:
    """Drop cached detectors before the interpreter unloads native modules."""
    _models.clear()


def _has_pipeline() -> bool:
    """Whether this image can process video.

    Returns:
        bool: ``True`` when the extraction stack is installed.
    """
    from importlib.util import find_spec

    return find_spec("exordium") is not None and find_spec("cv2") is not None


def _load(model_id: str) -> Any:  # noqa: ANN401 - BlinkDetector, imported lazily
    """Load a model by id, caching it.

    Args:
        model_id (str): A registered model id.

    Returns:
        BlinkDetector: Ready to score.

    Raises:
        HTTPException: 400 if the id is unknown, 503 if its weights are absent.
    """
    from blinklinmult import BlinkDetector
    from blinklinmult.detector import DetectorError
    from blinklinmult.registry import MODELS

    if model_id not in MODELS:
        raise HTTPException(400, f"Unknown model {model_id!r}; expected one of {sorted(MODELS)}.")
    if model_id not in _models:
        try:
            _models[model_id] = BlinkDetector.from_pretrained(model_id)
        except DetectorError:
            logger.exception("Loading %s failed", model_id)
            raise HTTPException(503, f"Weights for {model_id!r} are not available.") from None
    return _models[model_id]


def _weights_digest(model_id: str) -> str | None:
    """SHA-256 of the weights file backing a model.

    Returned with every prediction so a result traces to the exact artifact that
    produced it -- the difference between a reproducible container and one that
    merely looks deterministic.

    Args:
        model_id (str): A registered model id.

    Returns:
        str | None: Hex digest, or ``None`` if the file is not on disk.
    """
    from blinklinmult import WEIGHTS_DIR
    from blinklinmult.registry import spec

    candidates = list(Path(WEIGHTS_DIR).rglob(spec(model_id).filename))
    if not candidates:
        return None
    engine = hashlib.sha256()
    with candidates[0].open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            engine.update(block)
    return engine.hexdigest()


def _rule(threshold: float | None, low: float | None, model_id: str) -> Any:  # noqa: ANN401
    """Build the extraction rule a request asked for.

    Args:
        threshold (float | None): High threshold; the model's own if omitted.
        low (float | None): Low threshold for hysteresis.
        model_id (str): Used to fall back to the registered operating point.

    Returns:
        Extraction: The rule to apply.

    Raises:
        HTTPException: 400 if the thresholds cannot separate anything.
    """
    from blinklinmult.pipeline import Extraction, PipelineError
    from blinklinmult.registry import spec

    if threshold is None:
        return Extraction.fitted(spec(model_id))
    rule = Extraction(high=float(threshold), low=None if low is None else float(low))
    try:
        rule.validate()
    except PipelineError as error:
        raise HTTPException(400, str(error)) from None
    return rule


async def _read_upload(file: UploadFile) -> bytes:
    """Read an upload, refusing anything over the cap.

    Args:
        file (UploadFile): The incoming file.

    Returns:
        bytes: Its contents.

    Raises:
        HTTPException: 413 once the cap is passed.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1 << 20):
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        chunks.append(chunk)
    if not chunks:
        raise HTTPException(400, "Empty upload.")
    return b"".join(chunks)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Load the default model before the first request.

    A cold ONNX session on the first real request looks like a hang to whoever
    sent it, and to a health checker like a failure.

    Args:
        _app (FastAPI): Unused.

    Yields:
        None: While the app serves.
    """
    with suppress(HTTPException):
        _load(DEFAULT_MODEL_ID)
        logger.info("Warmed up %s", DEFAULT_MODEL_ID)
    yield


app = FastAPI(title="BlinkLinMulT", lifespan=_lifespan)
"""The ASGI application."""


class ScoreRequest(BaseModel):
    """Eye crops to score.

    Args:
        crops (list): ``(T, 3, H, W)`` float values in ``[0, 1]``, channel-first.
            **Not** pre-normalised: each model's own constants are applied here,
            and applying them twice costs accuracy silently.
        model_id (str): Which model to use.
        threshold (float | None): High threshold; the model's own if omitted.
        low_threshold (float | None): Low threshold, for hysteresis.
    """

    crops: list = Field(..., description="(T, 3, H, W) in [0, 1]")
    model_id: str = Field(default=DEFAULT_MODEL_ID)
    threshold: float | None = None
    low_threshold: float | None = None


@app.get("/ping")
def ping() -> dict[str, str]:
    """Liveness probe.

    Returns:
        dict[str, str]: ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@app.get("/manifest")
def manifest() -> dict[str, Any]:
    """What this image actually contains.

    Every model with the digest of the weights backing it, so a published result
    can be tied to an artifact rather than to a version number.

    Returns:
        dict[str, Any]: Per-model metadata and whether video is supported.
    """
    from blinklinmult.registry import MODELS

    return {
        "video_supported": _has_pipeline(),
        "default_model": DEFAULT_MODEL_ID,
        "models": {
            model_id: {
                "generation": entry.generation,
                "window": entry.window,
                "needs_features": entry.needs_features,
                "threshold": entry.threshold,
                "low_ratio": entry.low_ratio,
                "weights_sha256": _weights_digest(model_id),
            }
            for model_id, entry in MODELS.items()
        },
    }


@app.post("/score")
def score(request: ScoreRequest) -> dict[str, Any]:
    """Score eye crops.

    Args:
        request (ScoreRequest): Crops and the model to use.

    Returns:
        dict[str, Any]: The per-frame signal, the events, and the rule and
        weights digest that produced them.

    Raises:
        HTTPException: 400 on unusable input.
    """
    import numpy as np

    detector = _load(request.model_id)
    rule = _rule(request.threshold, request.low_threshold, request.model_id)

    # Checked before `np.asarray`, which would otherwise copy the whole body
    # into a second buffer before anyone could object to its size.
    if len(request.crops) > MAX_SCORE_FRAMES:
        raise HTTPException(
            413,
            f"{len(request.crops)} frames exceeds the {MAX_SCORE_FRAMES}-frame limit. "
            "Post a longer recording to /detect instead.",
        )

    try:
        crops = np.asarray(request.crops, dtype=np.float32)
        signal = detector.score_long(crops)
    except Exception:
        logger.exception("Scoring failed")
        raise HTTPException(400, GENERIC_FAILURE) from None

    from blinklinmult.train.events import to_intervals

    mask = np.isfinite(signal)
    events = to_intervals(np.nan_to_num(signal), mask, rule.high, rule.low)
    return {
        "model_id": request.model_id,
        "weights_sha256": _weights_digest(request.model_id),
        "extraction": {"high": rule.high, "low": rule.low},
        "signal": [round(float(value), 5) for value in signal],
        "events": [list(interval) for interval in events],
    }


@app.post("/invocations")
def invocations(request: ScoreRequest) -> dict[str, Any]:
    """SageMaker's expected entry point.

    An alias for :func:`score`, kept because SageMaker requires this exact path.

    Args:
        request (ScoreRequest): Crops and the model to use.

    Returns:
        dict[str, Any]: As :func:`score`.
    """
    return score(request)


@app.post("/detect")
async def detect(
    file: UploadFile,
    model_id: str = Form(default=DEFAULT_MODEL_ID),
    start: float = Form(default=0.0),
    duration: float = Form(default=10.0),
    threshold: float | None = Form(default=None),
    low_threshold: float | None = Form(default=None),
) -> dict[str, Any]:
    """Detect blinks in an uploaded video.

    Runs the seven-stage pipeline: face detection and tracking, head pose,
    landmarks, eye selection, inference, labelling. Both eyes are scored
    separately, and one turned away from the camera is suppressed rather than
    guessed at -- its signal is ``null``, not zero.

    Args:
        file (UploadFile): The video.
        model_id (str): Which model to use.
        start (float): Segment start, in seconds.
        duration (float): Segment length, in seconds.
        threshold (float | None): High threshold; the model's own if omitted.
        low_threshold (float | None): Low threshold, for hysteresis.

    Returns:
        dict[str, Any]: Per-eye signals and events, with the rule applied.

    Raises:
        HTTPException: 501 without the extraction stack, 413 for a large upload,
            400 for anything unprocessable.
    """
    if not _has_pipeline():
        raise HTTPException(
            501,
            "This image scores crops only. Video needs the extraction stack: "
            "use the image built with the `preprocess` extra.",
        )

    payload = await _read_upload(file)
    detector = _load(model_id)
    rule = _rule(threshold, low_threshold, model_id)

    from blinklinmult.data.schema import LEFT, RIGHT
    from blinklinmult.pipeline import NoFaceError, PipelineError, Stage, run

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix) as handle:
        handle.write(payload)
        handle.flush()
        try:
            result = None
            stages = []
            for item in run(handle.name, detector, extraction=rule, start=start, duration=duration):
                if isinstance(item, Stage):
                    stages.append(item.line())
                else:
                    result = item
        except NoFaceError as error:
            raise HTTPException(422, str(error)) from None
        except PipelineError:
            logger.exception("Pipeline failed")
            raise HTTPException(400, GENERIC_FAILURE) from None
        except Exception:
            logger.exception("Unexpected failure")
            raise HTTPException(400, GENERIC_FAILURE) from None

    if result is None:  # pragma: no cover - run always ends with a Result
        raise HTTPException(400, GENERIC_FAILURE)

    import numpy as np

    return {
        "model_id": model_id,
        "weights_sha256": _weights_digest(model_id),
        "extraction": {"high": rule.high, "low": rule.low},
        "fps": result.fps,
        "frames": len(result.frames),
        "first_frame": result.frames[0].index if result.frames else 0,
        "truncated": result.truncated,
        "stages": stages,
        "eyes": {
            side: {
                "signal": [
                    None if not np.isfinite(value) else round(float(value), 5)
                    for value in result.signal[side]
                ],
                "events": [list(interval) for interval in result.events[side]],
            }
            for side in (LEFT, RIGHT)
        },
    }


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """A human-readable summary of what this image serves.

    Returns:
        str: HTML listing the models and endpoints.
    """
    from blinklinmult.registry import MODELS

    rows = "".join(
        f"<tr><td><code>{model_id}</code></td><td>{entry.generation}</td>"
        f"<td>{entry.window or 'frame-wise'}</td><td>{entry.description}</td></tr>"
        for model_id, entry in MODELS.items()
    )
    video = "available" if _has_pipeline() else "not in this image (crops only)"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>BlinkLinMulT</title>
<style>body{{font-family:system-ui;margin:2rem;max-width:60rem}}
table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:.4rem .6rem}}</style>
</head><body>
<h1>BlinkLinMulT</h1>
<p>Eye-state recognition and blink detection. Default model:
<code>{DEFAULT_MODEL_ID}</code>. Video endpoints: {video}.</p>
<table><tr><th>id</th><th>generation</th><th>window</th><th>what it is</th></tr>
{rows}</table>
<h2>Endpoints</h2>
<ul>
<li><code>GET /ping</code> — liveness</li>
<li><code>GET /manifest</code> — models and weight digests</li>
<li><code>POST /score</code> — eye crops in, eye-state curve out</li>
<li><code>POST /invocations</code> — the same, under SageMaker's path</li>
<li><code>POST /detect</code> — video in, blink events out</li>
</ul>
<p>Every response carries the weights digest and the extraction rule applied, so
a result can be reproduced rather than merely repeated.</p>
</body></html>"""
