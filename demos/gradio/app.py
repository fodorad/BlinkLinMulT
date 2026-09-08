"""Gradio demo: blink detection on an uploaded video.

Runs locally and deploys unchanged to a Hugging Face Space::

    uv run --extra demo --extra preprocess --extra onnx python demos/gradio/app.py

Pick one of the four published models, upload a clip, and the seven-stage
pipeline in :mod:`blinklinmult.pipeline` reports each stage as it finishes. The
outputs are a two-panel plot -- left eye above right -- and the source video with
the face box, eye boxes, landmarks and a per-frame readout drawn on.

**Both eyes are scored separately**, and an eye turned away from the camera is
suppressed rather than guessed at: its box turns red, its readout reads ``--``,
and its curve breaks. A gap means "not looked at", which is a different claim
from "open".
"""

from __future__ import annotations

# `spaces` must be imported before torch touches CUDA, and is absent locally.
# The shim lets one file serve both without a branch at every call site.
try:  # pragma: no cover - exercised only on a Space
    import spaces
except ImportError:

    class spaces:  # type: ignore[no-redef]  # noqa: N801 - mimics the real module
        """Stand-in for the Spaces runtime when running locally."""

        @staticmethod
        def GPU(*args: object, **kwargs: object):  # noqa: ANN205, N802, ARG004 - mirrors the real API
            """No-op replacement for the ZeroGPU decorator.

            Args:
                *args (object): A function when used bare, else decorator args.
                **kwargs (object): Ignored.

            Returns:
                object: The function, or a pass-through decorator.
            """
            if args and callable(args[0]):
                return args[0]

            def decorator(function):  # noqa: ANN001, ANN202
                return function

            return decorator


import json
import logging
import tempfile
import time
from pathlib import Path

import gradio as gr

from blinklinmult import assets, pipeline
from blinklinmult.pipeline import MAX_SECONDS

logger = logging.getLogger(__name__)
"""Module-level logger."""

MODEL_LABELS = {
    "BlinkCNN": "blinkcnn-onnx",
    "BlinkDenseNet121": "densenet121-union",
    "BlinkLinT": "blinklint-union",
    "BlinkLinMulT": "blinklinmult-union",
}
"""Display name to registry id, current model first.

The dropdown shows the names the models are published under; the registry needs
the ids, which carry deployment detail (``-union``, ``-onnx``) that means nothing
to a reader. Mapping rather than prettifying ids with string surgery, because
``blinkcnn-onnx`` is simply called BlinkCNN -- the suffix is how it is stored.

**Every id here is an ONNX graph.** The PyTorch checkpoint path rebuilds the
module through ``blinklinmult.train.model``, which reaches linmult, omniloader
and the rest of the training stack; on a Space that surfaced as three separate
ModuleNotFoundError builds. The graphs need onnxruntime and nothing else.
"""

MODEL_IDS = tuple(MODEL_LABELS.values())
"""The registry ids behind :data:`MODEL_LABELS`, in the same order."""

EXAMPLE_VIDEO = assets.EXAMPLE_VIDEO
"""Built-in example, shipped inside the package.

TalkingFace's first 10 seconds, carrying three annotated blinks (frames 168, 225
and 274). Bundled rather than read from ``data/raw`` so the demo works from a
plain ``pip install`` -- on a Space or in a container there is no corpus to read.
"""

EXAMPLE_TAG = assets.EXAMPLE_TAG
"""The example's annotation, plotted in green beside the prediction.

Only the corpus clips ship one; an uploaded video is scored without a reference.
"""

STATS_FILENAME = "rn30_feature_stats.json"
"""Corpus feature statistics, needed only by ``blinklinmult-union``.

Its 160-d descriptor stream was trained standardised; fed raw values the model
never fires. See :meth:`~blinklinmult.detector.BlinkDetector.prepare_features`.
"""

STATS_PATH = Path(__file__).resolve().parents[2] / "artifacts/onnx" / STATS_FILENAME
"""Where a repository checkout keeps the statistics.

``parents[2]`` because this file lives at ``demos/gradio/app.py``. A Space has
no repository above it, so this path will not exist there and
:func:`_feature_stats` falls back to the Hub.
"""


def _feature_stats() -> dict:
    """Load the descriptor statistics, from the checkout or the Hub.

    A repository checkout has them under ``artifacts/onnx``; a Space has only
    the three files that were uploaded to it, so they come from the same public
    model repo the weights do. Six kilobytes, cached after the first call.

    Returns:
        dict: ``{"mean": [...], "std": [...]}``.

    Raises:
        gr.Error: If neither source has them, naming both places looked.
    """
    if STATS_PATH.is_file():
        return json.loads(STATS_PATH.read_text())

    from blinklinmult.registry import HF_MODEL_REPO

    try:
        from huggingface_hub import hf_hub_download

        downloaded = hf_hub_download(repo_id=HF_MODEL_REPO, filename=STATS_FILENAME)
    except Exception as error:  # noqa: BLE001 - network, auth, or a missing file
        raise gr.Error(
            f"{STATS_FILENAME} is needed for this model but was not found at "
            f"{STATS_PATH} and could not be downloaded from {HF_MODEL_REPO} ({error})."
        ) from error
    return json.loads(Path(downloaded).read_text())


SOURCE_CUSTOM = "Custom video..."
"""Dropdown label for a user-supplied clip. The default."""

SOURCE_EXAMPLE = "Example: TalkingFace video"
"""Dropdown label for the built-in clip."""

METHOD_FITTED = "Fitted hysteresis"
"""Use the model's registered operating point. The default."""

METHOD_THRESHOLD = "Threshold"
"""One cut, supplied by the user."""

METHOD_CUSTOM = "Custom range"
"""A user-supplied hysteresis pair."""

METHODS = (METHOD_FITTED, METHOD_THRESHOLD, METHOD_CUSTOM)
"""Ways of turning the eye-state curve into blink events."""

POSE_LABEL_GEOMETRIC = "Geometric (fast)"
"""Read the angles from the face detector's keypoints. The default."""

POSE_LABEL_6DREPNET = "6DRepNet (precise)"
"""Run the dedicated pose network: a forward pass per frame."""

POSE_LABELS = {
    POSE_LABEL_GEOMETRIC: pipeline.POSE_GEOMETRIC,
    POSE_LABEL_6DREPNET: pipeline.POSE_6DREPNET,
}
"""Display name to the ``head_pose`` argument :func:`~blinklinmult.pipeline.run` takes."""


def _forces_precise(model_id: str) -> bool:
    """Whether this model overrides the pose choice.

    A model consuming the 160-d iris descriptor can only run the precise route,
    and that descriptor encodes 6DRepNet's angles -- so the pose dropdown has no
    effect on it. Read from the registry rather than listed here, so adding such
    a model to :data:`MODEL_LABELS` does not silently leave the control enabled.

    Args:
        model_id (str): A registry id.

    Returns:
        bool: True if the pipeline will override the requested pose route.
    """
    from blinklinmult.registry import spec

    return bool(spec(model_id).needs_features)


_models: dict[str, object] = {}
"""Loaded detectors, keyed by model id. Loading is slow; switching should not be."""


GPU_SECONDS = 180
"""ZeroGPU allocation window for one analysis.

Seven stages over 300 frames, several models each, measured at roughly 5.8x
realtime on an M4 -- so ~30 s for a 10 s clip locally, and slower on a shared
A10G queue. 180 s leaves headroom without hoarding an allocation.
"""


@spaces.GPU(duration=GPU_SECONDS)
def _analyse_video(
    video_path: str,
    model_id: str,
    stats: dict | None,
    tag: Path | None,
    rule,  # noqa: ANN001 - Extraction, imported lazily
    start: float,
    duration: float,
    head_pose: str = pipeline.POSE_GEOMETRIC,
) -> tuple[list[str], object]:
    """Run the whole pipeline inside one GPU allocation.

    **This is the function ZeroGPU schedules**, so every model that wants the
    GPU must run inside it. The decorator cannot go on :func:`analyse`, which is
    a generator: ZeroGPU releases the allocation when the wrapped call returns,
    and a generator returns at its first ``yield`` -- the work after that would
    fall back to CPU with nothing to say so.

    The cost is that stage lines cannot stream while this runs; they are
    collected and returned together. Progress granularity for a GPU that is
    actually used is the right trade.

    Args:
        video_path (str): The clip to analyse.
        model_id (str): Which model to run.
        stats (dict | None): Corpus feature statistics, for the two-stream model.
        tag (Path | None): Annotation to plot alongside, if the clip has one.
        rule (Extraction | None): How to turn eye state into events.
        start (float): Segment start, in seconds.
        duration (float): Segment length, in seconds.
        head_pose (str): Which pose route to run, from
            :data:`~blinklinmult.pipeline.HEAD_POSES`. The pipeline overrides it
            for a model that consumes the iris descriptor.

    Returns:
        tuple[list[str], object]: The stage lines, and the pipeline ``Result``.
    """
    from blinklinmult.pipeline import Stage, run

    detector = _load(model_id)
    lines: list[str] = []
    result = None
    for item in run(
        video_path,
        detector,
        stats,
        tag_path=tag,
        extraction=rule,
        start=start,
        duration=duration,
        head_pose=head_pose,
    ):
        if isinstance(item, Stage):
            lines.append(item.line())
            logger.info(item.line())
        else:
            result = item
    return lines, result


def _load(model_id: str):  # noqa: ANN202 - BlinkDetector, imported lazily
    """Load a model, caching it.

    Args:
        model_id (str): One of :data:`MODEL_IDS`.

    Returns:
        BlinkDetector: Ready to score.
    """
    if model_id not in _models:
        from blinklinmult import BlinkDetector

        _models[model_id] = BlinkDetector.from_pretrained(model_id)
    return _models[model_id]


def analyse(  # noqa: PLR0913 - one argument per UI control
    model_label: str,
    video_path: str | None,
    source: str = SOURCE_CUSTOM,
    method: str = METHOD_FITTED,
    threshold: float = 0.5,
    low: float = 0.13,
    high: float = 0.53,
    start: float = 0.0,
    duration: float = 10.0,
    pose_label: str = POSE_LABEL_GEOMETRIC,
):
    """Run the pipeline, yielding the log as each stage completes.

    Args:
        model_label (str): A key of :data:`MODEL_LABELS` -- the display name the
            dropdown shows, not the registry id.
        video_path (str | None): The uploaded clip.
        source (str): Which dropdown entry produced it. The annotation is keyed
            off this rather than the path, because Gradio hands the handler a
            **cached copy** of the file -- comparing that path to the example's
            never matches, which is how the ground truth silently went missing.
        method (str): How to turn the eye-state curve into events.
        threshold (float): Single cut, when ``method`` asks for one.
        low (float): Low threshold, for a custom range.
        high (float): High threshold, for a custom range.
        start (float): Where in the video to begin, in seconds.
        duration (float): How much of it to analyse, in seconds.
        pose_label (str): A key of :data:`POSE_LABELS`. Ignored for a model that
            consumes the iris descriptor -- see :func:`_forces_precise`.

    Yields:
        tuple: ``(plot, annotated video, log)``. The heavy outputs stay ``None``
        until the final yield so the log streams while they are being made.

    Raises:
        gr.Error: If the input is unusable, or no face is found.
    """
    import matplotlib.pyplot as plt

    from blinklinmult import overlay
    from blinklinmult.pipeline import NoFaceError, PipelineError

    if model_label not in MODEL_LABELS:
        raise gr.Error(f"Unknown model {model_label!r}; expected one of {list(MODEL_LABELS)}.")
    model_id = MODEL_LABELS[model_label]
    head_pose = POSE_LABELS.get(pose_label, pipeline.POSE_GEOMETRIC)

    _validate(model_id, video_path, method, threshold, low, high, start, duration)

    lines: list[str] = []

    def log(message: str) -> str:
        lines.append(message)
        logger.info(message)
        return "\n".join(lines)

    # The label, not the id: the id's `-onnx`/`-union` suffix is storage detail.
    # Model and head pose get a line each: they are two independent choices, and
    # putting the route in parentheses after the model read as if it were part
    # of the model's name.
    yield None, None, log(f"Model: {model_label}")
    # A user who chose geometric and got 6DRepNet deserves to see why, on the
    # line that reports what actually ran.
    if _forces_precise(model_id):
        yield (
            None,
            None,
            log(f"Head pose: {POSE_LABEL_6DREPNET} — this model takes these angles as input"),
        )
    else:
        yield None, None, log(f"Head pose: {pose_label}")
    yield None, None, log("Loading model (cached after first use)...")

    detector = _load(model_id)
    stats = None
    if detector.spec.needs_features:  # type: ignore[attr-defined]
        stats = _feature_stats()

    # The window belongs to the model; the extraction rule does not. Printing
    # `spec.threshold` beside it claimed the model's registered value was in
    # use even when the user had chosen another -- the two are reported
    # separately now, and the rule line says which one actually applies.
    window = detector.spec.window  # type: ignore[attr-defined]
    yield None, None, log(f"Window: {window or 'frame-wise'}")

    rule = _extraction(method, threshold, low, high)
    from blinklinmult.pipeline import Extraction

    shown = rule if rule is not None else Extraction.fitted(detector.spec)
    origin = "model default" if rule is None else "your setting"
    yield None, None, log(f"Event extraction: {shown.describe()}  ({origin})")

    began = time.perf_counter()
    result = None
    try:
        # The annotation belongs to the example clip alone; an uploaded file of
        # the same name is not the same video.
        tag = EXAMPLE_TAG if source == SOURCE_EXAMPLE and EXAMPLE_TAG.is_file() else None
        if tag is not None:
            yield None, None, log(f"Ground truth: {tag.name}")

        stage_lines, result = _analyse_video(
            video_path, model_id, stats, tag, rule, start, duration, head_pose
        )
        for line in stage_lines:
            yield None, None, log(line)
    except NoFaceError as error:
        raise gr.Error(str(error)) from error
    except PipelineError as error:
        raise gr.Error(str(error)) from error
    except Exception as error:  # noqa: BLE001 - anything else is still the user's problem
        raise gr.Error(f"Analysis failed: {error}") from error

    if result is None:  # pragma: no cover - run() always ends with a Result
        raise gr.Error("The pipeline produced no result.")

    if result.truncated:
        # The segment is whatever the user asked for, clipped to what the video
        # holds -- saying "the first 10 seconds" would be wrong for any start
        # other than zero, and wrong about the length whenever the clip ran out.
        analysed = len(result.frames) / result.fps
        yield (
            None,
            None,
            log(
                f"Note: the video ended early -- analysed {analysed:.1f} s "
                f"from {start:g} s, not the {duration:g} s requested."
            ),
        )

    yield None, None, log("[7/7] Rendering plot and video...")
    figure = overlay.plot(result)

    from blinklinmult.pipeline import _read_video

    frames, _, _, _ = _read_video(video_path, start, duration)
    destination = Path(tempfile.mkstemp(suffix=".mp4")[1])
    overlay.render(frames, result, destination)

    elapsed = time.perf_counter() - began
    seconds = len(result.frames) / result.fps
    summary = log(
        f"Done. {elapsed:.1f} s for {seconds:.1f} s of video ({elapsed / seconds:.1f}x realtime)."
    )
    yield figure, str(destination), summary
    plt.close(figure)


def _on_method_change(method: str):
    """Show the inputs the chosen extraction method needs.

    Args:
        method (str): One of :data:`METHODS`.

    Returns:
        tuple: Updates for the single-threshold, low and high boxes.
    """
    single = method == METHOD_THRESHOLD
    custom = method == METHOD_CUSTOM
    # `interactive` is repeated on every update: a component that arrives as an
    # event output can otherwise be rendered read-only, which looks like a
    # disabled box rather than one waiting for input.
    return (
        gr.update(visible=single, interactive=True),
        gr.update(visible=custom, interactive=True),
        gr.update(visible=custom, interactive=True),
    )


def _extraction(method: str, threshold: float, low: float, high: float):
    """Build the extraction rule the user asked for.

    Args:
        method (str): One of :data:`METHODS`.
        threshold (float): The single cut, when that method is chosen.
        low (float): Low threshold, for a custom range.
        high (float): High threshold, for a custom range.

    Returns:
        Extraction | None: The rule, or ``None`` to use the model's own.

    Raises:
        gr.Error: If the supplied numbers cannot separate anything.
    """
    from blinklinmult.pipeline import Extraction, PipelineError

    if method == METHOD_FITTED:
        return None
    rule = (
        Extraction(high=float(threshold))
        if method == METHOD_THRESHOLD
        else Extraction(high=float(high), low=float(low))
    )
    try:
        rule.validate()
    except PipelineError as error:
        raise gr.Error(str(error)) from error
    return rule


def _validate(
    model_id: str,
    video_path: str | None,
    method: str,
    threshold: float,
    low: float,
    high: float,
    start: float = 0.0,
    duration: float = 10.0,
) -> None:
    """Reject unusable input before any model is loaded.

    Every problem is collected and reported together: fixing one thing only to
    be told about the next is a worse experience than being told both at once,
    and the pipeline is far too slow to discover a bad threshold at stage 6.

    Args:
        model_id (str): The chosen model's registry id.
        video_path (str | None): The uploaded clip.
        method (str): The extraction method.
        threshold (float): Single cut, when that method is chosen.
        low (float): Low threshold, for a custom range.
        high (float): High threshold, for a custom range.
        start (float): Segment start, in seconds.
        duration (float): Segment length, in seconds.

    Raises:
        gr.Error: Listing everything that is missing or out of range.
    """
    from blinklinmult.pipeline import Extraction, PipelineError

    problems: list[str] = []

    if not video_path:
        problems.append("no video — upload one, or pick the TalkingFace example")
    elif not Path(video_path).is_file():
        problems.append("the uploaded video could not be found on disk")

    if model_id not in MODEL_IDS:
        problems.append(f"unknown model {model_id!r}")

    if method not in METHODS:
        problems.append(f"unknown detection method {method!r}")
    elif method != METHOD_FITTED:
        rule = (
            Extraction(high=float(threshold))
            if method == METHOD_THRESHOLD
            else Extraction(high=float(high), low=float(low))
        )
        try:
            rule.validate()
        except PipelineError as error:
            problems.append(str(error).rstrip("."))

    if start < 0:
        problems.append(f"start timestamp must be at or after 0 s (got {start:g})")
    if duration <= 0:
        problems.append(f"duration must be positive (got {duration:g})")

    # A start past the end fails 30 seconds later, inside the decoder, after the
    # model has already loaded. Probing the video's length here costs
    # milliseconds and turns that into an immediate, specific message. A segment
    # that merely *overruns* the end is fine -- it is silently truncated to what
    # exists, which is what a user asking for "10 seconds from 2 s" of an 8 s
    # clip means.
    if video_path and start >= 0:
        try:
            from exordium.video.core.io import get_video_metadata

            meta = get_video_metadata(video_path)
            length = float(meta.get("num_frames") or 0) / float(meta.get("fps") or 25.0)
        except Exception:  # noqa: BLE001 - an unreadable video is reported below
            length = 0.0
        if length and start >= length:
            problems.append(
                f"start timestamp {start:g} s is past the end of the video, "
                f"which is {length:.1f} s long"
            )

    if problems:
        raise gr.Error("Cannot run: " + "; ".join(problems) + ".")


def _reveal(
    model_label: str,
    video_path: str | None,
    method: str,
    threshold: float,
    low: float,
    high: float,
    start: float = 0.0,
    duration: float = 10.0,
):
    """Validate, then uncover the output panels.

    Runs before :func:`analyse` in the click chain. Validation lives here as
    well as there so a bad input raises *before* anything is revealed, leaving
    the page as it was rather than showing three empty panels beside an error.

    Args:
        model_label (str): The chosen model's display name, as the dropdown
            shows it -- resolved to a registry id before validating.
        video_path (str | None): The uploaded clip.
        method (str): The extraction method.
        threshold (float): Single cut, when that method is chosen.
        low (float): Low threshold, for a custom range.
        high (float): High threshold, for a custom range.
        start (float): Segment start, in seconds.
        duration (float): Segment length, in seconds.

    Returns:
        tuple: Updates making the plot, video and log visible.

    Raises:
        gr.Error: If anything is missing or out of range.
    """
    _validate(
        MODEL_LABELS.get(model_label, model_label),
        video_path,
        method,
        threshold,
        low,
        high,
        start,
        duration,
    )
    return gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)


def _on_model_change(model_label: str):
    """Enable or disable the pose dropdown for the chosen model.

    A model consuming the iris descriptor runs 6DRepNet whatever the dropdown
    says, so leaving the control live would let a user pick "Geometric (fast)"
    and receive neither speed nor an explanation.

    Args:
        model_label (str): The newly chosen display name.

    Returns:
        dict: A Gradio update for the pose dropdown.
    """
    model_id = MODEL_LABELS.get(model_label)
    if model_id is not None and _forces_precise(model_id):
        return gr.update(
            value=POSE_LABEL_6DREPNET,
            interactive=False,
            info="Fixed: this model takes 6DRepNet's angles as part of its input.",
        )
    return gr.update(
        interactive=True,
        info=(
            "Geometric reads the angles from the detector's keypoints -- no extra "
            "forward pass, and roughly 6x faster end to end."
        ),
    )


def _on_source_change(source: str):
    """Show or fill the upload box when the source changes.

    Args:
        source (str): The chosen source label.

    Returns:
        gr.update: The video component's new state -- interactive and empty for
        a custom clip, filled and read-only for the example.
    """
    if source == SOURCE_EXAMPLE and EXAMPLE_VIDEO.is_file():
        return gr.update(value=str(EXAMPLE_VIDEO), interactive=False)
    return gr.update(value=None, interactive=True)


def build_demo() -> gr.Blocks:
    """Construct the UI.

    Separate from :func:`main` so a test can build it without starting a server.

    Returns:
        gr.Blocks: The demo.
    """
    with gr.Blocks(title="Blink detection demo") as demo:
        gr.Markdown("# Blink detection demo")
        gr.Markdown(
            "Upload a clip or pick the built-in example, and the largest face is tracked "
            "through its first 10 seconds while both eyes are scored separately; with "
            "an eye rotated away from the camera (|yaw| > 45\u00b0) suppressed rather "
            "than guessed at, since a turned head hides it."
        )

        with gr.Row():
            with gr.Column(scale=1):
                model_in = gr.Dropdown(
                    choices=list(MODEL_LABELS),
                    value=next(iter(MODEL_LABELS)),
                    label="Model",
                    info=(
                        "BlinkCNN is the current model; the other three are from the 2023 paper. "
                        "All four run as ONNX graphs."
                    ),
                )
                pose_in = gr.Dropdown(
                    choices=list(POSE_LABELS),
                    value=POSE_LABEL_GEOMETRIC,
                    label="Head pose",
                    info=(
                        "Geometric reads the angles from the detector's keypoints -- no extra "
                        "forward pass, and roughly 6x faster end to end. Roll and pitch track "
                        "6DRepNet closely (r=0.94); yaw is an approximation, and yaw is what "
                        "the occlusion gate uses. BlinkLinMulT takes 6DRepNet's angles as model "
                        "input, so it always runs them."
                    ),
                )
                source_in = gr.Dropdown(
                    choices=[SOURCE_CUSTOM, SOURCE_EXAMPLE],
                    value=SOURCE_CUSTOM,
                    label="Video source",
                    info="The example is TalkingFace, which blinks three times in 10 seconds.",
                )
                start_in = gr.Number(
                    value=0.0,
                    label="Start timestamp (sec)",
                    info="Where in the video to begin.",
                    minimum=0.0,
                    step=0.5,
                    interactive=True,
                )
                duration_in = gr.Number(
                    value=10.0,
                    label="Duration (sec)",
                    info=f"How much to analyse, up to {int(MAX_SECONDS)} s.",
                    minimum=0.5,
                    maximum=MAX_SECONDS,
                    step=0.5,
                    interactive=True,
                )
                method_in = gr.Dropdown(
                    choices=list(METHODS),
                    value=METHOD_FITTED,
                    label="Blink event detection",
                    info=(
                        "Eye state is continuous; a blink is a decision on top of it. "
                        "The fitted point was tuned on RN validation data, so it is a "
                        "transfer on any other footage."
                    ),
                )
                threshold_in = gr.Number(
                    value=0.5,
                    label="Threshold",
                    info="A run peaking above this counts as a blink.",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    visible=False,
                    interactive=True,
                )
                low_in = gr.Number(
                    value=0.13,
                    label="Low threshold",
                    info="A run extends while above this.",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    visible=False,
                    interactive=True,
                )
                high_in = gr.Number(
                    value=0.53,
                    label="High threshold",
                    info="A run must peak above this to count at all.",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.01,
                    visible=False,
                    interactive=True,
                )
            with gr.Column(scale=1):
                video_in = gr.Video(label="Video", sources=["upload"], interactive=True)

        source_in.change(_on_source_change, [source_in], [video_in])
        method_in.change(_on_method_change, [method_in], [threshold_in, low_in, high_in])

        run_button = gr.Button("Run pipeline to detect blink events", variant="primary")

        with gr.Row():
            log_out = gr.Textbox(
                label="Logs",
                lines=16,
                interactive=False,
                placeholder="Each pipeline stage reports as it finishes.",
                visible=False,
            )
            video_out = gr.Video(label="Annotated video", visible=False)

        plot_out = gr.Plot(
            label="Eye State Recognition and Blink Presence Detection", visible=False
        )

        # Every control the pipeline reads must appear here. Passing a subset is
        # silent: the missing arguments fall back to their defaults, so a
        # user-chosen threshold is quietly replaced by the fitted one.
        # `_reveal` runs first and only uncovers the panels; `analyse` fills
        # them. Chained rather than merged so a validation failure leaves the
        # outputs hidden -- an empty plot beside an error is worse than no plot.
        run_button.click(
            _reveal,
            [model_in, video_in, method_in, threshold_in, low_in, high_in, start_in, duration_in],
            [plot_out, video_out, log_out],
        ).then(
            analyse,
            [
                model_in,
                video_in,
                source_in,
                method_in,
                threshold_in,
                low_in,
                high_in,
                start_in,
                duration_in,
                pose_in,
            ],
            [plot_out, video_out, log_out],
        )

        # A control that silently does nothing is worse than one that explains
        # itself: BlinkLinMulT's iris descriptor encodes 6DRepNet's angles, so
        # the pipeline overrides this choice. Say so, and grey it out.
        model_in.change(_on_model_change, [model_in], [pose_in])
    return demo


def main() -> None:
    """Launch the demo."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    build_demo().launch()


if __name__ == "__main__":
    main()
