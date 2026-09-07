"""Where published weights live, and what each model expects at its input.

One table, :data:`MODELS`, describing every shipped model: which file holds its
weights, how crops must be normalised for it, and the operating point fitted for
turning its scores into blink events.

The normalisation entry is the reason this table exists rather than a set of
constants scattered across the model classes. The 1.x networks were trained on
ImageNet-standardised crops and ``BlinkCNN`` on plain ``/255``; applying the
wrong convention costs accuracy **without raising anything**, so the value has to
travel with the model id rather than be assumed by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

HF_MODEL_REPO = "fodorad/blink_detection"
"""Public Hugging Face repository holding the published weights."""

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
"""ImageNet channel means -- the 1.x normalisation."""

IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
"""ImageNet channel standard deviations -- the 1.x normalisation."""

UNIT_MEAN: tuple[float, float, float] = (0.0, 0.0, 0.0)
"""v2's normalisation: crops are already ``/255``, so nothing is subtracted."""

UNIT_STD: tuple[float, float, float] = (1.0, 1.0, 1.0)
"""v2's normalisation. See :data:`UNIT_MEAN`."""


V1_FEATURE_ORDER: tuple[tuple[int, int], ...] = ((157, 160), (0, 157))
"""How to reorder a v2 feature vector into the 1.x layout.

``blinklinmult-union`` was trained on ``[head_pose(3), landmarks(142),
iris(10), diameters(2), eyelid(2), ear(1)]`` -- **head pose first**. The v2
schema puts head pose last. These slices, concatenated in order, convert one to
the other.

Measured on an rn30 window with an annotated blink: feeding the v2 order gives a
peak probability of **0.0002** (the model never fires), and the v1 order with the
same standardisation gives **0.91**, bracketing the annotated frames.
"""


@dataclass(frozen=True)
class ModelSpec:
    """Everything needed to run one published model.

    Args:
        model_id (str): Short selector, e.g. ``"blinkcnn"``.
        filename (str): Its weight file in :data:`HF_MODEL_REPO`.
        generation (str): Which body of work produced the weights -- ``"paper"``
            for the 1.x publication, ``"v2"`` for the current rewrite. Provenance
            only; it does not decide how the file is loaded. See :data:`runtime`.
        runtime (str): What loads and executes the file -- ``"onnx"`` for a
            frozen graph run through onnxruntime, ``"pytorch"`` for a checkpoint
            rebuilt into an :class:`~torch.nn.Module`.

            **Separate from** ``generation`` **on purpose.** The two partitions
            coincided while every 1.x model was ONNX and every v2 model was
            torch, and one field answered both questions. ``blinkcnn`` shipping
            in both runtimes ends that: it is ``generation="v2"`` either way,
            and only ``runtime`` says which loader to call.
        mean (tuple[float, float, float]): Per-channel mean to subtract.
        std (tuple[float, float, float]): Per-channel divisor.
        image_size (int): Crop side the model was trained on.
        window (int | None): Frames per window the model was trained on, or
            ``None`` for the frame-wise models, which score each frame alone and
            accept any length. The 1.x sequence models were trained on 15-frame
            (~0.5 s) windows and **degrade markedly** when given longer ones --
            see the note on :data:`MODELS`.
        needs_features (bool): Whether it also takes the 160-d descriptor stream.
        standardise_features (bool): Whether that stream must be z-scored with
            corpus-wide statistics before use. The 1.x two-stream model was
            trained on standardised descriptors, and feeding it raw ones leaves
            its logits saturated near -9.9 -- it silently never fires.
        threshold (float): Score above which a run counts as a blink. The 1.x
            models carry the plain 0.5 default rather than a fitted value -- see
            the note on :data:`MODELS`.
        low_ratio (float | None): Hysteresis low threshold as a fraction of
            ``threshold``; ``None`` for a single cut.
        description (str): One line for a UI or a model card.
    """

    model_id: str
    filename: str
    generation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    runtime: str = "onnx"
    image_size: int = 64
    window: int | None = None
    needs_features: bool = False
    standardise_features: bool = False
    threshold: float = 0.5
    low_ratio: float | None = None
    description: str = ""


# **The 1.x sequence models expect 15-frame windows.** They were trained on
# ~0.5 s at 25 fps, and scoring a longer window costs a lot of accuracy. Measured
# over 25 rn30 windows carrying an annotated blink, scoring the blink's location
# to within 3 frames:
#
#     =====================  ==========  ==========
#     model                  45 frames   15 frames
#     =====================  ==========  ==========
#     blinklint-union            72%        **92%**
#     blinklinmult-union         52%        **84%**
#     =====================  ==========  ==========
#
# This -- not the feature pipeline -- is what a low score on these models usually
# means. Slide a 15-frame window along a longer recording rather than feeding the
# whole thing at once.
#
# **The 1.x entries carry 0.5 as a placeholder, not a fitted value.** The 1.x
# work reported eye-state recognition; event extraction was never its focus, so
# the released weights came with no operating point and 0.5 is simply the neutral
# midpoint of a sigmoid. Nothing measured it.
#
# So 1.x and v2 *event* counts are not comparable: v2's pair was swept on
# validation, theirs was picked for want of anything better. Frame-level
# `score()` output is unaffected and remains directly comparable, and the demo
# lets a user supply their own thresholds rather than inheriting this one.
MODELS: dict[str, ModelSpec] = {
    "densenet121-union": ModelSpec(
        model_id="densenet121-union",
        filename="densenet121-union.onnx",
        generation="paper",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        description="1.x frame-wise eye state (DenseNet121).",
    ),
    "blinklint-union": ModelSpec(
        model_id="blinklint-union",
        filename="blinklint-union.onnx",
        generation="paper",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        window=15,
        description="1.x sequence model (DenseNet121 + LinT).",
    ),
    "blinklinmult-union": ModelSpec(
        model_id="blinklinmult-union",
        filename="blinklinmult-union.onnx",
        generation="paper",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        window=15,
        needs_features=True,
        standardise_features=True,
        description="1.x two-stream model (crops + 160-d iris descriptors).",
    ),
    "blinkcnn": ModelSpec(
        model_id="blinkcnn",
        filename="blinkcnn.pt",
        generation="v2",
        runtime="pytorch",
        mean=UNIT_MEAN,
        std=UNIT_STD,
        # Fitted on validation by the hysteresis callback, not guessed. The
        # single-threshold variant of this same run scores 0.19 event F1 against
        # 0.52 for these two, so the pair must travel together.
        threshold=0.53,
        low_ratio=0.25,
        description="v2 frame-wise eye state (ConvNeXt-Femto).",
    ),
    # The same weights as `blinkcnn`, frozen into a graph. Both ship: the
    # checkpoint is the trainable artifact, the graph the deployable one.
    #
    # **The normalisation stays UNIT here, and that is not an oversight.**
    # `EyeEncoder.forward` standardises with ImageNet statistics *inside* the
    # network, so the traced graph carries that step within it and the external
    # contract is raw [0, 1] crops -- identical to the checkpoint's. Setting
    # IMAGENET_MEAN/STD here would normalise twice: the model would still run
    # and still return plausible probabilities, silently degraded. The parity
    # test in tests/test_detector.py is what holds these two entries together.
    "blinkcnn-onnx": ModelSpec(
        model_id="blinkcnn-onnx",
        filename="blinkcnn.onnx",
        generation="v2",
        runtime="onnx",
        mean=UNIT_MEAN,
        std=UNIT_STD,
        threshold=0.53,
        low_ratio=0.25,
        description="v2 frame-wise eye state (ConvNeXt-Femto), ONNX graph.",
    ),
}
"""Every shipped model, keyed by id."""


def spec(model_id: str) -> ModelSpec:
    """Look up one model's specification.

    Args:
        model_id (str): A key of :data:`MODELS`.

    Returns:
        ModelSpec: Its entry.

    Raises:
        KeyError: If the id is not registered.
    """
    if model_id not in MODELS:
        raise KeyError(f"Unknown model id {model_id!r}; expected one of {sorted(MODELS)}.")
    return MODELS[model_id]
