"""The BlinkLinMulT architecture: an eye-crop CNN feeding a LinMulT sequence model.

Three families share one construction path, because they are the paper's three
rows of the same table:

``cnn``
    The per-frame backbone with a classification head and no sequence model —
    the DenseNet121 eye-state baseline.
``lint``
    Backbone embeddings through a single-stream :class:`linmult.LinT` — the
    BlinkLinT baseline.
``linmult``
    Backbone embeddings *and* handcrafted eye features through the cross-modal
    :class:`linmult.LinMulT` — the published BlinkLinMulT.

**One sample is one eye.** The encoder embeds a single eye's window; the two
eyes of a frame are separate samples, recombined only at evaluation time. That
matches how the corpora annotate (per eye) and lets the corpora that supply one
eye, or label the whole face, be represented honestly.

**What changed from 1.x.** The old ``BlinkLinMulT.forward`` embedded frames with
a Python loop over the time axis, one backbone call per timestep, and hard-coded
a 1024-d DenseNet embedding into the LinMulT construction. Here every crop in
the batch is embedded in a single fused call
(:func:`~blinklinmult.data.collate.fold_time_into_batch`), and the embedding
width is *read from the constructed backbone* rather than assumed — so swapping
DenseNet121 for ResNet50 is a config edit, not a code edit.

**Heads.** Both tasks are sequence-level: they predict one value per timestep,
not one per clip, so both heads are linmult ``sequence`` heads. The frame- and
clip-level decisions the tasks are scored on are aggregations of the sequence
(see :mod:`blinklinmult.train.metrics`), which keeps the temporal localisation
the model learns rather than discarding it at the head.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from linmult import HeadConfig, LinMulT, LinMulTConfig, LinT, LinTConfig
from torch import nn
from torchvision import models

from blinklinmult.data.collate import (
    check_eye_images,
    fold_time_into_batch,
    unfold_time_from_batch,
)
from blinklinmult.data.schema import BLINK_PRESENCE, EYE_STATE

if TYPE_CHECKING:
    from collections.abc import Callable

    from blinklinmult.train.config import ModelConfig

logger = logging.getLogger(__name__)
"""Module-level logger."""

SEQUENCE_HEAD = "eye_openness"
"""Name of the single head every supervised target reads.

Both tasks are decided from one per-timestep sequence, exactly as in the paper:
eye state recognition reads it frame by frame, and blink presence detection
takes the ``max`` over a window. Two independent heads would fit the annotation
slightly better but could disagree -- a window scored "no frame shows a closed
eye" *and* "a blink occurred" is incoherent, and nothing in a two-head model
forbids it. One head makes that inconsistency unrepresentable.
"""

EVENT_HEAD = "blink_event"
"""Name of the optional head stacked on the ESR signal.

Reads :data:`SEQUENCE_HEAD`'s per-timestep closure score and predicts the blink
*interval* from it, rather than predicting both from one set of parameters.

**Why stack rather than share or split.** Sharing one head makes the two
annotations compete: a blink interval covers 3-4x more frames than actual
closure (measured P(closed | inside a blink) = 0.237 / 0.310 / 0.344 on RN15 /
RN30 / TalkingFace), so a corpus that annotates only intervals pulls the shared
parameters toward a wider target. On the video benchmark 72% of batches did
exactly that, and RN30's ESR F1 fell to 0.4107 against the frame-wise model's
0.6614.

Two *parallel* heads would remove the competition but allow incoherence -- one
head reporting "no frame shows a closed eye" while the other reports "a blink
occurred". Stacking keeps the event prediction a function of the closure signal,
so that contradiction stays unrepresentable, which was the reason the single
head was chosen in the first place.
"""

EVENT_HEAD_TYPES: frozenset[str] = frozenset({"conv", "attention"})
"""Event-head architectures.

Both see **neighbouring timesteps**, which a pointwise map cannot. That matters
because the closure-to-interval mapping is not a per-frame decision: a blink is
a *run* of elevated closure roughly 200-330 ms long, and the interval covers
3-4x more frames than the closure itself. A head reading one scalar at a time
could only learn a monotone rescaling -- a threshold, which the extractor
already applies for free.

``conv`` is a temporal convolution over the ESR sequence: local, cheap, and
exactly the smoothing-and-widening rule the mapping needs. ``attention`` is one
self-attention layer, which can additionally relate distant timesteps -- useful
if blink boundaries depend on the wider context rather than a fixed window.
"""

EVENT_CONV_KERNEL = 9
"""Temporal receptive field of the ``conv`` event head, in frames.

At 25-30 fps a blink spans roughly 5-10 frames, so a 9-frame window covers one
whole blink and its immediate surroundings -- enough to turn a closure apex into
the interval around it. Padded to preserve length, so the output stays aligned
with the input timestep for timestep.
"""

SUPPORTED_TARGETS: frozenset[str] = frozenset({BLINK_PRESENCE, EYE_STATE})
"""Targets the single head can serve."""

HEAD_OUTPUT_DIM = 1
"""Width of the head: one value per timestep.

A sample carries a single eye, so the eye axis lives in the sample identity
rather than in the head. Frame-level predictions are recovered by aggregating
the two eyes afterwards -- see
:class:`~blinklinmult.train.metrics.FrameAggregator`.
"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""Per-channel mean the pretrained backbones were trained against."""

IMAGENET_STD = (0.229, 0.224, 0.225)
"""Per-channel standard deviation the pretrained backbones were trained against."""


class ModelError(ValueError):
    """Raised when a model cannot be built as configured."""


class TimmBackbone(nn.Module):
    """Wraps a timm model so it pools to ``(N, D)`` like the torchvision path.

    timm's ``forward_features`` returns the spatial map, and its own pooling is
    entangled with a classifier this model does not use. Pooling here keeps both
    backbone families behind one interface.

    Args:
        network (nn.Module): A timm model created with ``num_classes=0``.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Embed a batch of crops.

        Args:
            images (torch.Tensor): ``(N, 3, H, W)``.

        Returns:
            torch.Tensor: ``(N, D)``.
        """
        # `nn.Module.__getattr__` returns `Tensor | Module`, and neither declares
        # `forward_features`, so the bound method is resolved through `getattr`.
        extract = cast(
            "Callable[[torch.Tensor], torch.Tensor]",
            getattr(self.get_submodule("network"), "forward_features"),  # noqa: B009
        )
        return torch.flatten(cast("torch.Tensor", self.pool(extract(images))), 1)


TIMM_WEIGHTS: dict[str, str] = {
    "mobilenetv4_conv_small": "mobilenetv4_conv_small.e2400_r224_in1k",
    "efficientnet_b0": "efficientnet_b0.ra4_e3600_r224_in1k",
    "convnext_femto": "convnext_femto.d1_in1k",
    "mobileone_s1": "mobileone_s1.apple_in1k",
}
"""Backbones served by timm, mapped to the pretrained tag used.

The tag is pinned rather than left to timm's default so a run is reproducible:
``timm`` moves its default weights between releases, and ``efficientnet_b0``
alone ships several checkpoints that differ by several points of ImageNet
accuracy.
"""


def _widen_stem(name: str, network: nn.Module) -> None:
    """Remove one stride-2 reduction so a 64px crop keeps a 4x4 feature map.

    Every ImageNet backbone reduces by 32 in total, which collapses a 64px eye
    crop to ``2x2`` before pooling -- four cells to describe an eye. Dropping
    one reduction doubles that to ``4x4``.

    **No pretrained parameter is discarded either way.** Two mechanisms are used
    depending on what the architecture offers: ``resnet``, ``shufflenet_v2`` and
    ``densenet121`` reduce through a **parameter-free max-pool**, which is
    replaced by an identity; ``mobilenetv4_conv_small`` and
    ``efficientnet_v2_s`` have no pooling layer, so the **stem convolution's
    stride** is set to 1 instead, which changes an attribute and not a weight.

    In both cases the state dict is byte-identical to the published checkpoint
    -- verified in the tests -- but the layers downstream see a feature map at a
    different sampling rate than they were trained on, so **either mechanism
    shifts the pretrained features and expects fine-tuning**. Measured cosine
    similarity between stock and widened embeddings runs 0.25-0.82 across these
    backbones, with no clean split by mechanism. Treat this as a fine-tuning
    change, not a free one.

    Args:
        name (str): Backbone name.
        network (nn.Module): The constructed backbone, modified in place.

    Raises:
        ModelError: If the backbone has no known reduction to remove.
    """
    # `nn.Module.__getattr__` is typed `Tensor | Module`, so every submodule
    # reach-through needs a cast to say which it is.
    if name in {"resnet18", "resnet50", "shufflenet_v2"}:
        network.maxpool = nn.Identity()
    elif name == "densenet121":
        cast("nn.Module", network.features).pool0 = nn.Identity()
    elif name in {"mobilenetv4_conv_small", "efficientnet_b0"}:
        cast("nn.Conv2d", network.conv_stem).stride = (1, 1)
    elif name == "efficientnet_v2_s":
        first_block = cast("nn.Sequential", cast("nn.Sequential", network.features)[0])
        cast("nn.Conv2d", first_block[0]).stride = (1, 1)
    elif name == "convnext_femto":
        # A patchify stem, not a stride-2 conv: it downsamples by 4 in one step,
        # so halving it to 2 is the equivalent single reduction. This one lands
        # on 3x3 rather than 4x4 -- the stem's kernel is 4 wide with stride 2, so
        # 64px leaves an odd edge that the later stages halve twice. Still more
        # than the 2x2 it starts from, which is the point.
        cast("nn.Conv2d", cast("nn.Sequential", network.stem)[0]).stride = (2, 2)
    elif name == "mobileone_s1":
        # The stem is a reparameterizable block with parallel kxk and scale
        # branches; both must change or their outputs stop aligning.
        stem = cast("nn.Module", network.stem)
        cast("nn.Conv2d", cast("nn.Sequential", stem.conv_kxk)[0].conv).stride = (1, 1)
        cast("nn.Conv2d", cast("nn.Module", stem.conv_scale).conv).stride = (1, 1)
    else:
        raise ModelError(
            f"backbone_wide_stem=true is not defined for {name!r}; it has no "
            "identified stride-2 reduction to remove."
        )


DENSENET_CUT = "denseblock4"
"""First DenseNet block dropped by ``densenet121_truncated``."""


def _truncate_densenet(stack: nn.Module) -> tuple[nn.Module, int]:
    """Drop DenseNet's last dense block, keeping the pretrained weights before it.

    **121 is the smallest pretrained DenseNet there is** -- torchvision and timm
    ship 121/161/169/201, and the others are all larger, so "a smaller DenseNet"
    has to be made rather than downloaded. Retraining a narrower one from scratch
    on 3.5k eye crops would lose exactly the ImageNet initialisation that makes
    this backbone work here.

    Truncation is the alternative, and the parameter profile says where to cut.
    Measured on a 64px crop, ``denseblock4`` holds **2.16M of the 6.95M
    parameters (31%) and runs entirely on a 2x2 feature map** -- with the wide
    stem, 4x4. That is a third of the network spent on sixteen spatial cells, at
    a depth whose ImageNet role is separating a thousand object classes rather
    than deciding whether an eyelid is shut.

    Cutting there gives a 4.79M-parameter backbone with a 512-d embedding whose
    every remaining weight is still the pretrained one. It is *not* much faster
    (about 63ms against 69ms per 26-crop batch): DenseNet's cost is the
    concatenation traffic through the earlier blocks, not the arithmetic in the
    last one. The reason to use it is size, and whatever regularisation comes
    from fine-tuning fewer parameters on a small corpus.

    Args:
        stack (nn.Module): DenseNet's ``features`` sequence.

    Returns:
        tuple[nn.Module, int]: The truncated stack and its output width.
    """
    kept = nn.Sequential()
    for child_name, child in stack.named_children():
        if child_name == DENSENET_CUT:
            break
        kept.add_module(child_name, child)

    # Read the width off the truncated stack rather than assuming it: the final
    # transition halves the channel count, and hard-coding 512 here would break
    # silently if the cut point ever moved.
    probe = torch.zeros(1, 3, 64, 64)
    was_training = kept.training
    kept.eval()
    with torch.no_grad():
        width = int(kept(probe).shape[1])
    kept.train(was_training)
    return kept, width


def build_backbone(
    name: str,
    pretrained: bool = True,
    wide_stem: bool = False,
    freeze_hint: bool = False,
) -> tuple[nn.Module, int]:
    """Construct an image backbone and report its embedding width.

    The width is read off the constructed network rather than hard-coded, so a
    library change to a backbone's final layer cannot silently produce a model
    whose projection expects the wrong size.

    Args:
        name (str): Backbone name.
        pretrained (bool): Start from ImageNet weights.
        wide_stem (bool): Drop one stride-2 reduction, so a 64px crop yields a
            ``4x4`` feature map instead of ``2x2``. See :func:`_widen_stem`;
            this expects fine-tuning.
        freeze_hint (bool): Whether the caller intends to freeze the backbone,
            used only to warn about the combination with ``wide_stem``.

    Returns:
        tuple[nn.Module, int]: The feature extractor, mapping ``(N, 3, H, W)`` to
        ``(N, D)``, and its width ``D``.

    Raises:
        ModelError: If the backbone name is unknown.
    """
    if name in TIMM_WEIGHTS:
        import timm

        network = timm.create_model(TIMM_WEIGHTS[name], pretrained=pretrained, num_classes=0)
        if wide_stem:
            _widen_stem(name, network)
        width = int(cast("int", network.num_features))
        features: nn.Module = TimmBackbone(network)
    elif name == "shufflenet_v2":
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        network = models.shufflenet_v2_x1_0(weights=weights)
        width = network.fc.in_features
        if wide_stem:
            _widen_stem(name, network)
        # ShuffleNet's forward hard-codes a mean over the spatial axes, so the
        # stages are re-composed rather than the classifier being replaced.
        features = nn.Sequential(
            network.conv1,
            network.maxpool,
            network.stage2,
            network.stage3,
            network.stage4,
            network.conv5,
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
    elif name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        network = models.resnet18(weights=weights)
        width = network.fc.in_features
        network.fc = nn.Identity()
        if wide_stem:
            _widen_stem(name, network)
        features = network
    elif name in {"densenet121", "densenet121_truncated"}:
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        network = models.densenet121(weights=weights)
        width = network.classifier.in_features
        if wide_stem:
            _widen_stem("densenet121", network)
        stack = cast("nn.Module", network.features)
        if name == "densenet121_truncated":
            stack, width = _truncate_densenet(stack)
        features = nn.Sequential(
            stack,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
    elif name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        network = models.resnet50(weights=weights)
        width = network.fc.in_features
        network.fc = nn.Identity()
        if wide_stem:
            _widen_stem(name, network)
        features = network
    elif name == "efficientnet_v2_s":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        network = models.efficientnet_v2_s(weights=weights)
        width = network.classifier[-1].in_features
        network.classifier = nn.Identity()
        if wide_stem:
            _widen_stem(name, network)
        features = network
    else:
        raise ModelError(f"Unknown backbone {name!r}.")

    if wide_stem and pretrained and freeze_hint:
        logger.warning(
            f"Backbone {name}: backbone_wide_stem changes the sampling rate the "
            "pretrained layers see, so a frozen backbone keeps weights that no "
            "longer match their input. Fine-tune, or set backbone_wide_stem=false."
        )
    logger.info(
        f"Backbone {name}: {width}-d embedding, pretrained={pretrained}, wide_stem={wide_stem}"
    )
    return features, width


MIN_IMAGE_SIZE = 32
"""Smallest eye crop the convolutional backbones accept.

All three backbones downsample by 32 in total, so a smaller crop collapses to a
zero-sized feature map inside their final pooling layer. Torch reports that as
``Calculated output size: (512x0x0)`` from somewhere deep in ``avg_pool2d``,
which says nothing about the crop size being the cause — hence the explicit
check.
"""


ENCODER_PREFIX = "model.encoder."
"""Prefix the eye encoder's weights carry inside a Lightning checkpoint.

``BlinkLightningModule`` holds a ``BlinkModel`` as ``self.model``, whose encoder
is ``self.encoder`` -- so a checkpoint's ``state_dict`` names them
``model.encoder.backbone.*`` and ``model.encoder.project.*``.
"""


def load_encoder_weights(encoder: nn.Module, path: str | Path) -> None:
    """Initialise an eye encoder from a trained frame-wise checkpoint.

    The frame-wise BlinkCNN and the sequence models share exactly one component:
    this encoder, backbone and projection together. Everything after it differs
    (BlinkCNN has a per-frame head; LinT and LinMulT have a transformer), so
    only the encoder transfers — and it transfers as a *starting point*, not
    frozen. The sequence models fine-tune it at ``optimizer.backbone_lr``.

    Loading is **strict about shape and lenient about nothing else**: a
    checkpoint trained with a different backbone or a different
    ``backbone_output_dim`` has tensors that cannot be assigned, and silently
    skipping them would leave a randomly-initialised encoder that looks
    pretrained. Any mismatch raises.

    Args:
        encoder (nn.Module): The freshly built :class:`EyeEncoder`.
        path (str | Path): A Lightning ``.ckpt`` written by a frame-wise run.

    Raises:
        ModelError: If the file is missing, holds no encoder weights, or its
            tensors do not fit this encoder.
    """
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise ModelError(
            f"model.encoder_weights={checkpoint_path} does not exist. Train a "
            "frame-wise model first, or clear the field to start from ImageNet."
        )

    # `weights_only=True`: only tensors are wanted, and the alternative
    # unpickles arbitrary objects from a file the config points at.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state = checkpoint.get("state_dict", checkpoint)

    weights = {
        key.removeprefix(ENCODER_PREFIX): value
        for key, value in state.items()
        if key.startswith(ENCODER_PREFIX)
    }
    if not weights:
        raise ModelError(
            f"{checkpoint_path} holds no {ENCODER_PREFIX}* tensors, so it is not a "
            "blinklinmult checkpoint. Point model.encoder_weights at a .ckpt "
            "written by a frame-wise (family: cnn) run."
        )

    try:
        missing, unexpected = encoder.load_state_dict(weights, strict=False)
    except RuntimeError as error:
        raise ModelError(
            f"{checkpoint_path} does not fit this encoder: {error}. The checkpoint's "
            "backbone and backbone_output_dim must match the ones configured here."
        ) from error

    # `strict=False` is needed because the encoder's normalisation buffers are
    # non-persistent and so absent from any checkpoint -- but a *parameter* that
    # failed to load is the silent failure this function exists to prevent.
    unloaded = [name for name in missing if not name.startswith("pixel_")]
    if unloaded or unexpected:
        raise ModelError(
            f"{checkpoint_path} does not fit this encoder: {len(unloaded)} weights "
            f"were not in the checkpoint ({unloaded[:3]}) and {len(unexpected)} were "
            f"not in the model ({list(unexpected)[:3]}). Check that model.backbone "
            "and model.backbone_output_dim match the frame-wise run."
        )

    logger.info(
        f"Encoder initialised from {checkpoint_path} ({len(weights)} tensors). "
        "Whether it then trains is decided by model.encoder_freeze, which is "
        "applied after this point -- see the trainable-parameter count logged "
        "by `build_model` for what actually receives gradient."
    )


class EyeEncoder(nn.Module):
    """Embeds a window of eye crops into a per-timestep feature sequence.

    Args:
        config (ModelConfig): Architecture settings.
        image_size (int): Side length of a square eye crop.

    Raises:
        ModelError: If the backbone cannot be built, or the crop is too small
            for it.
    """

    def __init__(self, config: ModelConfig, image_size: int):
        super().__init__()
        if image_size < MIN_IMAGE_SIZE:
            raise ModelError(
                f"image_size={image_size} is below the {MIN_IMAGE_SIZE}px minimum the "
                f"convolutional backbones accept: {config.backbone} downsamples by 32, "
                "so a smaller crop collapses to a zero-sized feature map. Re-extract "
                "the eye crops at 32px or larger."
            )

        self.image_size = image_size
        self.backbone, backbone_width = build_backbone(
            config.backbone,
            pretrained=config.backbone_pretrained,
            wide_stem=config.backbone_wide_stem,
            freeze_hint=config.backbone_freeze,
        )

        if config.backbone_freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            logger.info("Backbone frozen; only the projection and sequence model train.")

        # One projection for every eye, whichever side it is: a sample carries a
        # single eye, and sharing the projection is what keeps a left-eye and a
        # right-eye sample comparable -- including for MRL-Eye, which does not
        # say which side it supplied.
        self.project = nn.Sequential(
            nn.Linear(backbone_width, config.backbone_output_dim),
            nn.BatchNorm1d(config.backbone_output_dim),
            nn.GELU(),
            nn.Dropout(config.dropout_input),
        )
        self.output_dim = config.backbone_output_dim

        if config.encoder_weights:
            load_encoder_weights(self, config.encoder_weights)

        # Freezing is applied *after* loading, so a frozen encoder holds the
        # transferred weights rather than ImageNet's. `eval()` matters as much
        # as `requires_grad`: BatchNorm in `project` keeps updating its running
        # statistics in train mode even with no gradient, so a "frozen" encoder
        # would still drift and the arm would not be the ablation it claims.
        # `BlinkModel.train()` re-applies this, because Lightning puts the whole
        # module back into train mode at every epoch.
        self.frozen = bool(config.encoder_freeze)
        if self.frozen:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            self.eval()

        self.register_buffer(
            "pixel_mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "pixel_std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def train(self, mode: bool = True) -> EyeEncoder:
        """Keep a frozen encoder in eval mode.

        Lightning calls ``train()`` on the whole module at the start of every
        epoch, which would put a frozen encoder's BatchNorm back into training
        mode. Its running statistics would then keep updating even though no
        gradient flows, so the "frozen" arm would drift away from the
        checkpoint it was supposed to hold fixed -- and the comparison against
        the fine-tuned arm would measure something other than what it claims.

        Args:
            mode (bool): Requested mode, ignored while frozen.

        Returns:
            EyeEncoder: ``self``, as :meth:`torch.nn.Module.train` does.
        """
        # `getattr` because Lightning can call train() during __init__, before
        # the attribute is set.
        if getattr(self, "frozen", False):
            return super().train(False)
        return super().train(mode)

    def forward(self, eye_images: torch.Tensor) -> torch.Tensor:
        """Embed one batch of windows.

        Args:
            eye_images (torch.Tensor): ``(B, T, C, H, W)`` as loaded.

        Returns:
            torch.Tensor: ``(B, T, backbone_output_dim)``.
        """
        images = check_eye_images(eye_images, self.image_size)
        batch, time = images.shape[0], images.shape[1]

        crops = fold_time_into_batch(images)
        # `register_buffer` makes these Tensors, but nn.Module types every
        # attribute lookup as `Tensor | Module`.
        mean = cast("torch.Tensor", self.pixel_mean)
        std = cast("torch.Tensor", self.pixel_std)
        crops = (crops - mean) / std

        embedded = self.project(self.backbone(crops))
        return unfold_time_from_batch(embedded, batch, time)


def check_targets(target_names: list[str]) -> None:
    """Reject targets the single head cannot serve.

    Args:
        target_names (list[str]): Targets this run supervises.

    Raises:
        ModelError: If a target is not one this model predicts.
    """
    for name in target_names:
        if name not in SUPPORTED_TARGETS:
            raise ModelError(
                f"No head defined for target {name!r}; expected one of {sorted(SUPPORTED_TARGETS)}."
            )


def _heads(config: ModelConfig) -> list[HeadConfig]:
    """Build the one sequence head both targets read.

    Args:
        config (ModelConfig): Architecture settings.

    Returns:
        list[HeadConfig]: A single head; see :data:`SEQUENCE_HEAD` for why the
        count does not follow the target count.
    """
    return [
        HeadConfig(
            type="sequence",
            name=SEQUENCE_HEAD,
            output_dim=HEAD_OUTPUT_DIM,
            dropout=config.head_dropout,
            hidden_dim=config.head_hidden_dim,
            norm=config.head_norm,
        )
    ]


def _attention_kwargs(config: ModelConfig) -> dict[str, Any]:
    """Attention-variant fields common to both linmult config classes.

    Args:
        config (ModelConfig): Architecture settings.

    Returns:
        dict: Keyword arguments accepted by both configs.
    """
    return {
        "attention_type": config.attention_type,
        "flash_query_key_dim": config.flash_query_key_dim,
        "performer_num_random_features": config.performer_num_random_features,
        "bigbird_block_size": config.bigbird_block_size,
        "bigbird_num_global_tokens": config.bigbird_num_global_tokens,
        "bigbird_num_random_tokens": config.bigbird_num_random_tokens,
    }


class CnnHead(nn.Module):
    """Per-frame classification head for the ``cnn`` family.

    No sequence model: each timestep is classified from its own embedding,
    which is the DenseNet121 eye-state baseline of the paper. The output is
    still a sequence so that every family shares one training step.

    Args:
        input_dim (int): Encoder output width.
        config (ModelConfig): Architecture settings.
        target_names (list[str]): Targets to predict.
    """

    def __init__(self, input_dim: int, config: ModelConfig, target_names: list[str]):
        super().__init__()
        self.target_names = list(target_names)
        self.head = nn.Sequential(
            nn.Linear(input_dim, config.head_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.head_dropout),
            nn.Linear(config.head_hidden_dim, HEAD_OUTPUT_DIM),
        )

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        """Classify every timestep.

        Args:
            sequence (torch.Tensor): ``(B, T, D)`` per-timestep embeddings.

        Returns:
            dict[str, torch.Tensor]: The same ``(B, T, 1)`` logits under every
            supervised target key -- one decision, read two ways.
        """
        logits = self.head(sequence)
        return dict.fromkeys(self.target_names, logits)


class EventHead(nn.Module):
    """Predicts blink intervals from the ESR closure signal.

    Takes the ``(B, T, 1)`` per-timestep closure score and emits ``(B, T, 1)``
    interval logits. See :data:`EVENT_HEAD` for why this is stacked on the ESR
    head rather than sharing it or sitting beside it.

    **The input is detached by default.** Without that, the event loss reaches
    the ESR head through this module and the parameter competition returns by a
    longer path -- the thing stacking was meant to remove. Detached, the ESR
    head is supervised purely by closure and this module learns to interpret
    whatever signal it is handed.

    Attaching is left configurable rather than forbidden: a little event
    gradient may help the ESR head sharpen blink boundaries, which is exactly
    what RN15 under-specifies at 15 fps with a median 1.22 closed frames per
    blink. That is worth measuring rather than assuming either way.

    Args:
        kind (str): One of :data:`EVENT_HEAD_TYPES`.
        hidden_dim (int): Width of the internal representation.
        dropout (float): Dropout applied before the output projection.
        num_heads (int): Attention heads, for the ``attention`` variant.
        detach (bool): Cut the gradient to the ESR head.

    Raises:
        ModelError: If ``kind`` is not a known variant.
    """

    def __init__(
        self,
        kind: str,
        hidden_dim: int,
        dropout: float = 0.1,
        num_heads: int = 4,
        detach: bool = True,
    ):
        super().__init__()
        if kind not in EVENT_HEAD_TYPES:
            raise ModelError(
                f"model.event_head={kind!r} is unknown; expected one of "
                f"{sorted(EVENT_HEAD_TYPES)} or null to disable."
            )

        self.kind = kind
        self.detach = detach
        self.project = nn.Linear(HEAD_OUTPUT_DIM, hidden_dim)

        if kind == "conv":
            # Two layers so the field is wider than one kernel and the map is
            # not affine. `padding` keeps the output aligned with the input.
            self.conv = nn.Sequential(
                nn.Conv1d(
                    hidden_dim, hidden_dim, EVENT_CONV_KERNEL, padding=EVENT_CONV_KERNEL // 2
                ),
                nn.GELU(),
                nn.Conv1d(
                    hidden_dim, hidden_dim, EVENT_CONV_KERNEL, padding=EVENT_CONV_KERNEL // 2
                ),
            )
            self.norm = nn.LayerNorm(hidden_dim)

        if kind == "attention":
            # `batch_first` so the (B, T, D) convention holds throughout; the
            # rest of the model uses it and a silent transpose here would be a
            # subtle shape bug rather than a loud one.
            self.attention = nn.MultiheadAttention(
                hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)

        self.output = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, HEAD_OUTPUT_DIM),
        )

    def forward(self, esr_logits: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """Map a closure sequence to interval logits.

        Args:
            esr_logits (torch.Tensor): ``(B, T, 1)`` from the ESR head.
            mask (torch.Tensor | None): ``(B, T)``, ``True`` where the timestep
                is valid. Used by the attention variant to ignore padding.

        Returns:
            torch.Tensor: ``(B, T, 1)`` interval logits.
        """
        signal = esr_logits.detach() if self.detach else esr_logits
        hidden = self.project(signal)

        if self.kind == "conv":
            # Conv1d wants (B, C, T); the rest of the model speaks (B, T, C).
            moved = hidden.transpose(1, 2)
            hidden = self.norm(hidden + self.conv(moved).transpose(1, 2))

        if self.kind == "attention":
            # `key_padding_mask` marks positions to *ignore*, so it is the
            # inverse of the validity mask every other call site uses.
            padding = None if mask is None else ~mask.bool()
            attended, _ = self.attention(
                hidden, hidden, hidden, key_padding_mask=padding, need_weights=False
            )
            hidden = self.norm(hidden + attended)

        return self.output(hidden)


class BlinkModel(nn.Module):
    """The full model: eye encoder, optional sequence model, and task heads.

    Args:
        config (ModelConfig): Architecture settings.
        target_names (list[str]): Targets this run supervises, in head order.
        image_size (int): Side length of a square eye crop.
        eye_feature_dim (int | None): Width of the handcrafted eye features, or
            ``None`` when the run has none.

    Raises:
        ModelError: If the family and the available modalities disagree.
    """

    def __init__(
        self,
        config: ModelConfig,
        target_names: list[str],
        image_size: int,
        eye_feature_dim: int | None = None,
    ):
        super().__init__()
        check_targets(list(target_names))
        self.config = config
        self.target_names = list(target_names)
        self.family = config.family
        self.eye_feature_dim = eye_feature_dim

        self.encoder = EyeEncoder(config, image_size)
        encoder_dim = self.encoder.output_dim

        # Optional: predict blink intervals *from* the ESR signal rather than
        # from the same parameters. See `EVENT_HEAD`.
        self.event_head: EventHead | None = None
        if config.event_head is not None and BLINK_PRESENCE in target_names:
            self.event_head = EventHead(
                config.event_head,
                hidden_dim=config.event_head_hidden_dim,
                dropout=config.head_dropout,
                num_heads=config.num_heads,
                detach=config.event_head_detach,
            )

        if self.family == "cnn":
            self.sequence_model: nn.Module = CnnHead(encoder_dim, config, self.target_names)
            return

        common: dict[str, Any] = {
            "name": config.name,
            "d_model": config.d_model,
            "num_heads": config.num_heads,
            "cmt_num_layers": config.cmt_num_layers,
            "dropout_input": config.dropout_input,
            "dropout_output": config.dropout_output,
            "dropout_pe": config.dropout_pe,
            "dropout_ffn": config.dropout_ffn,
            "dropout_attention": config.dropout_attention,
            "add_module_tcn": config.add_module_tcn,
            "tcn_num_layers": config.tcn_num_layers,
            "tcn_kernel_size": config.tcn_kernel_size,
            "tcn_dropout": config.tcn_dropout,
            "add_module_ffn_fusion": config.add_module_ffn_fusion,
            "heads": _heads(config),
            **_attention_kwargs(config),
        }

        if self.family == "lint":
            self.sequence_model = LinT(LinTConfig(input_feature_dim=encoder_dim, **common))
        else:
            if eye_feature_dim is None:
                raise ModelError(
                    "model.family='linmult' is cross-modal but no dataset in this run "
                    "supplies handcrafted eye features. Use family='lint' for the "
                    "image-only model, or include a corpus with eye features."
                )
            self.sequence_model = LinMulT(
                LinMulTConfig(
                    input_feature_dim=[encoder_dim, eye_feature_dim],
                    branch_sat_num_layers=config.branch_sat_num_layers,
                    **common,
                )
            )

    @property
    def uses_eye_features(self) -> bool:
        """Whether this model consumes the handcrafted feature stream.

        Returns:
            bool: ``True`` only for the ``linmult`` family.
        """
        return self.family == "linmult"

    def forward(
        self,
        eye_images: torch.Tensor,
        eye_image_mask: torch.Tensor,
        eye_features: torch.Tensor | None = None,
        eye_feature_mask: torch.Tensor | None = None,
        eye_embedding: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict every supervised target over a batch of windows.

        Args:
            eye_images (torch.Tensor): ``(B, T, C, H, W)`` eye crops.
            eye_image_mask (torch.Tensor): ``(B, T)``, ``True`` = valid timestep.
            eye_features (torch.Tensor | None): ``(B, T, F)`` handcrafted
                features; required by the ``linmult`` family.
            eye_feature_mask (torch.Tensor | None): ``(B, T)`` validity mask.
            eye_embedding (torch.Tensor | None): ``(B, T, D)`` precomputed eye
                embeddings, replacing the encoder pass. Supplied only when a
                frozen encoder makes them constant across epochs; see
                :mod:`blinklinmult.data.embeddings`. ``None`` encodes live.

        Returns:
            dict[str, torch.Tensor]: Raw logits per target, ``(B, T, D)``.

        Raises:
            ModelError: If the cross-modal family is called without its second
                modality.
        """
        # The one seam the cache needs. A frozen encoder returns the same
        # vector for a given crop on every epoch, so a cached embedding is the
        # identical tensor this call would have produced -- everything
        # downstream is unchanged, because it only ever saw the embedding.
        embedded = eye_embedding if eye_embedding is not None else self.encoder(eye_images)

        if self.family == "cnn":
            # CnnHead already fans its one head out over the target names.
            return self.sequence_model(embedded)

        if self.family == "lint":
            output = self.sequence_model(embedded, eye_image_mask)
        elif eye_features is None or eye_feature_mask is None:
            raise ModelError(
                "model.family='linmult' was called without the handcrafted eye feature "
                "stream. The batch is missing 'eye_feature' or its mask."
            )
        else:
            output = self.sequence_model(
                [embedded, eye_features], [eye_image_mask, eye_feature_mask]
            )

        return self._fan_out(output, eye_image_mask)

    def _fan_out(
        self, output: dict[str, torch.Tensor], mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Expose the single head's logits under every supervised target key.

        Every target reads the *same* tensor, which is what makes the two
        protocols consistent by construction rather than by convention. See
        :data:`SEQUENCE_HEAD`.

        Args:
            output (dict[str, torch.Tensor]): The sequence model's output,
                keyed by head name.
            mask (torch.Tensor | None): ``(B, T)`` validity mask, passed to the
                event head so its attention variant can ignore padding.

        Returns:
            dict[str, torch.Tensor]: The same logits under each target name.

        Raises:
            ModelError: If the sequence model did not emit the expected head.
        """
        if SEQUENCE_HEAD not in output:
            raise ModelError(
                f"The sequence model emitted {sorted(output)} but no {SEQUENCE_HEAD!r} "
                "head. This is an internal inconsistency in head construction."
            )
        esr = output[SEQUENCE_HEAD]
        if self.event_head is None:
            # One head, read twice: the original behaviour, kept exactly so a
            # run without an event head is bit-identical to before.
            return dict.fromkeys(self.target_names, esr)

        # Stacked: closure comes from the ESR head, the interval from a module
        # reading it. `blink_presence` is the only target that moves; anything
        # else still reads the ESR signal directly.
        routed = dict.fromkeys(self.target_names, esr)
        if BLINK_PRESENCE in routed:
            routed[BLINK_PRESENCE] = self.event_head(esr, mask)
        return routed


def build_model(
    config: ModelConfig,
    target_names: list[str],
    image_size: int,
    eye_feature_dim: int | None = None,
) -> BlinkModel:
    """Construct the model for the configured family.

    Args:
        config (ModelConfig): Architecture settings.
        target_names (list[str]): Targets this run supervises.
        image_size (int): Side length of a square eye crop.
        eye_feature_dim (int | None): Handcrafted feature width, if any.

    Returns:
        BlinkModel: The model.
    """
    model = BlinkModel(config, target_names, image_size, eye_feature_dim)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"Built {config.name} ({config.family}): {trainable:,} trainable of "
        f"{total:,} parameters, heads={target_names}"
    )
    return model
