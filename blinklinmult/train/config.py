"""Typed configuration for the blink training pipeline.

Configuration is plain YAML parsed into frozen dataclasses, split across three
files per run — ``--data``, ``--model``, ``--train``. Two properties motivate
that over a framework:

* The same :class:`~blinklinmult.data.schema.DatasetSpec` objects describe a
  corpus to the HDF5 builder *and* to the dataloader. A framework deriving its
  schema from class constructors (Lightning CLI / jsonargparse) could not span
  both, because the builder is not a Lightning class.
* Nothing here changes the working directory or writes hidden output trees, so
  MLflow artifact paths stay predictable.

Unknown keys are rejected rather than ignored: a silently-dropped key produces a
run whose recorded config does not describe what it did, which is worse than a
crash. Retired 1.x model keys are rejected with an explicit rename hint.

Every knob a run depends on lives in one of these dataclasses, and
:meth:`ExperimentConfig.to_flat_dict` flattens the lot into MLflow params.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from blinklinmult.data.augment import AugmentConfig
from blinklinmult.data.schema import (
    BLINK_PRESENCE,
    DEFAULT_WINDOW_SECONDS,
    EYE_STATE,
    IMAGE_DATASETS,
    TARGET_KEYS,
    DatasetSpec,
    SchemaError,
)

logger = logging.getLogger(__name__)
"""Module-level logger."""

TASKS: frozenset[str] = frozenset({"blink_presence", "eye_state", "joint"})
"""The trainable tasks.

``blink_presence`` and ``eye_state`` are the paper's two tasks; ``joint`` trains
both heads at once over the union of corpora, which is the case OmniLoader's
masking exists to support and the main reason for the v2 rewrite.
"""

TASK_TARGETS: dict[str, tuple[str, ...]] = {
    "blink_presence": (BLINK_PRESENCE,),
    "eye_state": (EYE_STATE,),
    "joint": (BLINK_PRESENCE, EYE_STATE),
}
"""Target keys each task supervises."""

MODEL_FAMILIES: frozenset[str] = frozenset({"linmult", "lint", "cnn"})
"""Architecture families a model config may select.

``linmult`` is the multimodal cross-modal transformer over eye images plus
handcrafted features; ``lint`` is the single-stream transformer over eye images
alone (the BlinkLinT baseline); ``cnn`` is the per-frame image backbone with no
sequence model (the DenseNet121 baseline). All three are in the paper.
"""

BACKBONES: frozenset[str] = frozenset(
    {
        "mobilenetv4_conv_small",
        "shufflenet_v2",
        "resnet18",
        "densenet121",
        "resnet50",
        "efficientnet_v2_s",
        "efficientnet_b0",
        "convnext_femto",
        "mobileone_s1",
        "densenet121_truncated",
    }
)
"""Image backbones that embed one eye crop.

``convnext_femto`` is the default, chosen on measured CEW eye-state accuracy
(96.6% over three seeds, against 94.4% for ``densenet121``) at 4.8M parameters
and roughly 4x the throughput. ``densenet121`` is kept because it is the
published 1.x model and the paper's numbers refer to it;
``densenet121_truncated`` drops its last dense block, which holds 31% of the
parameters and runs on a 2x2 map at this crop size.
"""

OPTIMIZERS: frozenset[str] = frozenset({"adam", "adamw", "radam", "sgd"})
"""Supported optimizers."""

SCHEDULERS: frozenset[str] = frozenset({"onecycle", "cosine", "plateau", "none"})
"""Supported LR schedulers."""

LOSSES: frozenset[str] = frozenset({"bce", "focal", "dice_bce"})
"""Supported losses. All operate on masked binary targets."""

PRECISIONS: frozenset[str] = frozenset(
    {"32-true", "16-mixed", "bf16-mixed", "64-true", "16-true", "bf16-true"}
)
"""Lightning precision settings this project supports.

Validated here so a typo fails at config load with a readable message rather
than deep inside the Trainer.
"""


class ConfigError(ValueError):
    """Raised when a configuration is invalid or self-contradictory."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Parse a YAML file into a dict.

    Args:
        path (str | Path): File to read.

    Returns:
        dict: Parsed mapping.

    Raises:
        ConfigError: If the file is missing or is not a mapping.
    """
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    parsed = yaml.safe_load(path.read_text())
    if not isinstance(parsed, dict):
        raise ConfigError(f"{path}: expected a YAML mapping, got {type(parsed).__name__}.")
    return parsed


def _reject_unknown(cls: type, data: dict[str, Any], context: str) -> None:
    """Fail on config keys the dataclass does not declare.

    Args:
        cls (type): Target dataclass.
        data (dict): Incoming values.
        context (str): Human-readable source, for the error message.

    Raises:
        ConfigError: If any key is unknown.
    """
    # `fields` is typed to take a dataclass instance or type; every caller passes
    # a dataclass type, which the checker cannot verify from the bare `type`.
    known = {f.name for f in fields(cls)}  # ty: ignore[invalid-argument-type]
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"{context}: unknown keys {unknown}. Known keys: {sorted(known)}.")


@dataclass(frozen=True)
class DataConfig:
    """Dataloader configuration.

    Args:
        eval_datasets (list[str]): Extra corpora to include in the **test**
            split only, on top of ``datasets``. This is the cross-corpus
            generalisation protocol: train on what is annotated the way the
            model needs, then measure transfer to corpora never trained on.
            Empty means the test split matches ``datasets``.

            A corpus listed here that also appears in ``datasets`` is not
            duplicated -- it simply contributes its own test split as usual.
        datasets (list[str]): Corpora to include, by name. Each must have a
            declaration under ``config/data/<name>.yaml`` and a built HDF5 file.
        window_seconds (float | None): Analysis window applied to every corpus,
            which each converts to its own frame count from its rate. ``None``
            keeps each corpus's declared window.
        time_dim (int | None): Shared window length in frames, overriding what
            the corpora derive. ``None`` takes the longest any of them needs, so
            none is truncated.
        image_size (int | None): Shared eye-crop size. ``None`` requires the
            corpora to agree.
        strategy (str): Mixing strategy across corpora.
        strategy_kwargs (dict): Strategy-specific arguments. Empty by default
            rather than carrying the default strategy's knobs: a config that
            changed only ``strategy`` would otherwise pass the previous
            strategy's arguments to one that does not accept them.
        batch_size (int): Samples per batch.
        num_workers (int): Dataloader worker processes.
        pin_memory (bool): Page-lock host batches for faster device copies.
        persistent_workers (bool): Keep workers (and their HDF5 handles) alive
            between epochs.
        prefetch_factor (int): Batches prefetched per worker.
        cache_size (int): Per-worker LRU sample cache; ``0`` disables.
        preload (bool): Read whole splits into RAM. Fast for the small corpora,
            ruinous for MRL-Eye.
        seed (int): Seed for mixing, shuffling, and augmentation.
        stills (bool): Serve single frames instead of windows, for the
            frame-wise model. Video corpora are reduced to their individual
            annotated frames at load time; the still corpora are already
            single-frame and pass through unchanged.
        still_stride (int): In still mode, keep every Nth open-eye frame of a
            window. Closed-eye frames are always kept.
        still_open_to_closed (float): In still mode, open-eye frames kept per
            closed-eye frame in the training split.
        fit_fractions (dict[str, float]): Per-corpus fraction of the **train and
            validation** splits to keep, as ``{corpus: fraction}`` in
            ``(0, 1]``. Corpora not named are kept whole.

            **Test is never touched.** Capping it would score each arm on a
            different subset and make the arms incomparable, which is the whole
            reason for running them. That is the difference from
            ``train.limit_fit_batches``, which caps every corpus at once: this
            thins one corpus while the rest stay full, which is what a mix
            dominated by a single large corpus needs. MPEblink is 63 758 of the
            70 786 training windows here -- 90% -- so subsampling it alone
            changes the cost of an epoch by an order of magnitude while leaving
            the RN corpora, the headline benchmark, at full size.

            The subsample is a deterministic stride over the sorted sample ids,
            so a run is reproducible and every arm sees the same subset.
        embedding_cache (str | None): Directory of precomputed eye embeddings,
            or ``None`` to encode live.

            A **frozen** encoder computes the identical vector for a given crop
            on every epoch, so every epoch after the first recomputes a
            constant. Caching turns 360 crops per batch into 8 small matrices:
            measured 0.74 s/batch frozen against milliseconds cached, and 3.4 GB
            of embeddings replacing 51 GB of crops.

            **Valid only with a frozen encoder and augmentation off**, and both
            are refused rather than warned about. Augmentation perturbs the crop
            *before* the encoder, so a cache would freeze one augmented view per
            window forever -- keeping the distortion and losing the variety. An
            unfrozen encoder changes weights every step, so the cache is stale
            after the first one. Either mistake trains on embeddings from a
            different model and reports a plausible, wrong number.
    """

    datasets: list[str] = field(default_factory=list)
    eval_datasets: list[str] = field(default_factory=list)
    window_seconds: float | None = DEFAULT_WINDOW_SECONDS
    time_dim: int | None = None
    image_size: int | None = None
    stills: bool = False
    still_stride: int = 6
    still_open_to_closed: float = 1.0
    fit_fractions: dict[str, float] = field(default_factory=dict)
    embedding_cache: str | None = None
    embedding_dim: int | None = None
    augment: AugmentConfig | None = None
    model_reads_features: bool = False
    strategy: str = "temperature"
    strategy_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    cache_size: int = 0
    preload: bool = False
    seed: int = 42

    def __post_init__(self) -> None:
        """Validate the dataloader configuration.

        Raises:
            ConfigError: If any value is out of range or unrecognised.
        """
        # Imported here rather than at module scope: config.py is imported by the
        # builder's config path, and datamodule.py pulls in lightning + omniloader.
        from blinklinmult.data.datamodule import STRATEGIES

        if not self.datasets:
            raise ConfigError("data.datasets must name at least one corpus.")
        duplicates = sorted({d for d in self.datasets if self.datasets.count(d) > 1})
        if duplicates:
            raise ConfigError(f"data.datasets contains duplicates: {duplicates}.")
        if self.strategy not in STRATEGIES:
            raise ConfigError(f"data.strategy={self.strategy!r} is not one of {list(STRATEGIES)}.")
        if self.embedding_cache is not None and self.augment is not None:
            raise ConfigError(
                "data.embedding_cache cannot be used with data.augment. Augmentation "
                "perturbs the crop before the encoder, so a cached embedding would "
                "freeze one augmented view per window for the whole run -- keeping the "
                "distortion and losing the variety, which is worse than no augmentation. "
                "Set data.augment=null to use the cache."
            )
        for name, fraction in self.fit_fractions.items():
            if not 0.0 < fraction <= 1.0:
                raise ConfigError(
                    f"data.fit_fractions[{name!r}]={fraction} must be in (0, 1]. It is the "
                    "share of that corpus's train/valid split to keep; 1.0 keeps it whole, "
                    "and the test split is never subsampled."
                )
            if name not in self.datasets and name not in self.eval_datasets:
                raise ConfigError(
                    f"data.fit_fractions names {name!r}, which is not in data.datasets "
                    f"{self.datasets} or data.eval_datasets {self.eval_datasets}. A typo "
                    "here would silently subsample nothing."
                )
        if self.batch_size < 1:
            raise ConfigError(f"data.batch_size must be >= 1, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ConfigError(f"data.num_workers must be >= 0, got {self.num_workers}.")
        if self.time_dim is not None and self.time_dim < 1:
            raise ConfigError(f"data.time_dim must be >= 1, got {self.time_dim}.")
        if self.window_seconds is not None and self.window_seconds <= 0:
            raise ConfigError(
                f"data.window_seconds must be positive or null, got {self.window_seconds}."
            )
        if self.image_size is not None and self.image_size < 1:
            raise ConfigError(f"data.image_size must be >= 1, got {self.image_size}.")
        if self.cache_size < 0:
            raise ConfigError(f"data.cache_size must be >= 0, got {self.cache_size}.")
        if self.still_stride < 1:
            raise ConfigError(f"data.still_stride must be >= 1, got {self.still_stride}.")
        if isinstance(self.augment, dict):
            # YAML gives a mapping; the dataclass is what the rest of the code
            # expects, and building it here keeps the coercion in one place.
            object.__setattr__(self, "augment", AugmentConfig(**dict(self.augment)))
        if self.still_open_to_closed <= 0:
            raise ConfigError(
                f"data.still_open_to_closed must be positive, got {self.still_open_to_closed}."
            )
        if self.stills and self.time_dim not in (None, 1):
            raise ConfigError(
                f"data.stills serves one frame per sample, so data.time_dim must be 1 "
                f"or null, got {self.time_dim}."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DataConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            DataConfig: The validated config.
        """
        _reject_unknown(cls, data, "data config")
        return cls(**data)

    def resolve_specs(self, config_dir: Path) -> list[DatasetSpec]:
        """Load the declaration of every configured corpus, in config order.

        Args:
            config_dir (Path): Directory of per-dataset YAML declarations.

        Returns:
            list[DatasetSpec]: One spec per configured corpus.

        Raises:
            ConfigError: If a declaration is missing or invalid.
        """
        specs = []
        # Eval-only corpora are appended after the training ones, so the
        # training order (which the mixing strategy indexes) is unchanged.
        # `dict.fromkeys` dedupes while preserving order -- a corpus named in
        # both lists must contribute one spec, not two.
        ordered = list(dict.fromkeys([*self.datasets, *self.eval_datasets]))
        for name in ordered:
            path = config_dir / f"{name}.yaml"
            if not path.is_file():
                raise ConfigError(f"No dataset declaration for {name!r} at {path}.")
            raw = load_yaml(path)
            # These two keys belong to the builder, which owns where files live.
            for key in ("processed_dir", "h5_path"):
                raw.pop(key, None)
            try:
                spec = DatasetSpec.from_dict(raw)
            except SchemaError as error:
                raise ConfigError(f"{path}: {error}") from error
            # One window length for the whole run; each corpus re-derives its own
            # frame count from its rate, so every corpus spans the same duration.
            specs.append(spec.with_window(self.window_seconds))
        return specs


@dataclass(frozen=True)
class OptimConfig:
    """Optimizer and LR-schedule configuration.

    Args:
        name (str): Optimizer name.
        lr (float): Peak learning rate.
        backbone_lr (float | None): Separate learning rate for the image
            backbone. A pretrained CNN fine-tunes at a fraction of the rate the
            randomly-initialised transformer needs; ``None`` uses ``lr`` for
            everything.
        weight_decay (float): L2 / decoupled weight decay.
        momentum (float): SGD momentum; ignored by the Adam family.
        scheduler (str): Schedule name, or ``"none"``.
        warmup_ratio (float): Fraction of training spent warming up.
        scheduler_patience (int | None): Epochs without improvement before
            ``plateau`` cuts the learning rate. ``None`` uses a tenth of
            ``max_epochs``, which scales with the run but is usually too patient:
            at 28 minutes an epoch, waiting five epochs to react costs over two
            hours. Ignored by the other schedulers.
        scheduler_factor (float): Multiplier applied to the learning rate when
            ``plateau`` fires. Ignored by the other schedulers.
        gradient_clip_val (float): Gradient-norm clip; ``0`` disables.
    """

    name: str = "adamw"
    lr: float = 1e-3
    backbone_lr: float | None = 1e-4
    weight_decay: float = 0.01
    momentum: float = 0.9
    scheduler: str = "onecycle"
    scheduler_patience: int | None = None
    scheduler_factor: float = 0.5
    warmup_ratio: float = 0.1
    gradient_clip_val: float = 1.0

    def __post_init__(self) -> None:
        """Validate the optimizer configuration.

        Raises:
            ConfigError: If any value is out of range or unrecognised.
        """
        if self.name not in OPTIMIZERS:
            raise ConfigError(f"optimizer.name={self.name!r} not in {sorted(OPTIMIZERS)}.")
        if self.scheduler not in SCHEDULERS:
            raise ConfigError(
                f"optimizer.scheduler={self.scheduler!r} not in {sorted(SCHEDULERS)}."
            )
        if self.lr <= 0:
            raise ConfigError(f"optimizer.lr must be positive, got {self.lr}.")
        if self.backbone_lr is not None and self.backbone_lr <= 0:
            raise ConfigError(
                f"optimizer.backbone_lr must be positive or null, got {self.backbone_lr}."
            )
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ConfigError(f"optimizer.warmup_ratio must be in [0, 1), got {self.warmup_ratio}.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OptimConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            OptimConfig: The validated config.
        """
        _reject_unknown(cls, data, "optimizer config")
        return cls(**data)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture configuration.

    Field names track **linmult 2.x** exactly. The 1.x names the published code
    used (``input_modality_channels``, ``projected_modality_dim``,
    ``number_of_layers``, ``add_projection_fusion``, ``aggregation``) no longer
    exist upstream and are rejected with a rename hint rather than silently
    ignored.

    ``input_feature_dim`` is absent by design: it is derived from the backbone's
    embedding width and the dataset's feature width, so the model cannot
    disagree with its data.

    Args:
        name (str): Model name, recorded in the run.
        family (str): ``linmult``, ``lint``, or ``cnn``.
        backbone (str): Image backbone embedding one eye crop.
        backbone_pretrained (bool): Start the backbone from ImageNet weights.
        backbone_freeze (bool): Freeze the backbone's parameters.
        backbone_wide_stem (bool): Drop one stride-2 reduction so a 64px eye
            crop keeps a ``4x4`` feature map rather than collapsing to ``2x2``.
            Free for the backbones in
            :data:`~blinklinmult.train.model.BACKBONE_STEM_IS_FREE`, which
            reduce through a parameter-free max-pool; for the others it
            re-strides a convolution and perturbs the pretrained kernels.
        backbone_output_dim (int): Width the backbone's embedding is projected
            to before the sequence model. The raw backbone embedding is 960-d
            per eye for MobileNetV4, 1024-d for DenseNet121; projecting first
            keeps the transformer's input width independent of which backbone
            is chosen.
        encoder_weights (str | None): Path to a frame-wise ``.ckpt`` whose eye
            encoder initialises this model's. The encoder — backbone and
            projection — is the only component BlinkCNN and the sequence models
            share, so it is the only part that transfers; the transformer starts
            fresh either way. It is a *starting point*, not frozen:
            ``optimizer.backbone_lr`` fine-tunes it from there.

            A path rather than a baked-in default so that retraining the
            frame-wise model and pointing at the new checkpoint is a one-line
            config edit. ``None`` starts the encoder from ImageNet weights.
        encoder_freeze (bool): Hold the eye encoder fixed -- no gradient, and
            eval mode so its BatchNorm statistics stop moving too. Turns
            ``encoder_weights`` from a starting point into a fixed feature
            extractor, which is what separates the ``frozen`` arm from the
            ``fine-tuned`` one in the video benchmark.

            Worth measuring rather than assuming: the video corpora are small
            next to the 4.8M backbone parameters they would be fine-tuning, so
            freezing may well win. Meaningless without ``encoder_weights`` --
            freezing an ImageNet encoder that never saw an eye is not an arm
            anyone wants -- and rejected in that combination.
        event_head (str | None): Stack a blink-interval head on the ESR signal
            -- ``dense``, ``attention``, or ``None`` to keep one shared head.

            With one head both annotations train the same parameters, and a
            blink interval covers 3-4x more frames than actual closure, so a
            corpus annotating only intervals drags the closure prediction wider.
            Measured on the video benchmark, where 72% of batches supervised
            events alone: RN30's ESR F1 fell to 0.4107 against the frame-wise
            model's 0.6614.

            Stacking keeps the event prediction a *function of* the closure
            signal, so the two cannot contradict each other -- the reason a
            single head was chosen originally -- while giving each its own
            parameters. ``None`` reproduces the previous behaviour exactly.
        event_head_hidden_dim (int): Width inside the event head.
        event_head_detach (bool): Cut the gradient from the event loss to the
            ESR head. ``True`` is the point of the split; ``False`` is worth
            measuring, since a little event gradient may sharpen blink
            boundaries that RN15 under-specifies at 15 fps.
        d_model (int): Transformer internal width.
        num_heads (int): Attention heads.
        cmt_num_layers (int): Cross-modal (LinMulT) / encoder (LinT) depth.
        branch_sat_num_layers (int): Per-branch self-attention depth; LinMulT
            only.
        attention_type (str): ``linear`` (LinMulT) or ``softmax`` (MulT); this
            is the only architectural difference between the two baselines.
            ``flash``, ``performer``, and ``bigbird`` are also accepted.
        flash_query_key_dim (int | None): Flash-attention query/key width.
        performer_num_random_features (int | None): Performer feature count.
        bigbird_block_size (int): BigBird block size.
        bigbird_num_global_tokens (int): BigBird global-token count.
        bigbird_num_random_tokens (int): BigBird random-token count.
        dropout_input (float): Dropout before projection.
        dropout_output (float): Dropout after fusion.
        dropout_pe (float): Dropout after positional encoding.
        dropout_ffn (float): Dropout inside FFN blocks.
        dropout_attention (float): Attention-weight dropout.
        add_module_tcn (bool): Per-modality TCN after projection.
        tcn_num_layers (int): TCN depth, when enabled.
        tcn_kernel_size (int): TCN kernel size.
        tcn_dropout (float): TCN dropout.
        add_module_ffn_fusion (bool): FFN + residual after fusion.
        head_dropout (float): Dropout inside each head.
        head_hidden_dim (int): Head hidden width.
        head_norm (str): Head normalisation (``bn`` or ``ln``).
    """

    name: str = "BlinkLinMulT"
    family: str = "linmult"
    backbone: str = "convnext_femto"
    backbone_pretrained: bool = True
    backbone_freeze: bool = False
    backbone_wide_stem: bool = True
    backbone_output_dim: int = 256
    encoder_weights: str | None = None
    encoder_freeze: bool = False
    event_head: str | None = None
    event_head_hidden_dim: int = 64
    event_head_detach: bool = True
    d_model: int = 32
    num_heads: int = 8
    cmt_num_layers: int = 5
    branch_sat_num_layers: int = 5
    attention_type: str = "linear"
    flash_query_key_dim: int | None = None
    performer_num_random_features: int | None = None
    bigbird_block_size: int = 64
    bigbird_num_global_tokens: int = 16
    bigbird_num_random_tokens: int = 10
    dropout_input: float = 0.0
    dropout_output: float = 0.2
    dropout_pe: float = 0.1
    dropout_ffn: float = 0.2
    dropout_attention: float = 0.0
    add_module_tcn: bool = False
    tcn_num_layers: int = 3
    tcn_kernel_size: int = 3
    tcn_dropout: float = 0.1
    add_module_ffn_fusion: bool = True
    head_dropout: float = 0.2
    head_hidden_dim: int = 256
    head_norm: str = "bn"

    def __post_init__(self) -> None:
        """Validate the architecture configuration.

        Raises:
            ConfigError: If a value is out of range or a name is unknown.
        """
        if self.family not in MODEL_FAMILIES:
            raise ConfigError(
                f"model.family={self.family!r} is unknown; expected one of "
                f"{sorted(MODEL_FAMILIES)}."
            )
        if self.backbone not in BACKBONES:
            raise ConfigError(
                f"model.backbone={self.backbone!r} is unknown; expected one of {sorted(BACKBONES)}."
            )
        if self.d_model < 1:
            raise ConfigError(f"model.d_model must be >= 1, got {self.d_model}.")
        if self.d_model % self.num_heads:
            raise ConfigError(
                f"model.d_model={self.d_model} must be divisible by num_heads={self.num_heads}."
            )
        if self.backbone_output_dim < 1:
            raise ConfigError(
                f"model.backbone_output_dim must be >= 1, got {self.backbone_output_dim}."
            )
        if self.event_head is not None:
            from blinklinmult.train.model import EVENT_HEAD_TYPES

            if self.family == "cnn":
                raise ConfigError(
                    "model.event_head is not available for family='cnn'. A frame-wise "
                    "model trains in stills mode, where each window is collapsed to a "
                    "single timestep, so a temporal event head would have no context to "
                    "read -- and BlinkModel.forward returns from the cnn branch before "
                    "the event head runs, so setting it would silently do nothing. "
                    "Recover events from the per-frame signal with the hysteresis "
                    "extractor instead, which is how the frame-wise arms are scored."
                )

            if self.event_head not in EVENT_HEAD_TYPES:
                raise ConfigError(
                    f"model.event_head={self.event_head!r} is unknown; expected one of "
                    f"{sorted(EVENT_HEAD_TYPES)}, or null to keep the single shared head."
                )
            if self.event_head_hidden_dim < 1:
                raise ConfigError(
                    f"model.event_head_hidden_dim must be >= 1, got {self.event_head_hidden_dim}."
                )
        if self.encoder_freeze and not self.encoder_weights:
            raise ConfigError(
                "model.encoder_freeze=True needs model.encoder_weights: freezing an "
                "encoder that was never given trained weights pins it to ImageNet "
                "features that have never seen an eye, which is not an arm worth "
                "running. Point encoder_weights at a frame-wise checkpoint, or leave "
                "encoder_freeze off."
            )
        if self.backbone_freeze and not self.backbone_pretrained:
            raise ConfigError(
                "model.backbone_freeze=true with backbone_pretrained=false freezes a "
                "randomly-initialised backbone, which can never learn."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            ModelConfig: The validated config.

        Raises:
            ConfigError: If a retired linmult 1.x key is present.
        """
        legacy = {
            "input_modality_channels": "(derived from the backbone and dataset)",
            "projected_modality_dim": "d_model",
            "number_of_layers": "cmt_num_layers",
            "add_projection_fusion": "add_module_ffn_fusion",
            "aggregation": "(heads are sequence-level; no time reducer)",
            "input_dim": "(derived from the dataset's feature_dim)",
            "output_dim": "(derived from the task's targets)",
            "n_heads": "num_heads",
            "n_layers": "cmt_num_layers",
            "dropout_qkv": "dropout_attention",
            "weights": "(load a checkpoint with --resume instead)",
        }
        found = sorted(set(data) & set(legacy))
        if found:
            renames = ", ".join(f"{k!r} -> {legacy[k]}" for k in found)
            raise ConfigError(
                f"model config uses blinklinmult/linmult 1.x keys that no longer exist: "
                f"{renames}. Rewrite the config against the 2.x API."
            )
        _reject_unknown(cls, data, "model config")
        return cls(**data)


@dataclass(frozen=True)
class MLflowConfig:
    """MLflow tracking configuration.

    Args:
        tracking_uri (str): Backend store. A **SQLite database, not a
            directory**: MLflow 3.x puts the filesystem store in maintenance
            mode and raises rather than using it. Relative by default so no
            absolute local path leaks into a committed config.
        experiment_name (str): MLflow experiment to log under.
        run_name (str | None): Run name; MLflow generates one when ``None``.
        log_model (bool): Log the best checkpoint as an MLflow artifact.
    """

    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "blink"
    run_name: str | None = None
    log_model: bool = True

    def __post_init__(self) -> None:
        """Reject the retired filesystem store.

        Raises:
            ConfigError: If the tracking URI is a bare directory path.
        """
        if "://" not in self.tracking_uri:
            raise ConfigError(
                f"mlflow.tracking_uri={self.tracking_uri!r} looks like a filesystem "
                "store. MLflow 3.x has retired the file store and raises on it; use a "
                "database URI such as 'sqlite:///mlflow.db'."
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MLflowConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            MLflowConfig: The validated config.
        """
        _reject_unknown(cls, data, "mlflow config")
        return cls(**data)


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Early-stopping configuration.

    Args:
        enabled (bool): Whether to stop early at all.
        monitor (str): Metric to watch.
        mode (str): ``"max"`` or ``"min"``.
        patience (int): Epochs without improvement before stopping.
        min_delta (float): Smallest change counted as an improvement.
    """

    enabled: bool = True
    monitor: str = "valid/mean_f1"
    mode: str = "max"
    patience: int = 10
    min_delta: float = 0.0

    def __post_init__(self) -> None:
        """Validate the early-stopping configuration.

        Raises:
            ConfigError: If ``mode`` is not ``"min"`` or ``"max"``.
        """
        if self.mode not in {"min", "max"}:
            raise ConfigError(f"early_stopping.mode must be 'min' or 'max', got {self.mode!r}.")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EarlyStoppingConfig:
        """Build from a plain dict.

        Args:
            data (dict): Config values.

        Returns:
            EarlyStoppingConfig: The validated config.
        """
        _reject_unknown(cls, data, "early_stopping config")
        return cls(**data)


@dataclass(frozen=True)
class TrainConfig:
    """Training-loop configuration.

    Args:
        task (str): ``blink_presence``, ``eye_state``, or ``joint``.
        targets (list[str]): Target keys to supervise. Defaults to the task's.
        eval_targets (list[str]): Targets **scored but never trained**. The
            model grows no head for these and they reach no loss; the existing
            prediction is measured against a second annotation. This is how a
            frame-wise closure model is benchmarked on blink events, which is
            what the video corpora annotate.
        loss (str): Loss name.
        loss_kwargs (dict): Loss-specific arguments (e.g. focal ``gamma``).
        event_carrier_only (bool): In the event report, keep only the largest
            corpus and drop the rest as carriers. Correct when a single corpus
            is being scored through a carrier that supplies the head; **wrong
            on a joint split**, where it would discard real corpora because one
            happens to be larger. Off by default; the per-corpus eval scripts
            set it.
        occlusion_yaw (float | None): Absolute head yaw, in degrees, beyond
            which the far eye is treated as self-occluded and masked out of both
            loss and metrics. ``None`` disables the rule.

            Applied in the dataloader rather than baked in at build time, so the
            threshold stays sweepable without re-preprocessing. Only MPEblink is
            materially affected -- 35.7% of its frames sit beyond 45 degrees --
            while RN15, RN30 and TalkingFace are near-frontal and lose almost
            nothing. The right value is a question for inspection, 30 and 45
            both being defensible, which is why there is no default.
        smoothness_weight (float): Weight on
            :func:`~blinklinmult.train.losses.temporal_smoothness`, which
            penalises frame-to-frame chatter in the predicted signal. A blink is
            continuous motion, and a jagged signal is what makes interval
            extraction brittle -- every spurious crossing of the operating point
            becomes a spurious event boundary. Needs no annotation, so it
            applies to every corpus. ``0.0`` disables it, which is the default:
            it changes what the model optimises and so must be opted into.
        smoothness_margin (float): Frame-to-frame jumps at or below this are
            free, so a genuinely fast closure is not penalised.
        duration_weight (float): Weight on
            :func:`~blinklinmult.train.losses.duration_prior`, which penalises
            predicted closures far outside a plausible blink duration
            (~100-400 ms, about 3-12 frames at 30 fps). Also annotation-free.
            ``0.0`` disables it.
        task_weights (dict[str, float]): Per-target loss weight for a joint run.
            Absent targets default to ``1.0``.
        max_epochs (int): Training epochs.
        accelerator (str): Lightning accelerator.
        devices (list[int] | int | str): Lightning devices spec.
        precision (str): Lightning precision.
        seed (int): Global seed.
        deterministic (bool): Request deterministic kernels.
        output_dir (str): Root for run outputs.
        limit_batches (int | float | None): Cap the batches per epoch in every
            stage, for a fast but *complete* run. An integer is a batch count,
            a float in ``(0, 1]`` a fraction, ``None`` the whole split.
        limit_test_batches (int | float | None): Cap the **test** split only,
            leaving train and validation whole. The inverse of
            ``limit_fit_batches``, and what a threshold-fitting pass needs: it
            wants the full validation split to fit on and no test pass at all.
            Same units as ``limit_batches``.
        limit_fit_batches (int | float | None): Cap train and validation only,
            leaving **test at full size**. This is the knob for cheap
            diagnostic sweeps: capping test as well would score each arm on a
            different subset and make the arms incomparable, which defeats the
            purpose of running them. Same units as ``limit_batches``.

            Unlike ``--fast-dev-run``, this keeps the logger, the callbacks and
            the checkpointing, so every metric a full run produces is written
            and reaches MLflow — ``fast_dev_run`` replaces the logger with
            Lightning's ``DummyLogger`` and nothing is recorded at all. Use this
            when the point is to *inspect the metrics* quickly rather than to
            prove the code path runs.
        optimizer (OptimConfig): Optimizer settings.
        early_stopping (EarlyStoppingConfig): Early-stopping settings.
        mlflow (MLflowConfig): Tracking settings.
    """

    task: str = "joint"
    targets: list[str] = field(default_factory=list)
    eval_targets: list[str] = field(default_factory=list)
    loss: str = "bce"
    loss_kwargs: dict[str, Any] = field(default_factory=dict)
    occlusion_yaw: float | None = None
    event_carrier_only: bool = False
    smoothness_weight: float = 0.0
    smoothness_margin: float = 0.0
    duration_weight: float = 0.0
    task_weights: dict[str, float] = field(default_factory=dict)
    max_epochs: int = 30
    accelerator: str = "auto"
    devices: list[int] | int | str = "auto"
    precision: str = "32-true"
    seed: int = 42
    deterministic: bool = False
    output_dir: str = "results"
    limit_batches: int | float | None = None
    limit_fit_batches: int | float | None = None
    limit_test_batches: int | float | None = None
    optimizer: OptimConfig = field(default_factory=OptimConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)

    def __post_init__(self) -> None:
        """Validate the training configuration.

        Raises:
            ConfigError: If the task, targets, loss, or weights are invalid.
        """
        if self.task not in TASKS:
            raise ConfigError(f"train.task={self.task!r} not in {sorted(TASKS)}.")
        if self.loss not in LOSSES:
            raise ConfigError(f"train.loss={self.loss!r} not in {sorted(LOSSES)}.")
        if self.max_epochs < 1:
            raise ConfigError(f"train.max_epochs must be >= 1, got {self.max_epochs}.")
        if self.precision not in PRECISIONS:
            raise ConfigError(f"train.precision={self.precision!r} not in {sorted(PRECISIONS)}.")
        if self.limit_batches is not None:
            # Lightning reads an int as a count and a float as a fraction, so a
            # 0 or a negative silently means "no batches" rather than failing.
            positive = self.limit_batches > 0
            fraction_in_range = not isinstance(self.limit_batches, float) or (
                self.limit_batches <= 1.0
            )
            if not positive or not fraction_in_range:
                raise ConfigError(
                    f"train.limit_batches={self.limit_batches} is not usable: give a "
                    "positive integer for a batch count, or a float in (0, 1] for a "
                    "fraction of each split."
                )

        if not self.targets:
            object.__setattr__(self, "targets", list(TASK_TARGETS[self.task]))

        unknown = sorted(set(self.targets) - set(TARGET_KEYS))
        if unknown:
            raise ConfigError(
                f"train.targets contains unknown keys {unknown}. Available: {sorted(TARGET_KEYS)}."
            )

        expected = set(TASK_TARGETS[self.task])
        if set(self.targets) != expected:
            raise ConfigError(
                f"train.task={self.task!r} supervises {sorted(expected)}, but "
                f"targets={self.targets}. Change the task, or drop the explicit targets "
                "and let the task decide."
            )

        unknown_weights = sorted(set(self.task_weights) - set(self.targets))
        if unknown_weights:
            raise ConfigError(
                f"train.task_weights names targets {unknown_weights} that this task does "
                f"not supervise. Supervised targets: {self.targets}."
            )
        negative = sorted(k for k, v in self.task_weights.items() if v < 0)
        if negative:
            raise ConfigError(f"train.task_weights must be >= 0; negative for {negative}.")

    def weight_for(self, target: str) -> float:
        """Loss weight of one target.

        Args:
            target (str): Target key.

        Returns:
            float: The configured weight, or ``1.0``.
        """
        return float(self.task_weights.get(target, 1.0))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        """Build from a plain dict, including nested sections.

        Args:
            data (dict): Config values.

        Returns:
            TrainConfig: The validated config.
        """
        data = dict(data)
        _reject_unknown(cls, data, "train config")
        if "optimizer" in data:
            data["optimizer"] = OptimConfig.from_dict(data["optimizer"])
        if "early_stopping" in data:
            data["early_stopping"] = EarlyStoppingConfig.from_dict(data["early_stopping"])
        if "mlflow" in data:
            data["mlflow"] = MLflowConfig.from_dict(data["mlflow"])
        return cls(**data)


@dataclass(frozen=True)
class ExperimentConfig:
    """One run: its data, its model, and its training procedure.

    Args:
        data (DataConfig): Dataloader settings.
        model (ModelConfig): Architecture settings.
        train (TrainConfig): Training settings.
    """

    data: DataConfig
    model: ModelConfig
    train: TrainConfig

    def __post_init__(self) -> None:
        """Validate consistency across the three sections.

        Also derives the one data setting that depends on the model: whether the
        handcrafted descriptors are consumed, which decides if geometric
        augmentation is safe. Derived here rather than configured, so the two
        cannot drift apart.

        Raises:
            ConfigError: If the model family and the task disagree.
        """
        object.__setattr__(self.data, "model_reads_features", self.model.family == "linmult")
        # The schema must declare the embedding for it to survive collation, and
        # the schema is built before the model, so the width is copied here.
        object.__setattr__(
            self.data,
            "embedding_dim",
            self.model.backbone_output_dim if self.data.embedding_cache else None,
        )

        if self.data.embedding_cache is not None and not self.model.encoder_freeze:
            raise ConfigError(
                "data.embedding_cache needs model.encoder_freeze=true. An encoder that "
                "is still learning produces different embeddings after every optimiser "
                "step, so a cache built from its initial weights is stale from step one "
                "-- and training would silently continue against embeddings from a model "
                "that no longer exists."
            )

        if self.model.family == "cnn" and BLINK_PRESENCE in self.train.targets:
            raise ConfigError(
                "model.family='cnn' has no sequence model, so it cannot predict blink "
                "presence over a window. Use family='lint' or 'linmult', or train the "
                "eye_state task. To *evaluate* a frame-wise model against blink "
                "events -- recovering them from its per-frame closure signal -- put "
                "blink_presence in train.eval_targets instead, where it is scored "
                "but never trained."
            )

        # The dual guard: a still corpus has no temporal axis to model, so
        # feeding one to a sequence family trains it on degenerate one-frame
        # windows. The frame-wise model is where those corpora belong.
        if self.model.family != "cnn" and not self.data.stills:
            stills = [name for name in self.data.datasets if name in IMAGE_DATASETS]
            if stills:
                raise ConfigError(
                    f"model.family={self.model.family!r} is a sequence model, but "
                    f"data.datasets includes the still-image corpora {stills}. Still "
                    "images have no temporal axis to model. Train the sequence models "
                    "on the video corpora (config/data/all.yaml), and the frame-wise "
                    "model on config/data/stills.yaml."
                )

    @classmethod
    def from_files(
        cls,
        data_path: str | Path,
        model_path: str | Path,
        train_path: str | Path,
        overrides: dict[str, Any] | None = None,
    ) -> ExperimentConfig:
        """Load a run's three config files, applying dotted overrides.

        Args:
            data_path (str | Path): Data config.
            model_path (str | Path): Model config.
            train_path (str | Path): Train config.
            overrides (dict | None): Dotted ``section.key`` overrides, e.g.
                ``{"data.batch_size": 8}``.

        Returns:
            ExperimentConfig: The assembled configuration.
        """
        raw = {
            "data": load_yaml(data_path),
            "model": load_yaml(model_path),
            "train": load_yaml(train_path),
        }

        for dotted, value in (overrides or {}).items():
            apply_override(raw, dotted, value)

        return cls(
            data=DataConfig.from_dict(raw["data"]),
            model=ModelConfig.from_dict(raw["model"]),
            train=TrainConfig.from_dict(raw["train"]),
        )

    def to_flat_dict(self) -> dict[str, Any]:
        """Flatten every field to dotted keys, for ``mlflow.log_params``.

        Returns:
            dict: e.g. ``{"data.batch_size": 32, "train.optimizer.lr": 0.001}``.
        """
        return _flatten(asdict(self))


def apply_override(raw: dict[str, Any], dotted: str, value: Any) -> None:
    """Set a nested config value addressed by a dotted key.

    Args:
        raw (dict): Nested config mapping, modified in place.
        dotted (str): Key path, e.g. ``"train.optimizer.lr"``.
        value (Any): Value to set.
    """
    parts = dotted.split(".")
    node = raw
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_override(text: str) -> tuple[str, Any]:
    """Parse a ``key=value`` CLI override, typing the value via YAML.

    Args:
        text (str): e.g. ``"data.batch_size=8"``.

    Returns:
        tuple[str, Any]: The dotted key and its parsed value.

    Raises:
        ConfigError: If the text has no ``=``.
    """
    if "=" not in text:
        raise ConfigError(f"Override {text!r} must be of the form key=value.")
    key, _, raw = text.partition("=")
    value = yaml.safe_load(raw)

    # PyYAML implements YAML 1.1, whose float pattern requires a decimal point in
    # the mantissa: "1e-4" parses as the string "1e-4", not 0.0001. A learning
    # rate silently arriving as a string is exactly the kind of bug that only
    # surfaces deep inside the optimizer, so recover the intent here.
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            pass

    return key.strip(), value


def _flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Recursively flatten a nested mapping to dotted keys.

    Args:
        data (dict): Nested mapping.
        prefix (str): Accumulated key prefix.

    Returns:
        dict: Flattened mapping.
    """
    flat: dict[str, Any] = {}
    for key, value in data.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{path}."))
        elif isinstance(value, list):
            flat[path] = ",".join(str(v) for v in value)
        else:
            flat[path] = value
    return flat
