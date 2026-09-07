"""Lightning ``DataModule`` that mixes every blink corpus through OmniLoader.

This is where the six datasets become one training stream. Per split, each
corpus is opened as an :class:`omniloader.HDF5Dataset` over its own file and
paired with the schema generated from its
:class:`~blinklinmult.data.schema.DatasetSpec`; :class:`omniloader.OmniLoader`
takes the union, fills each corpus's gaps with placeholders and all-``False``
masks, and a mixing strategy balances corpora whose sizes differ by orders of
magnitude (MRL-Eye has ~85k stills; TalkingFace is one recording).

**Why not OmniLoader's own ``OmniDataModule``.** That class builds its splits
from an :class:`omniloader.OmniConfig`, which describes datasets by adapter name
and inline schema. Here the schemas are *derived* from the dataset specs the
HDF5 files were built with, so declaring them again in a config would be a
second source of truth that could drift. This module reuses every other piece of
OmniLoader — loader, sampler, strategies, transforms, collate — and only owns
the wiring.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightning as L
import torch
from omniloader import (
    AnnealedTemperatureStrategy,
    FixedWeightStrategy,
    HDF5Dataset,
    OmniLoader,
    ProportionalStrategy,
    RoundRobinStrategy,
    TemperatureStrategy,
    seed_worker,
    unified_collate,
)
from torch.utils.data import DataLoader

from blinklinmult.data.augment import EyeAugmentation
from blinklinmult.data.embeddings import (
    CacheKey,
    EmbeddingCacheError,
    read_shard,
    shard_path,
)
from blinklinmult.data.omni import build_schemas
from blinklinmult.data.schema import EYE_EMBEDDING, SAMPLE_KEY, SUBSETS, DatasetSpec
from blinklinmult.data.stills import StillConfig, StillFramesDataset

if TYPE_CHECKING:
    import numpy as np
    from omniloader import DatasetSchema, MixingStrategy
    from omniloader.loader import SizedDataset

    from blinklinmult.train.config import DataConfig

logger = logging.getLogger(__name__)
"""Module-level logger."""

STRATEGIES = ("proportional", "temperature", "annealed_temperature", "fixed", "round_robin")
"""Mixing strategies selectable from a data config."""

MAX_WORKERS = 8
"""Dataloader worker processes.

**Measured on the real corpus mix, not on one corpus.** An earlier value of 0
came from a genuine measurement on CEW -- a 123 MB corpus where worker startup
cost more than the workers saved, because 64px crops are already decoded in the
file and there is little per-sample work to parallelise. That measurement noted
its own limit: "Larger corpora may invert that trade."

It inverts. Re-measured over the six-corpus frame-wise mix (54 GB, of which
MPEblink is 45 GB), timing steady-state batches per second after a warm-up:

=============  =============  ==============
num_workers    batches/s      vs in-process
=============  =============  ==============
0                     10.87            1.00x
2                     19.63            1.81x
4                     35.14            3.23x
6                     40.18            3.70x
8                     43.18            3.97x
10                    46.65            4.29x
12                    45.08            4.15x
=============  =============  ==============

**Set to 8 rather than the fastest 10.** The machine has 10 logical cores (4
performance, 6 efficiency), so 10 workers leaves nothing for the training
process itself, and 12 is already slower than 8 -- the curve has turned. The
gain from 8 to 10 is 8%, against the risk of starving the main process on a
machine that is also running the model. 8 keeps two cores free and sits within
7% of the peak.

Reproduce with ``uv run python scripts/bench_workers.py``.
"""


class DataModuleError(RuntimeError):
    """Raised when the configured datasets cannot be served."""


class StridedSubset(torch.utils.data.Dataset):
    """An evenly-spaced share of a split, optionally shifting each epoch.

    A **stride** rather than a random draw. Sample ids are sorted by recording
    and frame, so striding spreads the retained windows across every recording
    instead of taking a contiguous prefix -- which on these corpora would keep
    whole recordings and drop others entirely.

    **On the training split the stride's offset advances every epoch**, so a
    run long enough eventually sees the whole corpus. At 10% and 30 epochs the
    model meets every one of MPEblink's 63 758 training windows rather than the
    same 6 376 thirty times, at identical per-epoch cost. The offset wraps, so
    the count per epoch never changes.

    **Validation and test keep a fixed offset.** A validation subset that moved
    between epochs would change the metric's meaning from one epoch to the
    next, and early stopping compares those numbers directly.

    Args:
        dataset (Any): The split to thin.
        keep (int): Windows to retain per epoch.
        shifting (bool): Advance the offset each epoch. Training only.

    Attributes:
        epoch (int): Current epoch, set by :meth:`set_epoch`.
    """

    def __init__(self, dataset: Any, keep: int, shifting: bool):
        self.dataset = dataset
        self.keep = keep
        self.shifting = shifting
        self.total = len(dataset)
        self.step = self.total / keep
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Advance the stride's offset.

        Args:
            epoch (int): The upcoming epoch.
        """
        self.epoch = epoch

    def __len__(self) -> int:
        """Windows retained per epoch.

        Returns:
            int: The fixed per-epoch count.
        """
        return self.keep

    def __getitem__(self, index: int) -> Any:
        """One window from this epoch's slice.

        Args:
            index (int): Position within the retained subset.

        Returns:
            Any: The underlying sample.
        """
        # The offset walks one window per epoch and wraps, so consecutive epochs
        # see disjoint slices until the whole corpus has been covered.
        offset = self.epoch if self.shifting else 0
        return self.dataset[(int(index * self.step) + offset) % self.total]


def _subsample(dataset: Any, fraction: float, name: str, subset: str) -> Any:
    """Keep an evenly-spaced share of a split.

    Args:
        dataset (Any): The split to thin.
        fraction (float): Share to keep, in ``(0, 1]``.
        name (str): Corpus name, for the log line.
        subset (str): Split name, for the log line.

    Returns:
        Any: A :class:`StridedSubset` view, or the dataset unchanged when the
        fraction would keep everything.
    """
    total = len(dataset)
    keep = max(1, int(round(total * fraction)))
    if keep >= total:
        return dataset

    shifting = subset == "train"
    coverage = "shifting each epoch" if shifting else "fixed"
    logger.info(
        f"{name}: {subset!r} subsampled to {keep} of {total} windows "
        f"({fraction:.0%}, stride {total / keep:.1f}, {coverage})."
    )
    return StridedSubset(dataset, keep, shifting)


class CachedEmbeddingDataset:
    """Serves a dataset's samples with a precomputed embedding attached.

    The wrapped dataset still supplies the labels, masks and metadata; only the
    encoder's output is substituted. That keeps every downstream consumer --
    losses, metrics, the event report -- reading exactly the fields it already
    read.

    **Keys are matched, not positions.** A cache row is paired with its sample
    by id, so a corpus rebuilt in a different order, or a subsampled split,
    cannot silently pair the wrong embedding with the wrong labels. A sample
    with no cached row is an error rather than a fallback: encoding it live
    would mix cached and fresh embeddings in one batch, and if the cache were
    stale that mix would be invisible.

    Args:
        dataset (Any): The corpus split.
        shard (dict[str, np.ndarray]): A loaded cache shard.
        name (str): Corpus name, for error messages.

    Raises:
        EmbeddingCacheError: If the shard does not cover the dataset.
    """

    def __init__(self, dataset: Any, shard: dict[str, np.ndarray], name: str):
        self.dataset = dataset
        self._rows = {key: index for index, key in enumerate(shard["keys"])}
        self._embedding = shard["embedding"]
        self.name = name

    def __len__(self) -> int:
        """Number of samples.

        Returns:
            int: Length of the wrapped dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """One sample, with its cached embedding attached.

        Args:
            index (int): Sample index.

        Returns:
            dict[str, Any]: The sample plus :data:`~blinklinmult.data.schema.EYE_EMBEDDING`.

        Raises:
            EmbeddingCacheError: If the sample has no cached row.
        """
        sample = self.dataset[index]
        key = sample.get(SAMPLE_KEY)
        row = self._rows.get(key)
        if row is None:
            raise EmbeddingCacheError(
                f"{self.name}: sample {key!r} has no cached embedding. The cache "
                "describes a different build of this corpus; rebuild it rather than "
                "encoding some samples live and reading others from a stale cache."
            )
        sample[EYE_EMBEDDING] = torch.from_numpy(self._embedding[row].astype("float32"))
        return sample


def pinning_helps() -> bool:
    """Whether pinned host memory does anything on this machine.

    Pinning is a CUDA optimisation: it stages batches in page-locked host memory
    so the copy to the device can run asynchronously. **MPS does not implement
    it.** PyTorch warns once per loader --

        UserWarning: 'pin_memory' argument is set as true but not supported on
        MPS now, device pinned memory won't be used.

    -- and then ignores the flag, so the setting is inert rather than harmful.
    The warning is still worth removing: a config that asks for something the
    hardware cannot do reads as an oversight, and a benchmark log full of
    ignorable warnings is one where a real warning goes unread.

    The configs keep ``pin_memory: true`` because it is correct on a CUDA host
    and these corpora are meant to be portable; this gate turns it off only
    where it would be a no-op.

    Returns:
        bool: ``True`` when a CUDA device is present, ``False`` otherwise.
    """
    return torch.cuda.is_available()


def worker_count(num_workers: int) -> int:
    """Clamp the configured worker count to what the reader supports.

    Args:
        num_workers (int): Worker count from the data config.

    Returns:
        int: The requested count, capped at :data:`MAX_WORKERS`; see it for the
        measurements behind that ceiling.
    """
    if num_workers > MAX_WORKERS:
        logger.warning(
            f"data.num_workers={num_workers} requested, using {MAX_WORKERS}: measured "
            "throughput peaks near 10 workers on this 10-core machine and falls again "
            "by 12, and leaving cores for the training process matters more than the "
            "last 8%."
        )
    return min(num_workers, MAX_WORKERS)


def build_strategy(name: str, sizes: list[int], kwargs: dict | None = None) -> MixingStrategy:
    """Construct a mixing strategy by config name.

    Args:
        name (str): One of :data:`STRATEGIES`.
        sizes (list[int]): Sample count per dataset, in dataset order.
        kwargs (dict | None): Strategy-specific keyword arguments.

    Returns:
        MixingStrategy: The strategy.

    Raises:
        DataModuleError: If the name is unknown.
    """
    classes = {
        "proportional": ProportionalStrategy,
        "temperature": TemperatureStrategy,
        "annealed_temperature": AnnealedTemperatureStrategy,
        "fixed": FixedWeightStrategy,
        "round_robin": RoundRobinStrategy,
    }
    if name not in classes:
        raise DataModuleError(f"Unknown mixing strategy {name!r}; expected {list(STRATEGIES)}.")

    try:
        return classes[name](sizes, **dict(kwargs or {}))
    except TypeError as error:
        # A strategy's knobs are its own. Reporting the mismatch here beats a
        # raw TypeError from deep inside OmniLoader, which does not say which
        # config key was wrong.
        raise DataModuleError(
            f"data.strategy_kwargs does not fit strategy {name!r}: {error}. "
            "Each strategy accepts only its own arguments."
        ) from error


class BlinkDataModule(L.LightningDataModule):
    """Serves the mixed blink corpora to the training loop.

    Args:
        config (DataConfig): Dataloader configuration.
        specs (list[DatasetSpec]): Corpora to include, in config order.
        root (Path): Repository root that dataset paths resolve against.

    Raises:
        DataModuleError: If no dataset is configured.
    """

    def __init__(self, config: DataConfig, specs: list[DatasetSpec], root: Path):
        super().__init__()
        if not specs:
            raise DataModuleError("At least one dataset must be configured.")

        self.config = config
        self.specs = list(specs)
        self.root = Path(root)

        # Set by the training entry point once the encoder exists, since the
        # cache is keyed by the encoder's own weights and the datamodule is
        # built before the model. `None` -- the default -- encodes live.
        self.embedding_key: CacheKey | None = None

        # Collected as splits are opened, so `set_epoch` can advance them
        # without knowing how OmniLoader composes its datasets.
        self._strided_subsets: list[StridedSubset] = []

        # In still mode every sample is one frame, whatever window the corpora
        # were built with, so the run's shared shape is fixed rather than
        # derived from the corpora's rates.
        schemas, time_dim, image_size, feature_dim = build_schemas(
            self.specs,
            time_dim=1 if config.stills else config.time_dim,
            image_size=config.image_size,
            stills=config.stills,
            embedding_dim=config.embedding_dim,
        )
        self.schemas: list[DatasetSchema] = schemas
        self.time_dim = time_dim
        self.image_size = image_size
        self.feature_dim = feature_dim

        self.loaders: dict[str, OmniLoader] = {}
        self._sampler = None

    def h5_path(self, spec: DatasetSpec) -> Path:
        """Path of one corpus's built HDF5 file.

        Args:
            spec (DatasetSpec): The corpus.

        Returns:
            Path: Absolute path to the file.
        """
        return self.root / "data" / "processed" / spec.name / f"{spec.name}.h5"

    def _open(self, spec: DatasetSpec, subset: str) -> SizedDataset | None:
        """Open one corpus's split, or report why it cannot be opened.

        A corpus legitimately need not populate every split — a single-recording
        corpus such as TalkingFace is held out entirely for testing — so an
        absent or empty split is skipped with a log line rather than an error.

        Args:
            spec (DatasetSpec): The corpus.
            subset (str): Split name.

        Returns:
            SizedDataset | None: The dataset, or ``None`` when this corpus has no
            samples in this split. In still mode a video corpus is wrapped in a
            :class:`~blinklinmult.data.stills.StillFramesDataset`.

        Raises:
            DataModuleError: If the corpus's file has not been built at all.
        """
        path = self.h5_path(spec)
        if not path.is_file():
            raise DataModuleError(
                f"{spec.name}: dataset not built at {path}. Build it first: "
                f"make preprocess-{spec.name.replace('_', '-')}"
            )

        try:
            dataset = HDF5Dataset(
                path,
                subset=subset,
                cache_size=self.config.cache_size,
                preload=self.config.preload,
            )
        except KeyError:
            logger.info(f"{spec.name}: no {subset!r} split in {path}; skipping.")
            return None

        if len(dataset) == 0:
            logger.info(f"{spec.name}: {subset!r} split is empty; skipping.")
            return None

        # Per-corpus subsampling, on the fitting splits only. `test` is never
        # thinned: scoring each arm on a different subset would make the arms
        # incomparable, which is the whole reason for running them.
        fraction = self.config.fit_fractions.get(spec.name, 1.0)
        if subset != "test" and fraction < 1.0:
            dataset = _subsample(dataset, fraction, spec.name, subset)
            if isinstance(dataset, StridedSubset) and dataset.shifting:
                self._strided_subsets.append(dataset)

        if self.embedding_key is not None:
            dataset = self._with_embeddings(dataset, spec.name, subset)

        if self.config.stills and spec.is_video:
            # Balance the training split only: validation and test must be
            # scored on the class distribution the model actually meets.
            return StillFramesDataset(
                dataset,
                StillConfig(
                    # Training thins the open frames and balances them against
                    # the closed ones; evaluation must not. The event protocol
                    # reassembles a *continuous* per-frame signal and thresholds
                    # it into intervals, so a strided or balanced eval split
                    # leaves a scatter of isolated frames -- every retained
                    # closed frame becomes its own one-frame "event" that
                    # matches trivially under every IoU criterion. Measured on a
                    # strided eval split: median 1 frame covered per recording,
                    # median blink length 1 frame, and all four criteria
                    # reporting an identical F1 at 100% recall.
                    stride=self.config.still_stride if subset == "train" else 1,
                    open_to_closed=self.config.still_open_to_closed,
                    seed=self.config.seed,
                ),
                name=spec.name,
                balance=subset == "train",
            )

        return dataset

    def _with_embeddings(self, dataset: Any, name: str, subset: str) -> Any:
        """Attach cached embeddings to a split, or leave it to encode live.

        A **missing** corpus falls back to live encoding with a log line, so
        adding a corpus does not invalidate the rest of the cache. A *present
        but mismatched* one raises: it describes a different encoder or input
        size, and reusing it would train against vectors from another model.

        Args:
            dataset (Any): The split.
            name (str): Corpus name.
            subset (str): Split name.

        Returns:
            Any: The split, wrapped when a cache shard covers it.
        """
        # Both are non-None here: `_open` only calls this when `embedding_key`
        # is set, and the key is only set when a cache directory is configured.
        root = self.config.embedding_cache
        key = self.embedding_key
        if root is None or key is None:
            return dataset

        path = shard_path(root, key, name, subset)
        if not path.is_file():
            logger.info(f"{name}: no cached embeddings for {subset!r}; encoding live.")
            return dataset

        shard = read_shard(path, key)
        logger.info(f"{name}: {subset!r} reading {len(shard['keys'])} cached embeddings.")
        return CachedEmbeddingDataset(dataset, shard, name)

    def _build_split(self, subset: str) -> OmniLoader | None:
        """Assemble the OmniLoader for one split.

        Args:
            subset (str): Split name.

        Returns:
            OmniLoader | None: The loader, or ``None`` when no corpus
            contributes to this split.
        """
        # Eval-only corpora contribute to the test split alone. Letting them
        # into train or valid would defeat the point: the question is how the
        # model transfers to data it never saw, and a corpus that leaked into
        # validation would also be steering early stopping and the operating
        # point.
        eval_only = set(self.config.eval_datasets) - set(self.config.datasets)

        datasets = []
        schemas = []
        for spec, schema in zip(self.specs, self.schemas, strict=True):
            if spec.name in eval_only and subset != "test":
                continue
            dataset = self._open(spec, subset)
            if dataset is not None:
                datasets.append(dataset)
                schemas.append(schema)

        if not datasets:
            logger.warning(f"No dataset contributes to the {subset!r} split.")
            return None

        training = subset == "train"
        loader = OmniLoader(
            datasets,
            schemas,
            training=training,
            seed=self.config.seed,
            # OmniLoader gates this on `training`, so validation and test see
            # the crops as built. It seeds the transform from
            # (seed, epoch, index), which is why `EpochPropagator` advances the
            # epoch: without it every epoch would augment identically.
            transform=self._augmentation(),
        )
        logger.info(
            f"{subset}: {len(loader)} samples from "
            f"{[s.name for s, d in zip(self.specs, datasets, strict=False)]}"
        )
        return loader

    def _augmentation(self) -> EyeAugmentation | None:
        """The training-time transform, or ``None`` when augmentation is off.

        **Geometry is dropped when the descriptors are actually consumed.** 152
        of the 160 dimensions are pixel-space landmark coordinates -- the eye
        region, the iris, the eyelid-pupil distances -- describing the crop as
        it was built. Rotating or shifting the image leaves them describing the
        *unrotated* eye, so the model would receive an image saying one thing
        and a feature vector saying another.

        Recomputing them per sample is not an option: running the iris
        landmarker in the dataloader would cost what preprocessing costs, per
        epoch. Transforming them analytically is possible -- a landmark rotates
        by a 2x2 multiply, and roll shifts by the angle -- and is the intended
        next step *if* geometric augmentation proves to matter for the
        image-only model. Until that is measured and the analytic transform is
        verified against recomputed descriptors, dropping geometry is the honest
        choice: a silently stale descriptor is worse than a weaker augmentation.

        Returns:
            EyeAugmentation | None: The transform OmniLoader applies per sample.
        """
        config = self.config.augment
        if config is None or not config.enabled:
            return None

        # Keyed on whether the *model* reads the descriptors, not on whether the
        # corpora supply them. RN15 and RN30 carry a 160-d vector, but BlinkCNN
        # takes eye crops alone -- gating on the corpora would drop rotation from
        # a run that could never be harmed by it.
        if config.geometric and self.config.model_reads_features:
            logger.info(
                "Augmentation: geometry disabled because this model reads handcrafted "
                "descriptors, whose landmark coordinates would no longer match the "
                "augmented crop. Brightness, contrast and blur still apply."
            )
            config = config.without_geometry()
            if not config.enabled:
                return None

        return EyeAugmentation(config)

    def setup(self, stage: str | None = None) -> None:
        """Open every split's datasets and build the training sampler.

        Args:
            stage (str | None): Lightning stage hint. Splits are built for
                whichever stages need them; ``None`` builds all.
        """
        wanted = {
            "fit": ("train", "valid"),
            "validate": ("valid",),
            "test": ("test",),
            "predict": ("test",),
        }.get(stage or "", SUBSETS)

        for subset in wanted:
            if subset not in self.loaders:
                loader = self._build_split(subset)
                if loader is not None:
                    self.loaders[subset] = loader

        train = self.loaders.get("train")
        if train is not None and self._sampler is None:
            strategy = build_strategy(
                self.config.strategy, train.dataset_sizes, self.config.strategy_kwargs
            )
            # OmniSampler infers rank and world size from torch.distributed, so
            # this is already DDP-correct without a distributed-specific branch.
            self._sampler = train.make_sampler(strategy)

    def set_epoch(self, epoch: int) -> None:
        """Advance the mixing draw and per-sample augmentation for a new epoch.

        Args:
            epoch (int): The upcoming epoch number.
        """
        for loader in self.loaders.values():
            loader.set_epoch(epoch)
        if self._sampler is not None:
            self._sampler.set_epoch(epoch)

        # `StridedSubset` shifts its offset per epoch so a subsampled corpus is
        # eventually seen in full. The subsets are nested inside the training
        # loader rather than held here, so they are reached by walking it --
        # OmniLoader's own `set_epoch` does not know about them.
        for subset in self._strided_subsets:
            subset.set_epoch(epoch)

    def _dataloader(self, subset: str, shuffle: bool) -> DataLoader:
        """Wrap one split's loader in a configured ``DataLoader``.

        Args:
            subset (str): Split name.
            shuffle (bool): Whether to shuffle. Ignored on the training split,
                which draws through the mixing sampler instead.

        Returns:
            DataLoader: The loader.

        Raises:
            DataModuleError: If the split was never set up.
        """
        loader = self.loaders.get(subset)
        if loader is None:
            raise DataModuleError(
                f"No {subset!r} split available. Call setup() first, and check that at "
                "least one configured dataset populates it."
            )

        sampler = self._sampler if subset == "train" else None
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)

        workers = worker_count(self.config.num_workers)
        return DataLoader(
            loader,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=workers,
            pin_memory=self.config.pin_memory and pinning_helps(),
            persistent_workers=(self.config.persistent_workers and workers > 0),
            prefetch_factor=(self.config.prefetch_factor if workers > 0 else None),
            drop_last=subset == "train",
            collate_fn=unified_collate,
            worker_init_fn=seed_worker,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        """Training loader, drawing through the mixing strategy.

        Returns:
            DataLoader: The loader.
        """
        return self._dataloader("train", shuffle=False)

    def val_dataloader(self) -> DataLoader:
        """Validation loader.

        Every corpus is iterated in full and in order, so a validation score is
        not itself a function of the mixing draw.

        Returns:
            DataLoader: The loader.
        """
        return self._dataloader("valid", shuffle=False)

    def test_dataloader(self) -> DataLoader:
        """Test loader.

        Returns:
            DataLoader: The loader.
        """
        return self._dataloader("test", shuffle=False)
