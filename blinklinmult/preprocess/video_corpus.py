"""The shared pipeline for the ``.tag``-annotated video corpora.

EyeBlink8, TalkingFace, RN, and HUST-LEBW differ only in where their files sit
and how fast they were recorded — the processing is identical: decode the
frames the annotation refers to, locate both eyes, describe them, cut windows,
write. That pipeline lives here once, and each corpus module supplies a
:class:`VideoCorpusLayout` describing its directory structure.

The 1.x code had this logic copied three times, with the copies having already
diverged (RN's cropped from the raw frames directory, EyeBlink8's from an
extracted one, TalkingFace's from a third path), which is precisely the kind of
drift that makes cross-corpus numbers incomparable.

**One command, one file.** A run decodes, extracts, crops, and writes
``data/processed/<name>/<name>.h5`` in a single pass. There is no staging tree
and no extracted-frame directory: the 1.x pipeline wrote every annotated frame
to disk as a PNG and every sample as a ``.npy``, which cost ~80x the final
artifact and could drift out of step with it.

**Frames are decoded once.** Sliding evaluation windows overlap, so extracting
per window would run the face detector repeatedly over the same pixels — and
that call is essentially the whole runtime. Each recording is swept once into a
small per-frame cache of crops and descriptors, and the windows are assembled
from it; see :mod:`blinklinmult.preprocess.stream`.

**Training and evaluation are sampled identically.** Every split is an
annotation-blind 50%-overlapping sweep, because a deployed model gets no oracle
telling it where the blinks are; the events are recovered afterwards from the
averaged per-frame signal (see :mod:`blinklinmult.train.events`). 1.x sampled
training separately — blink-centred, class-balanced windows — which fit the
model at a 50% blink prior against an evaluation prior nearer 10%, and never
showed it a blink cut by a window edge. See
:mod:`blinklinmult.preprocess.windows`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from blinklinmult.data.schema import DEFAULT_WINDOW_SECONDS, EYE_FEATURE_DIM, DatasetSpec
from blinklinmult.data.writer import H5Writer
from blinklinmult.preprocess.annotation import TagFile, parse_timestamps
from blinklinmult.preprocess.common import (
    PreprocessError,
    assign_splits,
    git_sha,
    list_files,
    progress,
)
from blinklinmult.preprocess.geometry import LEFT, RIGHT
from blinklinmult.preprocess.stream import build_cache
from blinklinmult.preprocess.windows import (
    Window,
    sliding_windows,
    window_frames,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from blinklinmult.preprocess.features import EyeFeatureExtractor
    from blinklinmult.preprocess.stream import FrameCache

logger = logging.getLogger(__name__)
"""Module-level logger."""

PROTOCOL = "sweep"
"""How every split is sampled -- training included.

Annotation-blind, 50%-overlapping windows. The window is the model's receptive
field and nothing more; blinks are recovered afterwards from the averaged
per-frame signal, as in Drutarovsky & Fogelton (2015), Fogelton & Beneš (2016),
and MPEblink (2023). Following it is what makes these numbers comparable with
published ones.
"""


@dataclass(frozen=True)
class VideoCorpusLayout:
    """Where one video corpus keeps its files, and how it is sampled.

    Args:
        name (str): Corpus name; must be a
            :data:`~blinklinmult.data.schema.DATASETS` member.
        raw_dir (Path): Root of the corpus under ``data/raw``.
        processed_dir (Path): Destination under ``data/processed``.
        fps (float): Native frame rate. The window's frame count is derived
            from it, so a 15 fps and a 30 fps corpus span the same real duration.
        tag_glob (str): Glob, relative to ``raw_dir``, matching every ``.tag``
            file in the corpus.
        image_size (int): Side length of the square eye crops to write.
        window_seconds (float): Analysis window duration.
        eval_stride (int | None): Frames between evaluation windows. ``None``
            takes half the window, the 50% overlap the protocol averages over.
        split_ratios (dict[str, float] | None): Group split proportions.
        seed (int): Seed for the group split draw.
        with_features (bool): Extract the 160-d handcrafted descriptors.
        video_id_from_stem (bool): Name a recording after its ``.tag`` file
            rather than its parent directory. TalkingFace keeps its files flat
            in the corpus root, where the parent is the corpus itself.
    """

    name: str
    raw_dir: Path
    processed_dir: Path
    fps: float
    tag_glob: str = "*/*.tag"
    image_size: int = 64
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    eval_stride: int | None = None
    split_ratios: dict[str, float] | None = None
    seed: int = 42
    with_features: bool = True
    video_id_from_stem: bool = False

    @property
    def window(self) -> int:
        """Window length in frames, derived from this corpus's rate.

        Returns:
            int: ``round(fps * window_seconds)``, at least 1.
        """
        return window_frames(self.fps, self.window_seconds)

    @property
    def stride(self) -> int:
        """Frames between evaluation windows.

        Returns:
            int: Half the window unless overridden — derived, so it stays a 50%
            overlap when the rate or the window duration changes.
        """
        return self.eval_stride if self.eval_stride is not None else max(1, self.window // 2)

    @property
    def h5_path(self) -> Path:
        """The single artifact this corpus builds.

        Returns:
            Path: ``data/processed/<name>/<name>.h5``.
        """
        return self.processed_dir / f"{self.name}.h5"

    def spec(self) -> DatasetSpec:
        """The corpus declaration this layout writes under.

        Returns:
            DatasetSpec: Shapes and annotated targets.
        """
        return DatasetSpec(
            name=self.name,
            fps=self.fps,
            window_seconds=self.window_seconds,
            image_size=self.image_size,
            feature_dim=EYE_FEATURE_DIM if self.with_features else None,
            has_blink_presence=True,
            has_eye_state=True,
            # Pose comes from the same extractor the descriptors do, so it is
            # available exactly when features are.
            has_head_pose=self.with_features,
            has_blink_ids=True,
            quality_signals=(
                ("eye_blur", "eye_exposure", "eye_jitter") if self.with_features else ()
            ),
        )

    def timestamp_path(self, tag_path: Path) -> Path:
        """Path of a recording's ``frame_id timestamp`` file.

        Args:
            tag_path (Path): The recording's ``.tag`` file.

        Returns:
            Path: The companion ``.txt`` file.
        """
        return tag_path.with_suffix(".txt")

    def video_path(self, tag_path: Path) -> Path | None:
        """Path of a recording's video file.

        Args:
            tag_path (Path): The recording's ``.tag`` file.

        Returns:
            Path | None: The first matching video, or ``None`` if none exists.
        """
        for suffix in (".avi", ".mp4", ".mov", ".mkv"):
            candidate = tag_path.with_suffix(suffix)
            if candidate.is_file():
                return candidate
        return None


def recording_id(tag_path: Path, layout: VideoCorpusLayout) -> str:
    """Name one recording.

    Args:
        tag_path (Path): The recording's ``.tag`` file.
        layout (VideoCorpusLayout): The corpus layout.

    Returns:
        str: The identifier, which becomes every sample key's first field.
    """
    return tag_path.stem if layout.video_id_from_stem else tag_path.parent.name


def find_tag_files(layout: VideoCorpusLayout) -> list[Path]:
    """Locate every annotated recording in a corpus.

    Args:
        layout (VideoCorpusLayout): The corpus layout.

    Returns:
        list[Path]: Sorted ``.tag`` paths.

    Raises:
        PreprocessError: If the raw directory is absent or holds no annotation.
    """
    if not layout.raw_dir.is_dir():
        raise PreprocessError(
            f"{layout.name}: raw data not found at {layout.raw_dir}. See the README's "
            "Data section for how to obtain it."
        )

    paths = list_files(layout.raw_dir, layout.tag_glob)
    if not paths:
        raise PreprocessError(
            f"{layout.name}: no .tag files match {layout.tag_glob!r} under "
            f"{layout.raw_dir}. Check the corpus layout."
        )

    logger.info(f"{layout.name}: found {len(paths)} annotated recordings.")
    return paths


def load_annotation(layout: VideoCorpusLayout, tag_path: Path) -> tuple[TagFile, dict[int, float]]:
    """Parse a recording's annotation and its timestamps.

    Args:
        layout (VideoCorpusLayout): The corpus layout.
        tag_path (Path): The recording's ``.tag`` file.

    Returns:
        tuple[TagFile, dict[int, float]]: The annotation and frame timestamps.

    Raises:
        PreprocessError: If the timestamp file is missing.
    """
    timestamp_path = layout.timestamp_path(tag_path)
    if not timestamp_path.is_file():
        raise PreprocessError(
            f"{tag_path.parent.name}: no timestamp file at {timestamp_path}. Each "
            "recording needs its companion 'frame_id timestamp' .txt."
        )

    tag = TagFile.from_path(tag_path)
    logger.info(f"{tag.video_id}: {tag.summary()}")
    return tag, parse_timestamps(timestamp_path)


def windows_for(layout: VideoCorpusLayout, tag: TagFile) -> list[Window]:
    """Sweep a recording into windows.

    **The same rule for every split**, which is why this takes no ``subset``:
    there is no branch left that could make training and evaluation diverge.
    See :mod:`blinklinmult.preprocess.windows` for what the 1.x blink-centred
    training sampler did and why it was removed.

    Args:
        layout (VideoCorpusLayout): The corpus layout.
        tag (TagFile): The recording's annotation.

    Returns:
        list[Window]: Windows covering the recording, in order.
    """
    return sliding_windows(tag, window=layout.window, stride=layout.stride)


def write_windows(
    writer: H5Writer,
    layout: VideoCorpusLayout,
    cache: FrameCache,
    tag: TagFile,
    subset: str,
    video_id: str,
) -> int:
    """Write one recording's windows, two samples each.

    The two eyes of a window share a ``frame_group``, which is what lets
    frame-level scoring recombine their predictions into one decision per frame.

    Args:
        writer (H5Writer): Destination.
        layout (VideoCorpusLayout): The corpus layout.
        cache (FrameCache): This recording's crops and descriptors.
        tag (TagFile): The recording's annotation.
        subset (str): Split this recording belongs to.
        video_id (str): Source recording.

    Returns:
        int: Samples written.
    """
    written = 0
    for window in windows_for(layout, tag):
        frame_ids = [int(value) for value in window.frame_ids]

        for eye_side in (LEFT, RIGHT):
            images, image_mask = cache.window_images(frame_ids, eye_side)
            features, feature_mask = (
                cache.window_features(frame_ids, eye_side) if layout.with_features else (None, None)
            )
            writer.add(
                subset=subset,
                video_id=video_id,
                frame_group=window.frame_group,
                eye_side=eye_side,
                eye_images=images,
                eye_image_mask=image_mask,
                blink_presence=window.labels,
                eye_state=window.eye_state_for(eye_side),
                # The annotation marks frames where the eye was not visible;
                # those carry no usable eye-state label, so they are masked
                # rather than trained against a value the annotator could not
                # see. Blink presence is a property of the window, not of one
                # eye's visibility, so it keeps full supervision.
                eye_state_mask=window.validity_for(eye_side),
                eye_features=features,
                eye_feature_mask=feature_mask,
                # A sweep window may hold two blinks -- consecutive rapid
                # blinking is real -- so the first is recorded rather than the
                # sample being refused. The full set stays in the annotation.
                blink_id=window.first_blink_id,
                head_pose=cache.window_pose(frame_ids),
                quality_signals=cache.window_quality(frame_ids, eye_side),
                # Per-frame event ids, which the window-level `blink_id` cannot
                # provide: it cannot tell one long blink from two adjacent ones
                # inside the same window.
                blink_ids=window.blink_ids,
            )
            written += 1
    return written


def process(
    layout: VideoCorpusLayout,
    limit: int | None = None,
    device_id: int | None = None,
    split_of: Callable[[Path], str] | None = None,
    name_of: Callable[[Path], str] | None = None,
) -> Path:
    """Build one video corpus's HDF5 file in a single pass.

    Args:
        layout (VideoCorpusLayout): The corpus layout.
        limit (int | None): Process only the first N recordings, for smoke runs.
        device_id (int | None): GPU index for the extractors; ``None`` is CPU.
        split_of (Callable[[Path], str] | None): Reads a recording's split from
            its ``.tag`` path. ``None`` assigns splits by hashing recording ids.
            RN ships its own participant-wise division and passes a reader for
            it, because re-deriving those splits would make published RN numbers
            incomparable with this project's.
        name_of (Callable[[Path], str] | None): Names a recording from its
            ``.tag`` path. ``None`` uses :func:`recording_id`. RN restarts its
            numbering inside every split directory, so it supplies a rule that
            keeps ``train/1`` and ``test/1`` apart.

    Returns:
        Path: The written HDF5 file.

    Raises:
        PreprocessError: If the corpus cannot be processed.
    """
    tag_paths = find_tag_files(layout)
    if limit is not None:
        tag_paths = tag_paths[:limit]

    # Split by recording, never by window: windows from one recording overlap
    # and share a subject, so splitting them individually leaks the test set.
    identify = name_of or partial(recording_id, layout=layout)

    # Every recording must have its own name. A corpus that restarts its
    # numbering per split -- RN does -- would otherwise give two recordings the
    # same id, and their sample keys would collide inside the HDF5.
    names = [identify(path) for path in tag_paths]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise PreprocessError(
            f"{layout.name}: recording ids are not unique: {duplicates[:5]}. Two "
            "recordings sharing a name would collide as HDF5 group keys. Pass "
            "name_of= to disambiguate them."
        )

    splits = (
        {identify(path): split_of(path) for path in tag_paths}
        if split_of is not None
        else assign_splits(
            (identify(path) for path in tag_paths),
            ratios=layout.split_ratios,
            salt=layout.name,
        )
    )

    extractor = _build_extractor(layout, device_id)
    spec = layout.spec()

    logger.info(
        f"{layout.name}: {layout.window}-frame windows "
        f"({layout.window_seconds:g}s at {layout.fps:g} fps), "
        f"evaluation stride {layout.stride}."
    )

    config_path = layout.processed_dir.parents[2] / "config" / "data" / f"{layout.name}.yaml"
    config_text = config_path.read_text() if config_path.is_file() else ""

    with H5Writer(
        spec,
        layout.h5_path,
        config_yaml=config_text,
        git_sha=git_sha(layout.processed_dir.parents[2]),
        extra_attrs={
            # One protocol, recorded once: every split is swept the same way,
            # so a separate train_protocol attribute would only invite drift.
            "protocol": PROTOCOL,
            "stride": layout.stride,
        },
    ) as writer:
        for tag_path in progress(tag_paths, f"{layout.name} recordings"):
            video_id = identify(tag_path)
            subset = splits[video_id]

            video_path = layout.video_path(tag_path)
            if video_path is None:
                raise PreprocessError(
                    f"{video_id}: no video beside {tag_path}. Expected one of .avi/.mp4/.mov/.mkv."
                )

            tag, timestamps = load_annotation(layout, tag_path)
            cache = build_cache(video_path, tag, timestamps, layout.image_size, extractor)
            written = write_windows(writer, layout, cache, tag, subset, video_id)
            logger.info(f"{video_id} -> {subset}: {written} samples.")

    return layout.h5_path


def _build_extractor(
    layout: VideoCorpusLayout, device_id: int | None
) -> EyeFeatureExtractor | None:
    """Construct the descriptor extractor, once per corpus run.

    Imported lazily: exordium pulls in multi-GB model weights, and a corpus
    built without features should not pay for loading them.

    Args:
        layout (VideoCorpusLayout): The corpus layout.
        device_id (int | None): GPU index, or ``None`` for CPU.

    Returns:
        EyeFeatureExtractor | None: The extractor, or ``None`` when this corpus
        supplies no handcrafted stream.
    """
    if not layout.with_features:
        return None

    from blinklinmult.preprocess.extractors import ExordiumExtractor

    logger.info(f"{layout.name}: loading the exordium extraction stack...")
    return ExordiumExtractor(device_id=device_id)


def add_cli_arguments(parser) -> None:
    """Add the options every video corpus shares.

    One definition so the four corpora cannot drift into slightly different
    flags for the same thing.

    Args:
        parser (argparse.ArgumentParser): The parser to extend.
    """
    from blinklinmult import PROJECT_ROOT

    parser.add_argument("--root", type=Path, default=PROJECT_ROOT, help="Repository root.")
    parser.add_argument("--image-size", type=int, default=64, help="Eye crop side length.")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Analysis window in seconds.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="Frames between evaluation windows; default is half the window.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Process only the first N recordings."
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip the exordium descriptors; the corpus is then image-only.",
    )
    parser.add_argument(
        "--device", type=int, default=None, help="GPU index for extraction; omit for CPU."
    )


def run_cli(
    build_layout,
    description: str,
    split_of: Callable[[Path], str] | None = None,
) -> Path:
    """Parse the shared arguments and build one corpus.

    Args:
        build_layout (Callable): A corpus's ``layout(root, **overrides)``.
        description (str): Help text for the parser.
        split_of (Callable[[Path], str] | None): Reads a recording's split from
            its path, for corpora whose division is given rather than derived.

    Returns:
        Path: The written HDF5 file.
    """
    import argparse

    parser = argparse.ArgumentParser(description=description)
    add_cli_arguments(parser)
    args = parser.parse_args()

    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    return process(
        build_layout(
            args.root,
            image_size=args.image_size,
            window_seconds=args.window_seconds,
            eval_stride=args.stride,
            with_features=not args.no_features,
        ),
        limit=args.limit,
        device_id=args.device,
        split_of=split_of,
    )
