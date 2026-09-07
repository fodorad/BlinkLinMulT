"""The common format every blink corpus is brought into.

Seven corpora annotate eye blinks in incompatible ways: EyeBlink8, TalkingFace,
and the two RN rates ship frame-level ``.tag`` files over continuous video; CEW
and MRL-Eye are collections of single still images with one label each;
HUST-LEBW is a set of short pre-cut clips. This module declares the one
representation all of them are mapped onto, and from which the per-dataset HDF5
files are written.

**One sample is one eye.** A sample carries a window of ``T`` frames of *a
single* eye — its crops, its optional handcrafted descriptors, and its labels —
with an :data:`EYE_SIDE` field saying which. That follows the annotation: the
``.tag`` format marks closure per eye (``LE_FC``, ``RE_FC``), so an eye-wise
sample trains on the actual ground truth rather than on a label manufactured by
collapsing two. It also makes the awkward corpora honest — MRL-Eye supplies one
eye and does not say which, and CEW labels the *face* — instead of duplicating a
crop into a second slot that does not exist.

Frame-level evaluation is recovered by aggregating the two eyes' predictions per
``(video_id, frame_id)``; see :mod:`blinklinmult.train.metrics`. Training is
eye-wise, reporting is frame-wise, and the same aggregation is applied to
validation so model selection optimises the number that gets reported.

**A still-image corpus is the ``T = 1`` case**, which is what lets one schema —
and therefore one joint dataloader — span video and image datasets.

**Windows are declared in seconds, not frames.** The corpora run at 15 and 30
fps, and in-the-wild video is worse. A fixed frame count would make a 15 fps
window span twice the wall-clock time of a 30 fps one, so the model would have
to learn two different notions of how fast a blink is. Each corpus therefore
derives its own ``time_dim`` from :attr:`DatasetSpec.fps`, every corpus spans the
same real duration, and OmniLoader pads the shorter ones with an all-``False``
mask. Nothing is resampled: interpolating a 15 fps clip up to 30 fps would
fabricate frames that were never captured, and a blink is only a handful of
frames long.

**Targets and their masks.** The two tasks do not both have supervision in every
corpus: HUST-LEBW annotates blink events per clip with no per-frame closure
marking, so it has no eye state to teach. Rather than train the wrong head on a
guessed label, each target carries a validity mask and a corpus declares only
what it actually annotates. OmniLoader fills the rest with a placeholder and an
all-``False`` mask, and the loss skips those positions — see
:mod:`blinklinmult.train.losses`.

CEW and MRL-Eye declare eye state alone, but for a different reason: on a still
image a blink is not a motion event, it is a closed eye, so the two tasks
coincide and one label answers both. Those corpora train the frame-wise model,
whose single classifier reads that label directly.

**Features carry their own masks, independently.** A frame whose eye crop is
perfectly usable may still yield no reliable handcrafted descriptors — the iris
landmarker can miss, or return low-confidence points on a blurred or
half-occluded eye. :data:`EYE_IMAGE` and :data:`EYE_FEATURE` therefore have
separate validity masks and the code never assumes they agree.

**Images travel as images.** ``eye_image`` is declared to OmniLoader with a
structured trailing ``shape=(C, H, W)`` rather than a flattened ``feature_dim``,
which needs **omniloader >= 1.1**. Before that the crops had to be stored as a
wide feature axis and reshaped at the model boundary; the schema then described a
12288-wide feature instead of an image, and nothing checked that the width
factorised correctly.

**Why this is not the PersonalityLinMulT layout.** That project stores one
columnar ``(N, T, D)`` array per feature and slices row ``i`` out of it. Here the
HDF5 is read by :class:`omniloader.HDF5Dataset`, whose contract is one *group per
sample* under a split group — ``/{subset}/{sample_id}/{key}``. The builder in
:mod:`blinklinmult.data.builder` writes that layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any

SCHEMA_VERSION = 3
"""Bumped whenever the on-disk HDF5 layout changes incompatibly.

Version 3 unifies the annotation vocabulary. Every corpus now supplies the same
per-frame **eye-state score** (:data:`EYE_STATE`), from which blink events are
derived; head pose becomes a named field rather than three anonymous dimensions
buried in :data:`EYE_FEATURE`; and every frame carries a per-eye confidence and
the id of the blink it belongs to.

Version 2 was the eye-wise layout: one sample per eye, with the eye axis removed
from every tensor and replaced by the :data:`EYE_SIDE` metadata field.
"""

SUBSETS: tuple[str, ...] = ("train", "valid", "test")
"""Dataset splits, in canonical order. These are the HDF5 top-level groups."""

DATASETS: tuple[str, ...] = (
    "talkingface",
    "rn15",
    "rn30",
    "cew",
    "mrl",
    "hust_lebw",
    "mpeblink",
)
"""Corpora this project supports.

RN is split by recording rate rather than kept as one corpus: the two rates are
reported separately in the benchmark, and a shared declaration could not carry
two different ``fps`` values. ``config/data/rn.yaml`` lists both for the
aggregate run.

RT-BENE / RT-GENE and ZJU are deliberately absent: they were used in the 1.x
paper but are not part of the v2 benchmark.
"""

VIDEO_DATASETS: tuple[str, ...] = (
    "talkingface",
    "rn15",
    "rn30",
    "hust_lebw",
    "mpeblink",
)
"""Corpora with a temporal axis, over which a blink is a multi-frame event.

These train the sequence models; see ``config/data/all.yaml``.
"""

IMAGE_DATASETS: tuple[str, ...] = ("cew", "mrl")
"""Still-image corpora: the ``T = 1`` case, where a closed eye is the label.

These train the frame-wise model, whose single classifier reads eye state
directly. A sequence model given one of them would be modelling a degenerate
one-frame window, which ``ExperimentConfig`` rejects.
"""

LEFT = "left"
"""Value of :data:`EYE_SIDE` for a left eye."""

RIGHT = "right"
"""Value of :data:`EYE_SIDE` for a right eye."""

UNKNOWN_EYE = "unknown"
"""Value of :data:`EYE_SIDE` when the corpus does not say which eye it supplied.

MRL-Eye is the case: it ships one eye per image without labelling the side.
Recorded honestly rather than guessed, so a per-side analysis can exclude it.
"""

EYE_SIDES: tuple[str, ...] = (LEFT, RIGHT, UNKNOWN_EYE)
"""Recognised values of the :data:`EYE_SIDE` metadata field."""

EYE_EMBEDDING = "eye_embedding"
"""Precomputed eye embedding, ``(T, D)``.

Not a stored corpus field: attached to a batch by the datamodule when a frozen
encoder makes the embedding constant across epochs, so the model can skip its
encoder pass. See :mod:`blinklinmult.data.embeddings`.
"""

EYE_IMAGE = "eye_image"
"""Feature key: one eye's cropped images, ``(T, C, H, W)``.

Stored and loaded in native image form. OmniLoader >= 1.1 lets a spec declare a
structured trailing ``shape``, so the array that is written is the array the
model receives — no flatten on the way in, no reshape on the way out.
"""

EYE_FEATURE = "eye_feature"
"""Feature key: one eye's per-frame handcrafted descriptors, ``(T, F)``.

The second modality of the published BlinkLinMulT: iris landmarks, iris
diameters, eyelid-pupil distances and eye-aspect-ratio from exordium, plus head
pose. A corpus without them declares the key absent rather than zero-filling,
and OmniLoader masks it out.
"""

BLINK_PRESENCE = "blink_presence"
"""Target key: per-frame blink label, ``(T,)``.

Stored per frame even though **blink presence detection is scored per window**,
because the model has one sequence head and the clip decision is a ``max`` over
its output. Keeping a single head is what makes the two tasks structurally
consistent: a window cannot come out "no eye ever closed" *and* "a blink
occurred", which two independent heads would permit.

A frame is a blink frame when its ``blink_id`` is not
:data:`~blinklinmult.preprocess.annotation.NO_BLINK`.
"""

EYE_STATE = "eye_state"
"""Target key: per-frame closed label for this sample's eye, ``(T,)``.

The eye-state recognition target, scored **per frame**.
"""

INVALID_BLINK = -2
"""Sentinel in :data:`BLINK_IDS`: this frame was never readable.

Distinct from :data:`NO_BLINK`, which asserts the eye was **open**. Without the
distinction an unseen frame and a confirmed open eye are the same value, and a
reader who forgets to cross-check ``eye_image_mask`` silently treats one as the
other. Making the field self-describing costs one integer.
"""

NO_BLINK = -1
"""``blink_id`` value meaning "this frame is not part of a blink".

Lives here rather than in :mod:`blinklinmult.preprocess.annotation` because the
builder and the HDF5 layout use it, and the data layer must not depend on the
preprocess package — which needs the heavyweight extraction stack and does not
ship in the wheel.
"""

BLINK_ID = "blink_id"
"""Per-sample field: which annotated blink a positive window was cut around.

Stored so a prediction can be traced to the event it was meant to detect, and so
the test protocol can assert that every annotated blink appears in exactly one
window. ``-1`` for a blink-free window.
"""

BLINK_IDS = "blink_ids"
"""Per-frame field: which annotated blink each frame belongs to.

The per-frame refinement of :data:`BLINK_ID`, which is one integer per window
and so cannot tell one long blink from two adjacent ones inside the same window.
That distinction is what event-level work needs: separating a double blink from
a single long closure, recognising an incomplete blink as its own event, and
supervising an event count rather than a frame label.

:data:`NO_BLINK` outside every event.

A **first-class per-frame field**, not a quality extra: quality fields are
written but never declared in the OmniLoader schema, so they never reach
training. It supervises nothing on its own -- it individuates the annotation
:data:`BLINK_PRESENCE` already carries -- but event-level work cannot run
without it.
"""

NO_CATEGORY = -1
"""Sentinel in :data:`BLINK_CATEGORY`: this frame has no category.

Either the frame is outside every blink, or the annotation gave no value -- 11
events across the corpus have ``None`` in the slot. The categories themselves
are non-negative, so the sentinel cannot collide with one.
"""

BLINK_CATEGORY = "blink_category"
"""MPEblink's undocumented third annotation field, per frame.

The corpus writes ``[start, end, category]`` and the category takes three
values, but the CVPR paper documents only the start and end frames, and the
authors' own converter reads elements ``[0]`` and ``[1]`` only. Measured over
8 072 segments the split is 83.8% / 6.5% / 9.8%, and it is **not** a duration
distinction: median lengths are 6, 7 and 7 frames.

Values across the whole corpus: **0** (15 083), **1** (1 210), **2** (1 406),
plus two anomalies written through as-is rather than corrected -- a single
**10** (``train/184`` person1, frames 284-292, almost certainly a typo) and 11
events with no value, which become :data:`NO_CATEGORY`.

Stored as a diagnostic rather than a target, because nothing is known about what
it supervises. Preserving it keeps the question answerable from the built corpus
-- 522 category-1 and 787 category-2 events can be inspected directly -- instead
of requiring the raw tree and a re-run.
"""

HEAD_POSE = "head_pose"
"""Per-frame field: ``(T, 3)`` head rotation in **degrees**, ``[yaw, pitch, roll]``.

Estimated once per face by 6DRepNet and shared by both eyes of a frame. Stored
as a named field in degrees rather than only as the trailing three dimensions of
:data:`EYE_FEATURE` divided by 90, which nothing decoded and no analysis could
read.

Yaw is what makes an eye **self-occluded**: past roughly 30-45 degrees the far
eye is behind the nose, and the crop shows something that is not an eye. That
rule is applied in the dataloader rather than baked in here, so the threshold
stays a hyperparameter -- see
:data:`~blinklinmult.data.stills.OCCLUSION_YAW`.

Absent on the still corpora, which have no face box to estimate pose from.
"""

HEAD_POSE_DIM = 3
"""Width of :data:`HEAD_POSE`: ``[yaw, pitch, roll]``."""

QUALITY_SIGNALS: tuple[str, ...] = (
    "eye_blur",
    "eye_exposure",
    "eye_contour_fit",
    "eye_jitter",
)
"""Per-frame quality signals, each ``(T,)`` in ``[0, 1]``, stored **separately**.

Deliberately not combined into one confidence score at build time. How to weigh
motion blur against foreshortening against a bad contour fit is an empirical
question, and the answer is only visible once the corpora are built and the
distributions can be inspected. A single number baked in here would freeze that
choice into tens of gigabytes; storing the components lets the dataloader decide
and lets the decision change without re-preprocessing.

``eye_blur``
    Sharpness, as Laplacian variance over intensity variance. Motion blur
    destroys the eyelid edge -- the one thing blink detection reads -- while
    every geometric check still passes.
``eye_exposure``
    Dynamic range. Catches near-black and blown-out patches.
``eye_contour_fit``
    Whether the fitted eyelid contour sits on real image structure or on flat
    skin. **The signal that catches the profile failure** where landmarks track
    an edge-on, invisible eye and every other check passes.
``eye_jitter``
    Box movement between frames in eye-widths. A detector losing lock produces
    misaligned crops that each look fine alone.

**Every signal describes one eye alone**, never a comparison against its
partner: a sample is one eye, so a signal read from the other would leak across
samples and would be undefined exactly when one eye is occluded.

All are computed at build time because they need the full-resolution crop and
the frame-level boxes, neither of which survives into the dataloader. Together
they enable curriculum learning: train on clean frames first, then admit the
rest. See :mod:`blinklinmult.preprocess.quality`.
"""

TASK_ESR = "eye_state"
"""Eye state recognition: is this eye closed in this frame?

Scored **frame-wise**, over every corpus. On the still-image corpora it is the
whole task: with no temporal axis, a closed eye is all a blink can mean, so this
protocol and blink presence detection ask the same question.
"""

TASK_BPD = "blink_presence"
"""Blink presence detection: does this window contain a blink?

Scored **window-wise**, over the video corpora, where a blink is a multi-frame
event: the corpora make it recoverable because every frame of one blink shares a
``blink_id``. On a still image the question degenerates to :data:`TASK_ESR` —
the window is one frame and a closed eye is the positive — so the still corpora
train the frame-wise model rather than being scored under this protocol.

More error-tolerant than frame-wise scoring, which is why it is the headline
protocol: the corpora do not annotate blink boundaries precisely, since a
closure begins before and ends after the frames marked as blinking.
"""

TASKS: tuple[str, ...] = (TASK_ESR, TASK_BPD)
"""The two evaluation protocols, which are distinct tasks over the same window."""

FEATURE_KEYS: tuple[str, ...] = (EYE_IMAGE, EYE_FEATURE)
"""Every feature key, in model input order."""

TARGET_KEYS: tuple[str, ...] = (BLINK_PRESENCE, EYE_STATE)
"""Every target key, in head order."""

SAMPLE_KEY = "key"
"""Metadata key holding a sample's unique identifier.

Named ``key`` rather than ``sample_id`` because that is what OmniLoader's
:data:`omniloader.schema.unify.METADATA_KEYS` copies verbatim through the
unifier; a differently-named field would be dropped before it reached the batch,
and every prediction would lose its provenance.
"""

SOURCE_KEY = "dataset"
"""Metadata key holding the corpus a sample came from.

Also passed through by OmniLoader, and what
:class:`~blinklinmult.train.callbacks.PerDatasetReport` groups the test scores
by.
"""

EYE_SIDE = "eye_side"
"""Per-sample field naming which eye this sample carries.

Stored in the HDF5 for offline inspection and written into the sample id, since
OmniLoader only forwards :data:`SAMPLE_KEY`, :data:`SOURCE_KEY` and ``subset``
into the batch. :func:`parse_sample_id` recovers it at evaluation time.
"""

FRAME_GROUP = "frame_group"
"""Per-sample field identifying the frames a window covers.

Two eye-wise samples describe the same frames of the same recording exactly when
their frame groups match, which is what lets
:class:`~blinklinmult.train.metrics.FrameAggregator` recombine them into one
prediction per frame.
"""

FACE_BOX = "face_box"
"""Per-frame face box, ``(T, 4)`` int32 as ``(x, y, w, h)``.

**Stored instead of the face crop itself.** A face stream at 112 px would add
~47 GB to MPEblink alone -- a 4x blowup -- and because a sample is one *eye*,
the ``|left`` and ``|right`` samples of a window would hold byte-identical
copies of it. The box is four numbers and reconstructs the crop exactly.

What it buys is reproducibility: the geometry is frozen at build time, so a
face crop taken later from the raw video is the same crop the builder saw,
rather than a fresh guess at the same preprocessing. Negative or overhanging
values are kept verbatim -- MPEblink tracks faces past the frame edge, and
clamping here would destroy that information.

``None`` for corpora that annotate no face box (CEW, MRL-Eye).
"""

EYE_ON_SCREEN = "eye_on_screen"
"""Per-frame fraction of an eye's landmarks inside the frame, ``(T,)`` float16.

``1.0`` is fully visible. The builder masks a frame below
:data:`~blinklinmult.preprocess.geometry.MIN_EYE_ON_SCREEN`, but the raw value
is kept so a marginal eye can be found and looked at rather than merely
trusted.
"""

EYE_SPAN = "eye_span"
"""Per-frame eye width in pixels, corner to corner, ``(T,)`` float16.

The crop's scale before resizing, so it says how much real detail a 64 px patch
holds: a 12 px eye upsampled to 64 is mostly interpolation. Films put faces at
every scale, so this varies by an order of magnitude within one corpus.
"""

EYE_ASPECT = "eye_aspect"
"""Per-frame eye height/width ratio, ``(T,)`` float16.

**The profile detector.** A face turned edge-on keeps a healthy eye *span* --
``test/183`` measures a perfectly ordinary 21.5 px median -- while the eye
itself becomes invisible, so span alone cannot find those clips. The landmark
contour flattens as the face rotates away, which this catches.

Low values also occur legitimately during a blink, so it is a ranking signal
for inspection, not a rejection rule.
"""

BOX_SHIFTED = "box_shifted"
"""Per-frame flag, ``(T,)`` bool: the crop was slid to fit inside the frame.

Set by :meth:`~blinklinmult.preprocess.geometry.EyeBox.shifted_into`. The eye is
present and the scale is unchanged, but it sits off centre, so a model that
learned to expect a centred eye may behave differently here. 0.46% of MPEblink
crops.
"""

LANDMARK_SOURCE = "landmark_source"
"""Per-sample string naming which landmark scheme located the eye.

``"wflw"`` (98-point) or ``"ibug"`` (68-point). MPEblink ships WFLW for most
clips but only iBUG for 30 of its test clips, and the two place the eye
slightly differently -- iBUG has no pupil point, so those crops centre on the
contour mean. Worth being able to score the two groups separately before
trusting a single test number.
"""

CONFIDENCE = "confidence"
"""Per-sample scalar in ``[0, 1]``, float32: how trustworthy this window looks.

A ranking aid for finding bad data, **not** a training weight and not a
probability. Pooled from the per-frame signals above so a notebook can sort
tracklets and render the worst; see
:func:`~blinklinmult.preprocess.quality.window_confidence`.

It exists because the automated checks are not sufficient on their own: the
``test/183`` profile crops scored *higher* contrast than good clips and were
still unusable. Only looking at them revealed it, and this is what makes
looking cheap.
"""

METADATA_KEYS: tuple[str, ...] = (SAMPLE_KEY, SOURCE_KEY, "subset")
"""Per-sample metadata that survives OmniLoader's unifier into the batch.

These three are exactly the keys OmniLoader copies verbatim; anything else is
dropped, which is why the eye side and the frame group are encoded into the
sample id rather than stored only as separate fields.
"""

TARGET_PLACEHOLDER = -1.0
"""Value filled in where a corpus does not supply a target.

Chosen outside ``[0, 1]`` so that a placeholder surviving into a metric is
visibly wrong rather than plausible. The mask is what the loss actually reads;
this is the second line of defence.
"""

IMAGE_CHANNELS = 3
"""Channel count of an eye crop. RGB, matching the ImageNet-pretrained backbones."""

DEFAULT_WINDOW_SECONDS = 1.5
"""Default analysis window, in seconds.

Matched to MPEblink, whose InstBlink++ runs inference on **36-frame clips with
a stride of 18** — a 50% overlap at ~24-30 fps, so roughly 1.2-1.5 s. Adopting
the same duration is what lets this project's numbers be read beside theirs.

**Not chosen to contain the blink.** Measured over MPEblink 2.0's 17 711
annotated events, the median blink is 6 frames and the 95th percentile 12, so a
15-frame window already contains 97.9% of them and 45 frames raises that only to
99.8%. The window is for *context*: what separates a blink from a downward
glance, a squint or motion blur is what the eye was doing before and after the
closure, not the closure itself.

1.x used 0.5 s (15 frames at 30 fps). Widening it costs preprocessing time and
memory in proportion, and no published ablation measures what the extra context
buys -- InstBlink++'s report asserts its clip length without one. Treat this as
a protocol choice for comparability, not as an established optimum.
"""

EYE_FEATURE_DIM = 160
"""Width of the handcrafted descriptor vector for one eye.

157 from exordium's :meth:`IrisWrapper.eye_to_feature` — 71x2 eye-region
landmarks, 5x2 iris landmarks, 2 iris diameters, 2 eyelid-pupil distances and 1
eye-aspect-ratio — plus 3 head-pose angles. See
:mod:`blinklinmult.preprocess.features`.
"""


class SchemaError(ValueError):
    """Raised when a dataset declaration is invalid or self-inconsistent."""


def frames_for(fps: float, window_seconds: float) -> int:
    """Frames spanning a duration at a given rate.

    Rounds **half up** rather than using :func:`round`, whose banker's rounding
    breaks an exact ``.5`` towards the even integer — so 25 fps over 0.5 s would
    give 12 frames while 30 fps over 0.5 s gives 15, and the shortfall would be
    invisible. Rounding up never gives a window fewer frames than its duration
    implies.

    Args:
        fps (float): Frame rate of the recording.
        window_seconds (float): Window duration.

    Returns:
        int: Frame count, at least 1.

    Raises:
        SchemaError: If either argument is not positive.
    """
    if fps <= 0:
        raise SchemaError(f"fps must be positive, got {fps}.")
    if window_seconds <= 0:
        raise SchemaError(f"window_seconds must be positive, got {window_seconds}.")
    return max(1, math.floor(fps * window_seconds + 0.5))


def build_sample_id(video_id: str, frame_group: str, eye_side: str) -> str:
    """Compose the identifier that carries a sample's provenance.

    OmniLoader forwards only :data:`SAMPLE_KEY`, :data:`SOURCE_KEY` and
    ``subset`` into a batch, so the eye side and the frame group have to travel
    inside the id to survive as far as the metrics.

    Args:
        video_id (str): Source recording.
        frame_group (str): Identifier of the frames the window covers, usually
            the first frame id.
        eye_side (str): One of :data:`EYE_SIDES`.

    Returns:
        str: e.g. ``"rn30_1|000881|left"``.

    Raises:
        SchemaError: If the eye side is unrecognised, or a component contains
            the ``|`` separator.
    """
    if eye_side not in EYE_SIDES:
        raise SchemaError(f"Unknown eye side {eye_side!r}; expected one of {list(EYE_SIDES)}.")
    for part, name in ((video_id, "video_id"), (frame_group, "frame_group")):
        if "|" in part:
            raise SchemaError(
                f"{name}={part!r} contains the '|' separator, which would make the "
                "sample id ambiguous."
            )
    return f"{video_id}|{frame_group}|{eye_side}"


def parse_sample_id(sample_id: str) -> tuple[str, str, str]:
    """Recover a sample's provenance from its identifier.

    Args:
        sample_id (str): An id built by :func:`build_sample_id`.

    Returns:
        tuple[str, str, str]: ``(video_id, frame_group, eye_side)``.

    Raises:
        SchemaError: If the id does not have the three expected components.
    """
    parts = sample_id.split("|")
    if len(parts) != 3:
        raise SchemaError(
            f"Sample id {sample_id!r} does not have the expected "
            "'<video_id>|<frame_group>|<eye_side>' form."
        )
    return parts[0], parts[1], parts[2]


@dataclass(frozen=True)
class DatasetSpec:
    """What one corpus provides, and in what shape.

    This is the single source of truth for a corpus: the builder writes what it
    declares, the OmniLoader schema is generated from it, and training reads it
    to know which heads that corpus can supervise. A corpus cannot be written
    under one interpretation and read under another.

    Args:
        name (str): Corpus name; must be one of :data:`DATASETS`. Becomes the
            ``data/processed/<name>`` directory and the HDF5 filename.
        fps (float | None): Native frame rate, or ``None`` for a still-image
            corpus. Together with the run's window length in seconds this
            determines :attr:`time_dim`, so that every corpus spans the same
            real duration whatever its rate.
        window_seconds (float | None): Analysis window for this corpus. ``None``
            means the corpus is not windowed (still images).
        image_size (int): Side length of a square eye crop.
        feature_dim (int | None): Width of :data:`EYE_FEATURE`, or ``None`` when
            the corpus supplies no handcrafted descriptors.
        has_blink_presence (bool): Whether the corpus annotates blink presence.
        has_eye_state (bool): Whether the corpus annotates per-eye closure.
        has_eye_side (bool): Whether the corpus says which eye a sample is.
            ``False`` for MRL-Eye, whose samples are recorded as
            :data:`UNKNOWN_EYE`.
        has_head_pose (bool): Whether the corpus supplies :data:`HEAD_POSE`.
            ``False`` for the still corpora, which have no face box to estimate
            pose from -- and so cannot be occlusion-filtered.
        has_blink_ids (bool): Whether the corpus supplies per-frame
            :data:`BLINK_IDS`, which event-level work needs to tell one long
            blink from two adjacent ones.
        quality_signals (tuple[str, ...]): Which of :data:`QUALITY_SIGNALS` this
            corpus supplies. Deliberately per-signal rather than all-or-nothing:
            a still-image corpus has one frame and so no
            :data:`~blinklinmult.preprocess.quality.box_jitter`. Empty means the
            corpus supplies none.

    Raises:
        SchemaError: If the declaration is invalid.
    """

    name: str
    fps: float | None = None
    window_seconds: float | None = DEFAULT_WINDOW_SECONDS
    image_size: int = 64
    feature_dim: int | None = None
    has_blink_presence: bool = False
    has_eye_state: bool = False
    has_eye_side: bool = True
    has_head_pose: bool = False
    has_blink_ids: bool = False
    quality_signals: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the declaration.

        Raises:
            SchemaError: If a field is out of range, the name is unknown, or the
                corpus annotates neither task.
        """
        if self.name not in DATASETS:
            raise SchemaError(f"Unknown dataset {self.name!r}; expected one of {sorted(DATASETS)}.")
        if self.image_size < 1:
            raise SchemaError(f"{self.name}: image_size must be >= 1, got {self.image_size}.")
        if self.feature_dim is not None and self.feature_dim < 1:
            raise SchemaError(
                f"{self.name}: feature_dim must be positive or None, got {self.feature_dim}."
            )
        if self.fps is not None and self.fps <= 0:
            raise SchemaError(f"{self.name}: fps must be positive or None, got {self.fps}.")
        unknown = [name for name in self.quality_signals if name not in QUALITY_SIGNALS]
        if unknown:
            raise SchemaError(
                f"{self.name}: unknown quality signals {unknown}; "
                f"expected a subset of {list(QUALITY_SIGNALS)}."
            )
        if self.window_seconds is not None and self.window_seconds <= 0:
            raise SchemaError(
                f"{self.name}: window_seconds must be positive or None, got {self.window_seconds}."
            )
        if not (self.has_blink_presence or self.has_eye_state):
            raise SchemaError(
                f"{self.name}: declares neither blink presence nor eye state. A corpus "
                "that supervises no task cannot contribute to training."
            )
        if self.has_blink_presence and self.fps is None:
            raise SchemaError(
                f"{self.name}: blink presence is a window-level target, but the corpus "
                "declares no fps and so has no window. On a still image a blink is a "
                "closed eye -- annotate it as eye state, which the frame-wise model "
                "reads directly."
            )
        if self.is_video and self.window_seconds is None:
            raise SchemaError(
                f"{self.name}: declares an fps but no window_seconds, so its window "
                "length is undefined."
            )

    @property
    def is_video(self) -> bool:
        """Whether this corpus has a temporal axis.

        Returns:
            bool: ``True`` when the corpus declares a frame rate.
        """
        return self.fps is not None

    @property
    def time_dim(self) -> int:
        """Frames spanned by this corpus's analysis window.

        Derived from the corpus's own rate rather than fixed, so a 15 fps and a
        30 fps recording cover the same wall-clock duration. A still-image
        corpus is the degenerate single-frame case.

        Returns:
            int: Window length in frames; ``1`` for a still-image corpus.
        """
        if self.fps is None or self.window_seconds is None:
            return 1
        return frames_for(self.fps, self.window_seconds)

    @property
    def has_quality_signals(self) -> bool:
        """Whether this corpus supplies any per-frame quality signal.

        Returns:
            bool: ``True`` when :attr:`quality_signals` is non-empty.
        """
        return bool(self.quality_signals)

    @property
    def has_eye_feature(self) -> bool:
        """Whether this corpus supplies handcrafted eye descriptors.

        Returns:
            bool: ``True`` when :attr:`feature_dim` is set.
        """
        return self.feature_dim is not None

    @property
    def feature_keys(self) -> list[str]:
        """Feature keys this corpus provides, in model input order.

        Returns:
            list[str]: A subset of :data:`FEATURE_KEYS`.
        """
        keys = [EYE_IMAGE]
        if self.has_eye_feature:
            keys.append(EYE_FEATURE)
        return keys

    @property
    def target_keys(self) -> list[str]:
        """Target keys this corpus annotates, in head order.

        Returns:
            list[str]: A subset of :data:`TARGET_KEYS`.
        """
        keys = []
        if self.has_blink_presence:
            keys.append(BLINK_PRESENCE)
        if self.has_eye_state:
            keys.append(EYE_STATE)
        return keys

    def with_window(self, window_seconds: float | None) -> DatasetSpec:
        """Return this spec with a different analysis window.

        Used by a run config to apply one window length across every corpus,
        each of which then re-derives its own frame count.

        Args:
            window_seconds (float | None): The new window length.

        Returns:
            DatasetSpec: A new spec; this one is unchanged.
        """
        if window_seconds is None or not self.is_video:
            return self
        values = {f.name: getattr(self, f.name) for f in fields(self)}
        values["window_seconds"] = window_seconds
        return DatasetSpec(**values)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetSpec:
        """Build a spec from a plain dict, e.g. parsed YAML.

        Args:
            data (dict): Declaration values.

        Returns:
            DatasetSpec: The validated spec.

        Raises:
            SchemaError: If keys are unknown or required ones are missing.
        """
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise SchemaError(
                f"{data.get('name', '<unnamed>')}: unknown keys {unknown}. "
                f"Known keys: {sorted(known)}."
            )
        if "name" not in data:
            raise SchemaError("Dataset declaration is missing 'name'.")
        return cls(**data)


@dataclass
class BuildStats:
    """Counters describing one builder run, for logging and tests.

    Args:
        dataset (str): Corpus that was built.
        per_subset (dict[str, int]): Samples written per split.
        bytes_written (int): Size of the resulting file.
        positive_fraction (dict[str, float]): Fraction of blink windows per
            split, which is what reveals a split that accidentally lost its
            positives.
    """

    dataset: str = ""
    per_subset: dict[str, int] = field(default_factory=dict)
    bytes_written: int = 0
    positive_fraction: dict[str, float] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        """Total samples written across every split.

        Returns:
            int: Sample count.
        """
        return sum(self.per_subset.values())
